# -*- coding: utf-8 -*-
"""
批量并行贝叶斯优化引擎
=====================

核心流程：
    [串行] TPE / 正态分布采样生成 N 组参数
        ↓
    [真多进程] ProcessPoolExecutor 并行跑所有回测
        ↓
    [串行] 按原始顺序 tell 回 study，更新 TPE 模型
        ↓
    循环直到 total trials 跑完

相比原项目 `study.optimize(..., n_jobs=N)` 的多线程方案，
本引擎使用 ProcessPoolExecutor 真多进程 + Constant Liar 补偿，
在 128 核服务器上可将 CPU 利用率从约 10% 提升到约 90%。
"""

from __future__ import annotations

import os
import sys
import time
import inspect
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)

# 添加当前目录到 Python path，保证子进程 import 时能找到 Optimizer 模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ==============================================================================
# 1. 子进程全局状态（每个 worker 进程启动时由 initializer 填充一次）
# ==============================================================================

_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(init_payload: Dict[str, Any]) -> None:
    """
    子进程初始化：只读一次数据、加载一次策略、构建一次 BacktestEngine。

    所有大对象放到 `_WORKER_STATE` 全局字典，`_worker_run` 直接使用，
    避免每个 trial 都 pickle 传递。

    Args:
        init_payload: 从主进程传入的初始化参数，包含数据路径、策略路径、经纪商配置等。
    """
    import importlib.util
    import pandas as pd
    import backtrader as bt
    import warnings
    warnings.filterwarnings("ignore")

    # 延迟导入，避免在主进程 import parallel_engine 时就触发重依赖
    from backtest_engine import BacktestEngine

    data_paths: List[str] = init_payload["data_paths"]
    strategy_path: str = init_payload["strategy_path"]
    objective: str = init_payload["objective"]
    broker_config = init_payload.get("broker_config")
    market_maker_config = init_payload.get("market_maker_config")
    data_frequency: Optional[str] = init_payload.get("data_frequency")
    initial_cash: float = init_payload.get("initial_cash", 100000.0)
    commission: float = init_payload.get("commission", 0.001)
    data_names: Optional[List[str]] = init_payload.get("data_names")
    is_multi_data: bool = init_payload.get("is_multi_data", False)

    # --- 加载数据 ---
    def _load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            for alt in ("date", "time_key", "time"):
                if alt in df.columns:
                    df.rename(columns={alt: "datetime"}, inplace=True)
                    break
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    if is_multi_data:
        data_obj = [_load_csv(p) for p in data_paths]
    else:
        data_obj = _load_csv(data_paths[0])

    # --- 动态加载策略 ---
    stem = os.path.splitext(os.path.basename(strategy_path))[0]
    module_name = f"strategy_module_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # 找策略类
    strategy_class = None
    custom_data_class = None
    custom_commission_class = None
    for name, obj in inspect.getmembers(module):
        if not inspect.isclass(obj):
            continue
        if obj.__module__ != module_name:
            continue
        if hasattr(obj, "params") and issubclass(obj, bt.Strategy):
            if strategy_class is None:
                strategy_class = obj
        elif issubclass(obj, bt.feeds.PandasData) and obj is not bt.feeds.PandasData:
            custom_data_class = obj
        elif issubclass(obj, bt.CommInfoBase) and obj is not bt.CommInfoBase:
            custom_commission_class = obj

    if strategy_class is None:
        raise RuntimeError(f"子进程未能在 {strategy_path} 中找到 bt.Strategy 子类")

    # 是否使用 trade_log 指标
    try:
        src = inspect.getsource(strategy_class)
        use_trade_log = "trade_log" in src
    except Exception:
        use_trade_log = False

    # strategy 是否接受 verbose
    try:
        sig = inspect.signature(strategy_class.__init__)
        accepts_verbose = "verbose" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except Exception:
        accepts_verbose = False

    # --- 构建 BacktestEngine（每个 worker 一份，只构建一次）---
    effective_freq = None if data_frequency in (None, "auto") else data_frequency
    engine = BacktestEngine(
        data=data_obj,
        strategy_class=strategy_class,
        initial_cash=initial_cash,
        commission=commission,
        data_frequency=effective_freq,
        custom_data_class=custom_data_class,
        custom_commission_class=custom_commission_class,
        strategy_module=module,
        use_trade_log_metrics=use_trade_log,
        broker_config=broker_config,
        market_maker_config=market_maker_config,
    )

    _WORKER_STATE.update(
        {
            "engine": engine,
            "strategy_class": strategy_class,
            "objective": objective,
            "accepts_verbose": accepts_verbose,
            "data_names": data_names,
        }
    )


# ==============================================================================
# 2. 子进程执行函数（必须是模块顶层函数，才能被 pickle）
# ==============================================================================

def _worker_run(task: Tuple[int, Dict[str, Any]]) -> Tuple[int, float, Optional[Dict[str, Any]], Optional[str]]:
    """
    在子进程内跑一次回测。

    Args:
        task: (trial_index, params_dict)

    Returns:
        (trial_index, score, metrics_dict, error_msg)
        失败时 score = -inf，metrics_dict = None，error_msg 含异常信息。
    """
    trial_index, params = task
    try:
        engine = _WORKER_STATE["engine"]
        strategy_class = _WORKER_STATE["strategy_class"]
        objective = _WORKER_STATE["objective"]
        accepts_verbose = _WORKER_STATE["accepts_verbose"]
        data_names = _WORKER_STATE.get("data_names")

        run_params = dict(params)
        if accepts_verbose:
            run_params["verbose"] = False

        result = engine.run_backtest(
            strategy_class=strategy_class,
            params=run_params,
            data_names=data_names,
        )
        if result is None:
            return trial_index, float("-inf"), None, "backtest returned None"

        score = engine.evaluate_objective(result, objective)
        metrics = {
            "sharpe_ratio": float(getattr(result, "sharpe_ratio", 0.0) or 0.0),
            "annual_return": float(getattr(result, "annual_return", 0.0) or 0.0),
            "max_drawdown": float(getattr(result, "max_drawdown", 0.0) or 0.0),
            "total_return": float(getattr(result, "total_return", 0.0) or 0.0),
            "trades_count": int(getattr(result, "trades_count", 0) or 0),
            "win_rate": float(getattr(result, "win_rate", 0.0) or 0.0),
        }
        return trial_index, float(score), metrics, None
    except Exception as exc:  # noqa: BLE001
        return trial_index, float("-inf"), None, f"{type(exc).__name__}: {exc}"


# ==============================================================================
# 3. 主进程：批量并行贝叶斯优化器
# ==============================================================================

@dataclass
class WorkerInitArgs:
    """传给子进程 initializer 的打包参数（主进程构造，可 pickle）。"""
    data_paths: List[str]
    strategy_path: str
    objective: str
    broker_config: Any = None
    market_maker_config: Any = None
    data_frequency: Optional[str] = None
    initial_cash: float = 100000.0
    commission: float = 0.001
    data_names: Optional[List[str]] = None
    is_multi_data: bool = False

    def as_payload(self) -> Dict[str, Any]:
        return {
            "data_paths": list(self.data_paths),
            "strategy_path": self.strategy_path,
            "objective": self.objective,
            "broker_config": self.broker_config,
            "market_maker_config": self.market_maker_config,
            "data_frequency": self.data_frequency,
            "initial_cash": self.initial_cash,
            "commission": self.commission,
            "data_names": list(self.data_names) if self.data_names else None,
            "is_multi_data": self.is_multi_data,
        }


@dataclass
class BatchTrialRecord:
    """一个 trial 的完整记录（主进程保留，用于生成 history / best_improvements）。"""
    trial_number: int
    phase: str  # 'exploration' | 'exploitation'
    params: Dict[str, Any]
    score: float
    metrics: Optional[Dict[str, Any]]
    error: Optional[str] = None


def _build_distributions(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    将项目内部的 SearchSpaceConfig 字典转为 optuna.distributions。

    search_space 的 value 是 strategy_analyzer.SearchSpaceConfig，含字段：
        param_type: 'int' | 'float'
        distribution: 'int_uniform' | 'uniform' | 'log_uniform'
        min_value, max_value, step
    """
    distributions: Dict[str, Any] = {}
    for name, cfg in search_space.items():
        if cfg.param_type == "int":
            step = int(cfg.step) if cfg.step else 1
            distributions[name] = IntDistribution(
                low=int(cfg.min_value),
                high=int(cfg.max_value),
                step=step,
            )
        elif cfg.param_type == "float":
            if cfg.distribution == "log_uniform":
                distributions[name] = FloatDistribution(
                    low=float(cfg.min_value),
                    high=float(cfg.max_value),
                    log=True,
                )
            else:
                step = float(cfg.step) if cfg.step else None
                distributions[name] = FloatDistribution(
                    low=float(cfg.min_value),
                    high=float(cfg.max_value),
                    step=step,
                )
        elif cfg.param_type == "bool":
            distributions[name] = CategoricalDistribution([True, False])
        else:
            raise ValueError(f"Unsupported param_type: {cfg.param_type}")
    return distributions


def create_parallel_study(
    direction: str = "maximize",
    seed: int = 42,
    n_startup_trials: int = 20,
    constant_liar: bool = True,
) -> optuna.Study:
    """
    创建一个适合批量并行的 optuna Study。

    - constant_liar=True: 对 batch 内尚未返回结果的 trial，TPE 用悲观估计占位，
      防止同一批内扎堆采到相同位置。
    - n_startup_trials=20: 前 20 个 trial 纯随机，积累样本后再启动 TPE 建模。
    """
    sampler = optuna.samplers.TPESampler(
        constant_liar=constant_liar,
        n_startup_trials=n_startup_trials,
        seed=seed,
    )
    return optuna.create_study(direction=direction, sampler=sampler)


class BatchParallelOptimizer:
    """
    批量并行贝叶斯优化器。

    使用方式：
        engine = BatchParallelOptimizer(
            study=study,
            search_space=search_space,
            worker_init_args=WorkerInitArgs(...),
            n_exploit_trials=exploit_trials,
            exploration_samples=samples,       # 可选：预生成的正态分布探索样本
            n_workers=120,
            explore_batch_size=128,
            exploit_batch_size=32,
            timeout_per_trial=300,
        )
        best_params, best_value, best_metrics = engine.run()

    设计要点见模块顶部的 docstring。
    """

    def __init__(
        self,
        study: optuna.Study,
        search_space: Dict[str, Any],
        worker_init_args: WorkerInitArgs,
        n_exploit_trials: int,
        exploration_samples: Optional[List[Dict[str, Any]]] = None,
        n_workers: int = 120,
        explore_batch_size: int = 128,
        exploit_batch_size: int = 32,
        timeout_per_trial: int = 300,
        verbose: bool = True,
        on_new_best: Optional[Callable[[BatchTrialRecord], None]] = None,
    ) -> None:
        self.study = study
        self.search_space = search_space
        self.distributions = _build_distributions(search_space)
        self.worker_init_args = worker_init_args
        self.n_exploit_trials = max(0, int(n_exploit_trials))
        self.exploration_samples = list(exploration_samples or [])
        self.n_workers = max(1, int(n_workers))
        self.explore_batch_size = max(1, int(explore_batch_size))
        self.exploit_batch_size = max(1, int(exploit_batch_size))
        self.timeout_per_trial = int(timeout_per_trial)
        self.verbose = verbose
        self.on_new_best = on_new_best

        # 运行期累积
        self.history: List[BatchTrialRecord] = []
        self.best_improvements: List[Dict[str, Any]] = []
        self._best_score: float = float("-inf")
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_metrics: Optional[Dict[str, Any]] = None
        self._global_trial_counter = 0

    # ------------------------------------------------------------------ public

    def run(self) -> Tuple[Dict[str, Any], float, Optional[Dict[str, Any]]]:
        """执行完整的批量并行优化，返回 (best_params, best_value, best_metrics)。"""
        total_explore = len(self.exploration_samples)
        total_exploit = self.n_exploit_trials
        total = total_explore + total_exploit

        if self.verbose:
            print(f"\n╔{'═'*78}╗")
            print(f"║ {'[BatchParallelOptimizer] 启动'.center(74)} ║")
            print(f"╠{'═'*78}╣")
            print(f"║ 总 trials: {total:<64} ║")
            print(f"║   • 探索阶段: {total_explore:<60} ║")
            print(f"║   • 利用阶段: {total_exploit:<60} ║")
            print(f"║ 并行进程数: {self.n_workers:<63} ║")
            print(f"║ 探索批大小: {self.explore_batch_size:<63} ║")
            print(f"║ 利用批大小: {self.exploit_batch_size:<63} ║")
            print(f"║ 单 trial 超时: {self.timeout_per_trial:<59}s ║")
            print(f"╚{'═'*78}╝")

        t_start = time.time()

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker,
            initargs=(self.worker_init_args.as_payload(),),
        ) as executor:
            if total_explore > 0:
                self._run_exploration(executor)
            if total_exploit > 0:
                self._run_exploitation(executor)

        elapsed = time.time() - t_start

        if self.verbose:
            print(f"\n╔{'═'*78}╗")
            print(f"║ {'[BatchParallelOptimizer] 完成'.center(74)} ║")
            print(f"╠{'═'*78}╣")
            print(f"║ 总 trials: {len(self.history):<64} ║")
            failed = sum(1 for r in self.history if r.score == float('-inf'))
            print(f"║ 失败 trials: {failed:<62} ║")
            print(f"║ 总耗时: {elapsed:<65.2f}s ║")
            print(f"║ 最优分数: {self._best_score:<63.4f} ║")
            print(f"╚{'═'*78}╝")

        best_params = self._best_params or {}
        return best_params, self._best_score, self._best_metrics

    # -------------------------------------------------------------- internals

    def _run_exploration(self, executor: ProcessPoolExecutor) -> None:
        """按 explore_batch_size 分批、并行评估预生成的探索样本。"""
        samples = self.exploration_samples
        n_batches = (len(samples) + self.explore_batch_size - 1) // self.explore_batch_size

        for batch_idx in range(n_batches):
            t_batch = time.time()
            start = batch_idx * self.explore_batch_size
            end = min(start + self.explore_batch_size, len(samples))
            batch_params = samples[start:end]

            tasks = []
            trial_numbers = []
            for params in batch_params:
                trial_num = self._global_trial_counter
                self._global_trial_counter += 1
                trial_numbers.append(trial_num)
                tasks.append((trial_num, params))

            results = self._execute_batch(executor, tasks)

            # 按原始顺序 tell 回 study（探索阶段也记录到 study，方便 TPE 冷启动样本）
            for (tn, params), (_, score, metrics, error) in zip(zip(trial_numbers, batch_params), results):
                try:
                    clipped = self._clip_to_space(params)
                    if score == float("-inf"):
                        trial = optuna.trial.create_trial(
                            params=clipped,
                            distributions=self.distributions,
                            value=None,
                            state=optuna.trial.TrialState.FAIL,
                        )
                    else:
                        trial = optuna.trial.create_trial(
                            params=clipped,
                            distributions=self.distributions,
                            value=score,
                            state=optuna.trial.TrialState.COMPLETE,
                        )
                    self.study.add_trial(trial)
                except Exception:
                    pass

                record = BatchTrialRecord(
                    trial_number=tn,
                    phase="exploration",
                    params=dict(params),
                    score=score,
                    metrics=metrics,
                    error=error,
                )
                self.history.append(record)
                self._maybe_update_best(record)

            self._print_batch_summary(
                phase="Explore",
                batch_idx=batch_idx + 1,
                n_batches=n_batches,
                batch_size=len(batch_params),
                elapsed=time.time() - t_batch,
                results=results,
            )

    def _run_exploitation(self, executor: ProcessPoolExecutor) -> None:
        """TPE 采样 → 并行回测 → tell 更新模型。"""
        remaining = self.n_exploit_trials
        batch_idx = 0
        n_batches = (remaining + self.exploit_batch_size - 1) // self.exploit_batch_size

        while remaining > 0:
            batch_idx += 1
            t_batch = time.time()
            this_batch = min(self.exploit_batch_size, remaining)

            trials: List[optuna.trial.Trial] = []
            batch_params: List[Dict[str, Any]] = []
            trial_numbers: List[int] = []

            for _ in range(this_batch):
                trial = self.study.ask(self.distributions)
                params = dict(trial.params)
                trials.append(trial)
                batch_params.append(params)
                trial_numbers.append(self._global_trial_counter)
                self._global_trial_counter += 1

            tasks = list(zip(trial_numbers, batch_params))
            results = self._execute_batch(executor, tasks)

            # 严格按原始顺序 tell，保证结果顺序确定性
            for trial, (tn, params), (_, score, metrics, error) in zip(
                trials, zip(trial_numbers, batch_params), results
            ):
                try:
                    if score == float("-inf"):
                        self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    else:
                        self.study.tell(trial, score)
                except Exception:
                    # tell 失败不应中断整个优化
                    pass

                record = BatchTrialRecord(
                    trial_number=tn,
                    phase="exploitation",
                    params=params,
                    score=score,
                    metrics=metrics,
                    error=error,
                )
                self.history.append(record)
                self._maybe_update_best(record)

            self._print_batch_summary(
                phase="Exploit",
                batch_idx=batch_idx,
                n_batches=n_batches,
                batch_size=this_batch,
                elapsed=time.time() - t_batch,
                results=results,
            )
            remaining -= this_batch

    def _execute_batch(
        self,
        executor: ProcessPoolExecutor,
        tasks: List[Tuple[int, Dict[str, Any]]],
    ) -> List[Tuple[int, float, Optional[Dict[str, Any]], Optional[str]]]:
        """
        提交一个 batch 到进程池，带超时保护，按原始顺序返回。
        """
        futures = [executor.submit(_worker_run, task) for task in tasks]
        results: List[Optional[Tuple[int, float, Optional[Dict[str, Any]], Optional[str]]]] = [None] * len(tasks)

        for idx, fut in enumerate(futures):
            trial_index = tasks[idx][0]
            try:
                results[idx] = fut.result(timeout=self.timeout_per_trial)
            except FutureTimeoutError:
                try:
                    fut.cancel()
                except Exception:
                    pass
                if self.verbose:
                    print(f"  ⚠️  Trial {trial_index} 超时 (>{self.timeout_per_trial}s)，记为 -inf")
                results[idx] = (trial_index, float("-inf"), None, "timeout")
            except Exception as exc:
                if self.verbose:
                    print(f"  ⚠️  Trial {trial_index} 异常: {exc}")
                results[idx] = (trial_index, float("-inf"), None, f"executor_error: {exc}")

        return results  # type: ignore[return-value]

    def _clip_to_space(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """裁剪参数到 search_space 的合法范围，供 create_trial 使用。"""
        clipped: Dict[str, Any] = {}
        for name, dist in self.distributions.items():
            if name not in params:
                continue
            v = params[name]
            if isinstance(dist, IntDistribution):
                clipped[name] = int(max(dist.low, min(dist.high, int(round(v)))))
            elif isinstance(dist, FloatDistribution):
                clipped[name] = float(max(dist.low, min(dist.high, float(v))))
            else:
                clipped[name] = v
        return clipped

    def _maybe_update_best(self, record: BatchTrialRecord) -> None:
        if record.score <= self._best_score:
            return
        self._best_score = record.score
        self._best_params = dict(record.params)
        self._best_metrics = dict(record.metrics) if record.metrics else None

        if record.metrics:
            bt_results = {
                "sharpe_ratio": round(record.metrics.get("sharpe_ratio", 0.0), 4),
                "annual_return": round(record.metrics.get("annual_return", 0.0), 2),
                "max_drawdown": round(record.metrics.get("max_drawdown", 0.0), 2),
            }
            is_dup = (
                self.best_improvements
                and self.best_improvements[-1]["backtest_results"] == bt_results
            )
            if not is_dup:
                self.best_improvements.append(
                    {"params": dict(record.params), "backtest_results": bt_results}
                )

        if self.verbose:
            phase_cn = "利用阶段" if record.phase == "exploitation" else "探索阶段"
            print(
                f"  🎯 [Trial {record.trial_number}/{phase_cn}] 新最优 "
                f"score={record.score:.4f}"
                + (
                    f" sharpe={record.metrics['sharpe_ratio']:.3f}"
                    f" ann={record.metrics['annual_return']:.2f}%"
                    f" dd={record.metrics['max_drawdown']:.2f}%"
                    if record.metrics else ""
                )
            )

        if self.on_new_best:
            try:
                self.on_new_best(record)
            except Exception:
                pass

    def _print_batch_summary(
        self,
        phase: str,
        batch_idx: int,
        n_batches: int,
        batch_size: int,
        elapsed: float,
        results: List[Tuple[int, float, Optional[Dict[str, Any]], Optional[str]]],
    ) -> None:
        if not self.verbose:
            return
        failed = sum(1 for r in results if r[1] == float("-inf"))
        best_in_batch = max((r[1] for r in results if r[1] != float("-inf")), default=float("-inf"))
        throughput = batch_size / elapsed if elapsed > 0 else 0.0
        print(
            f"[Batch {batch_idx}/{n_batches} | {phase}] "
            f"{batch_size} trials in {elapsed:.2f}s ({throughput:.1f} tps) | "
            f"best_in_batch={best_in_batch:.4f} | global_best={self._best_score:.4f} | "
            f"failed={failed}"
        )
