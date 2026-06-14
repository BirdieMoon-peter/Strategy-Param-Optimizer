# -*- coding: utf-8 -*-
"""
SPP 并行回测执行器。

父进程负责生成 Monte Carlo 参数样本和写报告；worker 只初始化一次数据、
策略和 BacktestEngine，然后批量执行只读回测任务。
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import backtrader as bt
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_WORKER_STATE: Dict[str, Any] = {}


@dataclass
class SPPWorkerInitArgs:
    """传给 SPP worker initializer 的可 pickle 参数。"""

    data_paths: List[str]
    strategy_path: str
    objective: str
    data_frequency: Optional[str] = None
    broker_config: Any = None
    data_names: Optional[List[str]] = None
    is_multi_data: bool = False
    initial_cash: Optional[float] = None
    commission: Optional[float] = None

    def as_payload(self) -> Dict[str, Any]:
        return {
            "data_paths": list(self.data_paths),
            "strategy_path": self.strategy_path,
            "objective": self.objective,
            "data_frequency": self.data_frequency,
            "broker_config": self.broker_config,
            "data_names": list(self.data_names) if self.data_names else None,
            "is_multi_data": self.is_multi_data,
            "initial_cash": self.initial_cash,
            "commission": self.commission,
        }


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        for col in ("date", "time_key", "time"):
            if col in df.columns:
                df.rename(columns={col: "datetime"}, inplace=True)
                break
    if "datetime" not in df.columns:
        raise ValueError(f"数据文件缺少 datetime/date/time_key/time 列: {path}")
    if pd.api.types.is_numeric_dtype(df["datetime"]):
        unit = "ms" if df["datetime"].iloc[0] > 1e12 else "s"
        df["datetime"] = pd.to_datetime(df["datetime"], unit=unit)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _load_strategy(strategy_path: str):
    stem = os.path.splitext(os.path.basename(strategy_path))[0]
    module_name = f"spp_strategy_module_{stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    strategy_class = None
    custom_data_class = None
    custom_commission_class = None
    for _, obj in inspect.getmembers(module):
        if not inspect.isclass(obj) or obj.__module__ != module_name:
            continue
        if hasattr(obj, "params") and issubclass(obj, bt.Strategy):
            if strategy_class is None:
                strategy_class = obj
        elif issubclass(obj, bt.feeds.PandasData) and obj is not bt.feeds.PandasData:
            custom_data_class = obj
        elif issubclass(obj, bt.CommInfoBase) and obj is not bt.CommInfoBase:
            custom_commission_class = obj

    if strategy_class is None:
        raise RuntimeError(f"未在策略文件中找到 bt.Strategy 子类: {strategy_path}")

    try:
        source = inspect.getsource(strategy_class)
        use_trade_log_metrics = "trade_log" in source
    except Exception:
        use_trade_log_metrics = False

    return (
        strategy_class,
        module,
        custom_data_class,
        custom_commission_class,
        use_trade_log_metrics,
    )


def _init_worker(init_payload: Dict[str, Any]) -> None:
    warnings.filterwarnings("ignore")
    from backtest_engine import BacktestEngine

    data_paths = init_payload["data_paths"]
    data_obj = [_load_csv(p) for p in data_paths] if init_payload.get("is_multi_data") else _load_csv(data_paths[0])
    (
        strategy_class,
        strategy_module,
        custom_data_class,
        custom_commission_class,
        use_trade_log_metrics,
    ) = _load_strategy(init_payload["strategy_path"])

    effective_freq = init_payload.get("data_frequency")
    if effective_freq == "auto":
        effective_freq = None

    broker_config = init_payload.get("broker_config")
    commission = init_payload.get("commission")
    if broker_config is not None and not getattr(broker_config, "is_futures", False):
        commission = broker_config.commission

    engine = BacktestEngine(
        data=data_obj,
        strategy_class=strategy_class,
        initial_cash=init_payload.get("initial_cash"),
        commission=commission,
        data_frequency=effective_freq,
        strategy_module=strategy_module,
        broker_config=broker_config,
        custom_data_class=custom_data_class,
        custom_commission_class=custom_commission_class,
        use_trade_log_metrics=use_trade_log_metrics,
    )

    _WORKER_STATE.update(
        {
            "engine": engine,
            "strategy_class": strategy_class,
            "data": data_obj,
            "objective": init_payload["objective"],
            "data_names": init_payload.get("data_names"),
        }
    )


def _worker_run(task: Tuple[int, Dict[str, Any]]) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    sample_id, params = task
    try:
        engine = _WORKER_STATE["engine"]
        result = engine.run_backtest(
            strategy_class=_WORKER_STATE["strategy_class"],
            data=_WORKER_STATE["data"],
            params=dict(params),
            calculate_yearly=False,
            data_names=_WORKER_STATE.get("data_names"),
        )
        if result is None:
            return sample_id, None, "backtest returned None"

        objective = _WORKER_STATE["objective"]
        obj_val = engine.evaluate_objective(result, objective)
        record = {
            "sample_id": sample_id,
            "params": dict(params),
            objective: obj_val,
            "annual_return": result.annual_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "trades_count": result.trades_count,
        }
        return sample_id, record, None
    except Exception as exc:  # noqa: BLE001
        return sample_id, None, f"{type(exc).__name__}: {exc}"


class SPPParallelEvaluator:
    """固定参数列表的并行 SPP 回测执行器。"""

    def __init__(
        self,
        worker_init_args: SPPWorkerInitArgs,
        n_workers: int,
        batch_size: int = 32,
        sample_timeout: int = 300,
        verbose: bool = True,
    ) -> None:
        self.worker_init_args = worker_init_args
        self.n_workers = max(1, int(n_workers))
        self.batch_size = max(1, int(batch_size))
        self.sample_timeout = int(sample_timeout)
        self.verbose = verbose
        self.failed_samples: List[Dict[str, Any]] = []

    def evaluate(self, param_list: List[Dict[str, Any]], desc: str = "蒙特卡洛") -> List[Dict[str, Any]]:
        if not param_list:
            return []

        records: List[Dict[str, Any]] = []
        total = len(param_list)
        start = time.time()

        with ProcessPoolExecutor(
            max_workers=min(self.n_workers, total),
            initializer=_init_worker,
            initargs=(self.worker_init_args.as_payload(),),
        ) as executor:
            for batch_start in range(0, total, self.batch_size):
                batch_params = param_list[batch_start:batch_start + self.batch_size]
                tasks = [
                    (batch_start + offset + 1, params)
                    for offset, params in enumerate(batch_params)
                ]
                futures = [executor.submit(_worker_run, task) for task in tasks]

                for idx, fut in enumerate(futures):
                    sample_id = tasks[idx][0]
                    try:
                        _, record, error = fut.result(timeout=self.sample_timeout)
                    except FutureTimeoutError:
                        fut.cancel()
                        record = None
                        error = f"timeout>{self.sample_timeout}s"
                    except Exception as exc:  # noqa: BLE001
                        record = None
                        error = f"executor_error: {exc}"

                    if record is not None:
                        records.append(record)
                    else:
                        self.failed_samples.append({
                            "sample_id": sample_id,
                            "error": error or "unknown error",
                            "params": tasks[idx][1],
                        })

                completed = min(batch_start + len(batch_params), total)
                if self.verbose:
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    print(
                        f"  [{desc}-并行] {completed}/{total} "
                        f"({100 * completed / total:.0f}%) 剩余 {remaining:.0f}s"
                    )

        records.sort(key=lambda r: int(r["sample_id"]))
        return records
