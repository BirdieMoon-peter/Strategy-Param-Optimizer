# -*- coding: utf-8 -*-
"""
贝叶斯优化器模块
基于Optuna实现多目标贝叶斯优化，支持LLM动态调整搜索空间
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt

from config import (
    BayesianOptConfig, DEFAULT_BAYESIAN_CONFIG,
    OPTIMIZATION_OBJECTIVES, OUTPUT_DIR
)
from strategy_analyzer import StrategyAnalyzer, SearchSpaceConfig, convert_to_optuna_space
from backtest_engine import BacktestEngine, BacktestResult
from llm_client import LLMClient, get_llm_client


@dataclass
class OptimizationResult:
    """单个目标的优化结果"""
    objective: str
    best_params: Dict[str, Any]
    best_value: float
    backtest_result: BacktestResult
    n_trials: int
    optimization_time: float


class BayesianOptimizer:
    """
    贝叶斯优化器
    结合LLM和Optuna进行智能超参数优化
    """
    
    def __init__(
        self,
        config: BayesianOptConfig = None,
        llm_client: LLMClient = None,
        use_llm: bool = True,
        backtest_engine: BacktestEngine = None,
        search_space = None,
        verbose: bool = True
    ):
        """
        初始化优化器
        
        Args:
            config: 贝叶斯优化配置
            llm_client: LLM客户端
            use_llm: 是否使用LLM动态调整
            backtest_engine: 外部传入的回测引擎（可选）
            search_space: 搜索空间配置（可选）
            verbose: 是否打印详细信息
        """
        self.config = config or DEFAULT_BAYESIAN_CONFIG
        self.verbose = verbose
        self.search_space = search_space
        
        # 使用外部传入的 backtest_engine 或创建新的
        self.backtest_engine = backtest_engine or BacktestEngine()
        
        # LLM 相关
        if llm_client is not None:
            self.llm_client = llm_client
            self.use_llm = use_llm and self.llm_client.check_connection()
        else:
            try:
                self.llm_client = get_llm_client()
                self.use_llm = use_llm and self.llm_client.check_connection()
            except:
                self.llm_client = None
                self.use_llm = False
        
        if self.llm_client:
            self.strategy_analyzer = StrategyAnalyzer(self.llm_client, use_llm)
        else:
            self.strategy_analyzer = None
        
        # 优化历史记录
        self.optimization_history = {}
        self.all_results = {}
        
        # 设置Optuna日志级别
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """创建采样器"""
        if self.config.sampler == "tpe":
            return TPESampler(seed=self.config.seed)
        elif self.config.sampler == "random":
            return RandomSampler(seed=self.config.seed)
        else:
            return TPESampler(seed=self.config.seed)
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """创建剪枝器"""
        if self.config.pruner == "median":
            return MedianPruner()
        elif self.config.pruner == "hyperband":
            return HyperbandPruner()
        else:
            return MedianPruner()
    
    def _suggest_params(
        self,
        trial: optuna.Trial,
        search_space: Dict[str, SearchSpaceConfig]
    ) -> Dict[str, Any]:
        """
        根据搜索空间配置建议参数
        
        Args:
            trial: Optuna试验对象
            search_space: 搜索空间配置
            
        Returns:
            建议的参数字典
        """
        params = {}
        
        for name, config in search_space.items():
            if config.param_type == "int":
                step = int(config.step) if config.step else 1
                params[name] = trial.suggest_int(
                    name,
                    int(config.min_value),
                    int(config.max_value),
                    step=step
                )
            elif config.param_type == "float":
                if config.distribution == "log_uniform":
                    params[name] = trial.suggest_float(
                        name,
                        config.min_value,
                        config.max_value,
                        log=True
                    )
                else:
                    step = config.step if config.step else None
                    params[name] = trial.suggest_float(
                        name,
                        config.min_value,
                        config.max_value,
                        step=step
                    )
            elif config.param_type == "bool":
                params[name] = trial.suggest_categorical(name, [True, False])
        
        return params
    
    def _create_objective_function(
        self,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        search_space: Dict[str, SearchSpaceConfig],
        objective: str,
        history_list: List[Dict]
    ) -> Callable:
        """
        创建Optuna目标函数
        
        Args:
            strategy_class: 策略类
            data: 行情数据
            search_space: 搜索空间
            objective: 优化目标
            history_list: 历史记录列表（用于存储）
            
        Returns:
            目标函数
        """
        def objective_fn(trial: optuna.Trial) -> float:
            # 建议参数
            params = self._suggest_params(trial, search_space)
            
            # 运行回测
            result = self.backtest_engine.run_backtest(
                strategy_class,
                data,
                params
            )
            
            if result is None:
                return float('-inf')
            
            # 获取目标值
            value = self.backtest_engine.evaluate_objective(result, objective)
            
            # 记录历史
            history_list.append({
                "trial": trial.number,
                "params": params.copy(),
                "value": value,
                "sharpe": result.sharpe_ratio,
                "annual_return": result.annual_return,
                "max_drawdown": result.max_drawdown
            })
            
            return value
        
        return objective_fn
    
    def optimize_single_objective(
        self,
        strategy_class: Type[bt.Strategy],
        strategy_name: str,
        data: pd.DataFrame,
        objective: str,
        search_space: Dict[str, SearchSpaceConfig] = None,
        n_trials: int = None,
        verbose: bool = True,
        default_params: Dict[str, Any] = None
    ) -> OptimizationResult:
        """
        单目标优化
        
        Args:
            strategy_class: 策略类
            strategy_name: 策略名称
            data: 行情数据
            objective: 优化目标
            search_space: 搜索空间（如果为None，自动生成）
            n_trials: 试验次数
            verbose: 是否打印进度
            default_params: 策略的默认参数，将作为第一个采样点
            
        Returns:
            优化结果
        """
        n_trials = n_trials or self.config.n_trials
        
        # 生成搜索空间
        if search_space is None:
            search_space = self.strategy_analyzer.generate_search_space(
                strategy_name,
                use_llm_recommendations=self.use_llm
            )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"开始优化: {strategy_name}")
            print(f"目标: {objective}")
            print(f"试验次数: {n_trials}")
            print(f"{'='*60}")
        
        # 初始化历史记录
        history_list = []
        
        # 创建Study
        direction = "maximize"  # 回撤已在evaluate_objective中取负
        
        study = optuna.create_study(
            direction=direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner()
        )
        
        # 将默认参数作为第一个采样点加入队列
        # 这样可以确保不会错过已经很优的默认参数
        if default_params:
            # 过滤出搜索空间中存在的参数
            enqueue_params = {k: v for k, v in default_params.items() if k in search_space}
            if enqueue_params:
                study.enqueue_trial(enqueue_params)
                if verbose:
                    print(f"[初始采样] 已将策略默认参数加入采样队列: {enqueue_params}")
        
        # 创建目标函数
        objective_fn = self._create_objective_function(
            strategy_class, data, search_space, objective, history_list
        )
        
        # 运行优化
        start_time = datetime.now()
        
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            show_progress_bar=verbose,
            n_jobs=self.config.n_jobs
        )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # 获取最佳结果
        best_params = study.best_params
        best_value = study.best_value
        
        # 重新运行最佳参数获取完整回测结果
        best_result = self.backtest_engine.run_backtest(
            strategy_class, data, best_params
        )
        
        if verbose:
            print(f"\n最佳参数: {best_params}")
            print(f"最佳{objective}: {best_value:.4f}")
            if best_result:
                summary = self.backtest_engine.get_result_summary(best_result)
                print("回测结果:")
                for k, v in summary.items():
                    print(f"  {k}: {v}")
        
        # 保存历史
        key = f"{strategy_name}_{objective}"
        self.optimization_history[key] = history_list
        
        return OptimizationResult(
            objective=objective,
            best_params=best_params,
            best_value=best_value if objective != "max_drawdown" else -best_value,
            backtest_result=best_result,
            n_trials=n_trials,
            optimization_time=optimization_time
        )
    
    def optimize_with_llm_feedback(
        self,
        strategy_class: Type[bt.Strategy],
        strategy_name: str,
        data: pd.DataFrame,
        objective: str,
        n_rounds: int = None,
        trials_per_round: int = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        带LLM反馈的多轮优化
        
        Args:
            strategy_class: 策略类
            strategy_name: 策略名称
            data: 行情数据
            objective: 优化目标
            n_rounds: 优化轮数
            trials_per_round: 每轮试验次数
            verbose: 是否打印进度
            
        Returns:
            最终优化结果
        """
        n_rounds = n_rounds or self.config.n_rounds
        trials_per_round = trials_per_round or (self.config.n_trials // n_rounds)
        
        # 初始搜索空间
        current_space = self.strategy_analyzer.generate_search_space(
            strategy_name,
            use_llm_recommendations=self.use_llm
        )
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"开始LLM引导的多轮优化")
            print(f"策略: {strategy_name}")
            print(f"目标: {objective}")
            print(f"轮数: {n_rounds}, 每轮试验: {trials_per_round}")
            print(f"{'#'*60}")
            
            self.strategy_analyzer.print_search_space(current_space)
        
        all_history = []
        best_result = None
        best_value = float('-inf')
        best_params = None
        
        for round_idx in range(n_rounds):
            if verbose:
                print(f"\n{'='*40}")
                print(f"第 {round_idx + 1}/{n_rounds} 轮优化")
                print(f"{'='*40}")
            
            # 运行这一轮优化
            result = self.optimize_single_objective(
                strategy_class,
                strategy_name,
                data,
                objective,
                search_space=current_space,
                n_trials=trials_per_round,
                verbose=verbose
            )
            
            # 获取这一轮的历史
            key = f"{strategy_name}_{objective}"
            round_history = self.optimization_history.get(key, [])
            all_history.extend(round_history)
            
            # 更新最佳结果
            current_value = result.best_value if objective != "max_drawdown" else -result.best_value
            compare_value = current_value if objective != "max_drawdown" else -current_value
            compare_best = best_value if objective != "max_drawdown" else -best_value
            
            if objective == "max_drawdown":
                # 回撤越小越好
                if best_result is None or result.best_value < best_value:
                    best_result = result
                    best_value = result.best_value
                    best_params = result.best_params.copy()
            else:
                if compare_value > compare_best:
                    best_result = result
                    best_value = current_value
                    best_params = result.best_params.copy()
            
            # 使用LLM调整搜索空间（除了最后一轮）
            if round_idx < n_rounds - 1 and self.use_llm:
                if verbose:
                    print("\n[LLM] 分析优化历史...")
                
                current_space = self.strategy_analyzer.adjust_search_space(
                    current_space,
                    all_history,
                    objective
                )
                
                if verbose:
                    self.strategy_analyzer.print_search_space(current_space)
        
        # 返回最终结果
        if best_result:
            best_result.best_params = best_params
            best_result.n_trials = len(all_history)
        
        return best_result
    
    def optimize_all_objectives(
        self,
        strategy_class: Type[bt.Strategy],
        strategy_name: str,
        data: pd.DataFrame,
        use_llm_feedback: bool = True,
        verbose: bool = True
    ) -> Dict[str, OptimizationResult]:
        """
        针对所有目标进行优化
        
        Args:
            strategy_class: 策略类
            strategy_name: 策略名称
            data: 行情数据
            use_llm_feedback: 是否使用LLM反馈
            verbose: 是否打印进度
            
        Returns:
            各目标的优化结果
        """
        results = {}
        objectives = ["sharpe_ratio", "annual_return", "max_drawdown"]
        
        for objective in objectives:
            if verbose:
                print(f"\n{'*'*60}")
                print(f"优化目标: {OPTIMIZATION_OBJECTIVES[objective].description}")
                print(f"{'*'*60}")
            
            if use_llm_feedback and self.use_llm:
                result = self.optimize_with_llm_feedback(
                    strategy_class,
                    strategy_name,
                    data,
                    objective,
                    verbose=verbose
                )
            else:
                result = self.optimize_single_objective(
                    strategy_class,
                    strategy_name,
                    data,
                    objective,
                    verbose=verbose
                )
            
            results[objective] = result
        
        # 保存结果
        self.all_results[strategy_name] = results
        self._save_results(strategy_name, results)
        
        return results
    
    def _save_results(self, strategy_name: str, results: Dict[str, OptimizationResult]):
        """保存优化结果到文件"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_optimization_{timestamp}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        save_data = {
            "strategy": strategy_name,
            "timestamp": timestamp,
            "results": {}
        }
        
        for objective, result in results.items():
            save_data["results"][objective] = {
                "best_params": result.best_params,
                "best_value": result.best_value,
                "n_trials": result.n_trials,
                "optimization_time": result.optimization_time,
                "backtest": {
                    "total_return": result.backtest_result.total_return,
                    "annual_return": result.backtest_result.annual_return,
                    "max_drawdown": result.backtest_result.max_drawdown,
                    "sharpe_ratio": result.backtest_result.sharpe_ratio,
                    "trades_count": result.backtest_result.trades_count,
                    "win_rate": result.backtest_result.win_rate
                } if result.backtest_result else None
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {filepath}")
    
    def get_optimization_summary(
        self,
        results: Dict[str, OptimizationResult]
    ) -> pd.DataFrame:
        """
        生成优化结果摘要表格
        
        Args:
            results: 优化结果字典
            
        Returns:
            DataFrame格式的摘要
        """
        summary_data = []
        
        for objective, result in results.items():
            row = {
                "优化目标": objective,
                "最佳值": result.best_value,
                "试验次数": result.n_trials,
                "优化时间(秒)": result.optimization_time
            }
            
            # 添加参数
            for param, value in result.best_params.items():
                row[f"参数_{param}"] = value
            
            # 添加回测结果
            if result.backtest_result:
                row["总收益率(%)"] = result.backtest_result.total_return
                row["年化收益率(%)"] = result.backtest_result.annual_return
                row["最大回撤(%)"] = result.backtest_result.max_drawdown
                row["夏普比率"] = result.backtest_result.sharpe_ratio
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # 测试代码
    from src.Aberration import AberrationStrategy
    
    optimizer = BayesianOptimizer(use_llm=False)
    engine = BacktestEngine()
    
    # 加载数据
    data = engine.load_data("BTC")
    
    if data is not None:
        print(f"数据加载完成: {len(data)} 条记录")
        
        # 简单测试
        result = optimizer.optimize_single_objective(
            AberrationStrategy,
            "AberrationStrategy",
            data,
            "sharpe_ratio",
            n_trials=20,
            verbose=True
        )
        
        print(f"\n优化完成!")
        print(f"最佳夏普比率: {result.best_value:.4f}")
        print(f"最佳参数: {result.best_params}")
