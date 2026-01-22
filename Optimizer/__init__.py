# -*- coding: utf-8 -*-
"""
Optimizer包初始化
"""

from .config import (
    STRATEGY_PARAMS,
    OPTIMIZATION_OBJECTIVES,
    LLMConfig,
    BacktestConfig,
    BayesianOptConfig,
    DEFAULT_LLM_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_BAYESIAN_CONFIG
)

from .llm_client import LLMClient, get_llm_client
from .strategy_analyzer import StrategyAnalyzer, SearchSpaceConfig
from .backtest_engine import BacktestEngine, BacktestResult
from .bayesian_optimizer import BayesianOptimizer, OptimizationResult
from .report_generator import ReportGenerator
from .main import QuantOptimizer

__version__ = "1.0.0"
__author__ = "Quant Optimizer"

__all__ = [
    # 配置
    "STRATEGY_PARAMS",
    "OPTIMIZATION_OBJECTIVES",
    "LLMConfig",
    "BacktestConfig",
    "BayesianOptConfig",
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_BACKTEST_CONFIG",
    "DEFAULT_BAYESIAN_CONFIG",
    # 类
    "LLMClient",
    "StrategyAnalyzer",
    "SearchSpaceConfig",
    "BacktestEngine",
    "BacktestResult",
    "BayesianOptimizer",
    "OptimizationResult",
    "ReportGenerator",
    "QuantOptimizer",
    # 函数
    "get_llm_client",
]
