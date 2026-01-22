# -*- coding: utf-8 -*-
"""
Optimizer配置文件
包含所有策略的参数定义、搜索空间配置和优化目标设置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==================== LLM配置 ====================
@dataclass
class LLMConfig:
    """Ollama轩辕大模型配置"""
    base_url: str = "http://localhost:11434"
    model_name: str = "xuanyuan"  # 轩辕大模型在ollama中的名称
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120


# ==================== 策略参数定义 ====================
@dataclass
class StrategyParam:
    """策略参数描述"""
    name: str
    param_type: str  # 'int', 'float', 'bool'
    default_value: Any
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None


# 各策略的参数定义
STRATEGY_PARAMS = {
    "AberrationStrategy": {
        "class_name": "AberrationStrategy",
        "module_path": "src.Aberration",
        "description": "纯布林带突破策略：价格突破上轨开多，突破下轨开空，跌破/突破中轨平仓",
        "params": [
            StrategyParam("period", "int", 35, "均线/标准差的计算周期", 10, 100, 5),
            StrategyParam("std_dev_upper", "float", 2.0, "上轨标准差乘数", 0.5, 4.0, 0.1),
            StrategyParam("std_dev_lower", "float", 2.0, "下轨标准差乘数", 0.5, 4.0, 0.1),
        ]
    },
    
    "BollingerBandit": {
        "class_name": "BollingerBandit",
        "module_path": "src.Bollinger_Bandit",
        "description": "布林带强盗策略：结合趋势过滤和动态平仓均线的布林带突破策略",
        "params": [
            StrategyParam("bband_period", "int", 50, "布林带计算周期", 10, 100, 5),
            StrategyParam("std_dev", "float", 1.0, "标准差倍数", 0.5, 3.0, 0.1),
            StrategyParam("init_exit_period", "int", 50, "初始平仓均线周期", 20, 100, 5),
            StrategyParam("min_exit_period", "int", 10, "最小平仓均线周期", 5, 30, 5),
            StrategyParam("trend_filter_period", "int", 30, "趋势过滤周期", 10, 60, 5),
        ]
    },
    
    "BollingerClassic": {
        "class_name": "BollingerClassic",
        "module_path": "src.Bollinger_Classic",
        "description": "经典布林带均值回归策略：价格触及下轨回升时买入，触及上轨回落时卖出",
        "params": [
            StrategyParam("period", "int", 20, "布林带周期", 10, 60, 5),
            StrategyParam("devfactor", "float", 2.0, "标准差倍数", 1.0, 4.0, 0.25),
        ]
    },
    
    "KingKeltnerStrategy": {
        "class_name": "KingKeltnerStrategy",
        "module_path": "src.King_Keltner",
        "description": "肯特纳通道策略：基于EMA和ATR构建通道，突破上轨开多，突破下轨开空",
        "params": [
            StrategyParam("ema_period", "int", 20, "EMA周期", 5, 50, 5),
            StrategyParam("atr_period", "int", 10, "ATR周期", 5, 30, 5),
            StrategyParam("atr_mult", "float", 2.0, "ATR乘数", 0.5, 4.0, 0.25),
        ]
    },
    
    "TrendModelSys": {
        "class_name": "TrendModelSys",
        "module_path": "src.TrendModelSys",
        "description": "趋势模型系统：基于MACD金叉死叉信号，结合ATR突破的趋势跟踪策略",
        "params": [
            StrategyParam("fast_ma", "int", 9, "MACD快速均线周期", 5, 20, 1),
            StrategyParam("slow_ma", "int", 26, "MACD慢速均线周期", 15, 50, 1),
            StrategyParam("macd_ma", "int", 12, "MACD信号线周期", 5, 20, 1),
            StrategyParam("ncos", "int", 4, "记录MACD穿越次数", 2, 8, 1),
            StrategyParam("nbars", "int", 50, "信号有效K线数", 20, 100, 10),
            StrategyParam("trail_bar", "int", 10, "追踪止损周期", 5, 30, 5),
            StrategyParam("atr_period", "int", 4, "ATR计算周期", 3, 20, 1),
            StrategyParam("atr_mult", "float", 0.5, "ATR乘数", 0.1, 2.0, 0.1),
        ]
    },
    
    "SidewinderStrategy": {
        "class_name": "SidewinderStrategy",
        "module_path": "src.Sidewinder",
        "description": "Sidewinder策略：三线EMA完美排列+均线发散度过滤的趋势跟踪策略",
        "params": [
            StrategyParam("s", "int", 10, "短期EMA周期", 5, 20, 1),
            StrategyParam("m", "int", 25, "中期EMA周期", 15, 40, 1),
            StrategyParam("l", "int", 50, "长期EMA周期", 30, 80, 5),
            StrategyParam("threshold", "float", 0.002, "均线发散度阈值", 0.001, 0.01, 0.001),
        ]
    },
}


# ==================== 优化目标配置 ====================
@dataclass
class OptimizationObjective:
    """优化目标定义"""
    name: str
    direction: str  # 'maximize' or 'minimize'
    description: str
    weight: float = 1.0


OPTIMIZATION_OBJECTIVES = {
    "sharpe_ratio": OptimizationObjective(
        name="sharpe_ratio",
        direction="maximize",
        description="夏普比率：衡量风险调整后的收益，越高越好"
    ),
    "annual_return": OptimizationObjective(
        name="annual_return",
        direction="maximize",
        description="年化收益率：策略的年化收益水平，越高越好"
    ),
    "max_drawdown": OptimizationObjective(
        name="max_drawdown",
        direction="minimize",
        description="最大回撤：策略的最大损失幅度，越低越好"
    ),
}


# ==================== 回测配置 ====================
@dataclass
class BacktestConfig:
    """回测配置"""
    initial_cash: float = 100000.0
    commission: float = 0.0005
    start_date: str = "2021-01-01"
    slippage: float = 0.0


# ==================== 贝叶斯优化配置 ====================
@dataclass
class BayesianOptConfig:
    """贝叶斯优化配置"""
    n_trials: int = 100  # 每轮优化的试验次数
    n_rounds: int = 3    # LLM动态调整的轮数
    sampler: str = "tpe"  # 采样器类型: 'tpe', 'gp', 'random'
    pruner: str = "median"  # 剪枝器类型: 'median', 'hyperband', 'none'
    seed: int = 42
    n_jobs: int = 1  # 并行任务数


# ==================== 默认配置实例 ====================
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
DEFAULT_BAYESIAN_CONFIG = BayesianOptConfig()
