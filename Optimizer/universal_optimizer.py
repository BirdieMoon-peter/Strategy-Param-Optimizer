# -*- coding: utf-8 -*-
"""
通用策略优化器
支持任意标的和策略的优化

v2.0 更新:
- 集成增强采样器（正态分布采样 + 并行探索）
- 动态试验次数根据参数量自动调整
- 增强的边界二次搜索功能
"""

import os
import sys
import json
import pandas as pd
import importlib.util
import inspect
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import backtrader as bt

from universal_llm_client import UniversalLLMClient, UniversalLLMConfig
from backtest_engine import BacktestEngine, BacktestResult
from bayesian_optimizer import BayesianOptimizer, OptimizationResult
from config import StrategyParam, BayesianOptConfig, MarketMakerConfig
from strategy_analyzer import SearchSpaceConfig as ParamSearchSpaceConfig
from param_space_optimizer import ParamSpaceOptimizer
from futures_config import BrokerConfig, create_commission_info

# 导入增强采样器
try:
    from enhanced_sampler import (
        SamplerConfig, DynamicTrialsCalculator, NormalDistributionSampler
    )
    ENHANCED_SAMPLER_AVAILABLE = True
except ImportError:
    ENHANCED_SAMPLER_AVAILABLE = False

# 导入并行引擎
try:
    from parallel_engine import (
        BatchParallelOptimizer,
        WorkerInitArgs,
        create_parallel_study,
    )
    PARALLEL_ENGINE_AVAILABLE = True
except ImportError:
    PARALLEL_ENGINE_AVAILABLE = False

# 定义内部 SearchSpaceConfig
@dataclass  
class SearchSpaceConfig:
    """搜索空间配置"""
    strategy_params: List[StrategyParam]
    constraints: List[str] = field(default_factory=list)


class UniversalOptimizer:
    """
    通用策略优化器
    
    功能:
    1. 支持任意CSV格式的标的数据
    2. 动态加载策略脚本
    3. 支持多种LLM API
    4. 输出JSON格式的优化结果
    5. 支持多种数据频率（日线、分钟线等）
    """
    
    def __init__(
        self,
        data_path: Any,
        strategy_path: str,
        objective: str = "sharpe_ratio",
        use_llm: bool = False,
        llm_config: Optional[UniversalLLMConfig] = None,
        output_dir: str = "./optimization_results",
        verbose: bool = True,
        target_params: Optional[List[str]] = None,
        custom_space: Optional[Dict[str, Dict]] = None,
        data_names: Optional[List[str]] = None,
        data_frequency: Optional[str] = None,
        broker_config: Optional[BrokerConfig] = None,
        market_maker_config: Optional[MarketMakerConfig] = None
    ):
        """
        初始化优化器
        
        Args:
            data_path: 标的数据CSV文件路径
            strategy_path: 策略脚本文件路径（.py文件）
            objective: 优化目标（sharpe_ratio, annual_return, etc.）
            use_llm: 是否使用LLM
            llm_config: LLM配置（如果use_llm为True则必须提供）
            output_dir: 输出目录
            verbose: 是否打印详细信息
            target_params: 指定要优化的参数列表，为None时优化所有参数
            custom_space: 自定义参数空间配置，格式: {param_name: {min, max, step, distribution}}
            data_frequency: 数据频率（'daily', '1m', '5m', '15m', '30m', 'hourly' 等）
                           为None或'auto'时自动检测
        """
        self.data_path = data_path
        self.strategy_path = strategy_path
        self.objective = objective
        self.use_llm = use_llm
        self.verbose = verbose
        self.target_params = target_params  # 指定要优化的参数
        self.custom_space = custom_space  # 自定义参数空间
        self.data_names = data_names  # 多数据源名称（可选）
        self.data_frequency = data_frequency  # 数据频率
        self.broker_config = broker_config  # 经纪商配置（期货/股票）
        self.market_maker_config = market_maker_config  # 做市商优化配置

        # 创建输出目录
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化参数空间优化器（需要在加载策略之前初始化）
        self.param_space_optimizer = ParamSpaceOptimizer(verbose=self.verbose)
        
        # 加载数据
        self.data = self._load_data()
        # 从文件名提取资产名称，去除 _processed 后缀
        if isinstance(data_path, (list, tuple)):
            if self.data_names:
                self.asset_name = "+".join(self.data_names)
            else:
                raw_names = [Path(p).stem.replace('_processed', '') for p in data_path]
                self.asset_name = "+".join(raw_names)
        else:
            raw_asset_name = Path(data_path).stem
            self.asset_name = raw_asset_name.replace('_processed', '')
        
        # 加载策略
        self.strategy_class, self.strategy_info = self._load_strategy()
        
        # 初始化LLM（如果需要）
        self.llm_client = None
        if use_llm:
            if llm_config is None:
                raise ValueError("使用LLM时必须提供llm_config")
            self.llm_client = UniversalLLMClient(llm_config)
            if self.verbose:
                print(f"[LLM] 初始化成功: {llm_config.api_type} - {llm_config.model_name}")
        
        # 初始化回测引擎（传递数据频率，如果是 'auto' 或 None 则自动检测）
        # 同时传入自定义数据类、手续费类等
        effective_freq = None if (data_frequency is None or data_frequency == 'auto') else data_frequency
        initial_cash = broker_config.initial_cash if broker_config else 100000.0
        commission = broker_config.commission if (broker_config and not broker_config.is_futures) else 0.001
        self.backtest_engine = BacktestEngine(
            data=self.data,
            strategy_class=self.strategy_class,
            initial_cash=initial_cash,
            commission=commission,
            data_frequency=effective_freq,
            custom_data_class=getattr(self, 'custom_data_class', None),
            custom_commission_class=getattr(self, 'custom_commission_class', None),
            strategy_module=getattr(self, 'strategy_module', None),
            use_trade_log_metrics=getattr(self, 'use_trade_log_metrics', False),
            broker_config=broker_config,
            market_maker_config=market_maker_config
        )
        
        # 保存检测到的数据频率
        self.detected_frequency = self.backtest_engine.config.data_frequency
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"通用策略优化器初始化完成")
            print(f"{'='*60}")
            print(f"标的: {self.asset_name}")
            print(f"策略: {self.strategy_info['class_name']}")
            print(f"优化目标: {objective}")
            print(f"数据频率: {self.detected_frequency}")
            print(f"使用LLM: {'是' if use_llm else '否'}")
            if getattr(self, 'custom_data_class', None):
                print(f"自定义数据类: {self.custom_data_class.__name__}")
            if getattr(self, 'custom_commission_class', None):
                print(f"自定义手续费类: {self.custom_commission_class.__name__}")
            if getattr(self, 'use_trade_log_metrics', False):
                print(f"指标计算: 基于交易日志 (trade_log)")
            if broker_config and broker_config.is_futures:
                print(f"资产类型: 期货")
                print(f"合约: {broker_config.contract_name or broker_config.contract_code} ({broker_config.contract_code})")
                print(f"合约乘数: {broker_config.mult}")
                print(f"保证金比例: {broker_config.margin*100:.1f}%")
                comm_desc = f"{broker_config.commission}元/手" if broker_config.comm_type == 'FIXED' else f"费率{broker_config.commission}"
                print(f"手续费: {comm_desc} ({'固定金额' if broker_config.comm_type == 'FIXED' else '百分比'})")
            else:
                print(f"资产类型: 股票")
            if isinstance(self.data, (list, tuple)):
                print(f"数据点数: {len(self.data[0])} (多数据源: {len(self.data)} 个)")
            else:
                print(f"数据点数: {len(self.data)}")
            print(f"{'='*60}\n")
    
    def _load_data(self) -> Any:
        """加载标的数据（支持单数据或多数据源）"""
        def _load_single(path: str) -> pd.DataFrame:
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据文件不存在: {path}")
            
            df = pd.read_csv(path)
            
            # 统一时间列
            if 'datetime' not in df.columns:
                if 'date' in df.columns:
                    df.rename(columns={'date': 'datetime'}, inplace=True)
                elif 'time_key' in df.columns:
                    df.rename(columns={'time_key': 'datetime'}, inplace=True)
            
            # 验证必需的列
            required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"数据文件缺少必需的列: {missing_columns}")
            
            # 转换datetime列
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df
        
        # 多数据源模式
        if isinstance(self.data_path, (list, tuple)):
            data_list = []
            for path in self.data_path:
                df = _load_single(path)
                data_list.append(df)
            
            if self.verbose:
                print(f"[数据] 成功加载多数据源: {len(data_list)} 个文件")
                for idx, path in enumerate(self.data_path, 1):
                    df = data_list[idx - 1]
                    print(f"  [{idx}] {path}")
                    print(f"       时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
            
            return data_list
        
        # 单数据源模式
        df = _load_single(self.data_path)
        
        if self.verbose:
            print(f"[数据] 成功加载: {self.data_path}")
            print(f"       时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        
        return df
    
    def _load_strategy(self) -> tuple:
        """
        动态加载策略类
        
        Returns:
            (策略类, 策略信息字典)
        """
        if not os.path.exists(self.strategy_path):
            raise FileNotFoundError(f"策略文件不存在: {self.strategy_path}")
        
        # 动态导入模块
        module_name = f"strategy_module_{Path(self.strategy_path).stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.strategy_path)
        module = importlib.util.module_from_spec(spec)
        # 重要：将模块添加到 sys.modules，backtrader 需要这个
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # 保存策略模块引用，用于后续查找自定义类
        self.strategy_module = module
        
        # 查找策略类（继承自backtrader.Strategy）
        strategy_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and hasattr(obj, 'params') 
                and obj.__module__ == module_name 
                and issubclass(obj, bt.Strategy)):
                strategy_classes.append(obj)
        
        if not strategy_classes:
            raise ValueError(f"未在策略文件中找到有效的策略类: {self.strategy_path}")
        
        if len(strategy_classes) > 1:
            if self.verbose:
                print(f"[警告] 发现多个策略类，将使用第一个: {strategy_classes[0].__name__}")
        
        strategy_class = strategy_classes[0]
        
        # 查找自定义数据类（继承自 bt.feeds.PandasData）
        self.custom_data_class = None
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                obj.__module__ == module_name and
                issubclass(obj, bt.feeds.PandasData) and
                obj is not bt.feeds.PandasData):
                self.custom_data_class = obj
                if self.verbose:
                    print(f"[策略] 发现自定义数据类: {obj.__name__}")
                break
        
        # 查找自定义手续费类（继承自 bt.CommInfoBase）
        self.custom_commission_class = None
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                obj.__module__ == module_name and
                issubclass(obj, bt.CommInfoBase) and
                obj is not bt.CommInfoBase):
                self.custom_commission_class = obj
                if self.verbose:
                    print(f"[策略] 发现自定义手续费类: {obj.__name__}")
                break
        
        # 检查策略是否有 trade_log 属性（用于决定是否使用 trade_log 模式计算指标）
        self.use_trade_log_metrics = hasattr(strategy_class, '__init__')
        # 通过检查源码判断是否记录 trade_log
        try:
            source = inspect.getsource(strategy_class)
            self.use_trade_log_metrics = 'trade_log' in source
            if self.use_trade_log_metrics and self.verbose:
                print(f"[策略] 检测到 trade_log，将使用交易日志计算指标")
        except:
            self.use_trade_log_metrics = False
        
        # 提取策略信息
        strategy_info = {
            'class_name': strategy_class.__name__,
            'description': strategy_class.__doc__ or "无描述",
            'params': self._extract_strategy_params(strategy_class)
        }
        
        if self.verbose:
            print(f"[策略] 成功加载: {strategy_info['class_name']}")
            print(f"       参数数量: {len(strategy_info['params'])}")
        
        return strategy_class, strategy_info
    
    def _extract_strategy_params(self, strategy_class) -> List[StrategyParam]:
        """提取并优化策略参数空间"""
        params = []
        all_param_names = []  # 记录所有参数名，用于验证
        
        # 不应该被优化的参数黑名单（这些通常是固定的配置参数）
        EXCLUDED_PARAMS = {
            'printlog', 'verbose',  # 日志相关
            'mult', 'margin',  # 期货合约固定参数
            'commission',  # 手续费（应在回测引擎层面设置）
            'percent',  # 资金分配比例（通常固定）
            'stocklike', 'commtype', 'percabs',  # 手续费类内部参数
        }
        
        if hasattr(strategy_class, 'params'):
            for param_name in dir(strategy_class.params):
                if not param_name.startswith('_'):
                    default_value = getattr(strategy_class.params, param_name)
                    
                    # 推断参数类型
                    param_type = type(default_value).__name__
                    
                    # 跳过非数值类型
                    if not isinstance(default_value, (int, float)):
                        continue
                    
                    # 记录所有数值参数（用于验证 target_params）
                    all_param_names.append(param_name)
                    
                    # 跳过黑名单中的参数
                    if param_name.lower() in EXCLUDED_PARAMS:
                        if self.verbose:
                            print(f"[跳过] 参数 '{param_name}' 在黑名单中，不会被优化")
                        continue
                    
                    # 如果指定了目标参数列表，只提取指定的参数
                    if self.target_params is not None:
                        if param_name not in self.target_params:
                            continue
                    
                    # 创建基础参数（不设置范围，将由优化器处理）
                    param = StrategyParam(
                        name=param_name,
                        param_type=param_type,
                        default_value=default_value,
                        description=f"{param_name} parameter",
                        min_value=None,
                        max_value=None,
                        step=None
                    )
                    params.append(param)
        
        # 验证目标参数是否都存在于策略中
        if self.target_params is not None:
            invalid_params = [p for p in self.target_params if p not in all_param_names]
            if invalid_params:
                if self.verbose:
                    print(f"[警告] 以下参数不存在于策略中，将被忽略: {invalid_params}")
                    print(f"[提示] 可用的参数: {all_param_names}")
            
            if self.verbose and params:
                print(f"[参数过滤] 仅优化指定参数: {[p.name for p in params]}")
        
        # 使用参数空间优化器生成智能的搜索空间
        if params:
            strategy_name = strategy_class.__name__
            params = self.param_space_optimizer.generate_space(
                params,
                strategy_type=strategy_name
            )
        
        # 应用自定义参数空间配置（如果提供）
        if self.custom_space and params:
            params = self._apply_custom_space(params)
        
        return params
    
    def _apply_custom_space(self, params: List[StrategyParam]) -> List[StrategyParam]:
        """
        应用自定义参数空间配置
        
        Args:
            params: 原始参数列表
            
        Returns:
            应用自定义配置后的参数列表
        """
        if not self.custom_space:
            return params
        
        updated_params = []
        for param in params:
            if param.name in self.custom_space:
                custom = self.custom_space[param.name]
                
                # 使用自定义配置覆盖默认值
                new_param = StrategyParam(
                    name=param.name,
                    param_type=param.param_type,
                    default_value=param.default_value,
                    description=custom.get('description', param.description),
                    min_value=custom.get('min', param.min_value),
                    max_value=custom.get('max', param.max_value),
                    step=custom.get('step', param.step)
                )
                updated_params.append(new_param)
                
                if self.verbose:
                    print(f"[自定义空间] {param.name}: [{new_param.min_value}, {new_param.max_value}]")
            else:
                updated_params.append(param)
        
        return updated_params
    
    def optimize(
        self,
        n_trials: int = 50,
        bayesian_config: Optional[BayesianOptConfig] = None,
        auto_expand_boundary: bool = True,
        max_expansion_rounds: int = 2,
        boundary_threshold: float = 0.1,
        expansion_factor: float = 1.5,
        use_enhanced_sampler: bool = True,
        enable_dynamic_trials: bool = True,
        use_parallel: bool = True,
        n_workers: int = 120,
        explore_batch_size: int = 128,
        exploit_batch_size: int = 32,
        trial_timeout: int = 300,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        执行优化（支持自动边界扩展、增强采样器和动态试验次数）
        
        Args:
            n_trials: 优化试验次数（基础值，可能被动态调整）
            bayesian_config: 贝叶斯优化配置
            auto_expand_boundary: 是否自动扩展边界参数
            max_expansion_rounds: 最大扩展轮数
            boundary_threshold: 边界阈值 (默认10%)
            expansion_factor: 扩展因子
            use_enhanced_sampler: 是否使用增强采样器（正态分布 + 并行）
            enable_dynamic_trials: 是否根据参数量动态调整试验次数
            
        Returns:
            优化结果字典
        """
        # 1. 构建初始搜索空间（紧凑范围）
        search_space_config = self._build_search_space()
        current_space = search_space_config.strategy_params.copy()
        n_params = len(current_space)
        
        # 2. 动态计算试验次数（传入搜索空间用于复杂度分析）
        actual_trials = n_trials
        exploration_trials = 0
        exploitation_trials = n_trials
        
        if enable_dynamic_trials and ENHANCED_SAMPLER_AVAILABLE:
            config = SamplerConfig()
            calculator = DynamicTrialsCalculator(config)
            
            # 构建搜索空间字典用于复杂度分析
            space_dict = {p.name: p for p in current_space}
            
            actual_trials, exploration_trials, exploitation_trials = \
                calculator.calculate_trials(n_params, search_space=space_dict, user_trials=n_trials)
            
            # 输出详细推荐信息
            if self.verbose:
                recommendation_msg = calculator.get_recommendation_message(
                    n_params, user_trials=n_trials, search_space=space_dict
                )
                print(recommendation_msg)
        
        if self.verbose:
            print(f"\n╔{'═'*78}╗")
            print(f"║ {'开始优化流程'.center(74)} ║")
            print(f"╠{'═'*78}╣")
            print(f"║ 参数数量: {n_params:<64} ║")
            if enable_dynamic_trials and ENHANCED_SAMPLER_AVAILABLE:
                print(f"║ 动态试验次数: {'启用':<59} ║")
                print(f"║   • 用户指定: {n_trials:<58} 次 ║")
                print(f"║   • 实际试验: {actual_trials:<58} 次 ║")
                print(f"║   • 探索阶段: {exploration_trials:<58} 次 ║")
                print(f"║   • 利用阶段: {exploitation_trials:<58} 次 ║")
            else:
                print(f"║ 试验次数: {actual_trials:<63} 次 ║")
            if use_enhanced_sampler and ENHANCED_SAMPLER_AVAILABLE:
                print(f"║ 采样策略: {'正态分布 + 贝叶斯优化':<59} ║")
            if use_parallel and PARALLEL_ENGINE_AVAILABLE:
                print(f"║ 并行模式: 多进程 ProcessPoolExecutor (workers={n_workers}) {'':25} ║")
                print(f"║   • 探索批大小: {explore_batch_size:<55} ║")
                print(f"║   • 利用批大小: {exploit_batch_size:<55} ║")
                print(f"║   • 单 trial 超时: {trial_timeout:<55}s ║")
            else:
                print(f"║ 并行模式: 串行 (study.optimize) {'':40} ║")
            if auto_expand_boundary:
                print(f"║ 边界二次搜索: 启用 (最多{max_expansion_rounds}轮) {'':40} ║")
            print(f"╚{'═'*78}╝")
        
        # 提取策略的默认参数，用于初始采样
        default_params = {}
        for param in self.strategy_info['params']:
            default_params[param.name] = param.default_value
        
        # 3. 配置贝叶斯优化
        if bayesian_config is None:
            bayesian_config = BayesianOptConfig(
                n_trials=actual_trials,
                n_rounds=1,
                sampler="tpe"
            )
        
        best_result = None
        best_params = None
        best_value = float('-inf')
        expansion_round = 0
        all_history = []
        all_best_improvements = []  # 收集所有轮次中每次发现更优参数的记录
        
        # 4. 优化循环（支持自动边界扩展）
        while True:
            round_label = f"第{expansion_round + 1}轮" if expansion_round > 0 else "初始优化"
            round_trials = actual_trials if expansion_round == 0 else int(actual_trials * 0.5)  # 二次搜索用一半试验
            
            if self.verbose and expansion_round > 0:
                print(f"\n╔{'═'*78}╗")
                print(f"║ {'🔄 边界二次搜索'.center(70)} ║")
                print(f"╠{'═'*78}╣")
                print(f"║ 轮次: {round_label:<69} ║")
                print(f"║ 试验次数: {round_trials:<63} 次 ║")
                print(f"╚{'═'*78}╝")
            
            # 转换搜索空间
            search_space = self._convert_search_space(
                SearchSpaceConfig(strategy_params=current_space)
            )

            # 确定初始采样点（首轮用默认参数，后续轮用上一轮最优）
            init_params = default_params if expansion_round == 0 else best_params

            # 执行优化（使用增强采样器，添加异常处理）
            try:
                if use_parallel and PARALLEL_ENGINE_AVAILABLE:
                    opt_result = self._run_parallel_round(
                        search_space=search_space,
                        n_trials=round_trials,
                        default_params=init_params,
                        use_enhanced_sampler=use_enhanced_sampler and ENHANCED_SAMPLER_AVAILABLE,
                        n_workers=n_workers,
                        explore_batch_size=explore_batch_size,
                        exploit_batch_size=exploit_batch_size,
                        trial_timeout=trial_timeout,
                        seed=seed + expansion_round,
                    )
                else:
                    optimizer = BayesianOptimizer(
                        config=bayesian_config,
                        backtest_engine=self.backtest_engine,
                        use_llm=False,
                        verbose=self.verbose
                    )
                    opt_result = optimizer.optimize_single_objective(
                        strategy_class=self.strategy_class,
                        strategy_name=self.strategy_info['class_name'],
                        data=self.data,
                        objective=self.objective,
                        search_space=search_space,
                        n_trials=round_trials,
                        verbose=self.verbose,
                        default_params=init_params,
                        use_enhanced_sampler=use_enhanced_sampler and ENHANCED_SAMPLER_AVAILABLE,
                        enable_dynamic_trials=enable_dynamic_trials
                    )
            except Exception as e:
                # 优化轮次失败，打印错误但尝试继续
                if self.verbose:
                    print(f"\n❌ [{round_label}] 优化失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # 如果是首轮失败且没有任何结果，抛出异常
                if expansion_round == 0 and best_result is None:
                    raise
                
                # 否则使用已有的最佳结果继续
                if self.verbose:
                    print(f"\n⚠️  将使用已有的最佳结果继续...")
                break
            
            # 更新最优结果
            current_value = opt_result.best_value
            if current_value > best_value:
                best_value = current_value
                best_params = opt_result.best_params
                best_result = opt_result.backtest_result
            
            # 收集该轮次的改进记录（跨轮次去重：按回测结果比较）
            if opt_result.best_improvements:
                for record in opt_result.best_improvements:
                    if not all_best_improvements or all_best_improvements[-1]["backtest_results"] != record["backtest_results"]:
                        all_best_improvements.append(record)
            
            # 检查是否需要扩展边界
            if not auto_expand_boundary or expansion_round >= max_expansion_rounds:
                break
            
            # 检测边界参数
            boundary_params = self.param_space_optimizer.check_boundary_params(
                opt_result.best_params,
                current_space,
                boundary_threshold=boundary_threshold
            )
            
            if not boundary_params:
                if self.verbose:
                    print(f"\n╔{'═'*78}╗")
                    print(f"║ {'✅ 无参数处于边界，优化完成'.center(70)} ║")
                    print(f"╚{'═'*78}╝")
                break
            
            # 有参数在边界，执行扩展
            if self.verbose:
                print(f"\n╔{'═'*78}╗")
                print(f"║ {'⚠️  边界参数检测'.center(70)} ║")
                print(f"╠{'═'*78}╣")
                print(f"║ 检测到 {len(boundary_params)} 个参数处于边界: {'':55} ║")
                for bp in boundary_params:
                    side_cn = "下界" if bp['side'] == 'lower' else "上界"
                    param_info = f"  • {bp['name']}: {bp['value']:.4f} (接近{side_cn} {bp['boundary']:.4f})"
                    print(f"║ {param_info:<76} ║")
                print(f"╠{'═'*78}╣")
                print(f"║ {'🔄 自动扩展边界参数，准备二次搜索...'.ljust(76)} ║")
                print(f"╚{'═'*78}╝")
            
            # 扩展边界
            current_space, expanded_names = self.param_space_optimizer.expand_boundary_params(
                opt_result.best_params,
                current_space,
                expansion_factor=expansion_factor,
                boundary_threshold=boundary_threshold
            )
            
            if self.verbose:
                print(f"\n╔{'═'*78}╗")
                print(f"║ {'📐 扩展后的参数空间'.center(70)} ║")
                print(f"╠{'═'*78}╣")
                for param in current_space:
                    if param.name in expanded_names:
                        param_info = f"  • {param.name}: [{param.min_value}, {param.max_value}] (已扩展)"
                        print(f"║ {param_info:<76} ║")
                print(f"╚{'═'*78}╝")
            
            expansion_round += 1
        
        # 5. 分析参数空间使用情况（添加异常保护）
        if self.verbose:
            print(f"\n╔{'═'*78}╗")
            print(f"║ {'参数空间分析'.center(74)} ║")
            print(f"╚{'═'*78}╝")
        
        try:
            param_analysis = self.param_space_optimizer.analyze_optimization_results(
                best_params,
                current_space
            )
        except Exception as e:
            if self.verbose:
                print(f"\n⚠️  参数空间分析失败: {str(e)}")
            param_analysis = {"suggestions": [], "boundary_params": [], "usage": {}}
        
        if self.verbose and param_analysis.get("suggestions"):
            print(f"\n╔{'═'*78}╗")
            print(f"║ {'💡 参数空间优化建议'.center(70)} ║")
            print(f"╠{'═'*78}╣")
            for suggestion in param_analysis["suggestions"]:
                print(f"║ • {suggestion:<74} ║")
            print(f"╚{'═'*78}╝")
        
        # 6. 生成详细结果（添加异常保护）
        try:
            result = self._generate_result(best_result)
            result["param_space_analysis"] = param_analysis
            result["optimization_info"]["expansion_rounds"] = expansion_round
            result["optimization_info"]["auto_expand_boundary"] = auto_expand_boundary
            result["optimization_info"]["total_trials"] = actual_trials
            result["optimization_info"]["exploration_trials"] = exploration_trials
            result["optimization_info"]["exploitation_trials"] = exploitation_trials
            result["optimization_info"]["use_enhanced_sampler"] = use_enhanced_sampler and ENHANCED_SAMPLER_AVAILABLE
            result["optimization_info"]["dynamic_trials_enabled"] = enable_dynamic_trials
            result["optimization_info"]["use_parallel"] = use_parallel and PARALLEL_ENGINE_AVAILABLE
            if use_parallel and PARALLEL_ENGINE_AVAILABLE:
                result["optimization_info"]["n_workers"] = n_workers
                result["optimization_info"]["explore_batch_size"] = explore_batch_size
                result["optimization_info"]["exploit_batch_size"] = exploit_batch_size
                result["optimization_info"]["trial_timeout"] = trial_timeout
            # 添加搜索过程中每次发现更优参数的记录
            result["best_improvements"] = all_best_improvements
        except Exception as e:
            if self.verbose:
                print(f"\n❌ 生成结果失败: {str(e)}")
                import traceback
                traceback.print_exc()
            raise
        
        # 7. 保存结果（添加异常保护）
        try:
            output_path = self._save_result(result)
        except Exception as e:
            if self.verbose:
                print(f"\n⚠️  保存结果失败: {str(e)}")
            output_path = None
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ 优化完成!")
            print(f"{'='*60}")
            print(f"总轮数: {expansion_round + 1}")
            print(f"总试验次数: {actual_trials}")
            if expansion_round > 0:
                print(f"  - 初始优化: {actual_trials} 次")
                print(f"  - 边界二次搜索: {expansion_round} 轮")
            print(f"结果已保存至: {output_path}")
            print(f"{'='*60}\n")
        
        return result
    
    def _run_parallel_round(
        self,
        search_space: Dict[str, ParamSearchSpaceConfig],
        n_trials: int,
        default_params: Dict[str, Any],
        use_enhanced_sampler: bool,
        n_workers: int,
        explore_batch_size: int,
        exploit_batch_size: int,
        trial_timeout: int,
        seed: int = 42,
    ) -> OptimizationResult:
        """
        使用 BatchParallelOptimizer 执行一轮真多进程并行优化。

        返回与 BayesianOptimizer.optimize_single_objective 兼容的 OptimizationResult。
        """
        from datetime import datetime
        start_time = datetime.now()

        n_params = len(search_space)

        # 探索/利用试验数拆分（与原逻辑保持一致）
        if use_enhanced_sampler:
            calculator = DynamicTrialsCalculator(SamplerConfig())
            total_trials, exploration_trials, exploitation_trials = \
                calculator.calculate_trials(n_params, search_space, n_trials)
        else:
            total_trials = n_trials
            exploration_trials = int(n_trials * 0.3)
            exploitation_trials = n_trials - exploration_trials

        if self.verbose:
            print(f"\n[并行优化] 总 trials={total_trials} "
                  f"(探索={exploration_trials}, 利用={exploitation_trials}) | "
                  f"workers={n_workers}")

        # 预生成探索阶段样本（Trial 0 = 策略默认参数）
        exploration_samples: List[Dict[str, Any]] = []
        if use_enhanced_sampler and exploration_trials > 0:
            sampler = NormalDistributionSampler(
                SamplerConfig(), seed=seed
            )
            exploration_samples, _ = sampler.generate_initial_samples(
                search_space=search_space,
                n_samples=exploration_trials,
                default_params=default_params,
                include_default=True,
            )

        # 打包 worker 初始化参数
        if isinstance(self.data_path, (list, tuple)):
            data_paths = [str(p) for p in self.data_path]
            is_multi = True
        else:
            data_paths = [str(self.data_path)]
            is_multi = False

        initial_cash = self.broker_config.initial_cash if self.broker_config else 100000.0
        commission = (
            self.broker_config.commission
            if (self.broker_config and not self.broker_config.is_futures)
            else 0.001
        )

        worker_init = WorkerInitArgs(
            data_paths=data_paths,
            strategy_path=str(self.strategy_path),
            objective=self.objective,
            broker_config=self.broker_config,
            market_maker_config=self.market_maker_config,
            data_frequency=self.data_frequency,
            initial_cash=initial_cash,
            commission=commission,
            data_names=self.data_names,
            is_multi_data=is_multi,
        )

        # 创建带 constant_liar 的 study
        study = create_parallel_study(
            direction="maximize",
            seed=seed,
            n_startup_trials=min(20, max(5, exploitation_trials // 5)),
            constant_liar=True,
        )

        engine = BatchParallelOptimizer(
            study=study,
            search_space=search_space,
            worker_init_args=worker_init,
            n_exploit_trials=exploitation_trials,
            exploration_samples=exploration_samples,
            n_workers=n_workers,
            explore_batch_size=explore_batch_size,
            exploit_batch_size=exploit_batch_size,
            timeout_per_trial=trial_timeout,
            verbose=self.verbose,
        )

        best_params, best_value, _ = engine.run()

        if not best_params:
            raise RuntimeError("并行优化未产生任何有效结果")

        # 在主进程用最佳参数跑一次完整回测（拿到 BacktestResult 对象）
        final_params = dict(best_params)
        try:
            sig = inspect.signature(self.strategy_class.__init__)
            accepts_verbose = 'verbose' in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except Exception:
            accepts_verbose = False
        if accepts_verbose:
            final_params['verbose'] = False

        best_result = self.backtest_engine.run_backtest(
            strategy_class=self.strategy_class,
            data=self.data,
            params=final_params,
            data_names=self.data_names,
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            objective=self.objective,
            best_params=dict(best_params),
            best_value=best_value,
            backtest_result=best_result,
            n_trials=len(engine.history),
            optimization_time=elapsed,
            best_improvements=list(engine.best_improvements),
        )

    def _build_search_space(self) -> SearchSpaceConfig:
        """构建搜索空间"""
        if self.use_llm and self.llm_client:
            if self.verbose:
                print("[LLM] 正在分析策略参数...")
            
            llm_recommendations = self.llm_client.analyze_strategy_params(
                self.strategy_info
            )
            
            if llm_recommendations and 'search_space' in llm_recommendations:
                if self.verbose:
                    print("[LLM] 成功获取参数推荐")
                return self._convert_llm_to_search_space(llm_recommendations)
        
        # 使用默认搜索空间
        if self.verbose:
            print("[搜索空间] 使用默认配置")
        
        return SearchSpaceConfig(
            strategy_params=self.strategy_info['params']
        )
    
    def _convert_llm_to_search_space(self, llm_recommendations: Dict) -> SearchSpaceConfig:
        """将LLM推荐转换为搜索空间配置"""
        updated_params = []
        
        for param in self.strategy_info['params']:
            if param.name in llm_recommendations['search_space']:
                rec = llm_recommendations['search_space'][param.name]
                
                # 更新参数范围
                param.min_value = rec.get('min', param.min_value)
                param.max_value = rec.get('max', param.max_value)
                
                if 'step' in rec:
                    param.step = rec['step']
            
            updated_params.append(param)
        
        return SearchSpaceConfig(
            strategy_params=updated_params,
            constraints=llm_recommendations.get('constraints', [])
        )
    
    def _convert_search_space(self, config: SearchSpaceConfig) -> Dict[str, ParamSearchSpaceConfig]:
        """将 SearchSpaceConfig 转换为 BayesianOptimizer 需要的格式"""
        search_space = {}
        
        for param in config.strategy_params:
            # 确定参数类型和分布
            if param.param_type == 'int':
                distribution = 'int_uniform'
                param_type = 'int'
            else:
                distribution = 'uniform'
                param_type = 'float'
            
            # 创建参数搜索空间配置
            search_space[param.name] = ParamSearchSpaceConfig(
                param_name=param.name,
                param_type=param_type,
                distribution=distribution,
                min_value=float(param.min_value),
                max_value=float(param.max_value),
                step=param.step,
                priority="medium"
            )
        
        return search_space
    
    def _generate_result(self, best_result: BacktestResult) -> Dict[str, Any]:
        """生成完整的结果字典"""
        # 选择用于展示的数据范围（多数据源时使用第一个数据源）
        data_for_range = self.data[0] if isinstance(self.data, (list, tuple)) else self.data
        
        result = {
            "optimization_info": {
                "asset_name": self.asset_name,
                "strategy_name": self.strategy_info['class_name'],
                "optimization_objective": self.objective,
                "optimization_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_range": {
                    "start": data_for_range['datetime'].min().strftime("%Y-%m-%d"),
                    "end": data_for_range['datetime'].max().strftime("%Y-%m-%d"),
                    "total_days": len(data_for_range)
                }
            },
            "best_parameters": best_result.params,
            "performance_metrics": {
                "sharpe_ratio": round(best_result.sharpe_ratio, 4),
                "annual_return": round(best_result.annual_return, 2),
                "max_drawdown": round(best_result.max_drawdown, 2),
                "total_return": round(best_result.total_return, 2),
                "final_value": round(best_result.final_value, 2),
                "trades_count": best_result.trades_count,
                "win_rate": round(best_result.win_rate, 2)
            },
            "yearly_performance": {}
        }

        # 添加经纪商配置信息（期货模式）
        if self.broker_config and self.broker_config.is_futures:
            result["broker_config"] = {
                "asset_type": self.broker_config.asset_type,
                "contract_code": self.broker_config.contract_code,
                "contract_name": self.broker_config.contract_name,
                "mult": self.broker_config.mult,
                "margin": self.broker_config.margin,
                "comm_type": self.broker_config.comm_type,
                "commission": self.broker_config.commission,
            }
        
        # 添加年度表现
        if best_result.yearly_returns:
            for year in sorted(best_result.yearly_returns.keys()):
                result["yearly_performance"][str(year)] = {
                    "return": round(best_result.yearly_returns.get(year, 0), 2),
                    "drawdown": round(best_result.yearly_drawdowns.get(year, 0), 2),
                    "sharpe_ratio": round(best_result.yearly_sharpe.get(year, 0), 4)
                }
        
        # LLM解释（如果启用）
        if self.use_llm and self.llm_client:
            if self.verbose:
                print("[LLM] 正在生成结果解释...")
            
            explanation = self.llm_client.explain_optimization_result(
                strategy_name=self.strategy_info['class_name'],
                best_params=best_result.params,
                backtest_result=result["performance_metrics"]
            )
            
            result["llm_explanation"] = explanation
        else:
            result["llm_explanation"] = {
                "parameter_explanation": "参数优化完成，以上为最优参数组合",
                "performance_analysis": f"策略在{self.objective}目标下表现最优",
                "risk_assessment": "建议进行样本外测试验证策略稳定性",
                "practical_suggestions": "实盘前请充分测试并评估风险",
                "key_insights": [
                    f"优化目标: {self.objective}",
                    f"回测期: {result['optimization_info']['data_range']['start']} 至 {result['optimization_info']['data_range']['end']}",
                    "历史表现不代表未来收益"
                ]
            }
        
        return result
    
    def _save_result(self, result: Dict[str, Any]) -> str:
        """保存结果为JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_{self.asset_name}_{self.strategy_info['class_name']}_{timestamp}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def batch_optimize(
        self,
        objectives: List[str],
        n_trials_per_objective: int = 50
    ) -> Dict[str, Any]:
        """
        批量优化（多个目标）
        
        Args:
            objectives: 优化目标列表
            n_trials_per_objective: 每个目标的试验次数
            
        Returns:
            批量优化结果
        """
        batch_results = {
            "batch_info": {
                "asset_name": self.asset_name,
                "strategy_name": self.strategy_info['class_name'],
                "objectives": objectives,
                "optimization_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": {}
        }
        
        for obj in objectives:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"优化目标: {obj}")
                print(f"{'='*60}\n")
            
            # 临时更改目标
            original_objective = self.objective
            self.objective = obj
            
            # 执行优化
            result = self.optimize(n_trials=n_trials_per_objective)
            batch_results["results"][obj] = result
            
            # 恢复原始目标
            self.objective = original_objective
        
        # 保存批量结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_optimization_{self.asset_name}_{self.strategy_info['class_name']}_{timestamp}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"批量优化完成")
            print(f"{'='*60}")
            print(f"结果已保存至: {filepath}")
            print(f"{'='*60}\n")
        
        return batch_results


def create_optimizer(
    data_path: str,
    strategy_path: str,
    objective: str = "sharpe_ratio",
    use_llm: bool = False,
    llm_config: Optional[UniversalLLMConfig] = None,
    **kwargs
) -> UniversalOptimizer:
    """
    创建优化器的工厂函数
    
    Args:
        data_path: 数据文件路径
        strategy_path: 策略文件路径
        objective: 优化目标
        use_llm: 是否使用LLM
        llm_config: LLM配置
        **kwargs: 其他参数
        
    Returns:
        优化器实例
    """
    return UniversalOptimizer(
        data_path=data_path,
        strategy_path=strategy_path,
        objective=objective,
        use_llm=use_llm,
        llm_config=llm_config,
        **kwargs
    )


if __name__ == "__main__":
    print("通用策略优化器")
    print("使用示例:")
    print("""
optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/my_strategy.py",
    objective="sharpe_ratio",
    use_llm=False
)

result = optimizer.optimize(n_trials=50)
print(json.dumps(result, indent=2, ensure_ascii=False))
""")
