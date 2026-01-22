# -*- coding: utf-8 -*-
"""
通用策略优化器
支持任意标的和策略的优化
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

from universal_llm_client import UniversalLLMClient, UniversalLLMConfig
from backtest_engine import BacktestEngine, BacktestResult
from bayesian_optimizer import BayesianOptimizer
from config import StrategyParam, BayesianOptConfig
from strategy_analyzer import SearchSpaceConfig as ParamSearchSpaceConfig
from param_space_optimizer import ParamSpaceOptimizer

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
    """
    
    def __init__(
        self,
        data_path: str,
        strategy_path: str,
        objective: str = "sharpe_ratio",
        use_llm: bool = False,
        llm_config: Optional[UniversalLLMConfig] = None,
        output_dir: str = "./optimization_results",
        verbose: bool = True,
        target_params: Optional[List[str]] = None
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
        """
        self.data_path = data_path
        self.strategy_path = strategy_path
        self.objective = objective
        self.use_llm = use_llm
        self.verbose = verbose
        self.target_params = target_params  # 指定要优化的参数
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化参数空间优化器（需要在加载策略之前初始化）
        self.param_space_optimizer = ParamSpaceOptimizer(verbose=self.verbose)
        
        # 加载数据
        self.data = self._load_data()
        # 从文件名提取资产名称，去除 _processed 后缀
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
        
        # 初始化回测引擎
        self.backtest_engine = BacktestEngine(
            data=self.data,
            strategy_class=self.strategy_class,
            initial_cash=100000.0,
            commission=0.001
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"通用策略优化器初始化完成")
            print(f"{'='*60}")
            print(f"标的: {self.asset_name}")
            print(f"策略: {self.strategy_info['class_name']}")
            print(f"优化目标: {objective}")
            print(f"使用LLM: {'是' if use_llm else '否'}")
            print(f"数据点数: {len(self.data)}")
            print(f"{'='*60}\n")
    
    def _load_data(self) -> pd.DataFrame:
        """加载标的数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # 验证必需的列
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"数据文件缺少必需的列: {missing_columns}")
        
        # 转换datetime列
        df['datetime'] = pd.to_datetime(df['datetime'])
        
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
        
        # 查找策略类（继承自backtrader.Strategy）
        strategy_classes = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, 'params') and obj.__module__ == module_name:
                strategy_classes.append(obj)
        
        if not strategy_classes:
            raise ValueError(f"未在策略文件中找到有效的策略类: {self.strategy_path}")
        
        if len(strategy_classes) > 1:
            if self.verbose:
                print(f"[警告] 发现多个策略类，将使用第一个: {strategy_classes[0].__name__}")
        
        strategy_class = strategy_classes[0]
        
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
        
        if hasattr(strategy_class, 'params'):
            for param_name in dir(strategy_class.params):
                if not param_name.startswith('_'):
                    default_value = getattr(strategy_class.params, param_name)
                    
                    # 推断参数类型
                    param_type = type(default_value).__name__
                    
                    # 跳过非数值类型
                    if not isinstance(default_value, (int, float)):
                        continue
                    
                    all_param_names.append(param_name)
                    
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
        
        return params
    
    def optimize(
        self,
        n_trials: int = 50,
        bayesian_config: Optional[BayesianOptConfig] = None
    ) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            n_trials: 优化试验次数
            bayesian_config: 贝叶斯优化配置
            
        Returns:
            优化结果字典
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"开始优化流程")
            print(f"{'='*60}\n")
        
        # 1. 构建搜索空间
        search_space_config = self._build_search_space()
        
        # 将 SearchSpaceConfig 转换为优化器需要的格式
        search_space = self._convert_search_space(search_space_config)
        
        # 2. 执行贝叶斯优化
        if bayesian_config is None:
            bayesian_config = BayesianOptConfig(
                n_trials=n_trials,
                n_rounds=1,  # 单轮优化
                sampler="tpe"
            )
        
        optimizer = BayesianOptimizer(
            config=bayesian_config,
            backtest_engine=self.backtest_engine,
            use_llm=False,  # 使用我们自己的 LLM 客户端
            verbose=self.verbose
        )
        
        # 调用优化方法
        opt_result = optimizer.optimize_single_objective(
            strategy_class=self.strategy_class,
            strategy_name=self.strategy_info['class_name'],
            data=self.data,
            objective=self.objective,
            search_space=search_space,
            n_trials=n_trials,
            verbose=self.verbose
        )
        
        # 提取回测结果
        best_result = opt_result.backtest_result
        best_params = opt_result.best_params  # 从优化结果中获取最优参数
        
        # 3. 分析参数空间使用情况
        if self.verbose:
            print(f"\n{'='*60}")
            print("参数空间分析")
            print(f"{'='*60}")
        
        param_analysis = self.param_space_optimizer.analyze_optimization_results(
            best_params,
            search_space_config.strategy_params
        )
        
        if self.verbose and param_analysis["suggestions"]:
            print("\n💡 参数空间优化建议:")
            for suggestion in param_analysis["suggestions"]:
                print(f"  • {suggestion}")
            print(f"{'='*60}\n")
        
        # 4. 生成详细结果（包含LLM解释和参数空间分析）
        result = self._generate_result(best_result)
        result["param_space_analysis"] = param_analysis
        
        # 5. 保存结果
        output_path = self._save_result(result)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"优化完成")
            print(f"{'='*60}")
            print(f"结果已保存至: {output_path}")
            print(f"{'='*60}\n")
        
        return result
    
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
        result = {
            "optimization_info": {
                "asset_name": self.asset_name,
                "strategy_name": self.strategy_info['class_name'],
                "optimization_objective": self.objective,
                "optimization_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_range": {
                    "start": self.data['datetime'].min().strftime("%Y-%m-%d"),
                    "end": self.data['datetime'].max().strftime("%Y-%m-%d"),
                    "total_days": len(self.data)
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
        filepath = self.output_dir / filename
        
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
        filepath = self.output_dir / filename
        
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
