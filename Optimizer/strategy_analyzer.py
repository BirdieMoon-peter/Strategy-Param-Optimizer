# -*- coding: utf-8 -*-
"""
策略分析模块
解析策略参数并与LLM交互生成搜索空间
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Any, Type, Optional
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STRATEGY_PARAMS, StrategyParam
from llm_client import LLMClient, get_llm_client


@dataclass
class SearchSpaceConfig:
    """搜索空间配置"""
    param_name: str
    param_type: str  # 'int', 'float'
    distribution: str  # 'uniform', 'log_uniform', 'int_uniform'
    min_value: float
    max_value: float
    step: Optional[float] = None
    priority: str = "medium"  # 'high', 'medium', 'low'


class StrategyAnalyzer:
    """
    策略分析器
    负责解析策略参数、与LLM交互、生成和调整搜索空间
    """
    
    def __init__(self, llm_client: LLMClient = None, use_llm: bool = True):
        """
        初始化策略分析器
        
        Args:
            llm_client: LLM客户端实例
            use_llm: 是否使用LLM（如果为False，使用默认配置）
        """
        self.llm_client = llm_client or get_llm_client()
        self.use_llm = use_llm and self.llm_client.check_connection()
        self.strategy_cache = {}
        
        if use_llm and not self.use_llm:
            print("[警告] LLM服务不可用，将使用默认搜索空间配置")
    
    def get_available_strategies(self) -> List[str]:
        """获取所有可用的策略名称"""
        return list(STRATEGY_PARAMS.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict]:
        """
        获取策略信息
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略信息字典
        """
        return STRATEGY_PARAMS.get(strategy_name)
    
    def load_strategy_class(self, strategy_name: str) -> Optional[Type]:
        """
        动态加载策略类
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略类
        """
        if strategy_name in self.strategy_cache:
            return self.strategy_cache[strategy_name]
        
        strategy_info = self.get_strategy_info(strategy_name)
        if not strategy_info:
            print(f"[错误] 未找到策略: {strategy_name}")
            return None
        
        try:
            module_path = strategy_info['module_path']
            class_name = strategy_info['class_name']
            
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            
            self.strategy_cache[strategy_name] = strategy_class
            return strategy_class
            
        except (ImportError, AttributeError) as e:
            print(f"[错误] 加载策略 {strategy_name} 失败: {e}")
            return None
    
    def generate_search_space(
        self, 
        strategy_name: str,
        use_llm_recommendations: bool = True
    ) -> Dict[str, SearchSpaceConfig]:
        """
        为策略生成搜索空间配置
        
        Args:
            strategy_name: 策略名称
            use_llm_recommendations: 是否使用LLM推荐
            
        Returns:
            搜索空间配置字典
        """
        strategy_info = self.get_strategy_info(strategy_name)
        if not strategy_info:
            return {}
        
        search_space = {}
        
        # 尝试使用LLM推荐
        llm_recommendations = None
        if use_llm_recommendations and self.use_llm:
            print(f"[LLM] 正在分析 {strategy_name} 的参数...")
            llm_recommendations = self.llm_client.analyze_strategy_params(strategy_info)
            
            if llm_recommendations:
                print(f"[LLM] 分析完成，获取到推荐配置")
                if 'recommendations' in llm_recommendations:
                    print(f"[LLM] 建议: {llm_recommendations['recommendations']}")
        
        # 构建搜索空间
        for param in strategy_info['params']:
            param_config = self._build_param_config(param, llm_recommendations)
            search_space[param.name] = param_config
        
        return search_space
    
    def _build_param_config(
        self, 
        param: StrategyParam, 
        llm_recommendations: Optional[Dict]
    ) -> SearchSpaceConfig:
        """
        构建单个参数的搜索空间配置
        
        Args:
            param: 策略参数对象
            llm_recommendations: LLM推荐配置
            
        Returns:
            SearchSpaceConfig对象
        """
        # 默认配置
        config = SearchSpaceConfig(
            param_name=param.name,
            param_type=param.param_type,
            distribution="int_uniform" if param.param_type == "int" else "uniform",
            min_value=param.min_value,
            max_value=param.max_value,
            step=param.step,
            priority="medium"
        )
        
        # 尝试应用LLM推荐
        if llm_recommendations and 'search_space' in llm_recommendations:
            llm_space = llm_recommendations['search_space']
            if param.name in llm_space:
                llm_param = llm_space[param.name]
                
                # 更新配置（保持类型安全）
                if 'min' in llm_param:
                    config.min_value = max(param.min_value, float(llm_param['min']))
                if 'max' in llm_param:
                    config.max_value = min(param.max_value, float(llm_param['max']))
                if 'distribution' in llm_param:
                    config.distribution = llm_param['distribution']
                if 'step' in llm_param:
                    config.step = float(llm_param['step'])
                if 'priority' in llm_param:
                    config.priority = llm_param['priority']
        
        # 确保min < max
        if config.min_value >= config.max_value:
            config.min_value = param.min_value
            config.max_value = param.max_value
        
        return config
    
    def adjust_search_space(
        self,
        current_space: Dict[str, SearchSpaceConfig],
        optimization_history: List[Dict],
        objective: str
    ) -> Dict[str, SearchSpaceConfig]:
        """
        根据优化历史动态调整搜索空间
        
        Args:
            current_space: 当前搜索空间配置
            optimization_history: 优化历史记录
            objective: 优化目标
            
        Returns:
            调整后的搜索空间配置
        """
        if not self.use_llm or len(optimization_history) < 10:
            return current_space
        
        # 将当前空间转换为LLM可读格式
        space_dict = {}
        for name, config in current_space.items():
            space_dict[name] = {
                "type": config.param_type,
                "distribution": config.distribution,
                "min": config.min_value,
                "max": config.max_value,
                "step": config.step
            }
        
        print(f"[LLM] 正在分析优化历史并调整搜索空间...")
        
        # 调用LLM分析
        adjustment = self.llm_client.analyze_optimization_history(
            optimization_history,
            {"search_space": space_dict},
            objective
        )
        
        # 应用调整
        adjusted_space = current_space.copy()
        
        if 'adjusted_space' in adjustment:
            for param_name, new_config in adjustment['adjusted_space'].items():
                if param_name in adjusted_space:
                    old_config = adjusted_space[param_name]
                    
                    # 更新配置
                    if 'min' in new_config:
                        new_min = float(new_config['min'])
                        # 不能超出原始边界太多
                        adjusted_space[param_name] = SearchSpaceConfig(
                            param_name=param_name,
                            param_type=old_config.param_type,
                            distribution=new_config.get('distribution', old_config.distribution),
                            min_value=new_min,
                            max_value=float(new_config.get('max', old_config.max_value)),
                            step=float(new_config.get('step', old_config.step)) if new_config.get('step') else old_config.step,
                            priority=new_config.get('priority', old_config.priority)
                        )
        
        # 打印关键发现
        if 'key_findings' in adjustment:
            print("[LLM] 关键发现:")
            for finding in adjustment['key_findings'][:5]:
                print(f"  - {finding}")
        
        if 'next_recommendations' in adjustment:
            print(f"[LLM] 下轮建议: {adjustment['next_recommendations']}")
        
        return adjusted_space
    
    def print_search_space(self, search_space: Dict[str, SearchSpaceConfig]):
        """打印搜索空间配置"""
        print("\n" + "="*60)
        print("搜索空间配置")
        print("="*60)
        
        for name, config in search_space.items():
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(config.priority, "⚪")
            print(f"\n{priority_emoji} {name}:")
            print(f"   类型: {config.param_type}")
            print(f"   分布: {config.distribution}")
            print(f"   范围: [{config.min_value}, {config.max_value}]")
            if config.step:
                print(f"   步长: {config.step}")
        
        print("\n" + "="*60)


def convert_to_optuna_space(search_space: Dict[str, SearchSpaceConfig]) -> Dict:
    """
    将SearchSpaceConfig转换为Optuna格式
    
    Args:
        search_space: 搜索空间配置字典
        
    Returns:
        Optuna可用的参数空间定义
    """
    optuna_space = {}
    
    for name, config in search_space.items():
        optuna_space[name] = {
            "type": config.param_type,
            "distribution": config.distribution,
            "low": config.min_value,
            "high": config.max_value,
            "step": config.step
        }
    
    return optuna_space


if __name__ == "__main__":
    # 测试代码
    analyzer = StrategyAnalyzer(use_llm=False)
    
    print("可用策略:")
    for strategy in analyzer.get_available_strategies():
        print(f"  - {strategy}")
    
    print("\n" + "="*60)
    
    # 测试生成搜索空间
    test_strategy = "AberrationStrategy"
    print(f"\n为 {test_strategy} 生成搜索空间:")
    
    space = analyzer.generate_search_space(test_strategy, use_llm_recommendations=False)
    analyzer.print_search_space(space)
