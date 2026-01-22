# -*- coding: utf-8 -*-
"""
参数空间优化器
提供智能的参数空间生成和优化功能
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import StrategyParam


@dataclass
class ParameterSpaceRule:
    """参数空间规则"""
    param_pattern: str  # 参数名称模式（支持正则）
    min_multiplier: float = 0.3  # 最小值倍数
    max_multiplier: float = 3.0  # 最大值倍数
    min_absolute: Optional[float] = None  # 绝对最小值
    max_absolute: Optional[float] = None  # 绝对最大值
    step_size: Optional[float] = None  # 步长
    distribution: str = "uniform"  # 分布类型: uniform, log_uniform, int_uniform
    priority: str = "medium"  # 优先级: high, medium, low
    description: str = ""


class ParamSpaceOptimizer:
    """
    参数空间优化器
    
    功能：
    1. 根据参数类型和名称智能生成搜索空间
    2. 支持自定义参数空间规则
    3. 处理参数间的约束关系
    4. 提供参数空间分析和建议
    """
    
    # 内置的参数空间规则
    BUILTIN_RULES = {
        # 周期类参数（period, window, length等）
        "period": ParameterSpaceRule(
            param_pattern=r".*period.*|.*window.*|.*length.*",
            min_multiplier=0.5,
            max_multiplier=2.5,
            min_absolute=5,
            max_absolute=200,
            distribution="int_uniform",
            priority="high",
            description="周期类参数：建议范围为默认值的50%-250%，最小5，最大200"
        ),
        
        # 标准差倍数（std_dev, devfactor等）
        "std_dev": ParameterSpaceRule(
            param_pattern=r".*std.*|.*dev.*factor.*",
            min_multiplier=0.5,
            max_multiplier=2.0,
            min_absolute=0.5,
            max_absolute=5.0,
            distribution="uniform",
            priority="high",
            description="标准差倍数：建议范围0.5-5.0"
        ),
        
        # 阈值类参数（threshold, limit等）
        "threshold": ParameterSpaceRule(
            param_pattern=r".*threshold.*|.*limit.*",
            min_multiplier=0.5,
            max_multiplier=2.0,
            min_absolute=0.01,
            max_absolute=0.5,
            distribution="uniform",
            priority="medium",
            description="阈值类参数：建议小范围调整"
        ),
        
        # RSI类阈值
        "rsi_threshold": ParameterSpaceRule(
            param_pattern=r".*rsi.*sold.*|.*rsi.*bought.*",
            min_multiplier=0.7,
            max_multiplier=1.3,
            min_absolute=10,
            max_absolute=90,
            distribution="int_uniform",
            priority="medium",
            description="RSI阈值：建议范围10-90"
        ),
        
        # 百分比参数（percent, ratio等）
        "percent": ParameterSpaceRule(
            param_pattern=r".*percent.*|.*ratio.*",
            min_multiplier=0.5,
            max_multiplier=2.0,
            min_absolute=0.01,
            max_absolute=1.0,
            distribution="uniform",
            priority="medium",
            description="百分比参数：建议范围0.01-1.0"
        ),
        
        # 快速均线周期
        "fast_period": ParameterSpaceRule(
            param_pattern=r".*fast.*|.*short.*",
            min_multiplier=0.5,
            max_multiplier=2.0,
            min_absolute=3,
            max_absolute=50,
            distribution="int_uniform",
            priority="high",
            description="快速周期：建议范围3-50"
        ),
        
        # 慢速均线周期
        "slow_period": ParameterSpaceRule(
            param_pattern=r".*slow.*|.*long.*",
            min_multiplier=0.5,
            max_multiplier=2.0,
            min_absolute=10,
            max_absolute=200,
            distribution="int_uniform",
            priority="high",
            description="慢速周期：建议范围10-200"
        ),
        
        # 止损止盈参数
        "stop_loss": ParameterSpaceRule(
            param_pattern=r".*stop.*loss.*|.*sl.*",
            min_multiplier=0.5,
            max_multiplier=2.0,
            min_absolute=0.01,
            max_absolute=0.2,
            distribution="uniform",
            priority="high",
            description="止损参数：建议范围1%-20%"
        ),
        
        "take_profit": ParameterSpaceRule(
            param_pattern=r".*take.*profit.*|.*tp.*",
            min_multiplier=0.5,
            max_multiplier=2.0,
            min_absolute=0.02,
            max_absolute=0.5,
            distribution="uniform",
            priority="high",
            description="止盈参数：建议范围2%-50%"
        ),
    }
    
    def __init__(
        self,
        custom_rules: Optional[Dict[str, ParameterSpaceRule]] = None,
        verbose: bool = True
    ):
        """
        初始化参数空间优化器
        
        Args:
            custom_rules: 自定义规则字典
            verbose: 是否打印详细信息
        """
        self.rules = self.BUILTIN_RULES.copy()
        if custom_rules:
            self.rules.update(custom_rules)
        self.verbose = verbose
        self.param_constraints = []  # 参数约束列表
    
    def generate_space(
        self,
        params: List[StrategyParam],
        strategy_type: Optional[str] = None
    ) -> List[StrategyParam]:
        """
        生成优化的参数空间
        
        Args:
            params: 原始参数列表
            strategy_type: 策略类型（可选，用于特殊处理）
            
        Returns:
            优化后的参数列表
        """
        optimized_params = []
        
        for param in params:
            optimized_param = self._optimize_param_space(param)
            optimized_params.append(optimized_param)
        
        # 处理参数约束
        optimized_params = self._apply_constraints(optimized_params)
        
        if self.verbose:
            self._print_space_summary(optimized_params)
        
        return optimized_params
    
    def _optimize_param_space(self, param: StrategyParam) -> StrategyParam:
        """
        优化单个参数的搜索空间
        
        Args:
            param: 原始参数
            
        Returns:
            优化后的参数
        """
        import re
        
        # 查找匹配的规则
        matched_rule = None
        for rule_name, rule in self.rules.items():
            if re.search(rule.param_pattern, param.name, re.IGNORECASE):
                matched_rule = rule
                break
        
        if matched_rule:
            # 使用规则生成参数空间
            if param.param_type == "int":
                min_val = max(
                    int(param.default_value * matched_rule.min_multiplier),
                    int(matched_rule.min_absolute) if matched_rule.min_absolute else 1
                )
                max_val = min(
                    int(param.default_value * matched_rule.max_multiplier),
                    int(matched_rule.max_absolute) if matched_rule.max_absolute else int(param.default_value * 10)
                )
                step = matched_rule.step_size if matched_rule.step_size else 1
            else:  # float
                min_val = max(
                    param.default_value * matched_rule.min_multiplier,
                    matched_rule.min_absolute if matched_rule.min_absolute else 0.0001
                )
                max_val = min(
                    param.default_value * matched_rule.max_multiplier,
                    matched_rule.max_absolute if matched_rule.max_absolute else param.default_value * 10
                )
                step = matched_rule.step_size
            
            # 确保 min < max
            if min_val >= max_val:
                if param.param_type == "int":
                    min_val = max(1, int(param.default_value * 0.5))
                    max_val = int(param.default_value * 2)
                else:
                    min_val = max(0.0001, param.default_value * 0.5)
                    max_val = param.default_value * 2
            
            # 创建优化后的参数
            optimized = StrategyParam(
                name=param.name,
                param_type=param.param_type,
                default_value=param.default_value,
                description=f"{param.description} [{matched_rule.description}]",
                min_value=min_val,
                max_value=max_val,
                step=step
            )
            
        else:
            # 使用默认规则（更保守的范围）
            if param.param_type == "int":
                min_val = max(1, int(param.default_value * 0.5))
                max_val = int(param.default_value * 2)
                step = 1
            else:
                min_val = max(0.0001, param.default_value * 0.5)
                max_val = param.default_value * 2
                step = None
            
            optimized = StrategyParam(
                name=param.name,
                param_type=param.param_type,
                default_value=param.default_value,
                description=param.description,
                min_value=min_val,
                max_value=max_val,
                step=step
            )
        
        return optimized
    
    def _apply_constraints(self, params: List[StrategyParam]) -> List[StrategyParam]:
        """
        应用参数约束
        
        常见约束：
        - 快速周期 < 慢速周期
        - RSI超卖 < RSI超买
        - 止损 < 止盈
        """
        param_dict = {p.name: p for p in params}
        
        # 快速/慢速周期约束
        fast_params = [p for p in params if 'fast' in p.name.lower() or 'short' in p.name.lower()]
        slow_params = [p for p in params if 'slow' in p.name.lower() or 'long' in p.name.lower()]
        
        if fast_params and slow_params:
            for fast_p in fast_params:
                for slow_p in slow_params:
                    if 'period' in fast_p.name.lower() and 'period' in slow_p.name.lower():
                        # 确保快速周期的最大值 < 慢速周期的最小值
                        if fast_p.max_value >= slow_p.min_value:
                            # 调整范围
                            mid_point = (fast_p.default_value + slow_p.default_value) / 2
                            fast_p.max_value = min(fast_p.max_value, int(mid_point) - 1)
                            slow_p.min_value = max(slow_p.min_value, int(mid_point) + 1)
                        
                        self.param_constraints.append({
                            "type": "less_than",
                            "param1": fast_p.name,
                            "param2": slow_p.name,
                            "description": f"{fast_p.name} 必须小于 {slow_p.name}"
                        })
        
        # RSI阈值约束
        rsi_oversold = [p for p in params if 'oversold' in p.name.lower()]
        rsi_overbought = [p for p in params if 'overbought' in p.name.lower()]
        
        if rsi_oversold and rsi_overbought:
            for oversold_p in rsi_oversold:
                for overbought_p in rsi_overbought:
                    # 确保超卖 < 超买
                    oversold_p.max_value = min(oversold_p.max_value, 45)
                    overbought_p.min_value = max(overbought_p.min_value, 55)
                    
                    self.param_constraints.append({
                        "type": "less_than",
                        "param1": oversold_p.name,
                        "param2": overbought_p.name,
                        "description": f"{oversold_p.name} 必须小于 {overbought_p.name}"
                    })
        
        # 上下轨标准差约束（可能相等或有关系）
        std_upper = [p for p in params if 'upper' in p.name.lower() and ('std' in p.name.lower() or 'dev' in p.name.lower())]
        std_lower = [p for p in params if 'lower' in p.name.lower() and ('std' in p.name.lower() or 'dev' in p.name.lower())]
        
        if std_upper and std_lower:
            for upper_p in std_upper:
                for lower_p in std_lower:
                    self.param_constraints.append({
                        "type": "independent",
                        "param1": upper_p.name,
                        "param2": lower_p.name,
                        "description": f"{upper_p.name} 和 {lower_p.name} 独立优化（可能不对称）"
                    })
        
        return params
    
    def _print_space_summary(self, params: List[StrategyParam]):
        """打印参数空间摘要"""
        print(f"\n{'='*70}")
        print("优化后的参数空间")
        print(f"{'='*70}")
        
        for param in params:
            range_str = f"[{param.min_value}, {param.max_value}]"
            if param.step:
                range_str += f" (步长: {param.step})"
            
            print(f"  {param.name:20s} | 类型: {param.param_type:6s} | 范围: {range_str}")
            if param.description and "[" in param.description:
                desc_parts = param.description.split("[")
                if len(desc_parts) > 1:
                    rule_desc = desc_parts[1].rstrip("]")
                    print(f"  {'':20s}   提示: {rule_desc}")
        
        if self.param_constraints:
            print(f"\n{'='*70}")
            print("参数约束")
            print(f"{'='*70}")
            for constraint in self.param_constraints:
                print(f"  - {constraint['description']}")
        
        print(f"{'='*70}\n")
    
    def add_custom_rule(self, name: str, rule: ParameterSpaceRule):
        """添加自定义规则"""
        self.rules[name] = rule
    
    def analyze_optimization_results(
        self,
        best_params: Dict[str, float],
        param_space: List[StrategyParam]
    ) -> Dict[str, Any]:
        """
        分析优化结果，判断参数空间是否合理
        
        Args:
            best_params: 最优参数
            param_space: 参数空间定义
            
        Returns:
            分析结果和建议
        """
        analysis = {
            "boundary_params": [],  # 在边界的参数
            "suggestions": [],  # 改进建议
            "space_utilization": {}  # 空间利用率
        }
        
        param_dict = {p.name: p for p in param_space}
        
        for param_name, param_value in best_params.items():
            if param_name not in param_dict:
                continue
            
            param_def = param_dict[param_name]
            param_range = param_def.max_value - param_def.min_value
            
            # 计算相对位置 (0-1)
            if param_range > 0:
                relative_pos = (param_value - param_def.min_value) / param_range
            else:
                relative_pos = 0.5
            
            analysis["space_utilization"][param_name] = {
                "value": param_value,
                "min": param_def.min_value,
                "max": param_def.max_value,
                "relative_position": relative_pos
            }
            
            # 检查是否在边界附近（10%阈值）
            if relative_pos < 0.1:
                analysis["boundary_params"].append({
                    "param": param_name,
                    "side": "lower",
                    "value": param_value,
                    "boundary": param_def.min_value
                })
                analysis["suggestions"].append(
                    f"参数 '{param_name}' 接近下界 ({param_value:.4f} ≈ {param_def.min_value:.4f})，"
                    f"建议扩大搜索范围到更小的值"
                )
            elif relative_pos > 0.9:
                analysis["boundary_params"].append({
                    "param": param_name,
                    "side": "upper",
                    "value": param_value,
                    "boundary": param_def.max_value
                })
                analysis["suggestions"].append(
                    f"参数 '{param_name}' 接近上界 ({param_value:.4f} ≈ {param_def.max_value:.4f})，"
                    f"建议扩大搜索范围到更大的值"
                )
        
        return analysis
    
    def suggest_refined_space(
        self,
        best_params: Dict[str, float],
        param_space: List[StrategyParam],
        expansion_factor: float = 1.5
    ) -> List[StrategyParam]:
        """
        基于优化结果建议改进的参数空间
        
        Args:
            best_params: 最优参数
            param_space: 当前参数空间
            expansion_factor: 扩展因子
            
        Returns:
            改进后的参数空间
        """
        refined_space = []
        param_dict = {p.name: p for p in param_space}
        
        for param_name, param_value in best_params.items():
            if param_name not in param_dict:
                continue
            
            param_def = param_dict[param_name]
            param_range = param_def.max_value - param_def.min_value
            relative_pos = (param_value - param_def.min_value) / param_range if param_range > 0 else 0.5
            
            new_min = param_def.min_value
            new_max = param_def.max_value
            
            # 如果参数在边界附近，扩展范围
            if relative_pos < 0.1:  # 接近下界
                if param_def.param_type == "int":
                    new_min = max(1, int(param_def.min_value / expansion_factor))
                else:
                    new_min = max(0.0001, param_def.min_value / expansion_factor)
            elif relative_pos > 0.9:  # 接近上界
                if param_def.param_type == "int":
                    new_max = int(param_def.max_value * expansion_factor)
                else:
                    new_max = param_def.max_value * expansion_factor
            else:
                # 参数在中间，可以缩小范围聚焦搜索
                center = param_value
                range_size = param_range * 0.5  # 缩小到原来的50%
                
                if param_def.param_type == "int":
                    new_min = max(1, int(center - range_size / 2))
                    new_max = int(center + range_size / 2)
                else:
                    new_min = max(0.0001, center - range_size / 2)
                    new_max = center + range_size / 2
            
            refined_param = StrategyParam(
                name=param_def.name,
                param_type=param_def.param_type,
                default_value=param_value,  # 使用最优值作为新的默认值
                description=param_def.description,
                min_value=new_min,
                max_value=new_max,
                step=param_def.step
            )
            refined_space.append(refined_param)
        
        return refined_space


# 使用示例
if __name__ == "__main__":
    # 创建测试参数
    test_params = [
        StrategyParam("fast_period", "int", 10, "快速均线周期"),
        StrategyParam("slow_period", "int", 30, "慢速均线周期"),
        StrategyParam("std_dev_upper", "float", 2.0, "上轨标准差"),
        StrategyParam("rsi_oversold", "int", 30, "RSI超卖阈值"),
        StrategyParam("rsi_overbought", "int", 70, "RSI超买阈值"),
    ]
    
    # 创建优化器
    optimizer = ParamSpaceOptimizer(verbose=True)
    
    # 生成优化的参数空间
    optimized = optimizer.generate_space(test_params)
    
    # 模拟优化结果
    best_params = {
        "fast_period": 5,  # 接近下界
        "slow_period": 45,  # 接近上界
        "std_dev_upper": 2.5,  # 中间
        "rsi_oversold": 28,
        "rsi_overbought": 72,
    }
    
    # 分析结果
    print("\n分析优化结果:")
    analysis = optimizer.analyze_optimization_results(best_params, optimized)
    
    if analysis["suggestions"]:
        print("\n建议:")
        for suggestion in analysis["suggestions"]:
            print(f"  - {suggestion}")
    
    # 生成改进的参数空间
    print("\n生成改进的参数空间:")
    refined = optimizer.suggest_refined_space(best_params, optimized)
    optimizer._print_space_summary(refined)
