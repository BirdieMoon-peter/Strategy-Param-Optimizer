# -*- coding: utf-8 -*-
"""
参数空间配置示例
演示如何自定义参数空间规则
"""

from param_space_optimizer import ParamSpaceOptimizer, ParameterSpaceRule
from config import StrategyParam


# ============================================
# 示例 1: 使用默认规则
# ============================================
def example_default_rules():
    """使用内置的默认规则"""
    print("=" * 70)
    print("示例 1: 使用默认规则")
    print("=" * 70)
    
    # 创建测试参数
    params = [
        StrategyParam("period", "int", 20, "布林带周期"),
        StrategyParam("devfactor", "float", 2.0, "标准差倍数"),
        StrategyParam("fast_period", "int", 10, "快速均线"),
        StrategyParam("slow_period", "int", 30, "慢速均线"),
    ]
    
    # 使用默认规则
    optimizer = ParamSpaceOptimizer(verbose=True)
    optimized = optimizer.generate_space(params)
    
    return optimized


# ============================================
# 示例 2: 添加自定义规则
# ============================================
def example_custom_rules():
    """添加自定义规则"""
    print("\n" + "=" * 70)
    print("示例 2: 添加自定义规则")
    print("=" * 70)
    
    # 创建优化器
    optimizer = ParamSpaceOptimizer(verbose=True)
    
    # 添加自定义规则：针对特定策略的参数
    custom_rule = ParameterSpaceRule(
        param_pattern=r"my_special_param",
        min_multiplier=0.8,
        max_multiplier=1.2,
        min_absolute=1.0,
        max_absolute=10.0,
        distribution="uniform",
        priority="high",
        description="我的特殊参数：窄范围搜索"
    )
    optimizer.add_custom_rule("my_special_param", custom_rule)
    
    # 测试参数
    params = [
        StrategyParam("my_special_param", "float", 5.0, "特殊参数"),
        StrategyParam("normal_param", "int", 20, "普通参数"),
    ]
    
    optimized = optimizer.generate_space(params)
    
    return optimized


# ============================================
# 示例 3: 分析优化结果
# ============================================
def example_analyze_results():
    """分析优化结果并获取建议"""
    print("\n" + "=" * 70)
    print("示例 3: 分析优化结果")
    print("=" * 70)
    
    # 创建参数空间
    params = [
        StrategyParam("period", "int", 20, "布林带周期"),
        StrategyParam("devfactor", "float", 2.0, "标准差倍数"),
    ]
    
    optimizer = ParamSpaceOptimizer(verbose=False)
    optimized = optimizer.generate_space(params)
    
    # 模拟优化结果（最优参数在边界上）
    best_params = {
        "period": 10,  # 接近下界
        "devfactor": 4.8,  # 接近上界
    }
    
    # 分析结果
    analysis = optimizer.analyze_optimization_results(best_params, optimized)
    
    print("\n参数空间利用情况:")
    for param_name, util in analysis["space_utilization"].items():
        print(f"  {param_name}:")
        print(f"    当前值: {util['value']:.4f}")
        print(f"    范围: [{util['min']:.4f}, {util['max']:.4f}]")
        print(f"    相对位置: {util['relative_position']:.2%}")
    
    if analysis["suggestions"]:
        print("\n💡 优化建议:")
        for suggestion in analysis["suggestions"]:
            print(f"  • {suggestion}")
    
    return analysis


# ============================================
# 示例 4: 生成改进的参数空间
# ============================================
def example_refined_space():
    """基于优化结果生成改进的参数空间"""
    print("\n" + "=" * 70)
    print("示例 4: 生成改进的参数空间")
    print("=" * 70)
    
    # 原始参数空间
    params = [
        StrategyParam("fast_period", "int", 10, "快速均线"),
        StrategyParam("slow_period", "int", 30, "慢速均线"),
        StrategyParam("threshold", "float", 0.02, "阈值"),
    ]
    
    optimizer = ParamSpaceOptimizer(verbose=False)
    original_space = optimizer.generate_space(params)
    
    print("\n原始参数空间:")
    optimizer._print_space_summary(original_space)
    
    # 第一次优化的最优参数
    best_params_round1 = {
        "fast_period": 5,      # 接近下界
        "slow_period": 48,     # 接近上界
        "threshold": 0.025,    # 在中间
    }
    
    # 生成改进的参数空间
    refined_space = optimizer.suggest_refined_space(
        best_params_round1,
        original_space,
        expansion_factor=1.5
    )
    
    print("\n改进后的参数空间:")
    optimizer._print_space_summary(refined_space)
    
    return refined_space


# ============================================
# 示例 5: 完整的优化流程
# ============================================
def example_complete_workflow():
    """完整的参数空间优化工作流"""
    print("\n" + "=" * 70)
    print("示例 5: 完整工作流")
    print("=" * 70)
    
    # 策略参数
    strategy_params = [
        StrategyParam("rsi_period", "int", 14, "RSI周期"),
        StrategyParam("rsi_oversold", "int", 30, "RSI超卖阈值"),
        StrategyParam("rsi_overbought", "int", 70, "RSI超买阈值"),
        StrategyParam("stop_loss", "float", 0.05, "止损比例"),
        StrategyParam("take_profit", "float", 0.10, "止盈比例"),
    ]
    
    # 第一轮优化
    print("\n【第一轮优化】")
    optimizer = ParamSpaceOptimizer(verbose=True)
    space_round1 = optimizer.generate_space(strategy_params)
    
    # 模拟第一轮优化结果
    best_params_round1 = {
        "rsi_period": 7,       # 接近下界
        "rsi_oversold": 28,    # 中间偏下
        "rsi_overbought": 72,  # 中间偏上
        "stop_loss": 0.025,    # 接近下界
        "take_profit": 0.095,  # 在中间
    }
    
    # 分析结果
    print("\n【参数空间分析】")
    analysis = optimizer.analyze_optimization_results(best_params_round1, space_round1)
    
    if analysis["suggestions"]:
        print("\n💡 建议:")
        for suggestion in analysis["suggestions"]:
            print(f"  • {suggestion}")
    
    # 第二轮优化（使用改进的参数空间）
    print("\n【第二轮优化 - 使用改进的参数空间】")
    space_round2 = optimizer.suggest_refined_space(
        best_params_round1,
        space_round1,
        expansion_factor=1.5
    )
    
    print("\n✅ 工作流完成！")
    print("在实际使用中，您可以使用 space_round2 进行第二轮优化。")
    
    return space_round2


# ============================================
# 示例 6: 针对特定策略类型的优化
# ============================================
def example_strategy_specific():
    """针对特定策略类型定制参数空间"""
    print("\n" + "=" * 70)
    print("示例 6: 策略类型特定优化")
    print("=" * 70)
    
    # 创建优化器并添加策略特定规则
    optimizer = ParamSpaceOptimizer(verbose=True)
    
    # 针对网格策略的规则
    grid_spacing_rule = ParameterSpaceRule(
        param_pattern=r"grid_spacing|grid_step",
        min_multiplier=0.7,
        max_multiplier=1.5,
        min_absolute=0.005,
        max_absolute=0.1,
        distribution="uniform",
        priority="high",
        description="网格间距：5%-10%之间"
    )
    optimizer.add_custom_rule("grid_spacing", grid_spacing_rule)
    
    # 针对马丁格尔策略的倍数规则
    martingale_rule = ParameterSpaceRule(
        param_pattern=r".*martingale.*|.*multiplier.*",
        min_multiplier=0.9,
        max_multiplier=1.1,
        min_absolute=1.5,
        max_absolute=3.0,
        distribution="uniform",
        priority="high",
        description="马丁格尔倍数：1.5-3.0之间（高风险参数）"
    )
    optimizer.add_custom_rule("martingale_multiplier", martingale_rule)
    
    # 测试参数
    params = [
        StrategyParam("grid_spacing", "float", 0.02, "网格间距"),
        StrategyParam("martingale_multiplier", "float", 2.0, "倍数"),
        StrategyParam("period", "int", 20, "周期"),
    ]
    
    optimized = optimizer.generate_space(params, strategy_type="GridMartingale")
    
    return optimized


# ============================================
# 运行所有示例
# ============================================
if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("参数空间优化器使用示例")
    print("🚀" * 35)
    
    # 运行示例
    example_default_rules()
    example_custom_rules()
    example_analyze_results()
    example_refined_space()
    example_complete_workflow()
    example_strategy_specific()
    
    print("\n" + "✅" * 35)
    print("所有示例运行完成！")
    print("✅" * 35 + "\n")
    
    print("提示：")
    print("1. 使用默认规则可以处理大多数常见参数")
    print("2. 针对特殊策略，可以添加自定义规则")
    print("3. 分析优化结果可以帮助改进参数空间")
    print("4. 迭代优化可以逐步缩小搜索范围，提高效率")
