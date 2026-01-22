# -*- coding: utf-8 -*-
"""
通用优化器使用示例
包含多种使用场景的完整示例代码
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from universal_optimizer import UniversalOptimizer
from universal_llm_client import UniversalLLMConfig


# ============================================================================
# 示例1: 最简单的使用 - 不使用LLM
# ============================================================================

def example_basic():
    """基本示例：不使用LLM的快速优化"""
    print("\n" + "="*60)
    print("示例1: 基本优化（不使用LLM）")
    print("="*60 + "\n")
    
    optimizer = UniversalOptimizer(
        data_path="data/BTC.csv",
        strategy_path="example_strategy.py",
        objective="sharpe_ratio",
        use_llm=False,
        output_dir="./demo_results",
        verbose=True
    )
    
    result = optimizer.optimize(n_trials=30)
    
    print("\n✅ 优化完成！")
    print("最优参数:", result['best_parameters'])
    print("夏普比率:", result['performance_metrics']['sharpe_ratio'])


# ============================================================================
# 示例2: 使用OpenAI LLM
# ============================================================================

def example_with_openai():
    """使用OpenAI GPT-4辅助优化"""
    print("\n" + "="*60)
    print("示例2: 使用OpenAI LLM辅助优化")
    print("="*60 + "\n")
    
    # 配置OpenAI
    llm_config = UniversalLLMConfig(
        api_type="openai",
        base_url="https://api.openai.com/v1",
        model_name="gpt-4",
        api_key="sk-your-api-key-here",  # 替换为实际密钥
        temperature=0.7
    )
    
    optimizer = UniversalOptimizer(
        data_path="data/BTC.csv",
        strategy_path="example_strategy.py",
        objective="sharpe_ratio",
        use_llm=True,
        llm_config=llm_config,
        output_dir="./demo_results",
        verbose=True
    )
    
    result = optimizer.optimize(n_trials=30)
    
    print("\n✅ 优化完成！")
    print("最优参数:", result['best_parameters'])
    
    # 查看LLM的解释
    explanation = result['llm_explanation']
    print("\n💡 LLM分析:")
    print("参数解释:", explanation['parameter_explanation'])
    print("\n关键洞察:")
    for insight in explanation['key_insights']:
        print(f"  • {insight}")


# ============================================================================
# 示例3: 使用本地Ollama
# ============================================================================

def example_with_ollama():
    """使用本地Ollama模型"""
    print("\n" + "="*60)
    print("示例3: 使用本地Ollama")
    print("="*60 + "\n")
    
    llm_config = UniversalLLMConfig(
        api_type="ollama",
        base_url="http://localhost:11434",
        model_name="qwen",  # 或 "xuanyuan"
        api_key="",
        timeout=120
    )
    
    optimizer = UniversalOptimizer(
        data_path="data/BTC.csv",
        strategy_path="example_strategy.py",
        objective="sharpe_ratio",
        use_llm=True,
        llm_config=llm_config,
        output_dir="./demo_results",
        verbose=True
    )
    
    result = optimizer.optimize(n_trials=30)
    
    print("\n✅ 优化完成！")


# ============================================================================
# 示例4: 批量优化多个目标
# ============================================================================

def example_batch_optimization():
    """批量优化：同时优化多个目标"""
    print("\n" + "="*60)
    print("示例4: 批量优化多个目标")
    print("="*60 + "\n")
    
    optimizer = UniversalOptimizer(
        data_path="data/BTC.csv",
        strategy_path="example_strategy.py",
        use_llm=False,
        output_dir="./demo_results",
        verbose=True
    )
    
    # 同时优化3个目标
    results = optimizer.batch_optimize(
        objectives=["sharpe_ratio", "annual_return", "calmar_ratio"],
        n_trials_per_objective=30
    )
    
    print("\n✅ 批量优化完成！")
    print("\n各目标的最优参数对比:")
    
    for obj, result in results['results'].items():
        print(f"\n目标: {obj}")
        print(f"  参数: {result['best_parameters']}")
        metrics = result['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  年化收益: {metrics['annual_return']:.2f}%")
        print(f"  最大回撤: {metrics['max_drawdown']:.2f}%")


# ============================================================================
# 示例5: 优化不同的标的和策略
# ============================================================================

def example_multiple_assets_strategies():
    """演示如何对不同标的和策略进行优化"""
    print("\n" + "="*60)
    print("示例5: 多标的、多策略优化")
    print("="*60 + "\n")
    
    # 定义要测试的配置
    configs = [
        {
            "data_path": "data/BTC.csv",
            "strategy_path": "example_strategy.py",
            "objective": "sharpe_ratio"
        },
        {
            "data_path": "data/ETH.csv",
            "strategy_path": "example_strategy.py",
            "objective": "sharpe_ratio"
        },
        {
            "data_path": "data/BTC.csv",
            "strategy_path": "strategies/rsi_strategy.py",
            "objective": "annual_return"
        }
    ]
    
    all_results = {}
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] 优化: {Path(config['data_path']).stem} - {Path(config['strategy_path']).stem}")
        
        try:
            optimizer = UniversalOptimizer(
                **config,
                use_llm=False,
                output_dir="./demo_results",
                verbose=False  # 关闭详细输出
            )
            
            result = optimizer.optimize(n_trials=20)
            
            key = f"{Path(config['data_path']).stem}_{Path(config['strategy_path']).stem}"
            all_results[key] = result
            
            print(f"  ✓ 完成 - 夏普比率: {result['performance_metrics']['sharpe_ratio']:.4f}")
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print("\n✅ 所有优化完成！")
    print(f"成功优化 {len(all_results)}/{len(configs)} 个配置")


# ============================================================================
# 示例6: 从JSON结果加载参数并应用
# ============================================================================

def example_load_and_apply_results():
    """演示如何加载优化结果并应用到实盘"""
    print("\n" + "="*60)
    print("示例6: 加载优化结果并应用")
    print("="*60 + "\n")
    
    import json
    import glob
    
    # 查找最新的优化结果
    result_files = glob.glob("./demo_results/optimization_*.json")
    
    if not result_files:
        print("未找到优化结果文件")
        return
    
    # 选择最新的文件
    latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
    
    print(f"加载结果文件: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    # 提取最优参数
    best_params = result['best_parameters']
    performance = result['performance_metrics']
    
    print("\n最优参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print("\n历史回测性能:")
    print(f"  夏普比率: {performance['sharpe_ratio']:.4f}")
    print(f"  年化收益: {performance['annual_return']:.2f}%")
    print(f"  最大回撤: {performance['max_drawdown']:.2f}%")
    print(f"  胜率: {performance['win_rate']:.2f}%")
    
    print("\n应用到实盘:")
    print(f"""
# 伪代码示例
from your_trading_system import TradingBot
from your_strategy import YourStrategy

# 使用优化的参数创建策略实例
strategy = YourStrategy(**{best_params})

# 启动实盘交易（谨慎！）
# bot = TradingBot(strategy=strategy)
# bot.start()

# 建议先进行样本外测试和模拟交易
""")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("通用策略优化器 - 使用示例集合")
    print("="*60)
    
    examples = {
        "1": ("基本优化（不使用LLM）", example_basic),
        "2": ("使用OpenAI LLM", example_with_openai),
        "3": ("使用本地Ollama", example_with_ollama),
        "4": ("批量优化多个目标", example_batch_optimization),
        "5": ("多标的、多策略优化", example_multiple_assets_strategies),
        "6": ("加载并应用结果", example_load_and_apply_results),
    }
    
    print("\n可用示例:")
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    print("  0. 运行所有示例")
    print("  q. 退出")
    
    choice = input("\n请选择示例 (1-6, 0, q): ").strip()
    
    if choice == 'q':
        print("退出")
        return
    
    if choice == '0':
        print("\n运行所有示例...")
        for desc, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"示例失败: {e}")
    elif choice in examples:
        desc, func = examples[choice]
        print(f"\n运行示例: {desc}")
        try:
            func()
        except Exception as e:
            print(f"示例失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("无效选择")


if __name__ == "__main__":
    # 运行主函数
    main()
    
    # 或者直接运行某个示例
    # example_basic()
    # example_with_openai()
    # example_with_ollama()
    # example_batch_optimization()
    # example_multiple_assets_strategies()
    # example_load_and_apply_results()
