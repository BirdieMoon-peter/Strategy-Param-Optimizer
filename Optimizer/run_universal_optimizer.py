# -*- coding: utf-8 -*-
"""
通用优化器运行脚本
使用示例和命令行接口
"""

import sys
import json
import argparse
from pathlib import Path

from universal_optimizer import UniversalOptimizer
from universal_llm_client import UniversalLLMConfig, PRESET_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        description="通用策略优化器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（不使用LLM）
  python run_universal_optimizer.py --data data/BTC.csv --strategy strategies/my_strategy.py
  
  # 使用OpenAI GPT-4
  python run_universal_optimizer.py --data data/BTC.csv --strategy strategies/my_strategy.py \\
      --use-llm --llm-type openai --llm-model gpt-4 --api-key sk-xxx
  
  # 使用本地Ollama
  python run_universal_optimizer.py --data data/BTC.csv --strategy strategies/my_strategy.py \\
      --use-llm --llm-type ollama --llm-model qwen
  
  # 批量优化多个目标
  python run_universal_optimizer.py --data data/BTC.csv --strategy strategies/my_strategy.py \\
      --batch --objectives sharpe_ratio annual_return calmar_ratio
  
  # 指定试验次数和输出目录
  python run_universal_optimizer.py --data data/BTC.csv --strategy strategies/my_strategy.py \\
      --trials 100 --output ./my_results
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--data",
        required=True,
        help="标的数据CSV文件路径"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="策略脚本文件路径（.py文件）"
    )
    
    # 优化参数
    parser.add_argument(
        "--objective",
        default="sharpe_ratio",
        choices=[
            "sharpe_ratio", "annual_return", "total_return",
            "max_drawdown", "calmar_ratio", "sortino_ratio"
        ],
        help="优化目标（默认: sharpe_ratio）"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="优化试验次数（默认: 50）"
    )
    
    # LLM参数
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="是否使用LLM辅助优化"
    )
    parser.add_argument(
        "--llm-type",
        choices=["openai", "ollama", "custom"],
        default="openai",
        help="LLM类型（默认: openai）"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4",
        help="LLM模型名称（默认: gpt-4）"
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API密钥（OpenAI或自定义API需要）"
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="API基础URL（可选，默认根据llm-type自动设置）"
    )
    
    # 批量优化
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量优化模式（优化多个目标）"
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["sharpe_ratio", "annual_return"],
        help="批量优化的目标列表"
    )
    
    # 输出参数
    parser.add_argument(
        "--output",
        default="./optimization_results",
        help="输出目录（默认: ./optimization_results）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（不打印详细信息）"
    )
    
    args = parser.parse_args()
    
    # 验证文件是否存在
    if not Path(args.data).exists():
        print(f"❌ 错误: 数据文件不存在: {args.data}")
        return 1
    
    if not Path(args.strategy).exists():
        print(f"❌ 错误: 策略文件不存在: {args.strategy}")
        return 1
    
    # 配置LLM
    llm_config = None
    if args.use_llm:
        # 设置默认base_url
        if not args.base_url:
            if args.llm_type == "openai":
                base_url = "https://api.openai.com/v1"
            elif args.llm_type == "ollama":
                base_url = "http://localhost:11434"
            else:
                print("❌ 错误: 使用custom类型时必须指定--base-url")
                return 1
        else:
            base_url = args.base_url
        
        llm_config = UniversalLLMConfig(
            api_type=args.llm_type,
            base_url=base_url,
            model_name=args.llm_model,
            api_key=args.api_key,
            temperature=0.7
        )
        
        if not args.quiet:
            print(f"🤖 LLM配置:")
            print(f"   类型: {args.llm_type}")
            print(f"   模型: {args.llm_model}")
            print(f"   URL: {base_url}")
            print()
    
    # 创建优化器
    try:
        optimizer = UniversalOptimizer(
            data_path=args.data,
            strategy_path=args.strategy,
            objective=args.objective,
            use_llm=args.use_llm,
            llm_config=llm_config,
            output_dir=args.output,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"❌ 创建优化器失败: {e}")
        return 1
    
    # 执行优化
    try:
        if args.batch:
            # 批量优化
            print(f"\n🚀 开始批量优化（目标: {', '.join(args.objectives)}）\n")
            result = optimizer.batch_optimize(
                objectives=args.objectives,
                n_trials_per_objective=args.trials
            )
        else:
            # 单目标优化
            print(f"\n🚀 开始优化（目标: {args.objective}）\n")
            result = optimizer.optimize(n_trials=args.trials)
        
        # 打印摘要
        if not args.quiet:
            print("\n" + "="*60)
            print("✅ 优化完成！")
            print("="*60)
            
            if args.batch:
                print(f"\n批量优化结果摘要:")
                for obj, obj_result in result.get("results", {}).items():
                    metrics = obj_result.get("performance_metrics", {})
                    print(f"\n目标: {obj}")
                    print(f"  夏普比率: {metrics.get('sharpe_ratio', 'N/A')}")
                    print(f"  年化收益: {metrics.get('annual_return', 'N/A')}%")
                    print(f"  最大回撤: {metrics.get('max_drawdown', 'N/A')}%")
            else:
                metrics = result.get("performance_metrics", {})
                params = result.get("best_parameters", {})
                
                print(f"\n最优参数:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
                
                print(f"\n性能指标:")
                print(f"  夏普比率: {metrics.get('sharpe_ratio', 'N/A')}")
                print(f"  年化收益: {metrics.get('annual_return', 'N/A')}%")
                print(f"  最大回撤: {metrics.get('max_drawdown', 'N/A')}%")
                print(f"  总收益率: {metrics.get('total_return', 'N/A')}%")
                print(f"  交易次数: {metrics.get('trades_count', 'N/A')}")
                print(f"  胜率: {metrics.get('win_rate', 'N/A')}%")
                
                # LLM解释
                if args.use_llm and "llm_explanation" in result:
                    explanation = result["llm_explanation"]
                    print(f"\n💡 LLM分析:")
                    print(f"  {explanation.get('parameter_explanation', '')}")
                    
                    if "key_insights" in explanation:
                        print(f"\n关键洞察:")
                        for insight in explanation["key_insights"]:
                            print(f"  • {insight}")
            
            print(f"\n详细结果已保存至: {args.output}")
            print("="*60 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  优化被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
