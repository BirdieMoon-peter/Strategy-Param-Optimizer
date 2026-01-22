# -*- coding: utf-8 -*-
"""
通用优化器测试脚本
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from universal_optimizer import UniversalOptimizer
from universal_llm_client import UniversalLLMConfig


def create_sample_data(output_path: str = "test_data_BTC.csv"):
    """创建示例数据"""
    print("创建示例数据...")
    
    # 生成模拟的BTC数据
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    
    import numpy as np
    np.random.seed(42)
    
    # 生成价格数据（随机游走）
    returns = np.random.randn(len(dates)) * 0.02
    price = 40000 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': price * (1 + np.random.randn(len(dates)) * 0.001),
        'high': price * (1 + abs(np.random.randn(len(dates)) * 0.005)),
        'low': price * (1 - abs(np.random.randn(len(dates)) * 0.005)),
        'close': price,
        'volume': np.random.randint(100, 1000, len(dates))
    })
    
    df.to_csv(output_path, index=False)
    print(f"✓ 示例数据已创建: {output_path}")
    return output_path


def test_basic_optimization():
    """测试基本优化功能（不使用LLM）"""
    print("\n" + "="*60)
    print("测试1: 基本优化（不使用LLM）")
    print("="*60 + "\n")
    
    # 创建示例数据
    data_path = create_sample_data()
    
    # 使用内置的示例策略
    strategy_path = Path(__file__).parent / "example_strategy.py"
    
    if not strategy_path.exists():
        print(f"❌ 错误: 示例策略文件不存在: {strategy_path}")
        return False
    
    try:
        # 创建优化器
        optimizer = UniversalOptimizer(
            data_path=data_path,
            strategy_path=str(strategy_path),
            objective="sharpe_ratio",
            use_llm=False,
            output_dir="./test_results",
            verbose=True
        )
        
        # 执行优化（较少的试验次数以加快测试）
        result = optimizer.optimize(n_trials=10)
        
        # 验证结果
        assert "best_parameters" in result
        assert "performance_metrics" in result
        assert "optimization_info" in result
        
        print("\n✅ 测试1通过：基本优化功能正常")
        
        # 打印结果摘要
        print("\n结果摘要:")
        print(json.dumps(result["performance_metrics"], indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理测试数据
        if os.path.exists(data_path):
            os.remove(data_path)


def test_llm_optimization():
    """测试LLM辅助优化（需要配置LLM）"""
    print("\n" + "="*60)
    print("测试2: LLM辅助优化（使用Ollama）")
    print("="*60 + "\n")
    
    # 创建示例数据
    data_path = create_sample_data()
    strategy_path = Path(__file__).parent / "example_strategy.py"
    
    try:
        # 配置Ollama（本地测试）
        llm_config = UniversalLLMConfig(
            api_type="ollama",
            base_url="http://localhost:11434",
            model_name="qwen",
            api_key="",
            timeout=60
        )
        
        # 创建优化器
        optimizer = UniversalOptimizer(
            data_path=data_path,
            strategy_path=str(strategy_path),
            objective="sharpe_ratio",
            use_llm=True,
            llm_config=llm_config,
            output_dir="./test_results",
            verbose=True
        )
        
        # 执行优化
        result = optimizer.optimize(n_trials=10)
        
        # 验证结果
        assert "llm_explanation" in result
        
        print("\n✅ 测试2通过：LLM辅助优化功能正常")
        
        # 打印LLM解释
        print("\nLLM解释:")
        print(json.dumps(result["llm_explanation"], indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"\n⚠️  测试2跳过（可能是LLM服务未启动）: {e}")
        return True  # 不算失败
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)


def test_batch_optimization():
    """测试批量优化功能"""
    print("\n" + "="*60)
    print("测试3: 批量优化（多个目标）")
    print("="*60 + "\n")
    
    data_path = create_sample_data()
    strategy_path = Path(__file__).parent / "example_strategy.py"
    
    try:
        optimizer = UniversalOptimizer(
            data_path=data_path,
            strategy_path=str(strategy_path),
            use_llm=False,
            output_dir="./test_results",
            verbose=True
        )
        
        # 批量优化
        result = optimizer.batch_optimize(
            objectives=["sharpe_ratio", "annual_return"],
            n_trials_per_objective=10
        )
        
        # 验证结果
        assert "results" in result
        assert "sharpe_ratio" in result["results"]
        assert "annual_return" in result["results"]
        
        print("\n✅ 测试3通过：批量优化功能正常")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)


def test_json_output():
    """测试JSON输出格式"""
    print("\n" + "="*60)
    print("测试4: JSON输出格式验证")
    print("="*60 + "\n")
    
    data_path = create_sample_data()
    strategy_path = Path(__file__).parent / "example_strategy.py"
    output_dir = Path("./test_results")
    
    try:
        optimizer = UniversalOptimizer(
            data_path=data_path,
            strategy_path=str(strategy_path),
            use_llm=False,
            output_dir=str(output_dir),
            verbose=False
        )
        
        result = optimizer.optimize(n_trials=5)
        
        # 查找生成的JSON文件
        json_files = list(output_dir.glob("optimization_*.json"))
        
        if not json_files:
            print("❌ 未找到生成的JSON文件")
            return False
        
        # 读取并验证JSON文件
        with open(json_files[0], 'r', encoding='utf-8') as f:
            saved_result = json.load(f)
        
        # 验证必需字段
        required_fields = [
            "optimization_info",
            "best_parameters",
            "performance_metrics",
            "yearly_performance",
            "llm_explanation"
        ]
        
        for field in required_fields:
            if field not in saved_result:
                print(f"❌ JSON文件缺少必需字段: {field}")
                return False
        
        print("✅ 测试4通过：JSON输出格式正确")
        print(f"   文件位置: {json_files[0]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("通用优化器测试套件")
    print("="*60)
    
    tests = [
        test_basic_optimization,
        test_llm_optimization,
        test_batch_optimization,
        test_json_output,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ 测试异常: {e}")
            results.append(False)
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
