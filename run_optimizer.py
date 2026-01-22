# -*- coding: utf-8 -*-
"""
通用策略优化测试脚本
支持命令行参数配置标的数据、策略脚本、优化目标、LLM等

使用示例:
  # 基本用法（不使用LLM）
  python run_optimizer.py --data project_trend/data/AG.csv --strategy project_trend/src/Aberration.py

  # 使用LLM
  python run_optimizer.py --data project_trend/data/BTC.csv --strategy project_trend/src/Aberration.py --use-llm

  # 指定优化目标和试验次数
  python run_optimizer.py --data project_trend/data/AG.csv --strategy project_trend/src/Aberration.py --objective annual_return --trials 100

  # 指定要优化的参数（通过params.txt文件）
  python run_optimizer.py --data project_trend/data/AG.csv --strategy project_trend/src/Aberration.py --params-file params.txt

  # 完整参数
  python run_optimizer.py --data project_trend/data/AG.csv --strategy project_trend/src/Aberration.py --objective sharpe_ratio --trials 50 --use-llm --llm-model xuanyuan --output ./my_results
"""

import sys
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加 Optimizer 到路径
optimizer_path = str(Path(__file__).parent / "Optimizer")
if optimizer_path not in sys.path:
    sys.path.insert(0, optimizer_path)

# 导入优化器模块
import universal_optimizer
import universal_llm_client
UniversalOptimizer = universal_optimizer.UniversalOptimizer
UniversalLLMConfig = universal_llm_client.UniversalLLMConfig


def load_target_params(params_file: str) -> list:
    """
    从文件加载要优化的参数列表
    
    Args:
        params_file: 参数文件路径，每行一个参数名
        
    Returns:
        参数名列表
    """
    if not Path(params_file).exists():
        raise FileNotFoundError(f"参数文件不存在: {params_file}")
    
    params = []
    with open(params_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除空白字符和注释
            param = line.strip()
            if param and not param.startswith('#'):
                params.append(param)
    
    if not params:
        raise ValueError(f"参数文件为空或没有有效参数: {params_file}")
    
    return params


def load_space_config(config_file: str) -> dict:
    """
    从 JSON 文件加载参数空间配置
    
    Args:
        config_file: 参数空间配置文件路径
        
    Returns:
        参数空间配置字典
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f"参数空间配置文件不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 验证配置格式
    if 'param_space' not in config:
        raise ValueError("配置文件必须包含 'param_space' 字段")
    
    param_space = config['param_space']
    
    # 验证每个参数的配置
    for param_name, param_config in param_space.items():
        if 'min' not in param_config or 'max' not in param_config:
            raise ValueError(f"参数 '{param_name}' 必须指定 'min' 和 'max'")
        if param_config['min'] >= param_config['max']:
            raise ValueError(f"参数 '{param_name}' 的 min 必须小于 max")
    
    return param_space


def prepare_data(data_path: str) -> str:
    """
    准备数据文件：确保有 datetime 列
    
    Args:
        data_path: 原始数据文件路径
        
    Returns:
        处理后的数据文件路径
    """
    df = pd.read_csv(data_path)
    
    # 检查并重命名日期列
    if 'date' in df.columns and 'datetime' not in df.columns:
        df.rename(columns={'date': 'datetime'}, inplace=True)
        print(f"[数据] 已将 'date' 列重命名为 'datetime'")
    
    if 'datetime' not in df.columns:
        raise ValueError("数据文件必须包含 'datetime' 或 'date' 列")
    
    # 转换日期格式
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 保存处理后的数据
    data_dir = Path(data_path).parent
    asset_name = Path(data_path).stem
    processed_path = data_dir / f"{asset_name}_processed.csv"
    df.to_csv(processed_path, index=False)
    
    print(f"[数据] 处理完成: {len(df)} 条记录")
    print(f"[数据] 时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    
    return str(processed_path)


def create_llm_config(args) -> UniversalLLMConfig:
    """
    创建LLM配置
    
    Args:
        args: 命令行参数
        
    Returns:
        LLM配置对象
    """
    return UniversalLLMConfig(
        api_type=args.llm_type,
        base_url=args.llm_url,
        model_name=args.llm_model,
        api_key=args.api_key,
        temperature=0.7,
        max_tokens=4096,
        timeout=args.timeout
    )


def print_results(result: dict, output_dir: Path, asset_name: str = None):
    """
    打印和保存优化结果
    
    Args:
        result: 优化结果字典
        output_dir: 输出目录
        asset_name: 资产名称（可选，用于覆盖结果中的名称）
    """
    print("\n" + "="*60)
    print("✅ 优化完成！")
    print("="*60)
    
    # 最优参数
    print("\n【最优参数】")
    best_params = result.get('best_parameters', {})
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    # 性能指标
    print("\n【性能指标】")
    metrics = result.get('performance_metrics', {})
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  年化收益率: {metrics.get('annual_return', 0):.2f}%")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"  总收益率: {metrics.get('total_return', 0):.2f}%")
    print(f"  交易次数: {metrics.get('trades_count', 0)}")
    print(f"  胜率: {metrics.get('win_rate', 0):.2f}%")
    
    # 逐年表现
    yearly = result.get('yearly_performance', {})
    if yearly:
        print("\n【逐年表现】")
        # 过滤掉收益为0且回撤为0的年份（可能是无交易年份）
        active_years = {y: p for y, p in yearly.items() 
                       if p.get('return', 0) != 0 or p.get('drawdown', 0) != 0}
        inactive_years = [y for y, p in yearly.items() 
                         if p.get('return', 0) == 0 and p.get('drawdown', 0) == 0]
        
        for year, perf in sorted(active_years.items()):
            ret = perf.get('return', 0)
            dd = perf.get('drawdown', 0)
            sr = perf.get('sharpe_ratio', 0)
            print(f"  {year}年: 收益 {ret:+.2f}%, 回撤 {dd:.2f}%, 夏普 {sr:.4f}")
        
        if inactive_years:
            print(f"  无交易年份: {', '.join(sorted(inactive_years))}")
    
    # LLM解释
    explanation = result.get('llm_explanation', {})
    if explanation and explanation.get('parameter_explanation'):
        print("\n【LLM 分析】")
        print(f"  {explanation.get('parameter_explanation', '')}")
        
        if explanation.get('key_insights'):
            print("\n关键洞察:")
            for i, insight in enumerate(explanation['key_insights'], 1):
                print(f"  {i}. {insight}")
    
    # 保存摘要
    summary_path = output_dir / "optimization_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("策略优化结果摘要\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"优化时间: {result.get('optimization_info', {}).get('optimization_time', '')}\n")
        f.write(f"标的: {result.get('optimization_info', {}).get('asset_name', '')}\n")
        f.write(f"策略: {result.get('optimization_info', {}).get('strategy_name', '')}\n")
        f.write(f"优化目标: {result.get('optimization_info', {}).get('optimization_objective', '')}\n\n")
        
        f.write("【最优参数】\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        
        f.write("\n【性能指标】\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\n结果摘要已保存至: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="通用策略优化器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python run_optimizer.py --data project_trend/data/AG.csv --strategy project_trend/src/Aberration.py

  # 使用本地 Ollama LLM
  python run_optimizer.py --data project_trend/data/AG.csv --strategy project_trend/src/Aberration.py --use-llm

  # 使用 OpenAI
  python run_optimizer.py --data project_trend/data/BTC.csv --strategy project_trend/src/Aberration.py --use-llm --llm-type openai --api-key sk-xxx

  # 指定优化目标
  python run_optimizer.py --data project_trend/data/AG.csv --strategy project_trend/src/Aberration.py --objective annual_return

优化目标选项:
  sharpe_ratio   - 夏普比率（默认，推荐）
  annual_return  - 年化收益率
  total_return   - 总收益率
  max_drawdown   - 最大回撤（最小化）
  calmar_ratio   - 卡玛比率
  sortino_ratio  - 索提诺比率
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="标的数据CSV文件路径（必须包含 datetime/date, open, high, low, close, volume 列）"
    )
    parser.add_argument(
        "--strategy", "-s",
        required=True,
        help="策略脚本文件路径（.py文件，必须包含继承 bt.Strategy 的策略类）"
    )
    
    # 优化参数
    parser.add_argument(
        "--objective", "-o",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "annual_return", "total_return", "max_drawdown", "calmar_ratio", "sortino_ratio"],
        help="优化目标（默认: sharpe_ratio）"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=50,
        help="优化试验次数（默认: 50）"
    )
    parser.add_argument(
        "--params-file", "-p",
        default=None,
        help="指定要优化的参数列表文件（每行一个参数名），不指定则优化所有参数"
    )
    parser.add_argument(
        "--space-config", "-S",
        default=None,
        help="参数空间配置文件（JSON格式），用于手动指定参数搜索范围，参考 space_config_example.json"
    )
    
    # LLM参数
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="是否使用LLM辅助优化"
    )
    parser.add_argument(
        "--llm-type",
        default="ollama",
        choices=["ollama", "openai", "custom"],
        help="LLM类型（默认: ollama）"
    )
    parser.add_argument(
        "--llm-model",
        default="xuanyuan",
        help="LLM模型名称（默认: xuanyuan）"
    )
    parser.add_argument(
        "--llm-url",
        default="http://localhost:11434",
        help="LLM API URL（默认: http://localhost:11434）"
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API密钥（OpenAI需要）"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="LLM请求超时时间（秒，默认: 180）"
    )
    
    # 输出参数
    parser.add_argument(
        "--output", "-O",
        default="./optimization_results",
        help="输出目录（默认: ./optimization_results）"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式（减少输出）"
    )
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not Path(args.data).exists():
        print(f"❌ 错误: 数据文件不存在: {args.data}")
        return 1
    
    if not Path(args.strategy).exists():
        print(f"❌ 错误: 策略文件不存在: {args.strategy}")
        return 1
    
    # 加载目标参数列表（如果指定了参数文件）
    target_params = None
    if args.params_file:
        if not Path(args.params_file).exists():
            print(f"❌ 错误: 参数文件不存在: {args.params_file}")
            return 1
        try:
            target_params = load_target_params(args.params_file)
        except Exception as e:
            print(f"❌ 错误: 读取参数文件失败: {e}")
            return 1
    
    # 加载参数空间配置（如果指定了配置文件）
    custom_space = None
    if args.space_config:
        if not Path(args.space_config).exists():
            print(f"❌ 错误: 参数空间配置文件不存在: {args.space_config}")
            return 1
        try:
            custom_space = load_space_config(args.space_config)
            print(f"[配置] 已加载自定义参数空间: {list(custom_space.keys())}")
        except Exception as e:
            print(f"❌ 错误: 读取参数空间配置文件失败: {e}")
            return 1
    
    # 打印配置信息
    if not args.quiet:
        print("\n" + "="*60)
        print("通用策略优化器")
        print("="*60)
        print(f"数据文件: {args.data}")
        print(f"策略文件: {args.strategy}")
        print(f"优化目标: {args.objective}")
        print(f"试验次数: {args.trials}")
        if target_params:
            print(f"指定参数: {target_params}")
        else:
            print(f"指定参数: 全部参数")
        if custom_space:
            print(f"自定义空间: {list(custom_space.keys())}")
        else:
            print(f"参数空间: 自动生成（智能规则）")
        print(f"使用LLM: {'是' if args.use_llm else '否'}")
        if args.use_llm:
            print(f"LLM类型: {args.llm_type}")
            print(f"LLM模型: {args.llm_model}")
        print("="*60 + "\n")
    
    try:
        # 1. 准备数据
        data_path = prepare_data(args.data)
        
        # 2. 配置LLM（如果需要）
        llm_config = None
        if args.use_llm:
            if args.llm_type == "openai" and not args.api_key:
                print("⚠️  警告: 使用OpenAI需要提供 --api-key")
            
            # 设置正确的URL
            if args.llm_type == "openai" and args.llm_url == "http://localhost:11434":
                args.llm_url = "https://api.openai.com/v1"
            
            llm_config = create_llm_config(args)
            
            if not args.quiet:
                print(f"[LLM] 配置: {args.llm_type} / {args.llm_model}")
        
        # 3. 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取原始资产名称（去除 _processed 后缀）
        original_asset_name = Path(args.data).stem.replace('_processed', '')
        
        # 4. 创建优化器
        if not args.quiet:
            print("\n[优化器] 初始化中...")
        
        optimizer = UniversalOptimizer(
            data_path=data_path,
            strategy_path=str(Path(args.strategy).absolute()),
            objective=args.objective,
            use_llm=args.use_llm,
            llm_config=llm_config,
            output_dir=str(output_dir),
            verbose=not args.quiet,
            target_params=target_params,
            custom_space=custom_space
        )
        
        # 5. 执行优化
        if not args.quiet:
            print(f"\n[优化] 开始优化（{args.trials} 次试验）...")
            print(f"[优化] 预计时间: {args.trials // 2} - {args.trials} 秒\n")
        
        result = optimizer.optimize(n_trials=args.trials)
        
        # 6. 打印和保存结果
        print_results(result, output_dir, original_asset_name)
        
        # 查找JSON文件
        json_files = list(output_dir.glob("optimization_*.json"))
        if json_files:
            print(f"完整JSON结果: {json_files[-1]}")
        
        print("\n" + "="*60)
        print("优化完成！")
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
