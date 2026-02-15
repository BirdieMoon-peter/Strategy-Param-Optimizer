# -*- coding: utf-8 -*-
"""
SPP (System Parameter Permutation) 鲁棒性分析 CLI 入口

读取优化器输出的 JSON 结果，对最优参数进行三维度鲁棒性分析。

使用示例:
  python run_spp_analysis.py \
    -r optimization_results/AG/optimization_AG_Aberration_20260202.json \
    -d project_trend/data/AG.csv \
    -s project_trend/src/Aberration.py

  # 小样本快速测试
  python run_spp_analysis.py \
    -r result.json -d data.csv -s strategy.py \
    --global-samples 50 --local-samples 30 --yearly-samples 30

  # 跳过逐年分析
  python run_spp_analysis.py \
    -r result.json -d data.csv -s strategy.py --no-short-run
"""

import sys
import os
import json
import argparse
import importlib.util
import inspect
from pathlib import Path
from datetime import datetime

import pandas as pd
import backtrader as bt

# 添加 optimizer/ 到路径
optimizer_path = str(Path(__file__).parent / "optimizer")
if optimizer_path not in sys.path:
    sys.path.insert(0, optimizer_path)

from backtest_engine import BacktestEngine
from config import StrategyParam
from spp_analyzer import SPPConfig, SPPAnalyzer

try:
    import futures_config as fc
except ImportError:
    fc = None


def load_strategy(strategy_path: str):
    """动态加载策略类，返回 (strategy_class, strategy_module)"""
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"策略文件不存在: {strategy_path}")

    module_name = f"strategy_module_{Path(strategy_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    strategy_classes = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and hasattr(obj, 'params')
                and obj.__module__ == module_name
                and issubclass(obj, bt.Strategy)):
            strategy_classes.append(obj)

    if not strategy_classes:
        raise ValueError(f"未在策略文件中找到有效的策略类: {strategy_path}")

    return strategy_classes[0], module


def load_data(data_path: str) -> pd.DataFrame:
    """加载并标准化 CSV 数据"""
    df = pd.read_csv(data_path)

    # 统一时间列
    if 'datetime' not in df.columns:
        for col in ('date', 'time_key', 'time'):
            if col in df.columns:
                df.rename(columns={col: 'datetime'}, inplace=True)
                break

    if 'datetime' not in df.columns:
        raise ValueError("数据文件必须包含 datetime/date/time_key 列")

    # 转换日期
    if pd.api.types.is_numeric_dtype(df['datetime']):
        unit = 'ms' if df['datetime'].iloc[0] > 1e12 else 's'
        df['datetime'] = pd.to_datetime(df['datetime'], unit=unit)
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])

    return df


def build_search_space(strategy_class, best_params: dict) -> dict:
    """从策略类提取参数并构建搜索空间 (±30% 范围)"""
    from param_space_optimizer import ParamSpaceOptimizer

    EXCLUDED = {'printlog', 'verbose', 'mult', 'margin', 'commission', 'percent',
                'stocklike', 'commtype', 'percabs'}

    raw_params = []
    if hasattr(strategy_class, 'params'):
        for pname in dir(strategy_class.params):
            if pname.startswith('_') or pname.lower() in EXCLUDED:
                continue
            default_val = getattr(strategy_class.params, pname)
            if not isinstance(default_val, (int, float)):
                continue
            raw_params.append(StrategyParam(
                name=pname,
                param_type=type(default_val).__name__,
                default_value=default_val,
                description=f"{pname} parameter",
            ))

    # 使用 ParamSpaceOptimizer 生成智能范围
    pso = ParamSpaceOptimizer(verbose=False)
    enriched = pso.generate_space(raw_params, strategy_type=strategy_class.__name__)

    return {p.name: p for p in enriched}


def main():
    parser = argparse.ArgumentParser(
        description="SPP 鲁棒性分析 (System Parameter Permutation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_spp_analysis.py \\
    -r optimization_results/AG/optimization_AG_Aberration_20260202.json \\
    -d project_trend/data/AG.csv \\
    -s project_trend/src/Aberration.py

  # 小样本快速测试
  python run_spp_analysis.py \\
    -r result.json -d data.csv -s strategy.py \\
    --global-samples 50 --local-samples 30 --yearly-samples 30
        """
    )

    # 必需参数
    parser.add_argument("-r", "--result", required=True, help="优化结果 JSON 路径")
    parser.add_argument("-d", "--data", required=True, help="CSV 数据文件路径")
    parser.add_argument("-s", "--strategy", required=True, help="策略 .py 文件路径")

    # 可选参数
    parser.add_argument("-o", "--objective", default=None,
                        help="分析指标 (默认: 从 JSON 读取)")
    parser.add_argument("--global-samples", type=int, default=500, help="全局采样数 (默认: 500)")
    parser.add_argument("--local-samples", type=int, default=200, help="局部采样数 (默认: 200)")
    parser.add_argument("--local-noise", type=float, default=0.10, help="局部扰动比例 (默认: 0.10)")
    parser.add_argument("--yearly-samples", type=int, default=200, help="每年采样数 (默认: 200)")
    parser.add_argument("--no-short-run", action="store_true", help="跳过逐年分析")
    parser.add_argument("--output", default="./spp_results", help="输出目录 (默认: ./spp_results)")
    parser.add_argument("--data-frequency", default=None, help="数据频率 (daily, 1m, 5m, ...)")
    parser.add_argument("--asset-type", default="stock", choices=["stock", "futures"])
    parser.add_argument("--contract-code", default=None, help="期货合约代码")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")

    args = parser.parse_args()

    # 1. 读取优化结果 JSON
    if not Path(args.result).exists():
        print(f"错误: JSON 文件不存在: {args.result}")
        return 1

    with open(args.result, 'r', encoding='utf-8') as f:
        opt_result = json.load(f)

    best_params = opt_result.get('best_parameters', {})
    best_metrics = opt_result.get('performance_metrics', {})
    opt_info = opt_result.get('optimization_info', opt_result.get('optimization_metadata', {}))

    if not best_params:
        print("错误: JSON 中未找到 best_parameters")
        return 1

    # 确定目标指标
    objective = args.objective or opt_info.get('optimization_objective', 'sharpe_ratio')

    # 确定资产和策略名称
    asset_name = opt_info.get('asset_name', Path(args.data).stem)
    strategy_name = opt_info.get('strategy_name', Path(args.strategy).stem)

    verbose = not args.quiet

    if verbose:
        print(f"\n{'='*60}")
        print(f"SPP 鲁棒性分析")
        print(f"{'='*60}")
        print(f"优化结果: {args.result}")
        print(f"数据文件: {args.data}")
        print(f"策略文件: {args.strategy}")
        print(f"分析指标: {objective}")
        print(f"最优参数: {best_params}")
        print(f"{'='*60}\n")

    # 2. 加载策略和数据
    try:
        strategy_class, strategy_module = load_strategy(str(Path(args.strategy).absolute()))
    except Exception as e:
        print(f"错误: 加载策略失败: {e}")
        return 1

    try:
        data = load_data(args.data)
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        return 1

    if verbose:
        print(f"[数据] {len(data)} 条记录")
        print(f"[策略] {strategy_class.__name__}")

    # 3. 构建经纪商配置
    broker_config = None
    if fc and args.asset_type == 'futures' and args.contract_code:
        try:
            broker_config = fc.build_broker_config(
                asset_type='futures',
                contract_code=args.contract_code,
            )
        except Exception as e:
            print(f"[警告] 期货配置失败: {e}，使用默认股票配置")

    # 4. 构建搜索空间
    search_space = build_search_space(strategy_class, best_params)
    if verbose:
        print(f"[搜索空间] {len(search_space)} 个参数:")
        for name, sp in search_space.items():
            print(f"  {name}: [{sp.min_value}, {sp.max_value}] "
                  f"step={sp.step} type={sp.param_type} default={sp.default_value}")

    # 5. 创建回测引擎
    effective_freq = None if (args.data_frequency is None or args.data_frequency == 'auto') else args.data_frequency
    engine = BacktestEngine(
        data=data,
        strategy_class=strategy_class,
        data_frequency=effective_freq,
        strategy_module=strategy_module,
        broker_config=broker_config,
    )

    # 6. 创建 SPP 分析器
    spp_config = SPPConfig(
        global_samples=args.global_samples,
        local_samples=args.local_samples,
        local_noise_level=args.local_noise,
        yearly_samples=args.yearly_samples,
        objective=objective,
    )

    analyzer = SPPAnalyzer(
        backtest_engine=engine,
        strategy_class=strategy_class,
        data=data,
        search_space=search_space,
        config=spp_config,
        verbose=verbose,
    )

    # 7. 运行完整分析
    output_dir = os.path.join(args.output, asset_name)
    result = analyzer.run_full_analysis(
        best_params=best_params,
        best_metrics=best_metrics,
        output_dir=output_dir,
        asset_name=asset_name,
        strategy_name=strategy_name,
        source_json=args.result,
        skip_short_run=args.no_short_run,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
