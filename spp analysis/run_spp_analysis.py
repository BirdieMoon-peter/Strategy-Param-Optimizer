# -*- coding: utf-8 -*-
"""
SPP (System Parameter Permutation) 鲁棒性分析 CLI 入口 — v3.0

以最优参数为中心进行蒙特卡洛采样，评估参数鲁棒性。
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

from backtest_engine import BacktestEngine
from config import StrategyParam
from spp_analyzer import SPPConfig, SPPAnalyzer
from spp_parallel_engine import SPPWorkerInitArgs

try:
    import futures_config as fc
except ImportError:
    fc = None


def load_strategy(strategy_path: str):
    """动态加载策略类，同时提取自定义数据类、手续费类和 trade_log 检测"""
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

    strategy_class = strategy_classes[0]

    # 查找自定义数据类（继承自 bt.feeds.PandasData）
    custom_data_class = None
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and obj.__module__ == module_name
                and issubclass(obj, bt.feeds.PandasData)
                and obj is not bt.feeds.PandasData):
            custom_data_class = obj
            break

    # 查找自定义手续费类（继承自 bt.CommInfoBase）
    custom_commission_class = None
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and obj.__module__ == module_name
                and issubclass(obj, bt.CommInfoBase)
                and obj is not bt.CommInfoBase):
            custom_commission_class = obj
            break

    # 检测策略是否使用 trade_log
    use_trade_log_metrics = False
    try:
        source = inspect.getsource(strategy_class)
        use_trade_log_metrics = 'trade_log' in source
    except Exception:
        pass

    return strategy_class, module, custom_data_class, custom_commission_class, use_trade_log_metrics


def load_data(data_path: str) -> pd.DataFrame:
    """加载并标准化 CSV 数据"""
    df = pd.read_csv(data_path)
    if 'datetime' not in df.columns:
        for col in ('date', 'time_key', 'time'):
            if col in df.columns:
                df.rename(columns={col: 'datetime'}, inplace=True)
                break
    if 'datetime' not in df.columns:
        raise ValueError("数据文件必须包含 datetime/date/time_key 列")
    if pd.api.types.is_numeric_dtype(df['datetime']):
        unit = 'ms' if df['datetime'].iloc[0] > 1e12 else 's'
        df['datetime'] = pd.to_datetime(df['datetime'], unit=unit)
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def load_params_file(path: str) -> dict:
    """解析 key=value 格式的参数文件，返回 {name: value} 字典"""
    params = {}
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                raise ValueError(f"第{lineno}行格式错误（需要 key = value）: {line}")
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            # 自动推断类型：int / float
            try:
                params[key] = int(val)
            except ValueError:
                try:
                    params[key] = float(val)
                except ValueError:
                    raise ValueError(f"第{lineno}行值无法解析为数字: {key} = {val}")
    return params


def validate_and_merge_center_params(base_params: dict, override_params: dict,
                                     search_space: dict):
    """校验并合并中心参数覆盖"""
    unknown = sorted(set(override_params) - set(search_space))
    if unknown:
        valid = ', '.join(sorted(search_space))
        raise ValueError(f"未知参数: {', '.join(unknown)}。可用参数: {valid}")

    normalized = {}
    warnings = []
    for name, value in override_params.items():
        sp = search_space[name]
        if sp.param_type == 'int':
            if isinstance(value, float) and not float(value).is_integer():
                raise ValueError(f"参数 {name} 需要整数，当前值为 {value}")
            normalized_value = int(value)
        else:
            normalized_value = float(value)

        if normalized_value < sp.min_value or normalized_value > sp.max_value:
            warnings.append(
                f"参数 {name}={normalized_value} 超出搜索空间 [{sp.min_value}, {sp.max_value}]"
            )
        normalized[name] = normalized_value

    merged = dict(base_params)
    merged.update(normalized)
    return merged, normalized, warnings


def build_metrics_from_backtest(result) -> dict:
    """从回测结果提取 SPP 所需指标"""
    return {
        'sharpe_ratio': result.sharpe_ratio,
        'annual_return': result.annual_return,
        'max_drawdown': result.max_drawdown,
        'sortino_ratio': result.sortino_ratio,
        'calmar_ratio': result.calmar_ratio,
        'total_return': result.total_return,
        'trades_count': result.trades_count,
        'win_rate': result.win_rate,
    }


def build_search_space(strategy_class, best_params: dict) -> dict:
    """从策略类提取参数并构建搜索空间"""
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
                name=pname, param_type=type(default_val).__name__,
                default_value=default_val, description=f"{pname} parameter"))
    pso = ParamSpaceOptimizer(verbose=False)
    enriched = pso.generate_space(raw_params, strategy_type=strategy_class.__name__)
    return {p.name: p for p in enriched}


def main():
    parser = argparse.ArgumentParser(
        description="SPP 蒙特卡洛鲁棒性分析 (v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用参数文件（推荐）
  python run_spp_analysis.py \\
    --params-file best_params.txt \\
    -d project_trend/data/AG.csv \\
    -s project_trend/src/Aberration.py

  # 使用优化结果 JSON
  python run_spp_analysis.py \\
    -r optimization_results/AG/optimization_AG_Aberration_20260202.json \\
    -d project_trend/data/AG.csv \\
    -s project_trend/src/Aberration.py

  # 使用优化结果 JSON + 覆盖中心参数
  python run_spp_analysis.py \
    -r optimization_results/AG/optimization_AG_Aberration_20260202.json \
    -d project_trend/data/AG.csv \
    -s project_trend/src/Aberration.py \
    --center-params-file center_params.txt

  # 多数据源（策略需要多个CSV）
  python run_spp_analysis.py \\
    --params-file best_params.txt \\
    -d data/BTC.csv data/ETH.csv \\
    -s strategy.py

  # 使用参数文件 + 覆盖中心参数
  python run_spp_analysis.py \
    --params-file best_params.txt \
    --center-params-file center_params.txt \
    -d data/BTC.csv data/ETH.csv \
    -s strategy.py

参数文件格式 (key = value):
  period = 35
  std_dev_upper = 2.1
  std_dev_lower = 1.8
        """
    )

    parser.add_argument("-r", "--result", default=None, help="优化结果 JSON 路径（与 --params-file 二选一）")
    parser.add_argument("--params-file", default=None,
                        help="参数文件路径（key=value 格式，与 -r 二选一）")
    parser.add_argument("-d", "--data", required=True, nargs='+',
                        help="CSV 数据文件路径（支持多个，空格分隔）")
    parser.add_argument("-s", "--strategy", required=True, help="策略 .py 文件路径")

    parser.add_argument("-n", "--samples", type=int, default=300,
                        help="蒙特卡洛采样次数 (默认: 300)")
    parser.add_argument("-p", "--perturbation", type=float, default=0.20,
                        help="扰动比例 (默认: 0.20)")
    parser.add_argument("-o", "--objective", default=None,
                        help="分析指标 (默认: 从 JSON 读取)")

    parser.add_argument("--center-params-file", default=None,
                        help="中心参数覆盖文件（key=value 格式，在基础参数来源上覆盖）")
    parser.add_argument("--use-llm", action="store_true",
                        help="启用 LLM 识别敏感参数")
    parser.add_argument("--llm-model", default="xuanyuan",
                        help="LLM 模型名 (默认: xuanyuan)")
    parser.add_argument("--sensitive-params", default=None,
                        help="手动指定敏感参数 (逗号分隔)")

    parser.add_argument("--output", default="./spp_results",
                        help="输出目录 (默认: ./spp_results)")
    parser.add_argument("--data-frequency", default=None,
                        help="数据频率 (daily, 1m, 5m, ...)")
    parser.add_argument("--asset-type", default="stock",
                        choices=["stock", "futures"])
    parser.add_argument("--contract-code", default=None, help="期货合约代码")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    parser.add_argument("--seed", type=int, default=None,
                        help="蒙特卡洛随机种子（默认: 自动生成）")
    parser.add_argument("--no-parallel", action="store_true",
                        help="禁用 SPP 多进程并行，使用串行回测")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="SPP 并行 worker 数（默认: CPU 核数 - 2，且不超过样本数）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="SPP 并行每批提交样本数（默认: 32）")
    parser.add_argument("--sample-timeout", type=int, default=300,
                        help="单个 SPP 样本回测超时秒数（默认: 300）")

    args = parser.parse_args()

    # 校验：-r 和 --params-file 必须提供其一
    if not args.result and not args.params_file:
        parser.error("必须提供 -r/--result 或 --params-file 其中之一")
    if args.result and args.params_file:
        parser.error("-r/--result 和 --params-file 不能同时使用")

    # 1. 读取参数来源
    best_metrics = {}  # params-file 模式下稍后自动回测获取
    source_label = ''
    parameter_source = ''
    parameter_source_path = ''
    center_override_file = args.center_params_file
    center_overrides = {}
    baseline_recomputed = False

    if args.params_file:
        # --- params-file 模式 ---
        if not Path(args.params_file).exists():
            print(f"错误: 参数文件不存在: {args.params_file}")
            return 1
        try:
            best_params = load_params_file(args.params_file)
        except ValueError as e:
            print(f"错误: 解析参数文件失败: {e}")
            return 1
        if not best_params:
            print("错误: 参数文件为空")
            return 1
        source_label = args.params_file
        parameter_source = 'params_file'
        parameter_source_path = args.params_file
        objective = args.objective or 'sharpe_ratio'
    else:
        # --- JSON 模式（原有逻辑） ---
        if not Path(args.result).exists():
            print(f"错误: JSON 文件不存在: {args.result}")
            return 1
        with open(args.result, 'r', encoding='utf-8') as f:
            opt_result = json.load(f)
        best_params = opt_result.get('best_parameters', {})
        best_metrics = opt_result.get('performance_metrics', {})
        opt_info = opt_result.get('optimization_info',
                                  opt_result.get('optimization_metadata', {}))
        if not best_params:
            print("错误: JSON 中未找到 best_parameters")
            return 1
        source_label = args.result
        parameter_source = 'result_json'
        parameter_source_path = args.result
        objective = args.objective or opt_info.get('optimization_objective', 'sharpe_ratio')

    asset_name = Path(args.data[0]).stem
    strategy_name = Path(args.strategy).stem
    verbose = not args.quiet
    multi_data = len(args.data) > 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"SPP 蒙特卡洛鲁棒性分析 (v3.0)")
        print(f"{'='*60}")
        print(f"参数来源: {source_label}")
        if center_override_file:
            print(f"中心参数覆盖: {center_override_file}")
        if multi_data:
            print(f"数据文件: {len(args.data)} 个")
            for dp in args.data:
                print(f"  - {dp}")
        else:
            print(f"数据文件: {args.data[0]}")
        print(f"策略文件: {args.strategy}")
        print(f"分析指标: {objective}")
        print(f"采样次数: {args.samples}, 扰动: ±{args.perturbation:.0%}")
        print(f"基础参数: {best_params}")
        print(f"{'='*60}\n")

    # 2. 加载策略和数据
    try:
        strategy_class, strategy_module, custom_data_class, custom_commission_class, use_trade_log_metrics = load_strategy(
            str(Path(args.strategy).absolute()))
    except Exception as e:
        print(f"错误: 加载策略失败: {e}")
        return 1

    try:
        if multi_data:
            data = [load_data(dp) for dp in args.data]
            data_names = [Path(dp).stem for dp in args.data]
        else:
            data = load_data(args.data[0])
            data_names = None
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        return 1

    if verbose:
        if multi_data:
            for dp, df in zip(args.data, data):
                print(f"[数据] {Path(dp).stem}: {len(df)} 条记录")
        else:
            print(f"[数据] {len(data)} 条记录")
        print(f"[策略] {strategy_class.__name__}")

    # 3. 构建经纪商配置
    broker_config = None
    if fc and args.asset_type == 'futures' and args.contract_code:
        try:
            broker_config = fc.build_broker_config(
                asset_type='futures', contract_code=args.contract_code)
        except Exception as e:
            print(f"[警告] 期货配置失败: {e}，使用默认股票配置")

    # 4. 构建搜索空间
    search_space = build_search_space(strategy_class, best_params)
    if verbose:
        print(f"[搜索空间] {len(search_space)} 个参数:")
        for name, sp in search_space.items():
            print(f"  {name}: [{sp.min_value}, {sp.max_value}] "
                  f"step={sp.step} type={sp.param_type}")

    # 5. 创建回测引擎
    effective_freq = None if (args.data_frequency is None
                              or args.data_frequency == 'auto') else args.data_frequency
    engine = BacktestEngine(
        data=data, strategy_class=strategy_class,
        data_frequency=effective_freq,
        strategy_module=strategy_module, broker_config=broker_config,
        custom_data_class=custom_data_class,
        custom_commission_class=custom_commission_class,
        use_trade_log_metrics=use_trade_log_metrics)

    if args.center_params_file:
        if not Path(args.center_params_file).exists():
            print(f"错误: 中心参数文件不存在: {args.center_params_file}")
            return 1
        try:
            raw_center_overrides = load_params_file(args.center_params_file)
            best_params, center_overrides, center_warnings = validate_and_merge_center_params(
                best_params, raw_center_overrides, search_space)
        except ValueError as e:
            print(f"错误: 解析中心参数文件失败: {e}")
            return 1
        if verbose:
            print(f"[中心参数] 覆盖文件: {args.center_params_file}")
            print(f"[中心参数] 生效覆盖: {center_overrides}")
            for warning in center_warnings:
                print(f"[警告] {warning}")

    # 5.1 params-file 模式：自动回测获取 best_metrics
    if not best_metrics or center_overrides:
        if verbose:
            print(f"[回测] 使用中心参数运行基准回测...")
        baseline = engine.run_backtest(strategy_class, data, best_params, calculate_yearly=False)
        if baseline is None:
            print("错误: 基准回测失败，无法获取 best_metrics")
            return 1
        best_metrics = build_metrics_from_backtest(baseline)
        baseline_recomputed = bool(center_overrides) or parameter_source == 'params_file'
        if verbose:
            print(f"[回测] 基准 {objective}: {best_metrics.get(objective, 'N/A')}")

    # 6. 解析敏感参数
    sensitive_list = None
    if args.sensitive_params:
        sensitive_list = [s.strip() for s in args.sensitive_params.split(',') if s.strip()]

    # 7. 创建 LLM 客户端（如需要）
    llm_client = None
    if args.use_llm:
        try:
            from universal_llm_client import UniversalLLMClient, PRESET_CONFIGS
            preset_key = f"ollama-{args.llm_model}"
            if preset_key in PRESET_CONFIGS:
                llm_client = UniversalLLMClient(PRESET_CONFIGS[preset_key])
            else:
                from universal_llm_client import UniversalLLMConfig
                llm_client = UniversalLLMClient(UniversalLLMConfig(
                    api_type="ollama", base_url="http://localhost:11434",
                    model_name=args.llm_model))
            if verbose:
                print(f"[LLM] 已启用: {args.llm_model}")
        except Exception as e:
            print(f"[警告] LLM 初始化失败: {e}，将使用全部参数")

    # 8. 创建 SPP 分析器
    spp_config = SPPConfig(
        n_samples=args.samples,
        perturbation_ratio=args.perturbation,
        objective=objective,
        use_llm=args.use_llm,
        sensitive_params=sensitive_list,
        random_seed=args.seed,
        record_samples=True,
        use_parallel=not args.no_parallel,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        sample_timeout=args.sample_timeout,
    )

    parallel_worker_args = None
    if spp_config.use_parallel:
        parallel_worker_args = SPPWorkerInitArgs(
            data_paths=[str(Path(dp).absolute()) for dp in args.data],
            strategy_path=str(Path(args.strategy).absolute()),
            objective=objective,
            data_frequency=effective_freq,
            broker_config=broker_config,
            data_names=data_names,
            is_multi_data=multi_data,
        )

    analyzer = SPPAnalyzer(
        backtest_engine=engine, strategy_class=strategy_class,
        data=data, search_space=search_space,
        config=spp_config, verbose=verbose, llm_client=llm_client,
        parallel_worker_args=parallel_worker_args)

    # 9. 运行分析
    output_dir = os.path.join(args.output, asset_name)
    result = analyzer.run_full_analysis(
        best_params=best_params, best_metrics=best_metrics,
        output_dir=output_dir, asset_name=asset_name,
        strategy_name=strategy_name, source_json=source_label,
        provenance={
            'parameter_source': parameter_source,
            'parameter_source_path': parameter_source_path,
            'center_override_file': center_override_file,
            'center_params_overridden': sorted(center_overrides),
            'baseline_recomputed': baseline_recomputed,
            'record_samples': spp_config.record_samples,
            'random_seed': spp_config.random_seed,
        })

    return 0


if __name__ == "__main__":
    sys.exit(main())
