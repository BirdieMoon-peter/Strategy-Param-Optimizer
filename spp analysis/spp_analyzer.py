# -*- coding: utf-8 -*-
"""
SPP (System Parameter Permutation) 鲁棒性分析模块 — v3.0

以贝叶斯优化找到的最优参数为中心，在邻域内进行蒙特卡洛采样，
评估参数鲁棒性。可选 LLM 辅助识别敏感参数，只扰动敏感参数以减少计算量。
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as _fm

# 按优先级查找可用中文字体文件
_CN_FONT_CANDIDATES = [
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
]
_cn_font_path = next((p for p in _CN_FONT_CANDIDATES if os.path.exists(p)), None)
if _cn_font_path:
    _fm.fontManager.addfont(_cn_font_path)
    _cn_font_name = _fm.FontProperties(fname=_cn_font_path).get_name()
    plt.rcParams['font.sans-serif'] = [_cn_font_name, 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['PingFang HK', 'STHeiti', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from backtest_engine import BacktestEngine, BacktestResult
from config import StrategyParam

try:
    from spp_parallel_engine import SPPParallelEvaluator
    SPP_PARALLEL_AVAILABLE = True
except ImportError:
    SPPParallelEvaluator = None
    SPP_PARALLEL_AVAILABLE = False


@dataclass
class SPPConfig:
    """SPP 分析配置"""
    n_samples: int = 300
    perturbation_ratio: float = 0.20
    objective: str = 'sharpe_ratio'
    use_llm: bool = False
    llm_config: Optional[Any] = None
    sensitive_params: Optional[List[str]] = None
    random_seed: Optional[int] = None
    record_samples: bool = True
    use_parallel: bool = True
    n_workers: Optional[int] = None
    batch_size: int = 32
    sample_timeout: int = 300


class SPPAnalyzer:
    """SPP 鲁棒性分析器 (v3.0) — 最优参数邻域蒙特卡洛采样"""

    def __init__(self, backtest_engine, strategy_class, data, search_space,
                 config=None, verbose=True, llm_client=None,
                 parallel_worker_args=None):
        self.engine = backtest_engine
        self.strategy_class = strategy_class
        self.data = data
        self.search_space = search_space
        self.config = config or SPPConfig()
        self.verbose = verbose
        self.llm_client = llm_client
        self.parallel_worker_args = parallel_worker_args
        self.parallel_info = {
            'use_parallel': False,
            'available': SPP_PARALLEL_AVAILABLE,
            'n_workers': 1,
            'batch_size': None,
            'sample_timeout': None,
            'failed_samples': 0,
            'failed_sample_details': [],
            'fallback_reason': None,
        }

    # ---- 敏感参数识别 ----

    def _identify_sensitive_params(self, best_params, best_metrics):
        """优先级: 手动指定 > LLM > 全部参数。返回 (list, method, reasoning)"""
        if self.config.sensitive_params:
            valid = [p for p in self.config.sensitive_params if p in self.search_space]
            if valid:
                if self.verbose:
                    print(f"  [敏感参数] 手动指定: {valid}")
                return valid, 'manual', ''
        if self.config.use_llm and self.llm_client is not None:
            try:
                params, reasoning = self._llm_extract_sensitive_params(best_params, best_metrics)
                if params:
                    if self.verbose:
                        print(f"  [敏感参数] LLM 识别: {params}")
                        if reasoning:
                            print(f"  [LLM 理由] {reasoning[:120]}...")
                    return params, 'llm', reasoning
            except Exception as e:
                if self.verbose:
                    print(f"  [警告] LLM 失败: {e}，fallback 到全部参数")
        all_p = list(self.search_space.keys())
        if self.verbose:
            print(f"  [敏感参数] 使用全部参数: {all_p}")
        return all_p, 'all', ''

    def _llm_extract_sensitive_params(self, best_params, best_metrics):
        """用 LLM 识别敏感参数。返回 (param_list, reasoning)"""
        sys_prompt = (
            '你是一个量化策略参数分析专家。给定策略的参数列表、最优值和搜索范围，'
            '判断哪些参数是"敏感参数"——即微小变化会显著影响策略表现的参数。'
            '通常与信号生成直接相关的参数（如周期、阈值）是敏感的，'
            '而与风控、仓位管理相关的参数相对不敏感。'
            '请返回 JSON: {"sensitive_params": ["p1","p2"], "reasoning": "..."}'
        )
        param_info = []
        for name, sp in self.search_space.items():
            param_info.append({
                'name': name, 'type': sp.param_type,
                'best_value': best_params.get(name, sp.default_value),
                'default_value': sp.default_value,
                'range': [sp.min_value, sp.max_value], 'step': sp.step,
            })
        mc = {k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()}
        user_prompt = (
            f"策略名称: {self.strategy_class.__name__}\n"
            f"最优参数性能: {json.dumps(mc, ensure_ascii=False)}\n"
            f"参数列表:\n{json.dumps(param_info, indent=2, ensure_ascii=False)}\n"
            f"请分析哪些是敏感参数，返回 JSON。"
        )
        resp = self.llm_client.client.call_llm([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])
        if not resp:
            return [], ''
        js, je = resp.find('{'), resp.rfind('}') + 1
        if js == -1 or je <= js:
            return [], ''
        r = json.loads(resp[js:je])
        return ([p for p in r.get('sensitive_params', []) if p in self.search_space],
                r.get('reasoning', ''))

    # ---- 蒙特卡洛采样 ----

    def _generate_mc_samples(self, best_params, sensitive_params):
        """对敏感参数均匀扰动采样，非敏感参数固定为最优值"""
        if self.config.random_seed is None:
            self.config.random_seed = int(np.random.SeedSequence().entropy)
        rng = np.random.default_rng(self.config.random_seed)
        ratio = self.config.perturbation_ratio
        samples = []
        for _ in range(self.config.n_samples):
            params = {}
            for name, sp in self.search_space.items():
                best_val = best_params.get(name, sp.default_value)
                if name not in sensitive_params:
                    params[name] = best_val
                    continue
                if sp.param_type == 'int':
                    int_ratio = ratio * 0.75  # int 用稍小扰动
                    lo = max(sp.min_value, best_val * (1 - int_ratio))
                    hi = min(sp.max_value, best_val * (1 + int_ratio))
                    if best_val == 0:
                        lo, hi = sp.min_value, sp.max_value
                    step = int(sp.step) if sp.step and sp.step >= 1 else 1
                    lo_i, hi_i = int(np.ceil(lo)), int(np.floor(hi))
                    possible = list(range(lo_i, hi_i + 1, step))
                    if not possible:
                        possible = [int(best_val)]
                    params[name] = int(rng.choice(possible))
                else:
                    lo = max(sp.min_value, best_val * (1 - ratio))
                    hi = min(sp.max_value, best_val * (1 + ratio))
                    if best_val == 0:
                        lo, hi = sp.min_value, sp.max_value
                    if lo >= hi:
                        lo, hi = sp.min_value, sp.max_value
                    if lo >= hi:
                        params[name] = float(best_val)
                        continue
                    val = rng.uniform(lo, hi)
                    if sp.step:
                        val = lo + round((val - lo) / sp.step) * sp.step
                        val = np.clip(val, lo, hi)
                    params[name] = float(round(val, 6))
            samples.append(params)
        return samples

    # ---- 批量回测 ----

    def _evaluate_batch_serial(self, param_list, desc=""):
        """串行批量回测，收集完整结果到 DataFrame"""
        self.parallel_info = {
            'use_parallel': False,
            'available': SPP_PARALLEL_AVAILABLE,
            'n_workers': 1,
            'batch_size': None,
            'sample_timeout': None,
            'failed_samples': 0,
            'failed_sample_details': [],
            'fallback_reason': None,
        }
        records = []
        total = len(param_list)
        start = time.time()
        for i, params in enumerate(param_list):
            result = self.engine.run_backtest(
                self.strategy_class, self.data, params, calculate_yearly=False
            )
            if result is not None:
                obj_val = self.engine.evaluate_objective(result, self.config.objective)
                records.append({
                    'sample_id': i + 1,
                    'params': params,
                    self.config.objective: obj_val,
                    'annual_return': result.annual_return,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'trades_count': result.trades_count,
                })
            if self.verbose and (i + 1) % max(1, total // 10) == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{desc}] {i+1}/{total} ({100*(i+1)/total:.0f}%) 剩余 {remaining:.0f}s")
        return pd.DataFrame(records)

    def _evaluate_batch_parallel(self, param_list, desc=""):
        """并行批量回测，worker 只负责回测，父进程统一汇总。"""
        total = len(param_list)
        n_workers = self.config.n_workers
        if n_workers is None:
            cpu_count = os.cpu_count() or 1
            n_workers = max(1, min(total, cpu_count - 2 if cpu_count > 2 else 1))
        else:
            n_workers = max(1, min(int(n_workers), total))

        if self.verbose:
            print(f"  [并行] workers={n_workers}, batch_size={self.config.batch_size}, "
                  f"timeout={self.config.sample_timeout}s")

        evaluator = SPPParallelEvaluator(
            worker_init_args=self.parallel_worker_args,
            n_workers=n_workers,
            batch_size=self.config.batch_size,
            sample_timeout=self.config.sample_timeout,
            verbose=self.verbose,
        )
        records = evaluator.evaluate(param_list, desc=desc)
        self.parallel_info = {
            'use_parallel': True,
            'available': True,
            'n_workers': n_workers,
            'batch_size': self.config.batch_size,
            'sample_timeout': self.config.sample_timeout,
            'failed_samples': len(evaluator.failed_samples),
            'failed_sample_details': evaluator.failed_samples[:20],
            'fallback_reason': None,
        }
        return pd.DataFrame(records)

    def _evaluate_batch(self, param_list, desc=""):
        """批量回测，按配置选择串行或多进程并行。"""
        if not param_list:
            return pd.DataFrame([])

        if not self.config.use_parallel:
            return self._evaluate_batch_serial(param_list, desc=desc)

        if not SPP_PARALLEL_AVAILABLE:
            if self.verbose:
                print("  [警告] SPP 并行执行器不可用，回退到串行")
            df = self._evaluate_batch_serial(param_list, desc=desc)
            self.parallel_info['fallback_reason'] = 'parallel_engine_unavailable'
            return df

        if self.parallel_worker_args is None:
            if self.verbose:
                print("  [警告] 未提供并行 worker 初始化参数，回退到串行")
            df = self._evaluate_batch_serial(param_list, desc=desc)
            self.parallel_info['fallback_reason'] = 'missing_worker_args'
            return df

        return self._evaluate_batch_parallel(param_list, desc=desc)

    def _serialize_mc_samples(self, mc_df):
        """将 Monte Carlo 结果转换为可序列化的精简样本列表"""
        if not self.config.record_samples or len(mc_df) == 0:
            return []
        samples = []
        for row in mc_df.to_dict(orient='records'):
            sample = {
                'sample_id': int(row['sample_id']),
                'params': row['params'],
                self.config.objective: round(float(row[self.config.objective]), 6),
                'annual_return': round(float(row['annual_return']), 6),
                'max_drawdown': round(float(row['max_drawdown']), 6),
                'sharpe_ratio': round(float(row['sharpe_ratio']), 6),
                'sortino_ratio': round(float(row['sortino_ratio']), 6),
                'trades_count': int(row['trades_count']),
            }
            samples.append(sample)
        return samples

    # ---- 蒙特卡洛分析 (核心) ----

    def run_monte_carlo_analysis(self, best_params, best_metrics):
        """
        以最优参数为中心的蒙特卡洛鲁棒性分析。
        返回 (DataFrame, sensitive_params, method, reasoning, sampling_metadata, sample_records)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP] 蒙特卡洛鲁棒性分析 ({self.config.n_samples} 组)")
            print(f"{'='*60}")

        sensitive_params, method, reasoning = self._identify_sensitive_params(best_params, best_metrics)
        samples = self._generate_mc_samples(best_params, sensitive_params)

        if self.verbose:
            print(f"  扰动比例: ±{self.config.perturbation_ratio:.0%}")
            print(f"  敏感参数 ({len(sensitive_params)}): {sensitive_params}")
            fixed = [n for n in self.search_space if n not in sensitive_params]
            if fixed:
                print(f"  固定参数 ({len(fixed)}): {fixed}")

        df = self._evaluate_batch(samples, desc="蒙特卡洛")
        sampling_metadata = {
            'n_samples': self.config.n_samples,
            'perturbation_ratio': self.config.perturbation_ratio,
            'random_seed': self.config.random_seed,
            'sensitive_params_method': method,
            'parallel_info': self.parallel_info,
        }
        sample_records = self._serialize_mc_samples(df)

        if self.verbose and len(df) > 0:
            obj = self.config.objective
            print(f"  有效样本: {len(df)}/{self.config.n_samples}")
            print(f"  {obj} 中位数: {df[obj].median():.4f}")
            print(f"  {obj} 均值:   {df[obj].mean():.4f}")
            print(f"  {obj} 标准差: {df[obj].std():.4f}")

        return df, sensitive_params, method, reasoning, sampling_metadata, sample_records

    # ---- 参数相关性 ----

    def _compute_param_correlations(self, df, sensitive_params):
        """计算各敏感参数与目标指标的 Pearson 相关系数"""
        obj = self.config.objective
        if len(df) < 5 or obj not in df.columns:
            return {}
        corrs = {}
        for pname in sensitive_params:
            vals = df['params'].apply(lambda d: d.get(pname))
            if vals.nunique() < 2:
                continue
            corrs[pname] = round(float(vals.astype(float).corr(df[obj])), 4)
        return corrs

    # ---- 多维度规则化分析 ----

    def _grade_robustness(self, decay_rate):
        """鲁棒性评级：基于衰减率"""
        d = abs(decay_rate)
        if d < 0.10:
            return '优', f'衰减率仅{d:.1%}，参数在邻域内表现极为稳定，微调参数几乎不影响策略效果'
        elif d < 0.25:
            return '良', f'衰减率{d:.1%}，参数具备一定鲁棒性，小幅偏离最优值仍可维持合理表现'
        elif d < 0.50:
            return '中', f'衰减率{d:.1%}，参数敏感性较高，偏离最优值后策略表现明显下降，需谨慎设定参数'
        else:
            return '差', f'衰减率高达{d:.1%}，策略高度依赖精确参数值，实盘中参数漂移风险极大'

    def _grade_overfit_risk(self, mc_df, best_obj, obj_key):
        """过拟合风险评级：基于最优值在 MC 分布中的分位数"""
        if len(mc_df) == 0 or obj_key not in mc_df.columns:
            return '中', '蒙特卡洛样本不足，无法准确评估过拟合风险'
        vals = mc_df[obj_key].dropna()
        if len(vals) == 0:
            return '中', '有效样本为零，无法评估'
        pct = float((vals < best_obj).sum()) / len(vals) * 100
        if pct < 75:
            return '优', f'最优值处于MC分布的{pct:.0f}%分位，大部分随机扰动也能达到相近水平，过拟合可能性低'
        elif pct < 90:
            return '良', f'最优值处于MC分布的{pct:.0f}%分位，优于多数扰动组合但未极端偏离，过拟合风险可控'
        elif pct < 98:
            return '中', f'最优值处于MC分布的{pct:.0f}%分位，显著优于绝大多数扰动结果，存在一定过拟合嫌疑'
        else:
            return '差', f'最优值处于MC分布的{pct:.0f}%分位，几乎是所有扰动中的极端值，过拟合风险很高'

    def _grade_trade_quality(self, trades_count, win_rate):
        """交易质量评级：基于交易次数和胜率"""
        wr = win_rate * 100 if win_rate <= 1 else win_rate  # 兼容小数/百分比
        if trades_count >= 30 and wr >= 40:
            return '优', f'共{trades_count}笔交易、胜率{wr:.1f}%，样本充足且胜率健康，统计意义可靠'
        elif trades_count >= 15 and wr >= 30:
            return '良', f'共{trades_count}笔交易、胜率{wr:.1f}%，交易频次尚可，但建议增加回测时间跨度以提高置信度'
        elif trades_count >= 5 and wr >= 20:
            return '中', f'共{trades_count}笔交易、胜率{wr:.1f}%，样本偏少或胜率偏低，结果参考价值有限'
        else:
            return '差', f'共{trades_count}笔交易、胜率{wr:.1f}%，交易次数过少或胜率过低，不具备统计显著性'

    def _grade_sensitivity(self, param_corrs, total_params):
        """敏感参数评级：基于高敏感参数占比 (|corr|>0.3)"""
        if total_params == 0:
            return '良', '无可分析参数'
        high_sens = sum(1 for v in param_corrs.values() if abs(v) > 0.3)
        ratio = high_sens / total_params
        high_names = [k for k, v in param_corrs.items() if abs(v) > 0.3]
        if ratio == 0:
            return '优', '所有参数与目标指标的相关性均较弱(|r|≤0.3)，策略不依赖单一参数驱动'
        elif ratio <= 0.25:
            return '良', f'高敏感参数({", ".join(high_names)})占比{ratio:.0%}，多数参数影响有限，整体可控'
        elif ratio <= 0.50:
            return '中', f'高敏感参数({", ".join(high_names)})占比{ratio:.0%}，策略表现受多个参数显著影响'
        else:
            return '差', f'高敏感参数({", ".join(high_names)})占比{ratio:.0%}，策略过度依赖参数精确取值'

    def _grade_risk_reward(self, sharpe, max_dd):
        """风险收益评级：基于 Sharpe 和最大回撤"""
        dd = abs(max_dd)  # 确保正值
        if sharpe >= 1.0 and dd <= 20:
            return '优', f'Sharpe={sharpe:.2f}、最大回撤={dd:.1f}%，风险收益比优秀，策略在风控和盈利间取得良好平衡'
        elif sharpe >= 0.5 and dd <= 35:
            return '良', f'Sharpe={sharpe:.2f}、最大回撤={dd:.1f}%，风险收益尚可，但仍有优化空间'
        elif sharpe >= 0.2 and dd <= 50:
            return '中', f'Sharpe={sharpe:.2f}、最大回撤={dd:.1f}%，收益勉强覆盖风险，需关注极端行情下的回撤控制'
        else:
            return '差', f'Sharpe={sharpe:.2f}、最大回撤={dd:.1f}%，风险收益比不理想，策略盈利能力或风控存在明显短板'

    def _grade_live_trading(self, grades):
        """实盘可用性评级：加权综合前 5 项"""
        score_map = {'优': 3, '良': 2, '中': 1, '差': 0}
        weights = {
            'robustness': 0.30, 'overfit_risk': 0.25,
            'trade_quality': 0.20, 'risk_reward': 0.15,
            'sensitivity': 0.10,
        }
        total = sum(score_map.get(grades.get(k, '中'), 1) * w
                    for k, w in weights.items())
        if total >= 2.5:
            grade = '优'
            text = f'综合加权得分{total:.2f}/3.00，各维度表现均衡且优秀，策略具备实盘部署条件'
        elif total >= 1.8:
            grade = '良'
            text = f'综合加权得分{total:.2f}/3.00，整体可用但部分维度存在短板，建议针对性优化后上线'
        elif total >= 1.0:
            grade = '中'
            text = f'综合加权得分{total:.2f}/3.00，策略存在较多风险点，不建议直接用于实盘'
        else:
            grade = '差'
            text = f'综合加权得分{total:.2f}/3.00，策略在多个维度表现不佳，需大幅改进后方可考虑实盘'
        return grade, text

    def _compute_analysis(self, mc_df, best_metrics, sensitive_params, param_corrs):
        """生成一段完整的中文分析文字，涵盖所有 SPP 指标"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)
        n_samples = self.config.n_samples
        ratio = self.config.perturbation_ratio

        # 无有效 MC 数据时的 fallback
        if len(mc_df) == 0 or obj not in mc_df.columns:
            return (f'本策略经SPP蒙特卡洛分析（{n_samples}组采样，±{ratio:.0%}扰动），'
                    f'未获得有效回测样本，无法评估参数鲁棒性。')

        vals = mc_df[obj].dropna()
        mc_median = float(vals.median())
        decay = (best_obj - mc_median) / abs(best_obj) if best_obj != 0 else 0

        # 中位数对应的统计结果：取 objective 最接近中位数的那一行
        median_idx = (mc_df[obj] - mc_median).abs().idxmin()
        median_row = mc_df.loc[median_idx]
        med_ar = median_row.get('annual_return', 0)
        med_dd = median_row.get('max_drawdown', 0)

        # 众数区间：对 MC 分布分 bin，取频次最高的区间
        n_bins = min(15, max(5, len(vals) // 5))
        counts, bin_edges = np.histogram(vals, bins=n_bins)
        mode_idx = int(np.argmax(counts))
        mode_lo, mode_hi = float(bin_edges[mode_idx]), float(bin_edges[mode_idx + 1])
        mode_pct = counts[mode_idx] / len(vals) * 100

        # 内部评级（基于中位数对应行的指标）
        med_trades = int(median_row.get('trades_count', 0))
        med_sharpe = float(median_row.get('sharpe_ratio', 0))

        g_rob, _ = self._grade_robustness(decay)
        g_ovf, _ = self._grade_overfit_risk(mc_df, best_obj, obj)
        g_trd, _ = self._grade_trade_quality(med_trades, 0)  # MC 无 win_rate
        g_sen, _ = self._grade_sensitivity(param_corrs, len(self.search_space))
        g_rr, _ = self._grade_risk_reward(med_sharpe, med_dd)
        grades = {
            'robustness': g_rob, 'overfit_risk': g_ovf,
            'trade_quality': g_trd, 'risk_reward': g_rr,
            'sensitivity': g_sen,
        }
        g_live, _ = self._grade_live_trading(grades)

        # ---- 拼接分析文字 ----
        parts = []

        # 1) 采样配置 + 最优 vs 中位数 + 衰减
        parts.append(
            f'本策略经SPP蒙特卡洛分析（{n_samples}组采样，±{ratio:.0%}扰动），'
            f'最优{obj}为{best_obj:.4f}，MC {obj}中位数为{mc_median:.4f}（衰减{abs(decay):.1%}）')

        # 2) 中位数对应的年化收益、最大回撤、交易次数（附带定性判断）
        med_desc = f'中位数对应年化收益{med_ar:.2f}%、最大回撤{abs(med_dd):.2f}%、交易{med_trades}笔'
        med_notes = []
        if g_trd in ('差', '中'):
            med_notes.append('交易样本不足')
        if g_rr in ('差', '中') and abs(med_dd) > 35:
            med_notes.append('回撤偏高')
        if med_notes:
            med_desc += f'（{"，".join(med_notes)}）'
        parts.append(med_desc)

        # 3) 众数区间
        parts.append(
            f'扰动结果集中在{mode_lo:.2f}~{mode_hi:.2f}区间（占比{mode_pct:.0f}%）')

        # 4) 最优值分位数 + 最优参数的统计指标
        pct = float((vals < best_obj).sum()) / len(vals) * 100
        best_ar = best_metrics.get('annual_return', 0)
        best_dd = abs(best_metrics.get('max_drawdown', 0))
        best_trades = int(best_metrics.get('trades_count', 0))
        parts.append(
            f'最优值处于MC分布{pct:.0f}%分位'
            f'（最优参数年化收益{best_ar:.2f}%、最大回撤{best_dd:.2f}%、交易{best_trades}笔）'
            + ('，过拟合风险较高' if g_ovf in ('差',) else ''))

        # 5) 综合评价
        eval_map = {
            '优': '参数鲁棒性强，各维度表现均衡，策略具备实盘部署条件',
            '良': '参数鲁棒性尚可，建议针对薄弱环节优化后再部署',
            '中': '策略存在明显短板，不建议直接用于实盘',
            '差': '策略在多个维度表现不佳，需大幅改进后方可考虑实盘',
        }
        parts.append(f'综合评价：{eval_map[g_live]}')

        return '。'.join(parts) + '。'

    # ---- 可视化报告 (2x2) ----

    def generate_report(self, mc_df, best_params, best_metrics,
                        sensitive_params, param_corrs, output_path, analysis=None):
        """生成 2x2 PNG 报告"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'SPP 蒙特卡洛鲁棒性分析 — {self.strategy_class.__name__}',
                     fontsize=14, fontweight='bold')

        # (0,0) MC 分布直方图 + KDE
        ax = axes[0, 0]
        if len(mc_df) > 0:
            vals = mc_df[obj].dropna()
            ax.hist(vals, bins=40, density=True, alpha=0.6,
                    color='seagreen', edgecolor='white')
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals)
                x_range = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(x_range, kde(x_range), 'g-', lw=2, label='KDE')
            except ImportError:
                pass
            median_val = vals.median()
            mean_val = vals.mean()
            ax.axvline(best_obj, color='red', ls='-', lw=2,
                       label=f'最优={best_obj:.3f}')
            ax.axvline(median_val, color='orange', ls='--', lw=2,
                       label=f'中位数={median_val:.3f}')
            decay = (best_obj - median_val) / abs(best_obj) if best_obj != 0 else 0
            ax.text(0.05, 0.95, f'衰减率={decay:.1%}',
                    transform=ax.transAxes, fontsize=11, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.legend(fontsize=9)
        ax.set_title(f'蒙特卡洛 {obj} 分布 (n={len(mc_df)}, '
                     f'扰动={self.config.perturbation_ratio:.0%})', fontsize=12)
        ax.set_xlabel(obj)
        ax.set_ylabel('密度')

        # (0,1) 参数敏感度柱状图
        ax = axes[0, 1]
        if param_corrs:
            names = list(param_corrs.keys())
            values = [param_corrs[n] for n in names]
            colors = ['#e74c3c' if abs(v) > 0.3 else '#3498db' for v in values]
            bars = ax.barh(names, values, color=colors, edgecolor='white')
            ax.axvline(0, color='gray', ls='-', lw=0.8)
            ax.axvline(0.3, color='red', ls=':', lw=1, alpha=0.5)
            ax.axvline(-0.3, color='red', ls=':', lw=1, alpha=0.5)
            for bar, v in zip(bars, values):
                ax.text(v + 0.02 if v >= 0 else v - 0.02, bar.get_y() + bar.get_height()/2,
                        f'{v:.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
        ax.set_title(f'参数敏感度 (与{obj}的相关系数)', fontsize=12)
        ax.set_xlabel('Pearson 相关系数')

        # (1,0) 风险-收益散点图
        ax = axes[1, 0]
        if len(mc_df) > 0 and 'max_drawdown' in mc_df.columns:
            ax.scatter(mc_df['max_drawdown'], mc_df['annual_return'],
                       alpha=0.4, s=15, c='seagreen')
            best_dd = best_metrics.get('max_drawdown', 0)
            best_ar = best_metrics.get('annual_return', 0)
            ax.scatter([best_dd], [best_ar], c='red', s=120, zorder=5,
                       marker='*', label='最优参数')
            ax.legend(fontsize=9)
        ax.set_title('风险-收益散点图 (MC采样)', fontsize=12)
        ax.set_xlabel('最大回撤 (%)')
        ax.set_ylabel('年化收益 (%)')

        # (1,1) 文字总结面板
        ax = axes[1, 1]
        ax.axis('off')
        summary_lines = self._build_text_summary(
            mc_df, best_params, best_metrics, sensitive_params, param_corrs,
            analysis or '')
        ax.text(0.05, 0.95, '\n'.join(summary_lines),
                transform=ax.transAxes, fontsize=10, va='top', ha='left',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.verbose:
            print(f"\n[SPP] 报告已保存: {output_path}")
        return output_path

    def _build_text_summary(self, mc_df, best_params, best_metrics,
                            sensitive_params, param_corrs, analysis):
        """构建文字总结面板内容"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)
        lines = ['═══ SPP 鲁棒性总结 ═══', '']

        # 最优参数
        lines.append('▸ 最优参数:')
        for k, v in best_params.items():
            marker = ' *' if k in sensitive_params else ''
            lines.append(f'    {k} = {v}{marker}')
        lines.append(f'▸ 最优 {obj}: {best_obj:.4f}')
        lines.append('')

        # MC 统计
        if len(mc_df) > 0:
            vals = mc_df[obj]
            mc_median = vals.median()
            decay = (best_obj - mc_median) / abs(best_obj) if best_obj != 0 else 0
            lines += [
                f'▸ MC中位数: {mc_median:.4f}',
                f'▸ 衰减率:   {decay:.1%}',
                '',
            ]

        # 分析文字（自动折行）
        lines.append('═══ 综合分析 ═══')
        if isinstance(analysis, str) and analysis:
            import textwrap
            for wrapped in textwrap.wrap(analysis, width=28):
                lines.append(wrapped)
        return lines

    # ---- 完整分析流程 ----

    def run_full_analysis(self, best_params, best_metrics, output_dir,
                          asset_name='ASSET', strategy_name='Strategy',
                          source_json='', provenance=None):
        """运行完整 SPP 分析并输出 JSON + PNG"""
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"# SPP 蒙特卡洛鲁棒性分析 (v3.0)")
            print(f"# 标的: {asset_name}  策略: {strategy_name}")
            print(f"# 目标: {self.config.objective}")
            print(f"# 采样: {self.config.n_samples} 次, 扰动: ±{self.config.perturbation_ratio:.0%}")
            print(f"{'#'*60}")

        # 蒙特卡洛分析
        mc_df, sensitive_params, sp_method, sp_reasoning, sampling_metadata, sample_records = \
            self.run_monte_carlo_analysis(best_params, best_metrics)

        elapsed = time.time() - start_time

        # 参数相关性
        param_corrs = self._compute_param_correlations(mc_df, sensitive_params)

        # 多维度分析（先于 generate_report，以便传入面板）
        analysis = self._compute_analysis(
            mc_df, best_metrics, sensitive_params, param_corrs)

        # 生成 PNG 报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(output_dir,
                                f'spp_report_{asset_name}_{timestamp}.png')
        self.generate_report(mc_df, best_params, best_metrics,
                             sensitive_params, param_corrs, png_path,
                             analysis=analysis)

        # 构建 JSON 结果
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)

        result = {
            'spp_info': {
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_seconds': round(elapsed, 1),
                'source_json': source_json,
                'asset': asset_name,
                'strategy': strategy_name,
                'config': {
                    'n_samples': self.config.n_samples,
                    'perturbation_ratio': self.config.perturbation_ratio,
                    'objective': self.config.objective,
                    'use_llm': self.config.use_llm,
                    'random_seed': self.config.random_seed,
                    'record_samples': self.config.record_samples,
                    'use_parallel': self.config.use_parallel,
                    'n_workers': self.config.n_workers,
                    'batch_size': self.config.batch_size,
                    'sample_timeout': self.config.sample_timeout,
                },
            },
            'best_parameters': best_params,
            'best_metrics': best_metrics,
            'monte_carlo': {
                'sampling_metadata': sampling_metadata,
                'samples_saved': len(sample_records),
                'samples': sample_records,
            },
        }
        if provenance:
            result['spp_info'].update(provenance)
        result['spp_info']['random_seed'] = self.config.random_seed
        result['spp_info']['record_samples'] = self.config.record_samples
        result['parallel_info'] = self.parallel_info

        # 敏感参数信息
        result['sensitive_params'] = {
            'method': sp_method,
            'params': sensitive_params,
            'llm_reasoning': sp_reasoning,
        }

        # MC 统计
        if len(mc_df) > 0:
            vals = mc_df[obj]
            mc_median = float(vals.median())
            decay = (best_obj - mc_median) / abs(best_obj) if best_obj != 0 else 0
            result['monte_carlo_stability'] = {
                'sample_count': len(mc_df),
                'perturbation_ratio': self.config.perturbation_ratio,
                'median': round(mc_median, 4),
                'mean': round(float(vals.mean()), 4),
                'std': round(float(vals.std()), 4),
                'p5': round(float(vals.quantile(0.05)), 4),
                'p25': round(float(vals.quantile(0.25)), 4),
                'p75': round(float(vals.quantile(0.75)), 4),
                'p95': round(float(vals.quantile(0.95)), 4),
                'decay_rate': round(float(decay), 4),
                'robustness_score': round(max(0, 1 - abs(decay)) * 100, 1),
                'param_correlations': param_corrs,
            }
        else:
            result['monte_carlo_stability'] = {}

        # 多维度分析
        result['analysis'] = analysis

        # 保存 JSON
        json_path = os.path.join(output_dir,
                                 f'spp_result_{asset_name}_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP] 分析完成! 耗时 {elapsed:.1f}s")
            print(f"[SPP] JSON: {json_path}")
            print(f"[SPP] PNG:  {png_path}")
            print(f"{'='*60}")
            print(f"\n  [分析] {analysis}")

        return result
