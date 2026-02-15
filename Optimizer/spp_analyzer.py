# -*- coding: utf-8 -*-
"""
SPP (System Parameter Permutation) 鲁棒性分析模块

基于 Dave Walton 的论文《Know Your System!》实现。
核心思想：不只看单一最优参数，而是看参数空间的整体分布（中位数、概率密度），
从而识别过拟合和数据挖掘偏差。

三个分析维度:
1. 全局分布 (Global Permutation) - 参数空间均匀采样，构建目标指标的抽样分布
2. 局部稳定性 (Local Stability) - 最优参数邻域扰动，衡量参数鲁棒性
3. 短期最坏情况 (Short-Run Worst-Case) - 逐年分析，识别最坏年份的尾部风险
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

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import BacktestEngine, BacktestResult
from enhanced_sampler import NormalDistributionSampler, ParallelExplorer, SamplerConfig
from config import StrategyParam


@dataclass
class SPPConfig:
    """SPP 分析配置"""
    global_samples: int = 500
    local_samples: int = 200
    local_noise_level: float = 0.10
    yearly_samples: int = 200
    objective: str = 'sharpe_ratio'
    min_year_bars: int = 60


class SPPAnalyzer:
    """
    SPP 鲁棒性分析器

    通过三个维度评估优化参数的鲁棒性:
    1. 全局分布 - 参数空间的整体表现分布
    2. 局部稳定性 - 最优参数邻域的表现衰减
    3. 短期最坏情况 - 逐年的尾部风险
    """

    def __init__(
        self,
        backtest_engine: BacktestEngine,
        strategy_class,
        data: Any,
        search_space: Dict[str, StrategyParam],
        config: SPPConfig = None,
        verbose: bool = True
    ):
        self.engine = backtest_engine
        self.strategy_class = strategy_class
        self.data = data
        self.search_space = search_space
        self.config = config or SPPConfig()
        self.verbose = verbose

    # ------------------------------------------------------------------
    # 采样工具
    # ------------------------------------------------------------------

    def _generate_uniform_samples(self, n: int) -> List[Dict[str, Any]]:
        """在搜索空间内均匀随机采样 n 组参数"""
        rng = np.random.default_rng()
        samples = []
        for _ in range(n):
            params = {}
            for name, sp in self.search_space.items():
                if sp.param_type == 'int':
                    step = int(sp.step) if sp.step and sp.step >= 1 else 1
                    possible = list(range(int(sp.min_value), int(sp.max_value) + 1, step))
                    params[name] = int(rng.choice(possible))
                else:
                    val = rng.uniform(sp.min_value, sp.max_value)
                    if sp.step:
                        val = sp.min_value + round((val - sp.min_value) / sp.step) * sp.step
                        val = np.clip(val, sp.min_value, sp.max_value)
                    params[name] = float(round(val, 6))
            samples.append(params)
        return samples

    def _evaluate_batch(
        self, param_list: List[Dict], desc: str = ""
    ) -> pd.DataFrame:
        """批量回测，收集完整结果到 DataFrame"""
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
                print(f"  [{desc}] {i+1}/{total} "
                      f"({100*(i+1)/total:.0f}%) "
                      f"剩余 {remaining:.0f}s")

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 维度 1: 全局分布
    # ------------------------------------------------------------------

    def run_global_distribution(self) -> pd.DataFrame:
        """在整个参数空间均匀采样，构建目标指标的抽样分布"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP 维度1] 全局分布分析 ({self.config.global_samples} 组)")
            print(f"{'='*60}")

        samples = self._generate_uniform_samples(self.config.global_samples)
        df = self._evaluate_batch(samples, desc="全局分布")

        if self.verbose and len(df) > 0:
            obj = self.config.objective
            print(f"  有效样本: {len(df)}/{self.config.global_samples}")
            print(f"  {obj} 中位数: {df[obj].median():.4f}")
            print(f"  {obj} 均值:   {df[obj].mean():.4f}")
            print(f"  {obj} 标准差: {df[obj].std():.4f}")

        return df

    # ------------------------------------------------------------------
    # 维度 2: 局部稳定性
    # ------------------------------------------------------------------

    def run_local_stability(self, best_params: Dict[str, Any]) -> pd.DataFrame:
        """以最优参数为中心，正态扰动采样，衡量局部鲁棒性"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP 维度2] 局部稳定性分析 ({self.config.local_samples} 组)")
            print(f"{'='*60}")

        # 构建以 best_params 为中心、std = noise_level * range 的采样器
        sampler_config = SamplerConfig(
            use_normal_distribution=True,
            normal_std_ratio=self.config.local_noise_level,
        )
        sampler = NormalDistributionSampler(sampler_config)

        samples = []
        for _ in range(self.config.local_samples):
            params = {}
            for name, sp in self.search_space.items():
                center = best_params.get(name, sp.default_value)
                params[name] = sampler.sample_single_param(
                    param_name=name,
                    min_value=sp.min_value,
                    max_value=sp.max_value,
                    param_type=sp.param_type,
                    default_value=center,
                    step=sp.step,
                )
            samples.append(params)

        df = self._evaluate_batch(samples, desc="局部稳定性")

        if self.verbose and len(df) > 0:
            obj = self.config.objective
            print(f"  有效样本: {len(df)}/{self.config.local_samples}")
            print(f"  {obj} 中位数: {df[obj].median():.4f}")
            print(f"  {obj} 均值:   {df[obj].mean():.4f}")

        return df

    # ------------------------------------------------------------------
    # 维度 3: 逐年最坏情况
    # ------------------------------------------------------------------

    def run_short_run_analysis(self) -> Dict[int, pd.DataFrame]:
        """按年切分数据，每年独立采样回测，输出分位数统计"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP 维度3] 逐年最坏情况分析 (每年 {self.config.yearly_samples} 组)")
            print(f"{'='*60}")

        # 获取数据的年份列表
        df_data = self.data
        if isinstance(df_data, (list, tuple)):
            df_data = df_data[0]

        df_copy = df_data.copy()
        if 'datetime' in df_copy.columns:
            df_copy['_dt'] = pd.to_datetime(df_copy['datetime'])
        elif 'date' in df_copy.columns:
            df_copy['_dt'] = pd.to_datetime(df_copy['date'])
        else:
            if self.verbose:
                print("  [警告] 无法识别日期列，跳过逐年分析")
            return {}

        df_copy['_year'] = df_copy['_dt'].dt.year
        years = sorted(df_copy['_year'].unique())

        yearly_results = {}
        for year in years:
            year_data = df_copy[df_copy['_year'] == year].drop(columns=['_dt', '_year'])
            if len(year_data) < self.config.min_year_bars:
                if self.verbose:
                    print(f"  {year}年: 数据不足 ({len(year_data)} bars < {self.config.min_year_bars})，跳过")
                continue

            if self.verbose:
                print(f"\n  --- {year}年 ({len(year_data)} bars) ---")

            # 为该年创建独立的 BacktestEngine
            year_engine = BacktestEngine(
                data=year_data,
                strategy_class=self.strategy_class,
                initial_cash=self.engine.config.initial_cash,
                commission=self.engine.config.commission,
                data_frequency=self.engine.config.data_frequency,
                custom_data_class=self.engine.custom_data_class,
                custom_commission_class=self.engine.custom_commission_class,
                strategy_module=self.engine.strategy_module,
                use_trade_log_metrics=self.engine.use_trade_log_metrics,
                broker_config=self.engine.broker_config,
            )

            samples = self._generate_uniform_samples(self.config.yearly_samples)
            records = []
            for params in samples:
                result = year_engine.run_backtest(
                    self.strategy_class, year_data, params, calculate_yearly=False
                )
                if result is not None:
                    obj_val = year_engine.evaluate_objective(result, self.config.objective)
                    records.append({
                        'params': params,
                        self.config.objective: obj_val,
                        'annual_return': result.annual_return,
                        'max_drawdown': result.max_drawdown,
                        'sharpe_ratio': result.sharpe_ratio,
                    })

            if records:
                year_df = pd.DataFrame(records)
                yearly_results[year] = year_df
                obj = self.config.objective
                if self.verbose:
                    print(f"  {year}年 有效样本: {len(year_df)}")
                    print(f"  {year}年 {obj} 5th: {year_df[obj].quantile(0.05):.4f}  "
                          f"中位数: {year_df[obj].median():.4f}  "
                          f"95th: {year_df[obj].quantile(0.95):.4f}")

        return yearly_results

    # ------------------------------------------------------------------
    # 可视化报告
    # ------------------------------------------------------------------

    def generate_report(
        self,
        global_df: pd.DataFrame,
        local_df: pd.DataFrame,
        yearly_dfs: Dict[int, pd.DataFrame],
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
        output_path: str,
    ) -> str:
        """生成 3x2 PNG 可视化报告，返回文件路径"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)

        plt.rcParams['font.sans-serif'] = [
            'Heiti TC',           # macOS 黑体
            'Hiragino Sans GB',   # macOS 冬青黑体
            'PingFang SC',        # macOS 苹方
            'Songti SC',          # macOS 宋体
            'STHeiti',            # macOS 华文黑体
            'Arial Unicode MS',   # macOS 通用 Unicode
            'SimHei',             # Windows 黑体
            'Microsoft YaHei',    # Windows 微软雅黑
            'WenQuanYi Micro Hei',# Linux 文泉驿
            'DejaVu Sans',        # 兜底
        ]
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        fig.suptitle('SPP 鲁棒性分析报告', fontsize=18, fontweight='bold', y=0.98)

        # (0,0) 全局分布直方图 + KDE
        ax = axes[0, 0]
        if len(global_df) > 0:
            vals = global_df[obj].dropna()
            ax.hist(vals, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='white')
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals)
                x_range = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(x_range, kde(x_range), 'b-', lw=2, label='KDE')
            except ImportError:
                pass
            median_val = vals.median()
            ax.axvline(median_val, color='orange', ls='--', lw=2, label=f'中位数={median_val:.3f}')
            ax.axvline(best_obj, color='red', ls='-', lw=2, label=f'最优={best_obj:.3f}')
            ax.legend(fontsize=9)
        ax.set_title(f'全局 {obj} 分布 (n={len(global_df)})', fontsize=12)
        ax.set_xlabel(obj)
        ax.set_ylabel('密度')

        # (0,1) 局部稳定性 KDE
        ax = axes[0, 1]
        if len(local_df) > 0:
            vals = local_df[obj].dropna()
            ax.hist(vals, bins=30, density=True, alpha=0.6, color='seagreen', edgecolor='white')
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals)
                x_range = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(x_range, kde(x_range), 'g-', lw=2, label='KDE')
            except ImportError:
                pass
            local_median = vals.median()
            decay = (best_obj - local_median) / abs(best_obj) if best_obj != 0 else 0
            ax.axvline(best_obj, color='red', ls='-', lw=2, label=f'最优={best_obj:.3f}')
            ax.axvline(local_median, color='orange', ls='--', lw=2, label=f'中位数={local_median:.3f}')
            ax.text(0.05, 0.95, f'衰减率={decay:.1%}', transform=ax.transAxes,
                    fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.legend(fontsize=9)
        ax.set_title(f'局部稳定性 (noise={self.config.local_noise_level:.0%}, n={len(local_df)})', fontsize=12)
        ax.set_xlabel(obj)
        ax.set_ylabel('密度')

        # (1,0) 风险-收益散点图
        ax = axes[1, 0]
        if len(global_df) > 0 and 'max_drawdown' in global_df.columns:
            ax.scatter(global_df['max_drawdown'], global_df['annual_return'],
                       alpha=0.4, s=15, c='steelblue')
            best_dd = best_metrics.get('max_drawdown', 0)
            best_ar = best_metrics.get('annual_return', 0)
            ax.scatter([best_dd], [best_ar], c='red', s=120, zorder=5,
                       marker='*', label=f'最优参数')
            ax.legend(fontsize=9)
        ax.set_title('风险-收益散点图 (全局采样)', fontsize=12)
        ax.set_xlabel('最大回撤 (%)')
        ax.set_ylabel('年化收益 (%)')

        # (1,1) 逐年箱线图
        ax = axes[1, 1]
        if yearly_dfs:
            sorted_years = sorted(yearly_dfs.keys())
            box_data = [yearly_dfs[y][obj].dropna().values for y in sorted_years]
            bp = ax.boxplot(box_data, labels=[str(y) for y in sorted_years],
                           patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            # 5th 分位线
            p5_vals = [yearly_dfs[y][obj].quantile(0.05) for y in sorted_years]
            ax.plot(range(1, len(sorted_years) + 1), p5_vals, 'rv--', ms=6, label='5th 分位')
            ax.legend(fontsize=9)
            ax.tick_params(axis='x', rotation=45)
        ax.set_title(f'逐年 {obj} 箱线图', fontsize=12)
        ax.set_ylabel(obj)

        # (2,0) 逐年分位数热力图
        ax = axes[2, 0]
        if yearly_dfs:
            sorted_years = sorted(yearly_dfs.keys())
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
            q_labels = ['5th', '25th', '50th', '75th', '95th']
            heatmap_data = []
            for y in sorted_years:
                row = [yearly_dfs[y][obj].quantile(q) for q in quantiles]
                heatmap_data.append(row)
            heatmap_arr = np.array(heatmap_data)
            im = ax.imshow(heatmap_arr.T, aspect='auto', cmap='RdYlGn')
            ax.set_xticks(range(len(sorted_years)))
            ax.set_xticklabels([str(y) for y in sorted_years], rotation=45)
            ax.set_yticks(range(len(q_labels)))
            ax.set_yticklabels(q_labels)
            # 在每个格子中标注数值
            for i in range(len(q_labels)):
                for j in range(len(sorted_years)):
                    ax.text(j, i, f'{heatmap_arr[j, i]:.2f}', ha='center', va='center', fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'逐年 {obj} 分位数热力图', fontsize=12)

        # (2,1) 文字总结面板
        ax = axes[2, 1]
        ax.axis('off')
        summary_lines = self._build_text_summary(global_df, local_df, yearly_dfs, best_params, best_metrics)
        ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.verbose:
            print(f"\n[SPP] 报告已保存: {output_path}")
        return output_path

    def _build_text_summary(
        self,
        global_df: pd.DataFrame,
        local_df: pd.DataFrame,
        yearly_dfs: Dict[int, pd.DataFrame],
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
    ) -> List[str]:
        """构建文字总结面板内容"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)
        lines = ['═══ SPP 鲁棒性总结 ═══', '']

        # 全局
        if len(global_df) > 0:
            g_median = global_df[obj].median()
            g_mean = global_df[obj].mean()
            profitable = (global_df[obj] > 0).mean() * 100
            percentile = (global_df[obj] < best_obj).mean() * 100
            lines += [
                f'▸ 全局中位数:    {g_median:.4f}',
                f'▸ 全局均值:      {g_mean:.4f}',
                f'▸ 盈利参数占比:  {profitable:.1f}%',
                f'▸ 最优分位排名:  {percentile:.1f}%',
                '',
            ]

        # 局部
        if len(local_df) > 0:
            l_median = local_df[obj].median()
            decay = (best_obj - l_median) / abs(best_obj) if best_obj != 0 else 0
            robust_score = max(0, 1 - abs(decay)) * 100
            lines += [
                f'▸ 局部中位数:    {l_median:.4f}',
                f'▸ 衰减率:        {decay:.1%}',
                f'▸ 鲁棒性评分:    {robust_score:.0f}/100',
                '',
            ]

        # 逐年
        if yearly_dfs:
            worst_year = None
            worst_p5 = float('inf')
            for y, ydf in yearly_dfs.items():
                p5 = ydf[obj].quantile(0.05)
                if p5 < worst_p5:
                    worst_p5 = p5
                    worst_year = y
            lines += [
                f'▸ 最差年份:      {worst_year}',
                f'▸ 最差年5th分位: {worst_p5:.4f}',
                '',
            ]

        # 综合判定
        verdict = self._compute_verdict(global_df, local_df, yearly_dfs, best_metrics)
        lines += [
            '═══ 综合判定 ═══',
            f'过拟合风险: {verdict["overfit_risk"]}',
            f'全局Edge:   {verdict["global_edge"]}',
            f'参数鲁棒:   {verdict["parameter_robust"]}',
        ]
        return lines

    def _compute_verdict(
        self,
        global_df: pd.DataFrame,
        local_df: pd.DataFrame,
        yearly_dfs: Dict[int, pd.DataFrame],
        best_metrics: Dict[str, Any],
    ) -> Dict[str, str]:
        """计算综合判定"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)
        verdict = {
            'overfit_risk': '未知',
            'global_edge': '未知',
            'parameter_robust': '未知',
            'worst_year_5pct': 0,
            'summary': '',
        }

        # 全局 edge
        if len(global_df) > 0:
            g_median = global_df[obj].median()
            percentile = (global_df[obj] < best_obj).mean() * 100
            if g_median > 0:
                verdict['global_edge'] = '有 (中位数>0)'
            else:
                verdict['global_edge'] = '无 (中位数<=0)'

            if percentile > 95:
                verdict['overfit_risk'] = '高 (>95th分位)'
            elif percentile > 80:
                verdict['overfit_risk'] = '中 (80-95th分位)'
            else:
                verdict['overfit_risk'] = '低 (<80th分位)'

        # 局部鲁棒性
        if len(local_df) > 0:
            l_median = local_df[obj].median()
            decay = (best_obj - l_median) / abs(best_obj) if best_obj != 0 else 0
            if abs(decay) < 0.15:
                verdict['parameter_robust'] = '强 (衰减<15%)'
            elif abs(decay) < 0.30:
                verdict['parameter_robust'] = '中 (衰减15-30%)'
            else:
                verdict['parameter_robust'] = '弱 (衰减>30%)'

        # 最差年份
        if yearly_dfs:
            worst_p5 = min(ydf[obj].quantile(0.05) for ydf in yearly_dfs.values())
            verdict['worst_year_5pct'] = float(worst_p5)

        # 总结
        parts = []
        if '高' in verdict['overfit_risk']:
            parts.append('过拟合风险较高')
        if '无' in verdict['global_edge']:
            parts.append('策略缺乏全局Edge')
        if '弱' in verdict['parameter_robust']:
            parts.append('参数敏感性高')
        verdict['summary'] = '；'.join(parts) if parts else '参数整体鲁棒，策略具备统计Edge'

        return verdict

    # ------------------------------------------------------------------
    # 完整分析流程
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
        output_dir: str,
        asset_name: str = 'ASSET',
        strategy_name: str = 'Strategy',
        source_json: str = '',
        skip_short_run: bool = False,
    ) -> dict:
        """运行完整 SPP 分析并输出 JSON + PNG"""
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"# SPP 鲁棒性分析")
            print(f"# 标的: {asset_name}  策略: {strategy_name}")
            print(f"# 目标: {self.config.objective}")
            print(f"{'#'*60}")

        # 维度 1
        global_df = self.run_global_distribution()

        # 维度 2
        local_df = self.run_local_stability(best_params)

        # 维度 3
        yearly_dfs = {}
        if not skip_short_run:
            yearly_dfs = self.run_short_run_analysis()

        elapsed = time.time() - start_time

        # 生成 PNG 报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(output_dir, f'spp_report_{asset_name}_{timestamp}.png')
        self.generate_report(global_df, local_df, yearly_dfs, best_params, best_metrics, png_path)

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
                    'global_samples': self.config.global_samples,
                    'local_samples': self.config.local_samples,
                    'local_noise_level': self.config.local_noise_level,
                    'yearly_samples': self.config.yearly_samples,
                    'objective': self.config.objective,
                },
            },
            'best_parameters': best_params,
            'best_metrics': best_metrics,
        }

        # 全局分布统计
        if len(global_df) > 0:
            vals = global_df[obj]
            result['global_distribution'] = {
                'sample_count': len(global_df),
                'median': round(float(vals.median()), 4),
                'mean': round(float(vals.mean()), 4),
                'std': round(float(vals.std()), 4),
                'p5': round(float(vals.quantile(0.05)), 4),
                'p25': round(float(vals.quantile(0.25)), 4),
                'p75': round(float(vals.quantile(0.75)), 4),
                'p95': round(float(vals.quantile(0.95)), 4),
                'profitability_rate': round(float((vals > 0).mean()) * 100, 1),
                'best_percentile': round(float((vals < best_obj).mean()) * 100, 1),
            }
        else:
            result['global_distribution'] = {}

        # 局部稳定性统计
        if len(local_df) > 0:
            lvals = local_df[obj]
            l_median = float(lvals.median())
            decay = (best_obj - l_median) / abs(best_obj) if best_obj != 0 else 0
            result['local_stability'] = {
                'sample_count': len(local_df),
                'median': round(l_median, 4),
                'mean': round(float(lvals.mean()), 4),
                'decay_rate': round(float(decay), 4),
                'robustness_score': round(max(0, 1 - abs(decay)) * 100, 1),
            }
        else:
            result['local_stability'] = {}

        # 逐年统计
        short_run = {}
        for y, ydf in sorted(yearly_dfs.items()):
            yvals = ydf[obj]
            short_run[str(y)] = {
                'p5': round(float(yvals.quantile(0.05)), 4),
                'p25': round(float(yvals.quantile(0.25)), 4),
                'median': round(float(yvals.median()), 4),
                'p75': round(float(yvals.quantile(0.75)), 4),
                'p95': round(float(yvals.quantile(0.95)), 4),
            }
        result['short_run'] = short_run

        # 综合判定
        result['verdict'] = self._compute_verdict(global_df, local_df, yearly_dfs, best_metrics)

        # 保存 JSON
        json_path = os.path.join(output_dir, f'spp_result_{asset_name}_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP] 分析完成! 耗时 {elapsed:.1f}s")
            print(f"[SPP] JSON: {json_path}")
            print(f"[SPP] PNG:  {png_path}")
            print(f"{'='*60}")

            # 打印关键结论
            v = result['verdict']
            print(f"\n  过拟合风险: {v['overfit_risk']}")
            print(f"  全局Edge:   {v['global_edge']}")
            print(f"  参数鲁棒:   {v['parameter_robust']}")
            if v['summary']:
                print(f"  总结: {v['summary']}")

        return result
