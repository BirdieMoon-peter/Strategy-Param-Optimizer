# -*- coding: utf-8 -*-
"""
回测引擎模块
封装backtrader回测逻辑，使用pyfolio/empyrical风格计算性能指标
"""

import os
import sys
import warnings
from typing import Dict, Any, Optional, Type, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# 禁用警告
warnings.filterwarnings("ignore")

# 设置matplotlib后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backtrader as bt

# pyfolio/empyrical 风格的指标计算
try:
    import empyrical as ep
    HAS_EMPYRICAL = True
except ImportError:
    HAS_EMPYRICAL = False
    print("[警告] empyrical 未安装，将使用内置指标计算。建议运行: pip install empyrical")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BacktestConfig, DEFAULT_BACKTEST_CONFIG


class PyfolioMetrics:
    """
    使用 empyrical (pyfolio核心库) 计算投资组合性能指标
    提供与 pyfolio 一致的专业级指标计算
    """
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None, 
                 risk_free_rate: float = 0.0, period: str = 'daily'):
        """
        初始化指标计算器
        
        Args:
            returns: 策略日收益率序列 (pd.Series, index为日期)
            benchmark_returns: 基准收益率序列 (可选)
            risk_free_rate: 无风险利率 (年化)
            period: 收益率周期 ('daily', 'weekly', 'monthly')
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.period = period
        
        # 年化因子
        self.annualization_factor = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12
        }.get(period, 252)
    
    def sharpe_ratio(self) -> float:
        """计算夏普比率 (年化)"""
        if HAS_EMPYRICAL:
            return ep.sharpe_ratio(self.returns, risk_free=self.risk_free_rate, 
                                   annualization=self.annualization_factor)
        else:
            return self._sharpe_ratio_fallback()
    
    def _sharpe_ratio_fallback(self) -> float:
        """内置夏普比率计算"""
        if len(self.returns) < 2:
            return 0.0
        excess_returns = self.returns - self.risk_free_rate / self.annualization_factor
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(self.annualization_factor) * excess_returns.mean() / excess_returns.std()
    
    def sortino_ratio(self) -> float:
        """计算索提诺比率 (只考虑下行风险)"""
        if HAS_EMPYRICAL:
            try:
                result = ep.sortino_ratio(self.returns, annualization=self.annualization_factor)
                if result is None or np.isnan(result) or np.isinf(result):
                    return self._sortino_ratio_fallback()
                return result
            except Exception:
                return self._sortino_ratio_fallback()
        else:
            return self._sortino_ratio_fallback()
    
    def _sortino_ratio_fallback(self) -> float:
        """内置索提诺比率计算"""
        if len(self.returns) < 2:
            return 0.0
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) == 0:
            # 没有负收益，返回一个较大的正值表示优秀
            return 10.0 if self.returns.mean() > 0 else 0.0
        downside_std = downside_returns.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 10.0 if self.returns.mean() > 0 else 0.0
        return np.sqrt(self.annualization_factor) * self.returns.mean() / downside_std
    
    def calmar_ratio(self) -> float:
        """计算卡玛比率 (年化收益 / 最大回撤)"""
        if HAS_EMPYRICAL:
            return ep.calmar_ratio(self.returns, annualization=self.annualization_factor)
        else:
            return self._calmar_ratio_fallback()
    
    def _calmar_ratio_fallback(self) -> float:
        """内置卡玛比率计算"""
        annual_ret = self.annual_return()
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0 if annual_ret <= 0 else np.inf
        return annual_ret / abs(max_dd)
    
    def max_drawdown(self) -> float:
        """计算最大回撤"""
        if HAS_EMPYRICAL:
            return ep.max_drawdown(self.returns)
        else:
            return self._max_drawdown_fallback()
    
    def _max_drawdown_fallback(self) -> float:
        """内置最大回撤计算"""
        if len(self.returns) == 0:
            return 0.0
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def annual_return(self) -> float:
        """计算年化收益率"""
        if HAS_EMPYRICAL:
            return ep.annual_return(self.returns, annualization=self.annualization_factor)
        else:
            return self._annual_return_fallback()
    
    def _annual_return_fallback(self) -> float:
        """内置年化收益率计算"""
        if len(self.returns) == 0:
            return 0.0
        total_return = (1 + self.returns).prod() - 1
        n_periods = len(self.returns)
        years = n_periods / self.annualization_factor
        if years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1
    
    def total_return(self) -> float:
        """计算总收益率"""
        if len(self.returns) == 0:
            return 0.0
        return (1 + self.returns).prod() - 1
    
    def annual_volatility(self) -> float:
        """计算年化波动率"""
        if HAS_EMPYRICAL:
            return ep.annual_volatility(self.returns, annualization=self.annualization_factor)
        else:
            return self.returns.std() * np.sqrt(self.annualization_factor)
    
    def omega_ratio(self, required_return: float = 0.0) -> float:
        """计算欧米伽比率"""
        if HAS_EMPYRICAL:
            return ep.omega_ratio(self.returns, required_return=required_return,
                                  annualization=self.annualization_factor)
        else:
            return self._omega_ratio_fallback(required_return)
    
    def _omega_ratio_fallback(self, required_return: float = 0.0) -> float:
        """内置欧米伽比率计算"""
        returns_above = self.returns[self.returns > required_return] - required_return
        returns_below = required_return - self.returns[self.returns <= required_return]
        if returns_below.sum() == 0:
            return np.inf if returns_above.sum() > 0 else 0.0
        return returns_above.sum() / returns_below.sum()
    
    def tail_ratio(self) -> float:
        """计算尾部比率 (95%分位数收益 / 5%分位数亏损)"""
        if HAS_EMPYRICAL:
            return ep.tail_ratio(self.returns)
        else:
            return self._tail_ratio_fallback()
    
    def _tail_ratio_fallback(self) -> float:
        """内置尾部比率计算"""
        if len(self.returns) == 0:
            return 0.0
        top = self.returns.quantile(0.95)
        bottom = abs(self.returns.quantile(0.05))
        if bottom == 0:
            return np.inf if top > 0 else 0.0
        return top / bottom
    
    def value_at_risk(self, cutoff: float = 0.05) -> float:
        """计算风险价值 VaR"""
        if HAS_EMPYRICAL:
            return ep.value_at_risk(self.returns, cutoff=cutoff)
        else:
            return self.returns.quantile(cutoff)
    
    def alpha_beta(self) -> Tuple[float, float]:
        """计算 Alpha 和 Beta (需要基准收益率)"""
        if self.benchmark_returns is None:
            return 0.0, 1.0
        
        if HAS_EMPYRICAL:
            alpha = ep.alpha(self.returns, self.benchmark_returns, 
                            risk_free=self.risk_free_rate,
                            annualization=self.annualization_factor)
            beta = ep.beta(self.returns, self.benchmark_returns)
            return alpha, beta
        else:
            # 内置计算
            aligned_returns, aligned_benchmark = self.returns.align(self.benchmark_returns, join='inner')
            if len(aligned_returns) < 2:
                return 0.0, 1.0
            covariance = aligned_returns.cov(aligned_benchmark)
            benchmark_var = aligned_benchmark.var()
            if benchmark_var == 0:
                return 0.0, 1.0
            beta = covariance / benchmark_var
            alpha = aligned_returns.mean() - beta * aligned_benchmark.mean()
            # 年化 alpha
            alpha = alpha * self.annualization_factor
            return alpha, beta
    
    def information_ratio(self) -> float:
        """计算信息比率 (需要基准收益率)"""
        if self.benchmark_returns is None:
            return 0.0
        
        aligned_returns, aligned_benchmark = self.returns.align(self.benchmark_returns, join='inner')
        active_returns = aligned_returns - aligned_benchmark
        
        if active_returns.std() == 0:
            return 0.0
        
        return np.sqrt(self.annualization_factor) * active_returns.mean() / active_returns.std()
    
    def get_all_metrics(self) -> Dict[str, float]:
        """获取所有核心指标"""
        metrics = {
            'sharpe_ratio': self._safe_metric(self.sharpe_ratio),
            'sortino_ratio': self._safe_metric(self.sortino_ratio),
            'calmar_ratio': self._safe_metric(self.calmar_ratio),
            'max_drawdown': self._safe_metric(self.max_drawdown) * 100,  # 转为百分比
            'annual_return': self._safe_metric(self.annual_return) * 100,  # 转为百分比
            'total_return': self._safe_metric(self.total_return) * 100,  # 转为百分比
            'annual_volatility': self._safe_metric(self.annual_volatility) * 100,  # 转为百分比
            'omega_ratio': self._safe_metric(self.omega_ratio),
            'tail_ratio': self._safe_metric(self.tail_ratio),
            'value_at_risk_5pct': self._safe_metric(lambda: self.value_at_risk(0.05)) * 100,
        }
        
        # 添加 alpha/beta (如果有基准)
        if self.benchmark_returns is not None:
            alpha, beta = self.alpha_beta()
            metrics['alpha'] = self._safe_value(alpha) * 100  # 转为百分比
            metrics['beta'] = self._safe_value(beta)
            metrics['information_ratio'] = self._safe_metric(self.information_ratio)
        
        return metrics
    
    def _safe_metric(self, func) -> float:
        """安全执行指标计算"""
        try:
            value = func()
            return self._safe_value(value)
        except Exception:
            return 0.0
    
    def _safe_value(self, value) -> float:
        """确保返回有效数值"""
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)


@dataclass
class BacktestResult:
    """
    回测结果 (使用 pyfolio/empyrical 风格的指标)
    """
    # 核心指标
    total_return: float        # 总收益率(%)
    annual_return: float       # 年化收益率(%)
    max_drawdown: float        # 最大回撤(%)
    sharpe_ratio: float        # 夏普比率
    final_value: float         # 最终资产
    trades_count: int          # 交易次数
    win_rate: float            # 胜率
    params: Dict[str, Any]     # 使用的参数
    
    # pyfolio 风格的额外指标
    sortino_ratio: float = 0.0       # 索提诺比率 (下行风险调整收益)
    calmar_ratio: float = 0.0        # 卡玛比率 (年化收益/最大回撤)
    annual_volatility: float = 0.0   # 年化波动率(%)
    omega_ratio: float = 0.0         # 欧米伽比率
    tail_ratio: float = 0.0          # 尾部比率
    value_at_risk: float = 0.0       # 5% VaR(%)
    
    # 逐年指标
    yearly_returns: Dict[int, float] = field(default_factory=dict)  # 每年的收益率(%)
    yearly_drawdowns: Dict[int, float] = field(default_factory=dict)  # 每年的最大回撤(%)
    yearly_sharpe: Dict[int, float] = field(default_factory=dict)  # 每年的夏普比率
    
    # 日收益率序列 (用于后续分析)
    daily_returns: pd.Series = field(default=None, repr=False)


class BacktestEngine:
    """
    回测引擎
    封装backtrader，提供简洁的API进行策略回测和性能评估
    """
    
    def __init__(
        self, 
        config: BacktestConfig = None,
        data: pd.DataFrame = None,
        strategy_class: Type[bt.Strategy] = None,
        initial_cash: float = None,
        commission: float = None
    ):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
            data: DataFrame格式的数据（新接口）
            strategy_class: 策略类（新接口）
            initial_cash: 初始资金（新接口）
            commission: 手续费率（新接口）
        """
        self.config = config or DEFAULT_BACKTEST_CONFIG
        self.data_cache = {}
        
        # 新接口支持
        self.data_df = data
        self.strategy_class = strategy_class
        if initial_cash is not None:
            self.config.cash = initial_cash
        if commission is not None:
            self.config.commission = commission
    
    def load_data(
        self, 
        asset_name: str, 
        data_dir: str = None,
        start_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        加载本地CSV数据
        
        Args:
            asset_name: 资产名称（对应CSV文件名）
            data_dir: 数据目录
            start_date: 起始日期
            
        Returns:
            DataFrame格式的行情数据
        """
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data"
            )
        
        if start_date is None:
            start_date = self.config.start_date
        
        cache_key = f"{asset_name}_{start_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        csv_path = os.path.join(data_dir, f"{asset_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"[错误] 未找到数据文件: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(
                csv_path,
                parse_dates=["date"],
                index_col="date"
            )
            
            # 列名标准化
            df.columns = [col.lower() for col in df.columns]
            required_cols = ["open", "high", "low", "close", "volume"]
            
            for col in required_cols:
                if col not in df.columns:
                    print(f"[错误] 数据文件缺少必要列: {col}")
                    return None
            
            # 过滤日期
            df = df[df.index >= pd.to_datetime(start_date)]
            df = df.dropna().sort_index()
            
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"[错误] 加载数据失败: {e}")
            return None
    
    def run_backtest(
        self,
        strategy_class: Type[bt.Strategy] = None,
        data: pd.DataFrame = None,
        params: Dict[str, Any] = None,
        asset_name: str = "ASSET",
        calculate_yearly: bool = True
    ) -> Optional[BacktestResult]:
        """
        运行单次回测
        
        Args:
            strategy_class: 策略类（可选，如果未提供则使用初始化时的策略）
            data: 行情数据（可选，如果未提供则使用初始化时的数据）
            params: 策略参数
            asset_name: 资产名称
            calculate_yearly: 是否计算每年的指标
            
        Returns:
            BacktestResult对象
        """
        # 使用提供的参数或初始化时的参数
        strategy_class = strategy_class or self.strategy_class
        data = data if data is not None else self.data_df
        
        if strategy_class is None:
            raise ValueError("必须提供 strategy_class 参数或在初始化时指定")
        if data is None:
            raise ValueError("必须提供 data 参数或在初始化时指定")
        
        params = params or {}
        
        try:
            # 初始化Cerebro
            cerebro = bt.Cerebro(stdstats=True)
            cerebro.broker.setcash(self.config.initial_cash)
            cerebro.broker.setcommission(commission=self.config.commission)
            
            # 准备数据：确保 datetime 是索引
            data_copy = data.copy()
            if 'datetime' in data_copy.columns:
                data_copy['datetime'] = pd.to_datetime(data_copy['datetime'])
                data_copy = data_copy.set_index('datetime')
            elif 'date' in data_copy.columns:
                data_copy['date'] = pd.to_datetime(data_copy['date'])
                data_copy = data_copy.set_index('date')
            
            # 确保列名小写
            data_copy.columns = [col.lower() for col in data_copy.columns]
            
            # 添加数据
            bt_data = bt.feeds.PandasData(dataname=data_copy, name=asset_name)
            cerebro.adddata(bt_data)
            
            # 添加策略
            cerebro.addstrategy(strategy_class, **params)
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                              riskfreerate=0.0, annualize=True)
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
            
            # 运行回测
            results = cerebro.run()
            strat = results[0]
            
            # 提取 backtrader 的时间收益率，用于 pyfolio 风格的计算
            timereturn = strat.analyzers.timereturn.get_analysis()
            trades = strat.analyzers.trades.get_analysis()
            final_value = cerebro.broker.getvalue()
            
            # 构建日收益率序列 (pyfolio 风格)
            if timereturn and isinstance(timereturn, dict) and len(timereturn) > 0:
                daily_returns = pd.Series(timereturn)
                daily_returns.index = pd.to_datetime(daily_returns.index)
                daily_returns = daily_returns.sort_index()
                # 过滤无效值
                daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
            else:
                daily_returns = pd.Series(dtype=float)
            
            # 使用 PyfolioMetrics 计算所有指标
            pyfolio_metrics = PyfolioMetrics(daily_returns, period='daily')
            metrics = pyfolio_metrics.get_all_metrics()
            
            # 交易统计
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 计算每年的指标
            yearly_returns = {}
            yearly_drawdowns = {}
            yearly_sharpe = {}
            yearly_sharpe = {}
            
            if calculate_yearly:
                yearly_stats = self._calculate_yearly_metrics(data, strategy_class, params, asset_name)
                yearly_returns = yearly_stats.get('returns', {})
                yearly_drawdowns = yearly_stats.get('drawdowns', {})
                yearly_sharpe = yearly_stats.get('sharpe', {})
            
            return BacktestResult(
                # 核心指标 (使用 pyfolio/empyrical 计算)
                total_return=metrics.get('total_return', 0),
                annual_return=metrics.get('annual_return', 0),
                max_drawdown=abs(metrics.get('max_drawdown', 0)),  # 转为正值
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                final_value=final_value,
                trades_count=total_trades,
                win_rate=win_rate,
                params=params,
                # pyfolio 风格的额外指标
                sortino_ratio=metrics.get('sortino_ratio', 0),
                calmar_ratio=metrics.get('calmar_ratio', 0),
                annual_volatility=metrics.get('annual_volatility', 0),
                omega_ratio=metrics.get('omega_ratio', 0),
                tail_ratio=metrics.get('tail_ratio', 0),
                value_at_risk=abs(metrics.get('value_at_risk_5pct', 0)),
                # 逐年指标
                yearly_returns=yearly_returns,
                yearly_drawdowns=yearly_drawdowns,
                yearly_sharpe=yearly_sharpe,
                # 日收益率序列
                daily_returns=daily_returns if len(daily_returns) > 0 else None
            )
            
        except Exception as e:
            import traceback
            print(f"[错误] 回测执行失败: {e}")
            traceback.print_exc()
            return None
    
    def _calculate_yearly_metrics(
        self,
        data: pd.DataFrame,
        strategy_class: Type[bt.Strategy],
        params: Dict[str, Any],
        asset_name: str
    ) -> Dict[str, Dict[int, float]]:
        """
        计算每年的性能指标
        
        Args:
            data: 行情数据
            strategy_class: 策略类
            params: 策略参数
            asset_name: 资产名称
            
        Returns:
            包含每年收益率、回撤和夏普比率的字典
        """
        yearly_returns = {}
        yearly_drawdowns = {}
        yearly_sharpe = {}
        
        # 确保数据有日期索引
        data_copy = data.copy()
        if 'datetime' in data_copy.columns:
            data_copy['datetime'] = pd.to_datetime(data_copy['datetime'])
            data_copy = data_copy.set_index('datetime')
        elif 'date' in data_copy.columns:
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            data_copy = data_copy.set_index('date')
        elif not isinstance(data_copy.index, pd.DatetimeIndex):
            # 如果没有日期列且索引不是日期，返回空字典
            return {'returns': {}, 'drawdowns': {}, 'sharpe': {}}
        
        # 获取所有年份
        years = sorted(data_copy.index.year.unique())
        
        for year in years:
            # 筛选该年的数据
            year_data = data_copy[data_copy.index.year == year]
            
            # 如果数据太少，跳过
            if len(year_data) < 20:
                continue
            
            try:
                # 为该年运行回测
                cerebro = bt.Cerebro(stdstats=True)
                cerebro.broker.setcash(self.config.initial_cash)
                cerebro.broker.setcommission(commission=self.config.commission)
                
                bt_data = bt.feeds.PandasData(dataname=year_data, name=asset_name)
                cerebro.adddata(bt_data)
                cerebro.addstrategy(strategy_class, **params)
                
                # 添加时间收益率分析器
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
                
                # 运行回测
                results = cerebro.run()
                strat = results[0]
                
                # 提取时间收益率
                timereturn = strat.analyzers.timereturn.get_analysis()
                
                # 构建日收益率序列并使用 PyfolioMetrics 计算
                if timereturn and isinstance(timereturn, dict) and len(timereturn) > 0:
                    daily_returns = pd.Series(timereturn)
                    daily_returns.index = pd.to_datetime(daily_returns.index)
                    daily_returns = daily_returns.sort_index()
                    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # 使用 PyfolioMetrics 计算年度指标
                    year_metrics = PyfolioMetrics(daily_returns, period='daily')
                    
                    year_return = year_metrics.total_return() * 100
                    year_drawdown = abs(year_metrics.max_drawdown()) * 100
                    year_sharpe_ratio = year_metrics._safe_value(year_metrics.sharpe_ratio())
                else:
                    year_return = 0
                    year_drawdown = 0
                    year_sharpe_ratio = 0
                
                yearly_returns[year] = year_return
                yearly_drawdowns[year] = year_drawdown
                yearly_sharpe[year] = year_sharpe_ratio
                
            except Exception as e:
                # 如果该年回测失败，记录空值
                yearly_returns[year] = 0
                yearly_drawdowns[year] = 0
                yearly_sharpe[year] = 0
        
        return {
            'returns': yearly_returns,
            'drawdowns': yearly_drawdowns,
            'sharpe': yearly_sharpe
        }
    
    def evaluate_objective(
        self,
        result: BacktestResult,
        objective: str
    ) -> float:
        """
        根据目标获取评估值 (支持 pyfolio 风格的多种目标)
        
        Args:
            result: 回测结果
            objective: 优化目标
            
        Returns:
            目标值（已处理方向）
        """
        if objective == "sharpe_ratio":
            return result.sharpe_ratio
        elif objective == "sortino_ratio":
            return result.sortino_ratio
        elif objective == "calmar_ratio":
            return result.calmar_ratio
        elif objective == "annual_return":
            return result.annual_return
        elif objective == "total_return":
            return result.total_return
        elif objective == "max_drawdown":
            # 最大回撤越小越好，但Optuna默认最大化
            # 返回负值让Optuna最大化时实际最小化回撤
            return -result.max_drawdown
        elif objective == "omega_ratio":
            return result.omega_ratio
        else:
            return result.sharpe_ratio
    
    def get_result_summary(self, result: BacktestResult) -> Dict[str, Any]:
        """
        获取回测结果摘要 (包含 pyfolio 风格的指标)
        
        Args:
            result: 回测结果
            
        Returns:
            结果摘要字典
        """
        return {
            # 核心指标
            "总收益率": f"{result.total_return:.2f}%",
            "年化收益率": f"{result.annual_return:.2f}%",
            "最大回撤": f"{result.max_drawdown:.2f}%",
            "夏普比率": f"{result.sharpe_ratio:.4f}",
            # pyfolio 风格指标
            "索提诺比率": f"{result.sortino_ratio:.4f}",
            "卡玛比率": f"{result.calmar_ratio:.4f}",
            "年化波动率": f"{result.annual_volatility:.2f}%",
            "欧米伽比率": f"{result.omega_ratio:.4f}",
            "尾部比率": f"{result.tail_ratio:.4f}",
            "VaR(5%)": f"{result.value_at_risk:.2f}%",
            # 交易统计
            "最终资产": f"${result.final_value:,.2f}",
            "交易次数": result.trades_count,
            "胜率": f"{result.win_rate:.1f}%"
        }
    
    def run_multi_asset_backtest(
        self,
        strategy_class: Type[bt.Strategy],
        assets: list,
        params: Dict[str, Any] = None,
        data_dir: str = None
    ) -> Dict[str, BacktestResult]:
        """
        在多个资产上运行回测
        
        Args:
            strategy_class: 策略类
            assets: 资产列表
            params: 策略参数
            data_dir: 数据目录
            
        Returns:
            各资产的回测结果
        """
        results = {}
        
        for asset in assets:
            data = self.load_data(asset, data_dir)
            if data is not None and len(data) > 50:  # 确保有足够数据
                result = self.run_backtest(strategy_class, data, params, asset)
                if result:
                    results[asset] = result
        
        return results
    
    def calculate_aggregate_metrics(
        self,
        results: Dict[str, BacktestResult]
    ) -> Dict[str, float]:
        """
        计算聚合指标（跨资产平均）
        
        Args:
            results: 各资产的回测结果
            
        Returns:
            聚合指标
        """
        if not results:
            return {
                "avg_sharpe": 0,
                "avg_annual_return": 0,
                "avg_max_drawdown": 0,
                "total_trades": 0
            }
        
        sharpes = [r.sharpe_ratio for r in results.values()]
        returns = [r.annual_return for r in results.values()]
        drawdowns = [r.max_drawdown for r in results.values()]
        trades = [r.trades_count for r in results.values()]
        
        return {
            "avg_sharpe": np.mean(sharpes),
            "avg_annual_return": np.mean(returns),
            "avg_max_drawdown": np.mean(drawdowns),
            "total_trades": sum(trades)
        }


class ObjectiveEvaluator:
    """
    目标评估器
    为Optuna提供目标函数
    """
    
    def __init__(
        self,
        engine: BacktestEngine,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        objective: str = "sharpe_ratio"
    ):
        """
        初始化目标评估器
        
        Args:
            engine: 回测引擎
            strategy_class: 策略类
            data: 行情数据
            objective: 优化目标
        """
        self.engine = engine
        self.strategy_class = strategy_class
        self.data = data
        self.objective = objective
        self.history = []
    
    def __call__(self, params: Dict[str, Any]) -> Tuple[float, BacktestResult]:
        """
        评估参数
        
        Args:
            params: 策略参数
            
        Returns:
            (目标值, 回测结果)
        """
        result = self.engine.run_backtest(
            self.strategy_class,
            self.data,
            params
        )
        
        if result is None:
            return float('-inf'), None
        
        value = self.engine.evaluate_objective(result, self.objective)
        
        # 记录历史
        self.history.append({
            "params": params.copy(),
            "value": value,
            "result": result
        })
        
        return value, result
    
    def get_history(self) -> list:
        """获取评估历史"""
        return self.history
    
    def get_best(self) -> Optional[Dict]:
        """获取最佳结果"""
        if not self.history:
            return None
        
        if self.objective == "max_drawdown":
            # 回撤是负值，所以取最大
            best = max(self.history, key=lambda x: x['value'])
        else:
            best = max(self.history, key=lambda x: x['value'])
        
        return best


if __name__ == "__main__":
    # 测试代码
    from src.Aberration import AberrationStrategy
    
    engine = BacktestEngine()
    
    # 加载数据
    data = engine.load_data("BTC")
    
    if data is not None:
        print(f"加载数据: {len(data)} 条记录")
        
        # 运行回测
        result = engine.run_backtest(
            AberrationStrategy,
            data,
            {"period": 35, "std_dev_upper": 2.0, "std_dev_lower": 2.0}
        )
        
        if result:
            print("\n回测结果:")
            for k, v in engine.get_result_summary(result).items():
                print(f"  {k}: {v}")
