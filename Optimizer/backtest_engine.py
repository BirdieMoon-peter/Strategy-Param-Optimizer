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

from config import BacktestConfig, DEFAULT_BACKTEST_CONFIG, get_annualization_factor
from futures_config import BrokerConfig, create_commission_info


def _is_intraday_frequency(data_frequency: str) -> bool:
    """判断是否是日内数据频率"""
    if not data_frequency:
        return False
    freq_lower = data_frequency.lower().strip()
    intraday_patterns = ['1m', '5m', '15m', '30m', 'min', 'minute', 'hourly', '1h', 'hour']
    return any(pattern in freq_lower for pattern in intraday_patterns)


def _resample_to_daily(returns: pd.Series) -> pd.Series:
    """
    将高频收益率重采样为日收益率
    
    对于分钟/小时数据，将每日的收益率累积为单日收益
    公式：日收益 = (1+r1)(1+r2)...(1+rn) - 1
    
    Args:
        returns: 高频收益率序列 (index 必须是 DatetimeIndex)
        
    Returns:
        日收益率序列
    """
    if len(returns) == 0:
        return returns
    
    if not isinstance(returns.index, pd.DatetimeIndex):
        return returns
    
    # 按日期分组，计算每日累积收益
    daily_returns = returns.groupby(returns.index.date).apply(
        lambda x: (1 + x).prod() - 1
    )
    daily_returns.index = pd.to_datetime(daily_returns.index)
    return daily_returns


class PyfolioMetrics:
    """
    使用 empyrical (pyfolio核心库) 计算投资组合性能指标
    提供与 pyfolio 一致的专业级指标计算
    支持多种数据频率（日线、分钟线等）
    
    注意：对于日内数据，推荐在 backtrader 层面使用 TimeReturn(timeframe=bt.TimeFrame.Days) 
    获取日收益率，然后以 'daily' 频率传入此类。如果传入高频收益率，会自动重采样为日收益率。
    """
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None, 
                 risk_free_rate: float = 0.0, period: str = 'daily',
                 data_frequency: str = None):
        """
        初始化指标计算器
        
        Args:
            returns: 策略收益率序列 (pd.Series, index为日期/时间)
                    对于日内数据，推荐传入已按日聚合的日收益率
            benchmark_returns: 基准收益率序列 (可选)
            risk_free_rate: 无风险利率 (年化)
            period: 收益率周期 ('daily', 'weekly', 'monthly') - 已弃用，建议使用 data_frequency
            data_frequency: 数据频率（如 'daily', '1m', '5m', '15m', 'hourly' 等）
                           如果传入高频数据会自动重采样为日收益率
        """
        self.raw_returns = returns.dropna()  # 保留原始高频收益率
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.period = period
        self.data_frequency = data_frequency
        self.is_intraday = _is_intraday_frequency(data_frequency)
        
        # 对于日内数据，重采样为日收益率用于年化指标计算
        if self.is_intraday and len(self.raw_returns) > 0:
            self.returns = _resample_to_daily(self.raw_returns)
            self.annualization_factor = 252  # 日线年化因子
        else:
            self.returns = self.raw_returns
            # 优先使用 data_frequency 计算年化因子
            if data_frequency:
                self.annualization_factor = get_annualization_factor(data_frequency)
            else:
                # 兼容旧的 period 参数
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
        """
        计算最大回撤
        注意：使用原始高频数据以获得更精确的回撤计算
        """
        # 使用原始数据计算最大回撤（更精确）
        returns_for_dd = self.raw_returns if hasattr(self, 'raw_returns') and len(self.raw_returns) > 0 else self.returns
        if HAS_EMPYRICAL:
            return ep.max_drawdown(returns_for_dd)
        else:
            return self._max_drawdown_fallback(returns_for_dd)
    
    def _max_drawdown_fallback(self, returns: pd.Series = None) -> float:
        """内置最大回撤计算"""
        if returns is None:
            returns = self.returns
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def annual_return(self) -> float:
        """
        计算年化收益率
        对于日内数据，使用重采样后的日收益率和日线年化因子
        """
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
        """
        计算总收益率
        注意：使用原始高频数据以获得精确的总收益
        """
        # 使用原始数据计算总收益率（更精确）
        returns_for_total = self.raw_returns if hasattr(self, 'raw_returns') and len(self.raw_returns) > 0 else self.returns
        if len(returns_for_total) == 0:
            return 0.0
        return (1 + returns_for_total).prod() - 1
    
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
    
    # 交易日志 (用于详细分析)
    trade_log: list = field(default_factory=list, repr=False)
    
    # 盈亏比
    profit_factor: float = 0.0


def calculate_metrics_from_trade_log(trade_log: list, initial_cash: float = 100000.0, 
                                     final_value: float = None) -> Dict[str, float]:
    """
    从策略的交易日志计算回测指标（用于分钟级策略的精确指标计算）
    
    Args:
        trade_log: 策略记录的交易日志 (list of dict)
        initial_cash: 初始资金
        final_value: 最终资金（可选，如不提供则从trade_log计算）
        
    Returns:
        指标字典
    """
    if not trade_log:
        return {
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'trades_count': 0,
            'avg_return': 0,
            'max_consecutive_losses': 0
        }
    
    df = pd.DataFrame(trade_log)
    
    # 基础统计
    total_trades = len(df)
    profitable_trades = len(df[df['pnl'] > 0])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 平均收益率
    avg_return = df['return_pct'].mean() if total_trades > 0 else 0
    
    # 盈亏比
    winning_trades = df[df['pnl'] > 0]['pnl']
    losing_trades = df[df['pnl'] < 0]['pnl']
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
    profit_factor = (avg_win / avg_loss) if avg_loss > 0 else (10.0 if avg_win > 0 else 0)
    
    # 最大连续亏损
    consecutive_losses = 0
    max_consecutive_losses = 0
    for pnl in df['pnl']:
        if pnl < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    # 总收益率
    if final_value is not None:
        total_return = ((final_value - initial_cash) / initial_cash) * 100
    elif 'final_portfolio_value' in df.columns:
        final_value = df['final_portfolio_value'].iloc[-1]
        total_return = ((final_value - initial_cash) / initial_cash) * 100
    else:
        total_return = df['pnl'].sum() / initial_cash * 100
        final_value = initial_cash + df['pnl'].sum()
    
    # 年化收益率 - 基于交易时间跨度
    start_date = pd.to_datetime(df['entry_datetime'].iloc[0])
    end_date = pd.to_datetime(df['exit_datetime'].iloc[-1])
    years = (end_date - start_date).days / 365.25
    if years > 0:
        annual_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100
    else:
        annual_return = 0
    
    # 计算日收益率用于夏普比率和回撤
    df['date'] = pd.to_datetime(df['entry_datetime']).dt.date
    daily_pnl = df.groupby('date')['pnl'].sum()
    
    # 构建每日资产曲线
    daily_values = [initial_cash]
    cumulative = initial_cash
    for pnl in daily_pnl:
        cumulative += pnl
        daily_values.append(cumulative)
    
    daily_values = pd.Series(daily_values)
    daily_returns = daily_values.pct_change().dropna()
    
    # 夏普比率
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # 最大回撤
    running_max = daily_values.cummax()
    drawdown = (daily_values - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades_count': total_trades,
        'avg_return': avg_return,
        'max_consecutive_losses': max_consecutive_losses
    }


class BacktestEngine:
    """
    回测引擎
    封装backtrader，提供简洁的API进行策略回测和性能评估
    支持多种数据频率（日线、分钟线等）
    支持自定义数据类、手续费类和从trade_log计算指标
    """
    
    def __init__(
        self, 
        config: BacktestConfig = None,
        data: pd.DataFrame = None,
        strategy_class: Type[bt.Strategy] = None,
        initial_cash: float = None,
        commission: float = None,
        data_frequency: str = None,
        custom_data_class: Type = None,
        custom_commission_class: Type = None,
        strategy_module: Any = None,
        use_trade_log_metrics: bool = False,
        broker_config: BrokerConfig = None
    ):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
            data: DataFrame格式的数据（新接口）
            strategy_class: 策略类（新接口）
            initial_cash: 初始资金（新接口）
            commission: 手续费率（新接口）
            data_frequency: 数据频率（如 'daily', '1m', '5m' 等），为None时自动检测
            custom_data_class: 自定义数据类（继承自 bt.feeds.PandasData）
            custom_commission_class: 自定义手续费类（继承自 bt.CommInfoBase）
            strategy_module: 策略模块（用于查找自定义类）
            use_trade_log_metrics: 是否优先使用策略的 trade_log 计算指标
        """
        self.config = config or DEFAULT_BACKTEST_CONFIG
        self.data_cache = {}
        
        # 新接口支持
        self.data_df = data
        self.strategy_class = strategy_class
        if initial_cash is not None:
            self.config.cash = initial_cash
            self.config.initial_cash = initial_cash
        if commission is not None:
            self.config.commission = commission
        
        # 自定义组件支持
        self.custom_data_class = custom_data_class
        self.custom_commission_class = custom_commission_class
        self.strategy_module = strategy_module
        self.use_trade_log_metrics = use_trade_log_metrics

        # 期货经纪商配置
        self.broker_config = broker_config
        self.fixed_strategy_params = {}
        if broker_config and broker_config.is_futures:
            self.fixed_strategy_params = {
                'mult': broker_config.mult,
                'margin': broker_config.margin,
            }
        
        # 设置数据频率
        if data_frequency:
            self.config.data_frequency = data_frequency
        elif data is not None:
            # 自动检测数据频率
            self.config.data_frequency = self._detect_data_frequency(data)
    
    def _detect_data_frequency(self, data: pd.DataFrame) -> str:
        """
        自动检测数据频率
        
        Args:
            data: DataFrame格式的数据
            
        Returns:
            检测到的数据频率（如 'daily', '1m', '5m' 等）
        """
        try:
            # 处理多数据源
            if isinstance(data, (list, tuple)):
                data = data[0] if data else None
            if isinstance(data, dict):
                data = list(data.values())[0] if data else None
            
            if data is None or len(data) < 2:
                return 'daily'
            
            # 获取时间索引
            df = data.copy()
            if 'datetime' in df.columns:
                df['_dt'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['_dt'] = pd.to_datetime(df['date'])
            elif 'time_key' in df.columns:
                df['_dt'] = pd.to_datetime(df['time_key'])
            elif isinstance(df.index, pd.DatetimeIndex):
                df['_dt'] = df.index
            else:
                return 'daily'
            
            # 计算相邻时间差的众数
            df = df.sort_values('_dt')
            time_diffs = df['_dt'].diff().dropna()
            
            if len(time_diffs) == 0:
                return 'daily'
            
            # 获取最常见的时间差
            median_diff = time_diffs.median()
            total_seconds = median_diff.total_seconds()
            
            # 根据时间差判断频率
            if total_seconds <= 60:  # <= 1分钟
                return '1m'
            elif total_seconds <= 300:  # <= 5分钟
                return '5m'
            elif total_seconds <= 900:  # <= 15分钟
                return '15m'
            elif total_seconds <= 1800:  # <= 30分钟
                return '30m'
            elif total_seconds <= 3600:  # <= 1小时
                return 'hourly'
            elif total_seconds <= 86400:  # <= 1天
                return 'daily'
            elif total_seconds <= 604800:  # <= 1周
                return 'weekly'
            else:
                return 'monthly'
                
        except Exception as e:
            print(f"[警告] 数据频率检测失败: {e}，使用默认日线")
            return 'daily'
    
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
        calculate_yearly: bool = True,
        data_names: Optional[list] = None
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
            
            # 设置手续费：优先使用期货配置 > 自定义手续费类 > 默认百分比
            if self.broker_config is not None and self.broker_config.is_futures:
                comminfo = create_commission_info(self.broker_config)
                cerebro.broker.addcommissioninfo(comminfo)
            elif self.custom_commission_class is not None:
                cerebro.broker.addcommissioninfo(self.custom_commission_class())
            else:
                cerebro.broker.setcommission(commission=self.config.commission)
            
            def _prepare_df(df: pd.DataFrame, use_custom_class: bool = False) -> pd.DataFrame:
                data_copy = df.copy()
                
                # 对于自定义数据类，保留完整的列结构，只设置索引
                if use_custom_class:
                    if 'time_key' in data_copy.columns:
                        data_copy['Datetime'] = pd.to_datetime(data_copy['time_key'])
                        data_copy = data_copy.set_index('Datetime')
                    elif 'datetime' in data_copy.columns:
                        data_copy['Datetime'] = pd.to_datetime(data_copy['datetime'])
                        data_copy = data_copy.set_index('Datetime')
                    elif 'date' in data_copy.columns:
                        data_copy['Datetime'] = pd.to_datetime(data_copy['date'])
                        data_copy = data_copy.set_index('Datetime')
                    # 不转换列名为小写，保留原始结构供自定义数据类使用
                    return data_copy
                
                # 标准数据类处理
                if 'datetime' in data_copy.columns:
                    data_copy['datetime'] = pd.to_datetime(data_copy['datetime'])
                    data_copy = data_copy.set_index('datetime')
                elif 'date' in data_copy.columns:
                    data_copy['date'] = pd.to_datetime(data_copy['date'])
                    data_copy = data_copy.set_index('date')
                elif 'time_key' in data_copy.columns:
                    data_copy['time_key'] = pd.to_datetime(data_copy['time_key'])
                    data_copy = data_copy.set_index('time_key')
                
                # 确保列名小写
                data_copy.columns = [col.lower() for col in data_copy.columns]
                return data_copy
            
            # 选择数据类：优先使用自定义数据类
            DataClass = self.custom_data_class if self.custom_data_class is not None else bt.feeds.PandasData
            use_custom = self.custom_data_class is not None
            
            # 添加数据（支持单数据/多数据源）
            if isinstance(data, dict):
                for name, df in data.items():
                    prepared = _prepare_df(df, use_custom)
                    bt_data = DataClass(dataname=prepared, name=name)
                    cerebro.adddata(bt_data)
            elif isinstance(data, (list, tuple)):
                # 确定数据源名称
                if data_names and len(data_names) == len(data):
                    names = data_names
                elif isinstance(asset_name, (list, tuple)) and len(asset_name) == len(data):
                    names = list(asset_name)
                else:
                    names = [f"ASSET{i+1}" for i in range(len(data))]
                
                for df, name in zip(data, names):
                    prepared = _prepare_df(df, use_custom)
                    bt_data = DataClass(dataname=prepared, name=name)
                    cerebro.adddata(bt_data)
            else:
                prepared = _prepare_df(data, use_custom)
                bt_data = DataClass(dataname=prepared, name=asset_name)
                cerebro.adddata(bt_data)
            
            # 添加策略（合并期货固定参数）
            merged_params = dict(self.fixed_strategy_params, **params) if self.fixed_strategy_params else params
            cerebro.addstrategy(strategy_class, **merged_params)
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                              riskfreerate=0.0, annualize=True)
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # 对于日内数据，使用 Daily timeframe 获取日收益率
            # 这样后续计算年化指标时可以直接使用日线年化因子(252)
            is_intraday = _is_intraday_frequency(self.config.data_frequency)
            if is_intraday:
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn',
                                   timeframe=bt.TimeFrame.Days)
            else:
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
            
            # 运行回测
            results = cerebro.run()
            strat = results[0]
            
            # 提取 backtrader 的时间收益率，用于 pyfolio 风格的计算
            timereturn = strat.analyzers.timereturn.get_analysis()
            trades = strat.analyzers.trades.get_analysis()
            final_value = cerebro.broker.getvalue()
            
            # 检查策略是否有 trade_log 属性，用于更精确的指标计算
            strategy_trade_log = getattr(strat, 'trade_log', None)
            has_trade_log = strategy_trade_log is not None and len(strategy_trade_log) > 0
            
            # 如果启用了 trade_log 指标计算且策略有 trade_log
            if self.use_trade_log_metrics and has_trade_log:
                # 使用策略自己记录的交易日志计算指标
                trade_log_metrics = calculate_metrics_from_trade_log(
                    strategy_trade_log, 
                    initial_cash=self.config.initial_cash,
                    final_value=final_value
                )
                
                # 计算每年的指标（简化版，只取主要指标）
                yearly_returns = {}
                yearly_drawdowns = {}
                yearly_sharpe = {}
                
                return BacktestResult(
                    # 核心指标 (使用 trade_log 计算)
                    total_return=trade_log_metrics.get('total_return', 0),
                    annual_return=trade_log_metrics.get('annual_return', 0),
                    max_drawdown=trade_log_metrics.get('max_drawdown', 0),
                    sharpe_ratio=trade_log_metrics.get('sharpe_ratio', 0),
                    final_value=final_value,
                    trades_count=trade_log_metrics.get('trades_count', 0),
                    win_rate=trade_log_metrics.get('win_rate', 0),
                    params=params,
                    # 额外指标
                    profit_factor=trade_log_metrics.get('profit_factor', 0),
                    sortino_ratio=0,  # trade_log 模式下暂不计算
                    calmar_ratio=trade_log_metrics.get('annual_return', 0) / max(trade_log_metrics.get('max_drawdown', 1), 0.01),
                    annual_volatility=0,
                    omega_ratio=0,
                    tail_ratio=0,
                    value_at_risk=0,
                    # 逐年指标
                    yearly_returns=yearly_returns,
                    yearly_drawdowns=yearly_drawdowns,
                    yearly_sharpe=yearly_sharpe,
                    # 收益率序列
                    daily_returns=None,
                    # 交易日志
                    trade_log=strategy_trade_log
                )
            
            # 构建收益率序列 (pyfolio 风格)
            if timereturn and isinstance(timereturn, dict) and len(timereturn) > 0:
                returns_series = pd.Series(timereturn)
                returns_series.index = pd.to_datetime(returns_series.index)
                returns_series = returns_series.sort_index()
                # 过滤无效值
                returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
            else:
                returns_series = pd.Series(dtype=float)
            
            # 使用 PyfolioMetrics 计算所有指标
            # 对于日内数据，TimeReturn 已经按日返回收益率，所以使用 'daily' 频率
            is_intraday = _is_intraday_frequency(self.config.data_frequency)
            metrics_frequency = 'daily' if is_intraday else self.config.data_frequency
            pyfolio_metrics = PyfolioMetrics(
                returns_series, 
                data_frequency=metrics_frequency
            )
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
                # 收益率序列（对于分钟数据也保持此字段名以保证兼容性）
                daily_returns=returns_series if len(returns_series) > 0 else None,
                # 交易日志
                trade_log=strategy_trade_log if has_trade_log else []
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
            data: 行情数据（可以是单个DataFrame或多个DataFrame的列表/字典）
            strategy_class: 策略类
            params: 策略参数
            asset_name: 资产名称
            
        Returns:
            包含每年收益率、回撤和夏普比率的字典
        """
        yearly_returns = {}
        yearly_drawdowns = {}
        yearly_sharpe = {}
        
        # 处理多数据源 - 获取第一个数据源用于确定年份范围
        if isinstance(data, dict):
            first_data = list(data.values())[0] if data else None
            data_dict = data
            is_multi = True
        elif isinstance(data, (list, tuple)):
            first_data = data[0] if data else None
            # 转换为字典格式
            if isinstance(asset_name, (list, tuple)):
                data_dict = {name: df for name, df in zip(asset_name, data)}
            else:
                data_dict = {f"ASSET{i+1}": df for i, df in enumerate(data)}
            is_multi = True
        else:
            first_data = data
            data_dict = {asset_name: data}
            is_multi = False
        
        if first_data is None:
            return {'returns': {}, 'drawdowns': {}, 'sharpe': {}}
        
        def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
            """准备数据帧，设置日期索引"""
            df_copy = df.copy()
            if 'datetime' in df_copy.columns:
                df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
                df_copy = df_copy.set_index('datetime')
            elif 'date' in df_copy.columns:
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                df_copy = df_copy.set_index('date')
            elif 'time_key' in df_copy.columns:
                df_copy['time_key'] = pd.to_datetime(df_copy['time_key'])
                df_copy = df_copy.set_index('time_key')
            return df_copy
        
        # 准备所有数据
        prepared_data = {name: _prepare_df(df) for name, df in data_dict.items()}
        first_prepared = list(prepared_data.values())[0]
        
        if not isinstance(first_prepared.index, pd.DatetimeIndex):
            return {'returns': {}, 'drawdowns': {}, 'sharpe': {}}
        
        # 获取所有年份
        years = sorted(first_prepared.index.year.unique())
        
        for year in years:
            try:
                # 为每个数据源筛选该年的数据
                year_data_dict = {}
                min_rows = float('inf')
                for name, df in prepared_data.items():
                    year_df = df[df.index.year == year]
                    year_data_dict[name] = year_df
                    min_rows = min(min_rows, len(year_df))
                
                # 如果数据太少，跳过
                if min_rows < 20:
                    continue
                
                # 为该年运行回测
                cerebro = bt.Cerebro(stdstats=True)
                cerebro.broker.setcash(self.config.initial_cash)
                cerebro.broker.setcommission(commission=self.config.commission)
                
                # 添加所有数据源
                for name, df in year_data_dict.items():
                    bt_data = bt.feeds.PandasData(dataname=df, name=name)
                    cerebro.adddata(bt_data)
                
                cerebro.addstrategy(strategy_class, **params)
                
                # 添加时间收益率分析器（对于日内数据使用 Daily timeframe）
                is_intraday = _is_intraday_frequency(self.config.data_frequency)
                if is_intraday:
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn',
                                       timeframe=bt.TimeFrame.Days)
                else:
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
                
                # 运行回测
                results = cerebro.run()
                strat = results[0]
                
                # 提取时间收益率
                timereturn = strat.analyzers.timereturn.get_analysis()
                
                # 构建收益率序列并使用 PyfolioMetrics 计算
                if timereturn and isinstance(timereturn, dict) and len(timereturn) > 0:
                    returns_series = pd.Series(timereturn)
                    returns_series.index = pd.to_datetime(returns_series.index)
                    returns_series = returns_series.sort_index()
                    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # 使用 PyfolioMetrics 计算年度指标（日内数据已转为日收益率，使用 daily）
                    metrics_frequency = 'daily' if is_intraday else self.config.data_frequency
                    year_metrics = PyfolioMetrics(
                        returns_series, 
                        data_frequency=metrics_frequency
                    )
                    
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
