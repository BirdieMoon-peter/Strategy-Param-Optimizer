# -*- coding: utf-8 -*-
"""
回测引擎模块
封装backtrader回测逻辑，使用empyrical/pyfolio计算性能指标
"""

import os
import sys
import warnings
from typing import Dict, Any, Optional, Type, Tuple, List
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

# 性能指标计算 (使用 empyrical 替代 backtrader analyzers)
try:
    import empyrical as ep
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    print("[警告] empyrical 未安装，将使用 backtrader 内置分析器")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BacktestConfig, DEFAULT_BACKTEST_CONFIG


class PortfolioValueAnalyzer(bt.Analyzer):
    """
    自定义分析器：记录每日收盘时的投资组合价值
    用于后续使用 empyrical 计算性能指标
    """
    
    def __init__(self):
        self.portfolio_values = []
        self.dates = []
    
    def next(self):
        # 记录当前日期和投资组合价值
        current_date = self.datas[0].datetime.date(0)
        current_value = self.strategy.broker.getvalue()
        
        # 只在新的一天记录（避免重复）
        if not self.dates or self.dates[-1] != current_date:
            self.dates.append(current_date)
            self.portfolio_values.append(current_value)
    
    def get_analysis(self):
        return {
            'dates': self.dates,
            'portfolio_values': self.portfolio_values
        }


class TradeRecorder(bt.Analyzer):
    """
    自定义分析器：记录交易信息
    """
    
    def __init__(self):
        self.trades = []
        self.total_trades = 0
        self.won_trades = 0
        self.lost_trades = 0
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_trades += 1
            if trade.pnl > 0:
                self.won_trades += 1
            elif trade.pnl < 0:
                self.lost_trades += 1
            
            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'size': trade.size,
                'price': trade.price,
                'value': trade.value,
            })
    
    def get_analysis(self):
        return {
            'total': self.total_trades,
            'won': self.won_trades,
            'lost': self.lost_trades,
            'trades': self.trades
        }


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float        # 总收益率(%)
    annual_return: float       # 年化收益率(%)
    max_drawdown: float        # 最大回撤(%)
    sharpe_ratio: float        # 夏普比率
    final_value: float         # 最终资产
    trades_count: int          # 交易次数
    win_rate: float            # 胜率
    params: Dict[str, Any]     # 使用的参数
    yearly_returns: Dict[int, float] = field(default_factory=dict)  # 每年的收益率(%)
    yearly_drawdowns: Dict[int, float] = field(default_factory=dict)  # 每年的最大回撤(%)
    yearly_sharpe: Dict[int, float] = field(default_factory=dict)  # 每年的夏普比率


class BacktestEngine:
    """
    回测引擎
    封装backtrader，提供简洁的API进行策略回测和性能评估
    """
    
    def __init__(
        self, 
        config: BacktestConfig = None,
        data: pd.DataFrame = None,
        data_list: List[pd.DataFrame] = None,
        data_names: List[str] = None,
        strategy_class: Type[bt.Strategy] = None,
        initial_cash: float = None,
        commission: float = None
    ):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
            data: DataFrame格式的数据（单数据源）
            data_list: 多个DataFrame格式的数据列表（多数据源）
            data_names: 数据源名称列表，与data_list对应
            strategy_class: 策略类（新接口）
            initial_cash: 初始资金（新接口）
            commission: 手续费率（新接口）
        """
        self.config = config or DEFAULT_BACKTEST_CONFIG
        self.data_cache = {}
        
        # 新接口支持
        if data is not None and data_list is not None:
            raise ValueError("data 和 data_list 不能同时提供")
        
        if data is not None:
            self.data_df = data
            self.data_list = [data]
            self.data_names = ["ASSET"]
            self.is_multi_data = False
        elif data_list is not None:
            self.data_df = None
            self.data_list = data_list
            self.data_names = data_names or [f"ASSET{i}" for i in range(len(data_list))]
            self.is_multi_data = len(data_list) > 1
        else:
            self.data_df = None
            self.data_list = None
            self.data_names = None
            self.is_multi_data = False
        
        self.strategy_class = strategy_class
        if initial_cash is not None:
            self.config.initial_cash = initial_cash
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
    
    def _detect_data_timeframe(self, data_df: pd.DataFrame) -> Tuple[int, int]:
        """
        自动检测数据的时间频率
        
        Args:
            data_df: 数据DataFrame（索引应为datetime）
            
        Returns:
            (timeframe, compression) 元组
            timeframe: bt.TimeFrame常量
            compression: 压缩倍数
        """
        if data_df.index.dtype != 'datetime64[ns]':
            # 如果索引不是datetime，尝试转换
            try:
                data_df.index = pd.to_datetime(data_df.index)
            except:
                # 无法转换，返回默认值（日线）
                return bt.TimeFrame.Days, 1
        
        # 计算前100个数据点的时间间隔
        sample_size = min(100, len(data_df))
        if sample_size < 2:
            return bt.TimeFrame.Days, 1
        
        time_deltas = data_df.index[1:sample_size] - data_df.index[0:sample_size-1]
        median_delta = time_deltas.median()
        
        # 转换为分钟数
        minutes = median_delta.total_seconds() / 60
        
        # 判断时间频率
        if minutes < 1.5:  # 1分钟
            return bt.TimeFrame.Minutes, 1
        elif minutes < 3:  # 2分钟
            return bt.TimeFrame.Minutes, 2
        elif minutes < 7:  # 5分钟
            return bt.TimeFrame.Minutes, 5
        elif minutes < 20:  # 15分钟
            return bt.TimeFrame.Minutes, 15
        elif minutes < 45:  # 30分钟
            return bt.TimeFrame.Minutes, 30
        elif minutes < 90:  # 60分钟
            return bt.TimeFrame.Minutes, 60
        elif minutes < 60 * 6:  # 4小时
            return bt.TimeFrame.Minutes, 240
        elif minutes < 60 * 24 * 1.5:  # 日线
            return bt.TimeFrame.Days, 1
        elif minutes < 60 * 24 * 8:  # 周线
            return bt.TimeFrame.Weeks, 1
        else:  # 月线
            return bt.TimeFrame.Months, 1
    
    def run_backtest(
        self,
        strategy_class: Type[bt.Strategy] = None,
        data: pd.DataFrame = None,
        data_list: List[pd.DataFrame] = None,
        data_names: List[str] = None,
        params: Dict[str, Any] = None,
        asset_name: str = "ASSET",
        calculate_yearly: bool = True
    ) -> Optional[BacktestResult]:
        """
        运行单次回测
        
        Args:
            strategy_class: 策略类（可选，如果未提供则使用初始化时的策略）
            data: 行情数据（可选，单数据源，如果未提供则使用初始化时的数据）
            data_list: 多个行情数据列表（可选，多数据源）
            data_names: 数据源名称列表，与data_list对应
            params: 策略参数
            asset_name: 资产名称（仅用于单数据源）
            calculate_yearly: 是否计算每年的指标
            
        Returns:
            BacktestResult对象
        """
        # 使用提供的参数或初始化时的参数
        strategy_class = strategy_class or self.strategy_class
        
        # 确定使用单数据源还是多数据源
        if data is not None:
            use_data_list = [data]
            use_data_names = [asset_name]
        elif data_list is not None:
            use_data_list = data_list
            use_data_names = data_names or [f"ASSET{i}" for i in range(len(data_list))]
        elif self.data_list is not None:
            use_data_list = self.data_list
            use_data_names = self.data_names
        elif self.data_df is not None:
            use_data_list = [self.data_df]
            use_data_names = [asset_name]
        else:
            raise ValueError("必须提供 data/data_list 参数或在初始化时指定")
        
        if strategy_class is None:
            raise ValueError("必须提供 strategy_class 参数或在初始化时指定")
        
        params = params or {}
        
        try:
            # 初始化Cerebro
            cerebro = bt.Cerebro(stdstats=True)
            cerebro.broker.setcash(self.config.initial_cash)
            cerebro.broker.setcommission(commission=self.config.commission)
            
            # 添加所有数据源
            for i, (data_df, data_name) in enumerate(zip(use_data_list, use_data_names)):
                # 准备数据：确保 datetime 是索引
                data_copy = data_df.copy()
                if 'datetime' in data_copy.columns:
                    data_copy['datetime'] = pd.to_datetime(data_copy['datetime'])
                    data_copy = data_copy.set_index('datetime')
                elif 'date' in data_copy.columns:
                    data_copy['date'] = pd.to_datetime(data_copy['date'])
                    data_copy = data_copy.set_index('date')
                
                # 确保列名小写
                data_copy.columns = [col.lower() for col in data_copy.columns]
                
                # 检测数据时间频率
                timeframe, compression = self._detect_data_timeframe(data_copy)
                
                # 添加数据（显式指定 timeframe 和 compression 以确保多数据源时间对齐）
                bt_data = bt.feeds.PandasData(
                    dataname=data_copy,
                    name=data_name,
                    timeframe=timeframe,
                    compression=compression
                )
                cerebro.adddata(bt_data)
            
            # 添加策略
            cerebro.addstrategy(strategy_class, **params)
            
            # 添加自定义分析器（用于 empyrical 计算）
            cerebro.addanalyzer(PortfolioValueAnalyzer, _name='portfolio')
            cerebro.addanalyzer(TradeRecorder, _name='trades')
            
            # 运行回测
            results = cerebro.run()
            strat = results[0]
            
            # 提取投资组合价值序列
            portfolio_analysis = strat.analyzers.portfolio.get_analysis()
            trade_analysis = strat.analyzers.trades.get_analysis()
            
            dates = portfolio_analysis['dates']
            portfolio_values = portfolio_analysis['portfolio_values']
            final_value = cerebro.broker.getvalue()
            
            # 使用 empyrical 计算指标
            if EMPYRICAL_AVAILABLE and len(portfolio_values) > 1:
                # 计算日收益率序列
                returns_series = self._calculate_returns_from_portfolio(dates, portfolio_values)
                
                # 使用 empyrical 计算指标
                total_return = ep.cum_returns_final(returns_series) * 100
                annual_return = ep.annual_return(returns_series) * 100
                max_drawdown = ep.max_drawdown(returns_series) * 100  # empyrical 返回负值
                max_drawdown = abs(max_drawdown)  # 转为正值
                sharpe_ratio = ep.sharpe_ratio(returns_series, risk_free=0.0, annualization=252)
                
                # 处理无效值
                if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                    sharpe_ratio = 0.0
            else:
                # 回退到简单计算
                total_return = ((final_value / self.config.initial_cash) - 1) * 100
                annual_return = self._simple_annual_return(dates, total_return)
                max_drawdown = self._simple_max_drawdown(portfolio_values)
                sharpe_ratio = self._simple_sharpe_ratio(portfolio_values)
            
            # 交易统计
            total_trades = trade_analysis['total']
            won_trades = trade_analysis['won']
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 计算每年的指标
            yearly_returns = {}
            yearly_drawdowns = {}
            yearly_sharpe = {}
            
            if calculate_yearly and len(use_data_list) == 1 and EMPYRICAL_AVAILABLE:
                # 使用 empyrical 计算逐年指标
                yearly_stats = self._calculate_yearly_metrics_empyrical(dates, portfolio_values)
                yearly_returns = yearly_stats.get('returns', {})
                yearly_drawdowns = yearly_stats.get('drawdowns', {})
                yearly_sharpe = yearly_stats.get('sharpe', {})
            
            return BacktestResult(
                total_return=total_return,
                annual_return=annual_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                final_value=final_value,
                trades_count=total_trades,
                win_rate=win_rate,
                params=params,
                yearly_returns=yearly_returns,
                yearly_drawdowns=yearly_drawdowns,
                yearly_sharpe=yearly_sharpe
            )
            
        except Exception as e:
            import traceback
            print(f"[错误] 回测执行失败: {e}")
            traceback.print_exc()
            return None
    
    def _calculate_returns_from_portfolio(
        self,
        dates: List,
        portfolio_values: List[float]
    ) -> pd.Series:
        """
        从投资组合价值序列计算日收益率序列
        
        Args:
            dates: 日期列表
            portfolio_values: 投资组合价值列表
            
        Returns:
            日收益率 Series (用于 empyrical)
        """
        if len(portfolio_values) < 2:
            return pd.Series(dtype=float)
        
        # 创建 DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'value': portfolio_values
        })
        df = df.set_index('date')
        
        # 计算日收益率
        df['returns'] = df['value'].pct_change()
        
        # 移除第一行 NaN
        returns = df['returns'].dropna()
        
        return returns
    
    def _calculate_yearly_metrics_empyrical(
        self,
        dates: List,
        portfolio_values: List[float]
    ) -> Dict[str, Dict[int, float]]:
        """
        使用 empyrical 计算每年的性能指标
        
        Args:
            dates: 日期列表
            portfolio_values: 投资组合价值列表
            
        Returns:
            包含每年收益率、回撤和夏普比率的字典
        """
        yearly_returns = {}
        yearly_drawdowns = {}
        yearly_sharpe = {}
        
        if len(dates) < 2:
            return {'returns': {}, 'drawdowns': {}, 'sharpe': {}}
        
        # 创建 DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'value': portfolio_values
        })
        df = df.set_index('date')
        df['returns'] = df['value'].pct_change()
        
        # 获取所有年份
        years = sorted(df.index.year.unique())
        
        for year in years:
            # 筛选该年的数据
            year_data = df[df.index.year == year]
            year_returns = year_data['returns'].dropna()
            
            # 如果数据太少，跳过
            if len(year_returns) < 5:
                continue
            
            try:
                # 使用 empyrical 计算指标
                year_return = ep.cum_returns_final(year_returns) * 100
                year_drawdown = abs(ep.max_drawdown(year_returns) * 100)
                year_sharpe_ratio = ep.sharpe_ratio(year_returns, risk_free=0.0, annualization=252)
                
                # 处理无效值
                if np.isnan(year_sharpe_ratio) or np.isinf(year_sharpe_ratio):
                    year_sharpe_ratio = 0.0
                if np.isnan(year_return):
                    year_return = 0.0
                if np.isnan(year_drawdown):
                    year_drawdown = 0.0
                
                yearly_returns[year] = year_return
                yearly_drawdowns[year] = year_drawdown
                yearly_sharpe[year] = year_sharpe_ratio
                
            except Exception:
                yearly_returns[year] = 0.0
                yearly_drawdowns[year] = 0.0
                yearly_sharpe[year] = 0.0
        
        return {
            'returns': yearly_returns,
            'drawdowns': yearly_drawdowns,
            'sharpe': yearly_sharpe
        }
    
    def _simple_annual_return(self, dates: List, total_return: float) -> float:
        """简单年化收益率计算（当 empyrical 不可用时）"""
        if len(dates) < 2:
            return 0.0
        
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        years = (end_date - start_date).days / 365.25
        
        if years <= 0:
            return total_return
        
        # 年化: (1 + total_return/100)^(1/years) - 1
        try:
            annual = ((1 + total_return / 100) ** (1 / years) - 1) * 100
            return annual if not np.isnan(annual) else 0.0
        except:
            return 0.0
    
    def _simple_max_drawdown(self, portfolio_values: List[float]) -> float:
        """简单最大回撤计算（当 empyrical 不可用时）"""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_dd = np.min(drawdown) * 100
        
        return abs(max_dd) if not np.isnan(max_dd) else 0.0
    
    def _simple_sharpe_ratio(self, portfolio_values: List[float]) -> float:
        """简单夏普比率计算（当 empyrical 不可用时）"""
        if len(portfolio_values) < 10:
            return 0.0
        
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        # 年化夏普比率
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0
    
    def _calculate_yearly_metrics(
        self,
        data: pd.DataFrame,
        strategy_class: Type[bt.Strategy],
        params: Dict[str, Any],
        asset_name: str
    ) -> Dict[str, Dict[int, float]]:
        """
        计算每年的性能指标（保留旧方法作为备用）
        当 empyrical 不可用时使用此方法
        
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
            return {'returns': {}, 'drawdowns': {}, 'sharpe': {}}
        
        years = sorted(data_copy.index.year.unique())
        
        for year in years:
            year_data = data_copy[data_copy.index.year == year]
            
            if len(year_data) < 20:
                continue
            
            try:
                cerebro = bt.Cerebro(stdstats=True)
                cerebro.broker.setcash(self.config.initial_cash)
                cerebro.broker.setcommission(commission=self.config.commission)
                
                # 检测数据时间频率
                timeframe, compression = self._detect_data_timeframe(year_data)
                
                # 添加数据（显式指定 timeframe 和 compression）
                bt_data = bt.feeds.PandasData(
                    dataname=year_data,
                    name=asset_name,
                    timeframe=timeframe,
                    compression=compression
                )
                cerebro.adddata(bt_data)
                cerebro.addstrategy(strategy_class, **params)
                
                # 使用自定义分析器
                cerebro.addanalyzer(PortfolioValueAnalyzer, _name='portfolio')
                
                results = cerebro.run()
                strat = results[0]
                
                portfolio_analysis = strat.analyzers.portfolio.get_analysis()
                pv = portfolio_analysis['portfolio_values']
                
                if len(pv) > 1:
                    year_return = ((pv[-1] / pv[0]) - 1) * 100
                    year_drawdown = self._simple_max_drawdown(pv)
                    year_sharpe_ratio = self._simple_sharpe_ratio(pv)
                else:
                    year_return = 0
                    year_drawdown = 0
                    year_sharpe_ratio = 0
                
                yearly_returns[year] = year_return
                yearly_drawdowns[year] = year_drawdown
                yearly_sharpe[year] = year_sharpe_ratio
                
            except Exception:
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
        根据目标获取评估值
        
        Args:
            result: 回测结果
            objective: 优化目标
            
        Returns:
            目标值（已处理方向）
        """
        if objective == "sharpe_ratio":
            return result.sharpe_ratio
        elif objective == "annual_return":
            return result.annual_return
        elif objective == "max_drawdown":
            # 最大回撤越小越好，但Optuna默认最大化
            # 返回负值让Optuna最大化时实际最小化回撤
            return -result.max_drawdown
        else:
            return result.sharpe_ratio
    
    def get_result_summary(self, result: BacktestResult) -> Dict[str, Any]:
        """
        获取回测结果摘要
        
        Args:
            result: 回测结果
            
        Returns:
            结果摘要字典
        """
        return {
            "总收益率": f"{result.total_return:.2f}%",
            "年化收益率": f"{result.annual_return:.2f}%",
            "最大回撤": f"{result.max_drawdown:.2f}%",
            "夏普比率": f"{result.sharpe_ratio:.2f}",
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
