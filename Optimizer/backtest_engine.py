# -*- coding: utf-8 -*-
"""
回测引擎模块
封装backtrader回测逻辑，提供统一的性能评估接口
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BacktestConfig, DEFAULT_BACKTEST_CONFIG


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
            
            # 提取结果
            ret = strat.analyzers.returns.get_analysis()
            dd = strat.analyzers.drawdown.get_analysis()
            sharpe = strat.analyzers.sharpe.get_analysis()
            trades = strat.analyzers.trades.get_analysis()
            
            # 计算指标
            total_return = (np.exp(ret.get('rtot', 0)) - 1) * 100
            annual_return = ret.get('rnorm100', 0)
            max_drawdown = dd.max.drawdown if dd.max.drawdown else 0
            sharpe_ratio = sharpe.get('sharperatio', 0) or 0
            final_value = cerebro.broker.getvalue()
            
            # 交易统计
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 计算每年的指标
            yearly_returns = {}
            yearly_drawdowns = {}
            yearly_sharpe = {}
            
            if calculate_yearly:
                yearly_stats = self._calculate_yearly_metrics(data, strategy_class, params, asset_name)
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
                
                # 添加分析器
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                                  riskfreerate=0.0, annualize=True)
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
                
                # 运行回测
                results = cerebro.run()
                strat = results[0]
                
                # 提取结果
                ret = strat.analyzers.returns.get_analysis()
                dd = strat.analyzers.drawdown.get_analysis()
                sharpe = strat.analyzers.sharpe.get_analysis()
                timereturn = strat.analyzers.timereturn.get_analysis()
                
                # 计算该年的指标
                # 使用 rtot (总对数收益率) 转换为百分比收益
                year_return = (np.exp(ret.get('rtot', 0)) - 1) * 100
                year_drawdown = dd.max.drawdown if dd.max.drawdown else 0
                
                # 计算年度夏普比率
                year_sharpe_ratio = sharpe.get('sharperatio', None)
                
                # 如果SharpeRatio分析器无法计算或为0，使用日收益率手动计算
                if year_sharpe_ratio is None or year_sharpe_ratio == 0 or np.isnan(year_sharpe_ratio):
                    # 使用TimeReturn计算日收益率，然后手动计算夏普比率
                    if timereturn and isinstance(timereturn, dict) and len(timereturn) > 1:
                        # TimeReturn返回字典，键是日期，值是收益率
                        daily_returns_list = [v for v in timereturn.values() if v is not None and not np.isnan(v)]
                        if len(daily_returns_list) > 1:
                            daily_returns_array = np.array(daily_returns_list)
                            mean_return = np.mean(daily_returns_array)
                            std_return = np.std(daily_returns_array)
                            if std_return > 0 and not np.isnan(std_return):
                                # 年化夏普比率 = (平均日收益率 / 日收益率标准差) * sqrt(252)
                                year_sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                                # 检查结果是否有效
                                if np.isnan(year_sharpe_ratio) or np.isinf(year_sharpe_ratio):
                                    year_sharpe_ratio = 0
                            else:
                                year_sharpe_ratio = 0
                        else:
                            year_sharpe_ratio = 0
                    else:
                        year_sharpe_ratio = 0
                
                # 确保返回有效值
                if year_sharpe_ratio is None or np.isnan(year_sharpe_ratio) or np.isinf(year_sharpe_ratio):
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
