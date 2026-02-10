# -*- coding: utf-8 -*-
"""
示例策略：简单的均线交叉策略
用于演示如何编写兼容通用优化器的策略
"""

import backtrader as bt


class SimpleMAStrategy(bt.Strategy):
    """
    简单双均线交叉策略
    
    策略逻辑:
    - 快线上穿慢线时买入
    - 快线下穿慢线时卖出
    """
    
    # 定义策略参数
    params = (
        ('fast_period', 10),    # 快速均线周期
        ('slow_period', 30),    # 慢速均线周期
    )
    
    def __init__(self):
        """初始化策略"""
        # 计算均线
        self.fast_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.slow_period
        )
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        """执行策略逻辑"""
        if not self.position:  # 无持仓
            if self.crossover > 0:  # 金叉
                self.buy()
        else:  # 有持仓
            if self.crossover < 0:  # 死叉
                self.sell()


class RSIStrategy(bt.Strategy):
    """
    RSI超买超卖策略
    
    策略逻辑:
    - RSI低于超卖线时买入
    - RSI高于超买线时卖出
    """
    
    params = (
        ('rsi_period', 14),      # RSI周期
        ('rsi_oversold', 30),    # 超卖阈值
        ('rsi_overbought', 70),  # 超买阈值
    )
    
    def __init__(self):
        """初始化策略"""
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_period
        )
    
    def next(self):
        """执行策略逻辑"""
        if not self.position:
            if self.rsi < self.params.rsi_oversold:
                self.buy()
        else:
            if self.rsi > self.params.rsi_overbought:
                self.sell()


class BollingerBandsStrategy(bt.Strategy):
    """
    布林带突破策略
    
    策略逻辑:
    - 价格突破下轨时买入
    - 价格突破上轨时卖出
    """
    
    params = (
        ('period', 20),      # 布林带周期
        ('devfactor', 2.0),  # 标准差倍数
    )
    
    def __init__(self):
        """初始化策略"""
        self.bbands = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.period,
            devfactor=self.params.devfactor
        )
    
    def next(self):
        """执行策略逻辑"""
        if not self.position:
            # 价格低于下轨
            if self.data.close < self.bbands.lines.bot:
                self.buy()
        else:
            # 价格高于上轨
            if self.data.close > self.bbands.lines.top:
                self.sell()


class MACDStrategy(bt.Strategy):
    """
    MACD策略
    
    策略逻辑:
    - MACD线上穿信号线时买入
    - MACD线下穿信号线时卖出
    """
    
    params = (
        ('fast_ema', 12),    # 快速EMA周期
        ('slow_ema', 26),    # 慢速EMA周期
        ('signal', 9),       # 信号线周期
    )
    
    def __init__(self):
        """初始化策略"""
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.fast_ema,
            period_me2=self.params.slow_ema,
            period_signal=self.params.signal
        )
        
        self.crossover = bt.indicators.CrossOver(
            self.macd.macd,
            self.macd.signal
        )
    
    def next(self):
        """执行策略逻辑"""
        if not self.position:
            if self.crossover > 0:  # MACD上穿信号线
                self.buy()
        else:
            if self.crossover < 0:  # MACD下穿信号线
                self.sell()


# 策略编写指南
"""
如何编写兼容通用优化器的策略：

1. 必须继承 backtrader.Strategy

2. 使用 params 元组定义所有可优化的参数：
   params = (
       ('param_name', default_value),
       ...
   )

3. 在 __init__ 方法中初始化指标和变量

4. 在 next 方法中实现交易逻辑

5. 参数命名建议：
   - 使用小写+下划线: fast_period, rsi_oversold
   - 使用有意义的名称
   - 提供合理的默认值

6. 数据访问：
   - self.data.close  # 收盘价
   - self.data.open   # 开盘价
   - self.data.high   # 最高价
   - self.data.low    # 最低价
   - self.data.volume # 成交量

7. 交易操作：
   - self.buy()   # 买入
   - self.sell()  # 卖出
   - self.position  # 当前持仓

8. 使用backtrader内置指标：
   - bt.indicators.SMA  # 简单移动平均
   - bt.indicators.EMA  # 指数移动平均
   - bt.indicators.RSI  # 相对强弱指标
   - bt.indicators.MACD  # MACD指标
   - bt.indicators.BollingerBands  # 布林带
   - 更多请参考backtrader文档

示例：

class MyStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('threshold', 0.02),
    )
    
    def __init__(self):
        self.ma = bt.indicators.SMA(self.data.close, period=self.params.period)
    
    def next(self):
        if not self.position:
            if self.data.close > self.ma * (1 + self.params.threshold):
                self.buy()
        else:
            if self.data.close < self.ma * (1 - self.params.threshold):
                self.sell()
"""
