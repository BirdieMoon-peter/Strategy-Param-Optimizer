# 通用策略优化器使用指南

## 📋 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [详细使用说明](#详细使用说明)
4. [LLM配置](#llm配置)
5. [策略编写指南](#策略编写指南)
6. [输出格式说明](#输出格式说明)
7. [高级功能](#高级功能)
8. [常见问题](#常见问题)

---

## 概述

通用策略优化器是一个灵活、强大的量化交易策略参数优化工具，支持：

✅ **任意标的数据** - 只需提供CSV格式的OHLCV数据  
✅ **任意策略** - 动态加载策略脚本，无需修改代码  
✅ **多种LLM API** - 支持OpenAI、Ollama、自定义API  
✅ **智能优化** - 贝叶斯优化 + 可选LLM辅助  
✅ **JSON输出** - 结构化结果，包含详细解释  

---

## 快速开始

### 1. 安装依赖

```bash
conda activate quant
pip install backtrader pandas numpy optuna requests
```

### 2. 准备数据文件

数据文件必须是CSV格式，包含以下列：

- `datetime`: 时间戳
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

示例：`data/BTC.csv`

```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00,42000,42500,41800,42300,1000000
2024-01-01 01:00:00,42300,42800,42200,42700,950000
...
```

### 3. 准备策略文件

创建一个Python文件，定义继承自`backtrader.Strategy`的策略类。

示例：`strategies/my_strategy.py`

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.sell()
```

### 4. 运行优化（不使用LLM）

```python
from universal_optimizer import UniversalOptimizer

# 创建优化器
optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/my_strategy.py",
    objective="sharpe_ratio",  # 优化目标：夏普比率
    use_llm=False,  # 不使用LLM
    output_dir="./results",
    verbose=True
)

# 执行优化
result = optimizer.optimize(n_trials=50)

print(f"优化完成！结果已保存")
```

### 5. 运行优化（使用LLM）

```python
from universal_optimizer import UniversalOptimizer
from universal_llm_client import UniversalLLMConfig

# 配置LLM（使用OpenAI）
llm_config = UniversalLLMConfig(
    api_type="openai",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4",
    api_key="sk-your-api-key-here",
    temperature=0.7
)

# 创建优化器
optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/my_strategy.py",
    objective="sharpe_ratio",
    use_llm=True,
    llm_config=llm_config,
    output_dir="./results",
    verbose=True
)

# 执行优化
result = optimizer.optimize(n_trials=50)
```

---

## 详细使用说明

### UniversalOptimizer 参数说明

```python
UniversalOptimizer(
    data_path: str,              # 数据文件路径（CSV格式）
    strategy_path: str,          # 策略文件路径（.py文件）
    objective: str,              # 优化目标（见下表）
    use_llm: bool,               # 是否使用LLM
    llm_config: Optional[UniversalLLMConfig],  # LLM配置
    output_dir: str,             # 输出目录
    verbose: bool                # 是否打印详细信息
)
```

### 优化目标选项

| 目标 | 说明 |
|------|------|
| `sharpe_ratio` | 夏普比率（风险调整后收益） |
| `annual_return` | 年化收益率 |
| `total_return` | 总收益率 |
| `max_drawdown` | 最大回撤（负值，最小化） |
| `calmar_ratio` | 卡玛比率（年化收益/最大回撤） |
| `sortino_ratio` | 索提诺比率（下行风险调整后收益） |

### optimize() 方法参数

```python
optimizer.optimize(
    n_trials: int = 50,                        # 优化试验次数
    bayesian_config: Optional[BayesianOptConfig] = None  # 贝叶斯优化配置
)
```

---

## LLM配置

### 支持的LLM类型

#### 1. OpenAI API

```python
from universal_llm_client import UniversalLLMConfig

llm_config = UniversalLLMConfig(
    api_type="openai",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4",  # 或 "gpt-3.5-turbo"
    api_key="sk-your-api-key-here",
    temperature=0.7,
    max_tokens=4096,
    timeout=120
)
```

#### 2. Ollama（本地）

```python
llm_config = UniversalLLMConfig(
    api_type="ollama",
    base_url="http://localhost:11434",
    model_name="qwen",  # 或其他本地模型
    api_key="",  # Ollama不需要API密钥
    temperature=0.7
)
```

#### 3. 自定义API

```python
llm_config = UniversalLLMConfig(
    api_type="custom",
    base_url="https://your-api-endpoint.com/chat",
    model_name="your-model",
    api_key="your-api-key",
    temperature=0.7
)
```

### 预设配置

```python
from universal_llm_client import PRESET_CONFIGS

# 使用预设配置
config = PRESET_CONFIGS["openai-gpt4"]
config.api_key = "sk-your-key"

# 可用预设：
# - "openai-gpt4"
# - "openai-gpt35"
# - "ollama-xuanyuan"
# - "ollama-qwen"
```

### 自定义System Prompt

LLM客户端内置了三个专用的system prompt：

1. **STRATEGY_ANALYSIS_PROMPT** - 用于分析策略参数并推荐搜索空间
2. **OPTIMIZATION_HISTORY_PROMPT** - 用于根据历史结果调整搜索空间
3. **RESULT_EXPLANATION_PROMPT** - 用于解释优化结果

如果需要自定义，可以在调用时传入：

```python
# 在optimizer内部，LLM客户端会自动使用内置prompt
# 如果需要完全自定义，可以修改 universal_llm_client.py 中的 PROMPT 常量
```

---

## 策略编写指南

### 基本结构

```python
import backtrader as bt

class YourStrategy(bt.Strategy):
    """策略描述（会被LLM使用）"""
    
    # 1. 定义参数
    params = (
        ('param1', default_value1),
        ('param2', default_value2),
    )
    
    # 2. 初始化
    def __init__(self):
        # 计算指标
        self.indicator = bt.indicators.SomeIndicator(...)
    
    # 3. 交易逻辑
    def next(self):
        if not self.position:
            # 买入条件
            if condition:
                self.buy()
        else:
            # 卖出条件
            if condition:
                self.sell()
```

### 参数定义规范

1. **使用 params 元组**：所有可优化参数必须在`params`中定义
2. **参数命名**：使用小写+下划线，如`fast_period`
3. **合理默认值**：提供有意义的默认值
4. **参数类型**：
   - 整数参数：`('period', 20)`
   - 浮点参数：`('threshold', 0.02)`

### 常用指标

```python
# 移动平均
bt.indicators.SMA(data, period=20)  # 简单移动平均
bt.indicators.EMA(data, period=20)  # 指数移动平均

# 震荡指标
bt.indicators.RSI(data, period=14)  # RSI
bt.indicators.Stochastic(data)      # KDJ

# 趋势指标
bt.indicators.MACD(data)            # MACD
bt.indicators.ADX(data)             # ADX

# 波动率指标
bt.indicators.BollingerBands(data, period=20, devfactor=2)
bt.indicators.ATR(data, period=14)  # ATR

# 交叉信号
bt.indicators.CrossOver(line1, line2)
```

### 完整示例

参见 `example_strategy.py` 文件，包含：
- SimpleMAStrategy - 双均线策略
- RSIStrategy - RSI超买超卖策略
- BollingerBandsStrategy - 布林带策略
- MACDStrategy - MACD策略

---

## 输出格式说明

优化完成后，会生成一个JSON文件，包含以下内容：

```json
{
  "optimization_info": {
    "asset_name": "BTC",
    "strategy_name": "SimpleMAStrategy",
    "optimization_objective": "sharpe_ratio",
    "optimization_time": "2024-01-15 10:30:00",
    "data_range": {
      "start": "2023-01-01",
      "end": "2024-01-01",
      "total_days": 365
    }
  },
  "best_parameters": {
    "fast_period": 12,
    "slow_period": 26
  },
  "performance_metrics": {
    "sharpe_ratio": 1.85,
    "annual_return": 35.2,
    "max_drawdown": -12.5,
    "total_return": 35.2,
    "final_value": 135200.0,
    "trades_count": 45,
    "win_rate": 62.5
  },
  "yearly_performance": {
    "2023": {
      "return": 35.2,
      "drawdown": -12.5,
      "sharpe_ratio": 1.85
    }
  },
  "llm_explanation": {
    "parameter_explanation": "该参数组合在趋势市场中表现优异...",
    "performance_analysis": "策略在2023年实现了35.2%的年化收益...",
    "risk_assessment": "最大回撤控制在12.5%以内，风险可控...",
    "practical_suggestions": "建议在实盘前进行样本外测试...",
    "key_insights": [
      "快线周期12与慢线周期26的配比最优",
      "在趋势明确的市场环境中表现最佳",
      "建议配合风险管理措施使用"
    ]
  }
}
```

### 字段说明

- **optimization_info** - 优化基本信息
- **best_parameters** - 最优参数组合
- **performance_metrics** - 总体性能指标
- **yearly_performance** - 逐年性能明细
- **llm_explanation** - LLM生成的详细解释（如果启用LLM）

---

## 高级功能

### 批量优化（多个目标）

```python
optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/my_strategy.py",
    use_llm=False
)

# 同时优化多个目标
results = optimizer.batch_optimize(
    objectives=["sharpe_ratio", "annual_return", "calmar_ratio"],
    n_trials_per_objective=50
)

# 结果会保存为 batch_optimization_*.json
```

### 自定义贝叶斯优化配置

```python
from config import BayesianOptConfig

bayesian_config = BayesianOptConfig(
    n_trials=100,            # 总试验次数
    n_startup_trials=20,     # 随机试验次数
    objective_focus="sharpe_ratio",  # 主优化目标
    n_jobs=4,                # 并行任务数
    timeout=3600             # 超时时间（秒）
)

result = optimizer.optimize(bayesian_config=bayesian_config)
```

### 从命令行运行

创建脚本 `run_optimization.py`：

```python
import sys
import json
from universal_optimizer import UniversalOptimizer
from universal_llm_client import UniversalLLMConfig

def main():
    if len(sys.argv) < 3:
        print("用法: python run_optimization.py <数据文件> <策略文件> [--llm]")
        return
    
    data_path = sys.argv[1]
    strategy_path = sys.argv[2]
    use_llm = "--llm" in sys.argv
    
    llm_config = None
    if use_llm:
        llm_config = UniversalLLMConfig(
            api_type="openai",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4",
            api_key="your-api-key"
        )
    
    optimizer = UniversalOptimizer(
        data_path=data_path,
        strategy_path=strategy_path,
        objective="sharpe_ratio",
        use_llm=use_llm,
        llm_config=llm_config
    )
    
    result = optimizer.optimize(n_trials=50)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

运行：

```bash
python run_optimization.py data/BTC.csv strategies/my_strategy.py
```

---

## 常见问题

### Q1: 如何选择优化目标？

**A:** 根据你的交易风格选择：
- 稳健型：`sharpe_ratio` 或 `calmar_ratio`
- 激进型：`annual_return` 或 `total_return`
- 风险厌恶型：最小化 `max_drawdown`

### Q2: 优化需要多少次试验？

**A:** 建议：
- 参数较少（2-3个）：30-50次
- 参数中等（4-6个）：50-100次
- 参数较多（7+个）：100-200次

### Q3: 是否必须使用LLM？

**A:** 不是必须的。
- **不使用LLM**：使用默认参数范围，速度快
- **使用LLM**：智能推荐参数范围，可能找到更好的解，但速度较慢

### Q4: LLM连接失败怎么办？

**A:** 检查：
1. API密钥是否正确
2. 网络连接是否正常
3. base_url是否正确
4. 对于Ollama，确保服务已启动：`ollama serve`

### Q5: 数据格式有什么要求？

**A:** 必须包含列：`datetime, open, high, low, close, volume`
- datetime可以是字符串或时间戳
- 价格和成交量必须是数值
- 数据应按时间升序排列

### Q6: 如何处理多时间周期策略？

**A:** backtrader支持多数据源：

```python
class MultiTimeframeStrategy(bt.Strategy):
    params = (
        ('period1', 10),
        ('period2', 20),
    )
    
    def __init__(self):
        # self.data0 是主数据
        # 可以在优化器中添加更多数据源
        self.indicator1 = bt.indicators.SMA(self.data0.close, period=self.params.period1)
```

### Q7: 优化结果过拟合怎么办？

**A:** 建议：
1. 保留样本外数据进行验证
2. 使用Walk-Forward分析
3. 增加正则化约束
4. 减少参数数量
5. 参考LLM的风险评估建议

### Q8: 如何集成到实盘交易系统？

**A:** 优化结果JSON文件可以直接被读取：

```python
import json

with open("optimization_BTC_Strategy_*.json", 'r') as f:
    result = json.load(f)

best_params = result['best_parameters']

# 在实盘中使用这些参数
strategy = YourStrategy(**best_params)
```

---

## 技术支持

如有问题，请检查：

1. **日志输出** - 设置 `verbose=True` 查看详细日志
2. **依赖版本** - 确保所有库版本兼容
3. **数据质量** - 检查数据是否有缺失值或异常值
4. **策略逻辑** - 确认策略代码无语法错误

---

## 更新日志

### v1.0.0 (2024-01-15)
- ✨ 初始发布
- ✅ 支持任意标的和策略
- ✅ 支持多种LLM API
- ✅ JSON格式输出
- ✅ 完整的LLM解释功能

---

**祝你交易顺利！** 🚀
