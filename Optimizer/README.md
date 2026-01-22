# 通用策略优化器 (Universal Strategy Optimizer)

<div align="center">

**强大、灵活、智能的量化交易策略参数优化工具**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[快速开始](#快速开始) • [功能特性](#功能特性) • [文档](#文档) • [示例](#示例)

</div>

---

## 📖 简介

通用策略优化器是一个封装完善的量化交易策略参数优化工具，支持：

✅ **任意标的数据** - 只需提供CSV格式的OHLCV数据  
✅ **任意策略** - 动态加载策略脚本，无需修改代码  
✅ **多种LLM API** - 支持OpenAI、Ollama、自定义API  
✅ **智能优化** - 贝叶斯优化 + 可选LLM辅助  
✅ **JSON输出** - 结构化结果，包含详细解释  

---

## 🚀 快速开始

### 1. 环境准备

```bash
conda activate quant
```

### 2. 三种使用方式

#### 方式1: 交互式启动（推荐新手）

```bash
./quick_start.sh
```

#### 方式2: Python脚本

```python
from universal_optimizer import UniversalOptimizer

optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="example_strategy.py",
    objective="sharpe_ratio",
    use_llm=False
)

result = optimizer.optimize(n_trials=50)
print("最优参数:", result['best_parameters'])
```

#### 方式3: 命令行

```bash
python run_universal_optimizer.py \
    --data data/BTC.csv \
    --strategy example_strategy.py \
    --trials 50
```

---

## ✨ 功能特性

### 输入支持
- **CSV数据文件**: 任意标的，包含 datetime, open, high, low, close, volume
- **Python策略脚本**: 继承 `backtrader.Strategy` 的策略类
- **优化目标**: sharpe_ratio, annual_return, calmar_ratio, 等
- **LLM选项**: 可选择使用 OpenAI / Ollama / 自定义API

### LLM智能辅助
- 分析策略参数并推荐搜索空间
- 解释优化结果
- 提供风险评估和实战建议
- 内置专业 System Prompt

### 输出格式
- **JSON结构化数据**: 最优参数、性能指标、逐年表现
- **LLM详细解释**: 参数有效性、性能分析、实战建议
- **易于集成**: 可直接应用到实盘系统

---

## 📚 文档

### 新手入门
- **[快速开始.md](./快速开始.md)** ⭐ - 5分钟快速上手（中文）
- **[项目总览.md](./项目总览.md)** - 完整项目结构说明

### 深入学习
- **[UNIVERSAL_OPTIMIZER_GUIDE.md](./UNIVERSAL_OPTIMIZER_GUIDE.md)** - 完整使用手册（英文）
- **[README_UNIVERSAL.md](./README_UNIVERSAL.md)** - 详细功能介绍

### 实践指南
- **[example_strategy.py](./example_strategy.py)** - 4个策略示例 + 编写指南
- **[使用示例.py](./使用示例.py)** - 6个完整使用场景
- **[config_example.py](./config_example.py)** - 配置示例

### 参考文档
- **[封装完成说明.md](./封装完成说明.md)** - 封装总结和使用方法

---

## 📝 示例

### 示例1: 基本优化（不使用LLM）

```python
from universal_optimizer import UniversalOptimizer

optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="example_strategy.py",
    objective="sharpe_ratio",
    use_llm=False
)

result = optimizer.optimize(n_trials=50)
```

### 示例2: 使用OpenAI LLM

```python
from universal_optimizer import UniversalOptimizer
from universal_llm_client import UniversalLLMConfig

llm_config = UniversalLLMConfig(
    api_type="openai",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4",
    api_key="sk-your-api-key"
)

optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="example_strategy.py",
    objective="sharpe_ratio",
    use_llm=True,
    llm_config=llm_config
)

result = optimizer.optimize(n_trials=50)

# 查看LLM解释
print(result['llm_explanation']['parameter_explanation'])
```

### 示例3: 批量优化多个目标

```python
optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="example_strategy.py",
    use_llm=False
)

results = optimizer.batch_optimize(
    objectives=["sharpe_ratio", "annual_return", "calmar_ratio"],
    n_trials_per_objective=50
)
```

---

## 🧪 运行测试

```bash
python test_universal_optimizer.py
```

测试包括：
- ✅ 基本优化功能
- ✅ LLM辅助优化
- ✅ 批量优化
- ✅ JSON输出格式验证

---

## 📋 文件结构

```
Optimizer/
├── universal_optimizer.py          # 通用优化器主类
├── universal_llm_client.py         # 通用LLM客户端
├── backtest_engine.py              # 回测引擎
├── bayesian_optimizer.py           # 贝叶斯优化器
├── config.py                       # 配置文件
├── __init__.py                     # 包初始化
├── requirements.txt                # 依赖列表
│
├── run_universal_optimizer.py      # 命令行入口
├── quick_start.sh                  # 快速启动脚本
│
├── example_strategy.py             # 策略示例
├── 使用示例.py                     # 使用示例集合
├── config_example.py               # 配置示例
├── test_universal_optimizer.py     # 测试套件
│
├── 快速开始.md                     # 5分钟入门
├── UNIVERSAL_OPTIMIZER_GUIDE.md    # 完整手册
├── README_UNIVERSAL.md             # 详细说明
├── 封装完成说明.md                 # 封装总结
├── 项目总览.md                     # 项目结构
│
└── output/                         # 输出目录
```

---

## 📊 输入输出示例

### 输入要求

**数据文件 (CSV)**:
```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00,42000,42500,41800,42300,1000000
```

**策略文件 (Python)**:
```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('period', 20),
    )
    
    def __init__(self):
        self.ma = bt.indicators.SMA(self.data.close, period=self.params.period)
    
    def next(self):
        if not self.position and self.data.close > self.ma:
            self.buy()
        elif self.position and self.data.close < self.ma:
            self.sell()
```

### 输出格式 (JSON)

```json
{
  "optimization_info": {...},
  "best_parameters": {"period": 25},
  "performance_metrics": {
    "sharpe_ratio": 1.85,
    "annual_return": 35.2,
    "max_drawdown": -12.5,
    ...
  },
  "yearly_performance": {...},
  "llm_explanation": {
    "parameter_explanation": "...",
    "performance_analysis": "...",
    "risk_assessment": "...",
    "practical_suggestions": "...",
    "key_insights": [...]
  }
}
```

---

## 🎯 支持的优化目标

- `sharpe_ratio` - 夏普比率（推荐）
- `annual_return` - 年化收益率
- `calmar_ratio` - 卡玛比率
- `sortino_ratio` - 索提诺比率
- `max_drawdown` - 最大回撤
- `total_return` - 总收益率

---

## 🤖 支持的LLM

- OpenAI (GPT-4, GPT-3.5)
- Ollama (本地模型)
- 自定义API

---

## ⚙️ 依赖

```
backtrader
pandas
numpy
optuna
requests
```

已在 `quant` conda 环境中安装。

---

## 💡 常见问题

### Q: 如何选择优化目标？
**A:** 稳健型选 `sharpe_ratio`，激进型选 `annual_return`。

### Q: 需要多少次试验？
**A:** 2-3个参数：30-50次；4-6个参数：50-100次；7+个参数：100-200次。

### Q: LLM有什么用？
**A:** 智能推荐参数范围、解释优化结果、提供风险评估和实战建议。

### Q: 必须使用LLM吗？
**A:** 不必须。不使用LLM速度更快，但没有详细解释。

更多问题请参考 [UNIVERSAL_OPTIMIZER_GUIDE.md](./UNIVERSAL_OPTIMIZER_GUIDE.md)

---

## 🎓 学习路径

1. **入门**: 阅读 [快速开始.md](./快速开始.md)（5分钟）
2. **测试**: 运行 `python test_universal_optimizer.py`
3. **示例**: 运行 `python 使用示例.py`（交互式）
4. **实践**: 使用自己的数据和策略
5. **进阶**: 阅读 [UNIVERSAL_OPTIMIZER_GUIDE.md](./UNIVERSAL_OPTIMIZER_GUIDE.md)

---

## 📄 版本信息

- **版本**: v1.0.0
- **状态**: ✅ 生产就绪
- **Python**: 3.8+
- **许可**: MIT

---

## 🙏 致谢

感谢以下开源项目：
- [Backtrader](https://www.backtrader.com/)
- [Optuna](https://optuna.org/)
- [OpenAI](https://openai.com/)
- [Ollama](https://ollama.ai/)

---

<div align="center">

**🚀 开始你的量化交易优化之旅！**

[快速开始](./快速开始.md) • [完整手册](./UNIVERSAL_OPTIMIZER_GUIDE.md) • [运行测试](#运行测试)

</div>
