# 通用策略优化器 (Universal Strategy Optimizer)

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**强大、灵活、智能的量化交易策略参数优化工具**

[快速开始](#快速开始) • [功能特性](#功能特性) • [使用文档](#使用文档) • [示例](#示例) • [FAQ](#常见问题)

</div>

---

## 📖 简介

通用策略优化器是一个封装完善的量化交易策略参数优化工具，支持任意标的、任意策略、多种LLM API接入，提供JSON格式的结构化输出和详细的性能分析。

### ✨ 核心特性

🎯 **通用性强**
- ✅ 支持任意标的数据（只需CSV格式）
- ✅ 支持任意策略（动态加载.py文件）
- ✅ 支持多种优化目标

🤖 **LLM智能辅助**
- ✅ 支持OpenAI GPT-4/3.5
- ✅ 支持本地Ollama
- ✅ 支持自定义API
- ✅ 内置专业system prompt

⚡ **高效优化**
- ✅ 贝叶斯优化算法（Optuna）
- ✅ 可选LLM参数推荐
- ✅ 并行计算支持

📊 **结构化输出**
- ✅ JSON格式结果
- ✅ 详细性能指标
- ✅ 逐年表现分析
- ✅ LLM生成的解释和建议

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate quant

# 安装依赖（如果还没安装）
pip install backtrader pandas numpy optuna requests
```

### 2. 最简示例

```python
from universal_optimizer import UniversalOptimizer

# 创建优化器
optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",        # 你的数据文件
    strategy_path="my_strategy.py",  # 你的策略文件
    objective="sharpe_ratio",         # 优化目标
    use_llm=False                     # 暂不使用LLM
)

# 执行优化
result = optimizer.optimize(n_trials=50)

print("最优参数:", result['best_parameters'])
print("性能指标:", result['performance_metrics'])
```

### 3. 命令行方式

```bash
python run_universal_optimizer.py \
    --data data/BTC.csv \
    --strategy my_strategy.py \
    --objective sharpe_ratio \
    --trials 50
```

---

## 📁 项目结构

```
Optimizer/
├── universal_optimizer.py          # 通用优化器主类
├── universal_llm_client.py         # 通用LLM客户端
├── run_universal_optimizer.py      # 命令行入口
├── example_strategy.py             # 示例策略集合
├── test_universal_optimizer.py     # 测试脚本
├── UNIVERSAL_OPTIMIZER_GUIDE.md    # 完整使用指南
├── 快速开始.md                      # 5分钟快速上手
└── README_UNIVERSAL.md             # 本文件
```

---

## 📚 使用文档

### 输入要求

#### 1. 数据文件（CSV格式）

必需列：`datetime, open, high, low, close, volume`

```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00,42000,42500,41800,42300,1000000
2024-01-01 01:00:00,42300,42800,42200,42700,950000
```

#### 2. 策略文件（Python脚本）

必须继承 `backtrader.Strategy`，在 `params` 中定义参数：

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('period', 20),      # 参数1
        ('threshold', 0.02), # 参数2
    )
    
    def __init__(self):
        # 初始化指标
        pass
    
    def next(self):
        # 交易逻辑
        pass
```

详细策略编写指南见 `example_strategy.py`

### 优化目标

| 目标 | 说明 | 推荐场景 |
|------|------|----------|
| `sharpe_ratio` | 夏普比率 | ⭐ 推荐，平衡收益与风险 |
| `annual_return` | 年化收益率 | 追求高收益 |
| `calmar_ratio` | 卡玛比率 | 收益/回撤 |
| `sortino_ratio` | 索提诺比率 | 关注下行风险 |
| `max_drawdown` | 最大回撤 | 风险控制 |

### LLM配置

#### OpenAI

```python
from universal_llm_client import UniversalLLMConfig

llm_config = UniversalLLMConfig(
    api_type="openai",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4",
    api_key="sk-your-key"
)
```

#### Ollama（本地）

```python
llm_config = UniversalLLMConfig(
    api_type="ollama",
    base_url="http://localhost:11434",
    model_name="qwen",
    api_key=""
)
```

#### 自定义API

```python
llm_config = UniversalLLMConfig(
    api_type="custom",
    base_url="https://your-api.com/chat",
    model_name="your-model",
    api_key="your-key"
)
```

### 输出格式

优化完成后生成JSON文件：

```json
{
  "optimization_info": {
    "asset_name": "BTC",
    "strategy_name": "MyStrategy",
    "optimization_objective": "sharpe_ratio",
    "optimization_time": "2024-01-15 10:30:00",
    "data_range": {...}
  },
  "best_parameters": {
    "period": 20,
    "threshold": 0.02
  },
  "performance_metrics": {
    "sharpe_ratio": 1.85,
    "annual_return": 35.2,
    "max_drawdown": -12.5,
    "total_return": 35.2,
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
    "parameter_explanation": "...",
    "performance_analysis": "...",
    "risk_assessment": "...",
    "practical_suggestions": "...",
    "key_insights": [...]
  }
}
```

---

## 💡 示例

### 示例1：基本优化

```python
from universal_optimizer import UniversalOptimizer

optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/ma_cross.py",
    objective="sharpe_ratio",
    use_llm=False,
    output_dir="./results"
)

result = optimizer.optimize(n_trials=50)
```

### 示例2：使用LLM

```python
from universal_optimizer import UniversalOptimizer
from universal_llm_client import UniversalLLMConfig

llm_config = UniversalLLMConfig(
    api_type="openai",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4",
    api_key="sk-xxx"
)

optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/ma_cross.py",
    objective="sharpe_ratio",
    use_llm=True,
    llm_config=llm_config
)

result = optimizer.optimize(n_trials=50)

# 查看LLM解释
print(result['llm_explanation']['parameter_explanation'])
print(result['llm_explanation']['key_insights'])
```

### 示例3：批量优化

```python
optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/ma_cross.py",
    use_llm=False
)

# 同时优化3个目标
results = optimizer.batch_optimize(
    objectives=["sharpe_ratio", "annual_return", "calmar_ratio"],
    n_trials_per_objective=50
)

# 比较不同目标下的最优参数
for obj, result in results['results'].items():
    print(f"\n目标: {obj}")
    print(f"参数: {result['best_parameters']}")
    print(f"夏普: {result['performance_metrics']['sharpe_ratio']}")
```

### 示例4：命令行批量优化

```bash
python run_universal_optimizer.py \
    --data data/ETH.csv \
    --strategy strategies/rsi.py \
    --batch \
    --objectives sharpe_ratio annual_return \
    --trials 100 \
    --use-llm \
    --llm-type openai \
    --llm-model gpt-4 \
    --api-key sk-xxx
```

---

## 🧪 测试

运行测试套件：

```bash
cd /Users/peter/Desktop/量化/project_trend
conda activate quant
python Optimizer/test_universal_optimizer.py
```

测试内容：
- ✅ 基本优化功能
- ✅ LLM辅助优化
- ✅ 批量优化
- ✅ JSON输出格式验证

---

## 📖 详细文档

- **快速开始.md** - 5分钟入门指南
- **UNIVERSAL_OPTIMIZER_GUIDE.md** - 完整使用手册
- **example_strategy.py** - 策略编写指南和示例

---

## ❓ 常见问题

### Q: 如何选择优化目标？

根据交易风格选择：
- **稳健型**：`sharpe_ratio` 或 `calmar_ratio`
- **激进型**：`annual_return`
- **风险厌恶**：最小化 `max_drawdown`

### Q: 需要多少次试验？

建议根据参数数量：
- 2-3个参数：30-50次
- 4-6个参数：50-100次
- 7+个参数：100-200次

### Q: LLM有什么用？

LLM可以：
- 智能推荐参数搜索范围（提高优化效率）
- 解释为什么这组参数有效
- 分析策略在不同市场环境下的表现
- 提供风险评估和实战建议

### Q: 不使用LLM可以吗？

完全可以！不使用LLM时：
- 使用默认参数范围（根据默认值自动生成）
- 优化速度更快
- 仍然会生成完整的JSON结果（但解释部分较简单）

### Q: 如何避免过拟合？

建议：
1. 保留样本外数据验证
2. 不要过度优化（试验次数适中）
3. 关注策略逻辑合理性
4. 查看LLM的风险评估
5. 使用多个优化目标交叉验证

### Q: 支持哪些backtrader指标？

所有backtrader内置指标都支持：
- 移动平均：SMA, EMA, WMA
- 震荡指标：RSI, Stochastic, CCI
- 趋势指标：MACD, ADX, Aroon
- 波动率：BollingerBands, ATR, Keltner
- 更多请参考 [backtrader文档](https://www.backtrader.com/docu/indautoref/)

---

## 🔧 技术栈

- **回测引擎**: Backtrader
- **优化算法**: Optuna (贝叶斯优化)
- **LLM接口**: OpenAI API / Ollama / 自定义
- **数据处理**: Pandas, NumPy
- **输出格式**: JSON

---

## 📄 许可证

MIT License

---

## 🙏 致谢

感谢以下开源项目：
- [Backtrader](https://www.backtrader.com/) - 强大的回测框架
- [Optuna](https://optuna.org/) - 先进的超参数优化库
- [OpenAI](https://openai.com/) - GPT系列模型
- [Ollama](https://ollama.ai/) - 本地LLM解决方案

---

## 📮 联系方式

如有问题或建议，欢迎提出Issue。

---

<div align="center">

**开始你的量化交易优化之旅！** 🚀

[快速开始](#快速开始) • [查看文档](./UNIVERSAL_OPTIMIZER_GUIDE.md) • [运行测试](#测试)

</div>
