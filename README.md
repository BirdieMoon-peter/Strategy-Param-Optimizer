# run_optimizer.py 使用指南

## 📖 简介

`run_optimizer.py` 是一个通用的量化策略优化命令行工具，支持任意标的数据和策略脚本的参数优化。它集成了贝叶斯优化算法和可选的LLM辅助分析功能。

### 核心特性

✅ **通用性强** - 支持任意CSV格式的标的数据和任意Backtrader策略  
✅ **多目标优化** - 支持夏普比率、年化收益率、最大回撤等多种优化目标  
✅ **智能参数空间** - 🆕 自动根据参数类型生成合理的搜索范围，提升优化效率  
✅ **参数空间分析** - 🆕 自动分析优化结果，给出参数空间改进建议  
✅ **LLM集成** - 可选集成大语言模型进行智能参数分析  
✅ **命令行友好** - 简单易用的命令行接口，支持批处理  
✅ **详细输出** - 生成JSON格式结果和可读的文本摘要

---

## 🚀 快速开始

### 最简单的用法

```bash
python run_optimizer.py -d project_trend/data/AG.csv -s project_trend/src/Aberration.py
```

这将使用默认参数（夏普比率优化，50次试验）对AG标的运行Aberration策略的参数优化。

---

## 📦 环境要求

### 依赖包

确保已安装以下Python包：

```bash
pip install pandas backtrader optuna matplotlib requests
```

或使用项目的 `requirements.txt`：

```bash
cd Optimizer
pip install -r requirements.txt
```

### Python版本

- Python 3.8 或更高版本

---

## 📋 参数说明

### 必需参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--data` | `-d` | 标的数据CSV文件路径 | `project_trend/data/BTC.csv` |
| `--strategy` | `-s` | 策略脚本文件路径 | `project_trend/src/Aberration.py` |

### 优化参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--objective` | `-o` | `sharpe_ratio` | 优化目标 |
| `--trials` | `-t` | `50` | 优化试验次数 |

**可选的优化目标：**
- `sharpe_ratio` - 夏普比率（默认，推荐）
- `annual_return` - 年化收益率
- `total_return` - 总收益率
- `max_drawdown` - 最大回撤（最小化）
- `calmar_ratio` - 卡玛比率
- `sortino_ratio` - 索提诺比率

### LLM参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-llm` | `False` | 是否使用LLM辅助优化 |
| `--llm-type` | `ollama` | LLM类型（ollama/openai/custom） |
| `--llm-model` | `xuanyuan` | LLM模型名称 |
| `--llm-url` | `http://localhost:11434` | LLM API URL |
| `--api-key` | - | API密钥（OpenAI需要） |
| `--timeout` | `180` | LLM请求超时时间（秒） |

### 输出参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-O` | `./optimization_results` | 输出目录 |
| `--quiet` | `-q` | `False` | 静默模式（减少输出） |

---

## 💡 使用示例

### 1. 基本用法（不使用LLM）

最简单的调用方式，使用默认配置：

```bash
python run_optimizer.py \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py
```

### 2. 指定优化目标

优化年化收益率而不是夏普比率：

```bash
python run_optimizer.py \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --objective annual_return
```

### 3. 调整试验次数

增加试验次数以获得更好的结果（但需要更长时间）：

```bash
python run_optimizer.py \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py \
  --trials 100
```

### 4. 使用本地Ollama LLM

启用LLM辅助优化（需要先启动Ollama服务）：

```bash
# 先启动Ollama（在另一个终端）
ollama serve

# 运行优化
python run_optimizer.py \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --use-llm
```

### 5. 使用OpenAI API

使用OpenAI的GPT模型进行LLM辅助：

```bash
python run_optimizer.py \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py \
  --use-llm \
  --llm-type openai \
  --llm-model gpt-4 \
  --api-key sk-your-api-key-here
```

### 6. 指定输出目录

将结果保存到自定义目录：

```bash
python run_optimizer.py \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --output ./my_optimization_results
```

### 7. 静默模式

减少输出信息，适合批处理：

```bash
python run_optimizer.py \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py \
  --quiet
```

### 8. 完整参数示例

使用所有主要参数的完整示例：

```bash
python run_optimizer.py \
  --data project_trend/data/BTC.csv \
  --strategy project_trend/src/Aberration.py \
  --objective sharpe_ratio \
  --trials 100 \
  --use-llm \
  --llm-type ollama \
  --llm-model xuanyuan \
  --output ./results \
  --quiet
```

---

## 📊 数据格式要求

### CSV文件格式

您的数据CSV文件必须包含以下列（列名不区分大小写）：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `datetime` 或 `date` | 日期时间 | 时间戳 | `2024-01-01 09:30:00` |
| `open` | 浮点数 | 开盘价 | `100.5` |
| `high` | 浮点数 | 最高价 | `102.3` |
| `low` | 浮点数 | 最低价 | `99.8` |
| `close` | 浮点数 | 收盘价 | `101.2` |
| `volume` | 整数 | 成交量 | `1000000` |

### 示例CSV文件

```csv
date,open,high,low,close,volume
2024-01-01,100.0,102.0,99.0,101.0,1000000
2024-01-02,101.0,103.0,100.5,102.5,1200000
2024-01-03,102.5,104.0,102.0,103.5,1100000
```

**注意：**
- 脚本会自动将 `date` 列重命名为 `datetime`
- 日期格式会自动解析
- 数据会自动按时间排序

---

## 📁 输出文件说明

### 输出目录结构

```
optimization_results/
├── optimization_BTC_AberrationStrategy_20260122_105954.json  # 完整JSON结果
└── optimization_summary.txt                                   # 文本摘要
```

### JSON文件内容

```json
{
  "optimization_info": {
    "asset_name": "BTC",
    "strategy_name": "AberrationStrategy",
    "optimization_objective": "sharpe_ratio",
    "optimization_time": "2026-01-22 10:59:54",
    "data_range": {
      "start": "2017-08-17",
      "end": "2025-12-31",
      "total_days": 3059
    }
  },
  "best_parameters": {
    "period": 103,
    "std_dev_upper": 2.47,
    "std_dev_lower": 3.46,
    "percent": 0.35,
    "allow_short": 2
  },
  "performance_metrics": {
    "sharpe_ratio": 1.3385,
    "annual_return": 10.51,
    "max_drawdown": 19.53,
    "total_return": 235.65,
    "final_value": 335648.71,
    "trades_count": 11,
    "win_rate": 72.73
  },
  "yearly_performance": {
    "2017": {"return": 12.17, "drawdown": 15.58, "sharpe_ratio": 1.0225},
    "2018": {"return": 8.44, "drawdown": 5.24, "sharpe_ratio": 0.8543},
    ...
  },
  "llm_explanation": {
    "parameter_explanation": "参数优化完成，以上为最优参数组合",
    "key_insights": [
      "优化目标: sharpe_ratio",
      "回测期: 2017-08-17 至 2025-12-31",
      "历史表现不代表未来收益"
    ]
  }
}
```

### 文本摘要内容

`optimization_summary.txt` 包含易读的优化结果摘要：

```
============================================================
策略优化结果摘要
============================================================

优化时间: 2026-01-22 10:59:54
标的: BTC
策略: AberrationStrategy
优化目标: sharpe_ratio

【最优参数】
  period: 103
  std_dev_upper: 2.4702
  std_dev_lower: 3.455
  percent: 0.3474
  allow_short: 2.0

【性能指标】
  sharpe_ratio: 1.3385
  annual_return: 10.51
  max_drawdown: 19.53
  ...
```

---

## 🔧 策略脚本要求

### 基本要求

您的策略脚本必须：

1. 定义一个继承自 `backtrader.Strategy` 的策略类
2. 使用 `params` 定义可优化的参数

### 策略示例

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    """我的自定义策略"""
    
    params = (
        ('period', 20),           # 周期参数
        ('threshold', 0.02),      # 阈值参数
        ('stop_loss', 0.05),      # 止损参数
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.period)
    
    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0] * (1 + self.params.threshold):
                self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.sell()
```

### 参数命名规范

- 参数名使用小写字母和下划线
- 整数参数（如周期）会在 [min, max] 范围内以整数步长搜索
- 浮点数参数会在 [min, max] 范围内连续搜索

---

## ❓ 常见问题

### Q1: 脚本运行很慢怎么办？

**A:** 可以减少试验次数：

```bash
python run_optimizer.py -d data.csv -s strategy.py --trials 20
```

或使用静默模式减少输出开销：

```bash
python run_optimizer.py -d data.csv -s strategy.py --quiet
```

### Q2: LLM连接超时怎么办？

**A:** 增加超时时间：

```bash
python run_optimizer.py -d data.csv -s strategy.py --use-llm --timeout 300
```

或检查Ollama服务是否正常运行：

```bash
curl http://localhost:11434/api/tags
```

### Q3: 数据文件格式错误怎么办？

**A:** 确保CSV文件包含必需的列：`datetime/date, open, high, low, close, volume`

如果列名不同，可以预处理数据：

```python
import pandas as pd
df = pd.read_csv('original.csv')
df = df.rename(columns={'时间': 'datetime', '开盘': 'open', ...})
df.to_csv('processed.csv', index=False)
```

### Q4: 如何批量优化多个标的？

**A:** 使用bash脚本循环：

```bash
#!/bin/bash
for asset in BTC ETH SOL; do
  python run_optimizer.py \
    -d "project_trend/data/${asset}.csv" \
    -s "project_trend/src/Aberration.py" \
    --output "./results/${asset}"
done
```

### Q5: 优化结果不理想怎么办？

**A:** 尝试以下方法：

1. **增加试验次数**：`--trials 200`
2. **更改优化目标**：`--objective annual_return`
3. **使用LLM辅助**：`--use-llm`
4. **检查策略逻辑**：确保策略参数范围合理
5. **检查数据质量**：确保数据完整无误

### Q6: 如何解读年度表现？

**A:** 输出中的年度表现包括：

- **收益** - 该年的收益率（%）
- **回撤** - 该年的最大回撤（%）
- **夏普** - 该年的夏普比率

如果某年显示 "无交易"，说明该年策略未产生交易信号。

---

## 🎯 进阶用法

### 1. 与其他工具集成

#### 与Jupyter Notebook集成

```python
import subprocess
import json

# 运行优化
result = subprocess.run([
    'python', 'run_optimizer.py',
    '-d', 'data/BTC.csv',
    '-s', 'strategies/my_strategy.py',
    '--quiet'
], capture_output=True, text=True)

# 读取结果
with open('optimization_results/optimization_*.json') as f:
    data = json.load(f)
    
print(f"最佳夏普比率: {data['performance_metrics']['sharpe_ratio']}")
```

#### 与Airflow集成

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG('strategy_optimization', start_date=datetime(2024, 1, 1))

optimize_task = BashOperator(
    task_id='optimize_strategy',
    bash_command='python run_optimizer.py -d data.csv -s strategy.py',
    dag=dag
)
```

### 2. 自定义优化目标

如果需要自定义优化目标，可以修改 `Optimizer/config.py`，添加新的目标函数。

### 3. 并行优化

使用GNU Parallel进行多策略并行优化：

```bash
parallel python run_optimizer.py -d data/{1}.csv -s src/{2}.py -O results/{1}_{2} \
  ::: BTC ETH SOL \
  ::: Aberration Bollinger Keltner
```

### 4. 参数敏感性分析

连续运行多次优化，分析参数稳定性：

```bash
for i in {1..10}; do
  python run_optimizer.py -d data.csv -s strategy.py --output "results/run_${i}"
done
```

然后分析所有运行的最优参数分布。

---

## 📝 最佳实践

### 1. 数据准备

- ✅ 确保数据完整、无缺失值
- ✅ 数据按时间正序排列
- ✅ 检查异常值和错误数据点
- ✅ 使用足够长的历史数据（至少2年）

### 2. 参数设置

- ✅ 从较少的试验次数开始（20-50次）
- ✅ 根据初步结果调整试验次数
- ✅ 选择合适的优化目标（通常用夏普比率）
- ✅ 对于快速测试使用 `--quiet` 模式

### 3. 结果验证

- ✅ 检查年度表现的稳定性
- ✅ 关注交易次数（过少或过多都不好）
- ✅ 注意过拟合风险（过于完美的结果）
- ✅ 在样本外数据上验证结果
- ✅ 考虑实际交易成本和滑点

### 4. LLM使用建议

- ✅ 仅在参数空间复杂时使用LLM
- ✅ 本地Ollama适合频繁使用
- ✅ OpenAI API适合高质量分析但成本较高
- ✅ 增加超时时间避免连接问题

---

## 📞 技术支持

### 查看帮助信息

```bash
python run_optimizer.py --help
```

### 调试模式

如果遇到问题，移除 `--quiet` 参数以查看详细输出：

```bash
python run_optimizer.py -d data.csv -s strategy.py
```

### 相关文档

- [Optimizer模块总览](Optimizer/项目总览.md)
- [通用优化器指南](Optimizer/UNIVERSAL_OPTIMIZER_GUIDE.md)
- [Backtrader官方文档](https://www.backtrader.com/docu/)

---

## 📜 许可证

本工具是量化交易研究项目的一部分，仅供学习和研究使用。

---

## 🎓 示例工作流

### 完整的策略优化流程

```bash
# 1. 准备环境
conda activate quant

# 2. 快速测试（10次试验）
python run_optimizer.py \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --trials 10

# 3. 如果结果合理，增加试验次数
python run_optimizer.py \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --trials 100 \
  --output ./results/btc_aberration

# 4. 尝试其他优化目标
python run_optimizer.py \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --objective annual_return \
  --trials 100 \
  --output ./results/btc_aberration_return

# 5. 使用LLM进行深度分析
python run_optimizer.py \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --use-llm \
  --trials 100 \
  --output ./results/btc_aberration_llm

# 6. 比较结果
ls -lh results/*/optimization_*.json
```

---

**更新时间**: 2026-01-22  
**版本**: 1.0.0  
**作者**: Peter
