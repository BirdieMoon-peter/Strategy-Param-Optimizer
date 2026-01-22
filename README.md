# run_optimizer.py 使用指南

## 📖 简介

`run_optimizer.py` 是一个通用的量化策略优化命令行工具，支持任意标的数据和策略脚本的参数优化。它集成了贝叶斯优化算法和可选的LLM辅助分析功能。

### 核心特性

✅ **通用性强** - 支持任意CSV格式的标的数据和任意Backtrader策略  
✅ **多目标优化** - 支持夏普比率、年化收益率、最大回撤等多种优化目标  
✅ **智能参数空间** - 🆕 自动根据参数类型生成合理的搜索范围，提升优化效率  
✅ **参数空间分析** - 🆕 自动分析优化结果，给出参数空间改进建议  
✅ **选择性优化** - 🆕 支持指定要优化的参数，其他参数保持默认值  
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
| `--params-file` | `-p` | - | 指定要优化的参数列表文件 🆕 |

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

### 基本用法

```bash
# 1. 最简单用法（优化所有参数）
python run_optimizer.py -d data.csv -s strategy.py

# 2. 指定试验次数
python run_optimizer.py -d data.csv -s strategy.py --trials 100

# 3. 只优化指定参数（推荐）
echo "period" > params.txt
echo "devfactor" >> params.txt
python run_optimizer.py -d data.csv -s strategy.py --params-file params.txt

# 4. 更改优化目标
python run_optimizer.py -d data.csv -s strategy.py --objective annual_return

# 5. 使用 LLM 辅助
python run_optimizer.py -d data.csv -s strategy.py --use-llm
```

### 参数文件格式

创建 `params.txt`，每行一个参数名：

```txt
# 这是注释，以 # 开头
period
devfactor
# 空行会被忽略
```

**注意：** 参数名必须与策略中定义的完全一致

### 完整示例

```bash
python run_optimizer.py \
  --data project_trend/data/BTC.csv \
  --strategy project_trend/src/Aberration.py \
  --params-file params.txt \
  --objective sharpe_ratio \
  --trials 100 \
  --output ./results
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

## 📁 输出文件

优化完成后会在输出目录生成两个文件：

```
optimization_results/
├── optimization_BTC_Strategy_20260122_105954.json  # 完整JSON结果
└── optimization_summary.txt                         # 易读的文本摘要
```

**JSON文件包含：**
- 优化信息（标的、策略、目标、时间）
- 最优参数
- 性能指标（夏普比率、收益率、回撤等）
- 逐年表现
- 参数空间分析建议 🆕

**文本摘要包含：**
- 最优参数值
- 关键性能指标
- 逐年表现摘要
- 优化建议

---

## 🔧 策略脚本要求

策略脚本必须：
1. 继承自 `backtrader.Strategy`
2. 使用 `params` 元组定义参数

### 简单示例

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('period', 20),      # 整数参数
        ('threshold', 0.02), # 浮点参数
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.period)
    
    def next(self):
        if not self.position and self.data.close > self.sma * 1.02:
            self.buy()
        elif self.position and self.data.close < self.sma:
            self.sell()
```

**参数命名：** 使用小写+下划线，如 `fast_period`、`stop_loss`

---

## ❓ 常见问题

### Q1: 如何只优化部分参数？

创建 `params.txt` 文件，每行一个参数名，然后使用 `--params-file` 参数。

**适用场景：** 策略参数多（>5个）、已知某些参数合理值、加快优化速度

### Q2: 参数空间分析建议如何理解？

优化后系统会提示参数是否在边界：
- **在边界** → 扩大搜索范围重新优化
- **在中间** → 参数空间设置合理
- **都在边界** → 可能策略逻辑有问题

### Q3: 优化运行慢怎么办？

- 减少试验次数：`--trials 20`
- 使用参数文件只优化关键参数
- 使用静默模式：`--quiet`

### Q4: 数据文件格式错误？

确保CSV包含必需列：`datetime/date, open, high, low, close, volume`

列名不同时预处理：
```python
import pandas as pd
df = pd.read_csv('original.csv')
df.rename(columns={'时间': 'datetime', '开盘': 'open'}, inplace=True)
df.to_csv('processed.csv', index=False)
```

### Q5: 如何批量优化？

```bash
for asset in BTC ETH SOL; do
  python run_optimizer.py -d data/${asset}.csv -s strategy.py -O results/${asset}
done
```

---

## 🎯 进阶用法

### 1. 迭代优化

第一轮发现参数在边界 → 第二轮针对性优化：

```bash
# 第一轮
python run_optimizer.py -d data.csv -s strategy.py --trials 50

# 如果 period 在边界，第二轮只优化它
echo "period" > params.txt
python run_optimizer.py -d data.csv -s strategy.py --params-file params.txt --trials 100
```

### 2. 参数敏感性分析

逐个优化参数，找出影响大的关键参数：

```bash
# 只优化 period
echo "period" > params_period.txt
python run_optimizer.py -d data.csv -s strategy.py --params-file params_period.txt

# 只优化 devfactor
echo "devfactor" > params_devfactor.txt
python run_optimizer.py -d data.csv -s strategy.py --params-file params_devfactor.txt

# 比较性能提升，找出敏感参数
```

### 3. 并行优化多标的

```bash
# 使用 GNU Parallel
parallel python run_optimizer.py -d data/{}.csv -s strategy.py -O results/{} \
  ::: BTC ETH SOL

# 或使用 bash 循环
for asset in BTC ETH SOL; do
  python run_optimizer.py -d data/${asset}.csv -s strategy.py -O results/${asset} &
done
wait
```

### 4. 参数稳定性验证

多次运行检验参数稳定性：

```bash
for i in {1..5}; do
  python run_optimizer.py -d data.csv -s strategy.py -O results/run_${i}
done
# 比较各次最优参数是否接近
```

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
- ✅ 策略参数多（>5个）时，考虑使用 `--params-file` 只优化关键参数
- ✅ 优先优化对策略影响大的参数（如周期、阈值）

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

- [参数空间优化指南](参数空间优化指南.md) 🆕
- [Optimizer模块总览](Optimizer/项目总览.md)
- [通用优化器指南](Optimizer/UNIVERSAL_OPTIMIZER_GUIDE.md)
- [Backtrader官方文档](https://www.backtrader.com/docu/)

---

## 🧠 智能参数空间说明

### 自动参数空间生成

系统会根据参数类型和名称自动生成合理的搜索范围：

| 参数类型 | 识别模式 | 默认范围 | 示例 |
|---------|---------|---------|------|
| 周期参数 | `period`, `window`, `length` | [默认值×0.5 ~ 2.5], 限制[5,200] | `period=20` → [10,50] |
| 标准差倍数 | `std`, `devfactor` | [0.5, 5.0] | `devfactor=2.0` → [1.0,4.0] |
| 快速周期 | `fast`, `short` | [3, 50] | `fast_period=10` → [5,20] |
| 慢速周期 | `slow`, `long` | [10, 200] | `slow_period=30` → [15,75] |
| RSI阈值 | `rsi.*sold`, `rsi.*bought` | [10, 90] | `rsi_oversold=30` → [21,39] |

**优势：**
- 🎯 根据参数语义设置合理范围
- ⚡ 避免盲目搜索，提升效率
- 📊 自动处理参数约束（如 fast < slow）
- 💡 优化后提供参数空间改进建议

**详细文档：** 参见 [参数空间优化指南.md](参数空间优化指南.md)

---

## 📜 许可证

本工具是量化交易研究项目的一部分，仅供学习和研究使用。

---

## 🎓 典型工作流

```bash
# 1. 快速测试（确认可运行）
python run_optimizer.py -d data.csv -s strategy.py --trials 20

# 2. 创建参数文件（只优化关键参数）
echo "period" > params.txt
echo "devfactor" >> params.txt

# 3. 正式优化
python run_optimizer.py -d data.csv -s strategy.py --params-file params.txt --trials 100

# 4. 查看结果和建议
cat optimization_results/optimization_summary.txt

# 5. 如有参数在边界，调整后重新优化
# （根据建议修改参数空间或重点优化特定参数）
```

**更多示例：** 参见 [QUICK_START.md](QUICK_START.md)

---

## 🆕 版本更新

### v1.1.0 (2026-01-22)

**新增功能：**
- ✨ 指定参数优化功能（`--params-file`）
- ✨ 智能参数空间自动生成
- ✨ 参数空间使用情况分析
- ✨ 参数约束自动处理（如 fast < slow）
- 📊 优化后自动提供参数空间改进建议

**改进：**
- 🚀 根据参数类型智能设置搜索范围
- 📈 提升优化效率，减少无效搜索
- 💡 新增参数敏感性分析示例
- 📖 完善文档和使用示例

---

**更新时间**: 2026-01-22  
**版本**: 1.1.0  
**作者**: Peter
