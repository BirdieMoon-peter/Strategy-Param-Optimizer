# run_optimizer.py 使用指南

## 📖 简介

`run_optimizer.py` 是一个通用的量化策略优化命令行工具，支持任意标的数据和策略脚本的参数优化。它集成了贝叶斯优化算法和可选的LLM辅助分析功能。

### 核心特性

✅ **通用性强** - 支持任意CSV格式的标的数据和任意Backtrader策略  
✅ **多数据源支持** - 🆕 支持多个CSV文件输入，适用于配对交易、跨市场套利等策略  
✅ **多目标优化** - 支持夏普比率、年化收益率、最大回撤等多种优化目标  
✅ **智能参数空间** - 🆕 自动根据参数类型生成合理的搜索范围，提升优化效率  
✅ **参数空间分析** - 🆕 自动分析优化结果，给出参数空间改进建议  
✅ **选择性优化** - 🆕 支持指定要优化的参数，其他参数保持默认值  
✅ **专业性能指标** - 🆕 使用 empyrical 库计算标准化的性能指标  
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
| `--data` | `-d` | 标的数据CSV文件路径（支持多次指定） | `-d data1.csv -d data2.csv` |
| `--strategy` | `-s` | 策略脚本文件路径 | `project_trend/src/Aberration.py` |

**多数据源支持 🆕**

对于需要多个数据源的策略（如配对交易、跨市场套利等），可以多次使用 `-d` 参数指定多个CSV文件：

```bash
# 示例：使用QQQ和TQQQ两个数据源
python run_optimizer.py \
  -d data_1m_QQQ.csv \
  -d data_1m_TQQQ.csv \
  -n QQQ -n TQQQ \
  -s multivwap2.py
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `--data-names` | `-n` | 数据源名称，与`--data`一一对应（可选） |

- 如果不指定 `--data-names`，系统会自动使用文件名作为数据源名称
- 在策略中可以通过 `self.datas[0]`、`self.datas[1]` 等访问不同的数据源

### 优化参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--objective` | `-o` | `sharpe_ratio` | 优化目标 |
| `--trials` | `-t` | `50` | 优化试验次数 |
| `--params-file` | `-p` | - | 指定要优化的参数列表文件 |
| `--space-config` | `-S` | - | 手动指定参数空间配置（JSON文件）🆕 |

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

### 多数据源策略优化 🆕

对于需要多个数据源的策略（如配对交易、跨市场套利、杠杆ETF策略等），可以指定多个CSV文件：

```bash
# 示例1：QQQ + TQQQ 杠杆ETF策略
python run_optimizer.py \
  -d data_1m_QQQ.csv \
  -d data_1m_TQQQ.csv \
  -n QQQ -n TQQQ \
  -s multivwap2.py \
  --trials 100

# 示例2：配对交易策略（不指定data-names，使用文件名）
python run_optimizer.py \
  -d gold.csv \
  -d silver.csv \
  -s pairs_trading.py

# 示例3：多市场套利策略
python run_optimizer.py \
  -d btc_binance.csv \
  -d btc_coinbase.csv \
  -d btc_kraken.csv \
  -n binance -n coinbase -n kraken \
  -s arbitrage_strategy.py
```

**注意事项：**
- 所有CSV文件必须包含相同的列（datetime, open, high, low, close, volume）
- 数据源的顺序很重要，在策略中通过 `self.datas[0]`、`self.datas[1]` 等按顺序访问
- 如果指定 `-n` 参数，数量必须与 `-d` 参数一致
- 时间范围会自动对齐到所有数据源的交集

### 参数文件格式

创建 `params.txt`，每行一个参数名：

```txt
# 这是注释，以 # 开头
period
devfactor
# 空行会被忽略
```

**注意：** 参数名必须与策略中定义的完全一致

### 手动配置参数空间 🆕

当正则表达式无法匹配参数、LLM 无法给出合适的参数空间，或需要精确控制搜索范围时，可以使用 JSON 配置文件手动指定参数空间。

#### 📝 使用场景

1. **正则表达式搜不到参数**：参数名不符合内置规则模式
2. **LLM 返回不合适**：LLM 给出的范围不符合预期
3. **需要精确控制**：根据经验或分析需要特定搜索范围
4. **特殊参数类型**：参数有特殊的业务约束

#### 🚀 快速开始

```bash
# 使用自定义参数空间
python run_optimizer.py -d data.csv -s strategy.py --space-config my_space_config.json

# 或使用简写
python run_optimizer.py -d data.csv -s strategy.py -S my_space_config.json
```

#### 📋 配置文件格式

创建 JSON 配置文件（如 `my_space_config.json`）：

```json
{
    "param_space": {
        "period": {
            "min": 10,
            "max": 50,
            "step": 1,
            "distribution": "int_uniform"
        },
        "devfactor": {
            "min": 1.0,
            "max": 4.0,
            "step": null,
            "distribution": "uniform"
        }
    }
}
```

#### ⚙️ 配置字段说明

| 字段 | 类型 | 必需 | 说明 | 示例 |
|------|------|------|------|------|
| `min` | number | ✅ | 参数最小值 | `10` |
| `max` | number | ✅ | 参数最大值 | `50` |
| `step` | number/null | 可选 | 步长（整型必需，浮点型可设为 `null`） | `1` 或 `null` |
| `distribution` | string | 可选 | 分布类型 | `"int_uniform"` |
| `description` | string | 可选 | 参数描述（仅用于文档） | `"周期参数"` |

**分布类型（distribution）：**
- `int_uniform` - 整数均匀分布（整型参数，如周期、窗口）
- `uniform` - 连续均匀分布（浮点型参数，如倍数、比率）
- `log_uniform` - 对数均匀分布（跨数量级参数，如学习率）

#### 💡 配置示例

**示例 1：布林带策略参数**

```json
{
    "param_space": {
        "period": {
            "min": 15,
            "max": 30,
            "step": 1,
            "distribution": "int_uniform"
        },
        "devfactor": {
            "min": 1.5,
            "max": 3.0,
            "step": null,
            "distribution": "uniform"
        }
    }
}
```

**示例 2：RSI 策略参数**

```json
{
    "param_space": {
        "rsi_period": {
            "min": 5,
            "max": 30,
            "step": 1,
            "distribution": "int_uniform"
        },
        "rsi_oversold": {
            "min": 20,
            "max": 40,
            "step": 1,
            "distribution": "int_uniform"
        },
        "rsi_overbought": {
            "min": 60,
            "max": 80,
            "step": 1,
            "distribution": "int_uniform"
        }
    }
}
```


#### 🎯 使用技巧

1. **只配置需要的参数**：未配置的参数会自动使用智能规则生成范围
2. **混合使用**：可以只配置部分参数，其他参数使用自动规则
3. **保存配置**：将常用配置保存为模板，方便复用
4. **验证配置**：运行前检查 `min < max`，确保配置合理

#### ❓ 常见问题

**Q: 配置文件中的参数名必须和策略中完全一致吗？**  
A: 是的，参数名必须与策略类中定义的参数名完全一致（大小写敏感）。

**Q: 如果配置了不存在的参数会怎样？**  
A: 系统会忽略不存在的参数，只应用存在的参数配置。

**Q: 可以只配置部分参数吗？**  
A: 可以，未配置的参数会使用智能规则自动生成范围。

**Q: `step` 什么时候设为 `null`？**  
A: 浮点型参数通常设为 `null`（连续值），整型参数需要指定步长（通常为 1）。

#### 📁 参考文件

- 示例配置文件：`my_space_config.json`
- 完整示例：查看项目根目录下的配置文件示例

### 完整示例

**示例 1：基本优化**
```bash
python run_optimizer.py \
  --data project_trend/data/BTC.csv \
  --strategy project_trend/src/Aberration.py \
  --params-file params.txt \
  --objective sharpe_ratio \
  --trials 100 \
  --output ./results
```

**示例 2：使用自定义参数空间**
```bash
python run_optimizer.py \
  --data project_trend/data/BTC.csv \
  --strategy project_trend/src/Aberration.py \
  --space-config my_space_config.json \
  --objective sharpe_ratio \
  --trials 100 \
  --output ./results
```

**示例 3：组合使用（指定参数 + 自定义空间）**
```bash
# 只优化 period 和 devfactor，并使用自定义范围
python run_optimizer.py \
  --data project_trend/data/BTC.csv \
  --strategy project_trend/src/Aberration.py \
  --params-file params.txt \
  --space-config my_space_config.json \
  --trials 100
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

### Q6: 正则表达式搜不到参数怎么办？

如果参数名不符合内置规则模式（如 `custom_param`、`param1`），系统会自动使用默认规则（0.5x - 2.0x）。

**解决方案：**
1. **使用自定义参数空间配置（推荐）**：
   ```bash
   # 创建配置文件 my_space_config.json
   {
       "param_space": {
           "custom_param": {
               "min": 10,
               "max": 100,
               "step": 1,
               "distribution": "int_uniform"
           }
       }
   }
   
   # 使用配置
   python run_optimizer.py -d data.csv -s strategy.py --space-config my_space_config.json
   ```

2. **修改参数名**：将参数名改为符合规则的模式（如 `custom_period` → 会匹配周期规则）

3. **使用默认规则**：如果默认范围（0.5x - 2.0x）可以接受，直接使用即可

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

## 🧠 智能参数空间与优化原理

### 1. 参数空间确定方法

系统使用**智能规则匹配**自动确定参数的搜索范围：

#### 🔍 参数识别流程

```
策略参数 → 提取参数名和默认值 → 规则匹配 → 计算搜索范围
```

**计算公式：**
```
min_value = max(默认值 × min_multiplier, min_absolute)
max_value = min(默认值 × max_multiplier, max_absolute)
```

#### 📊 内置识别规则

| 参数类型 | 识别模式 | 相对范围 | 绝对边界 | 示例 |
|---------|---------|---------|---------|------|
| 周期参数 | `period`, `window`, `length` | 0.5x ~ 2.5x | [5, 200] | `period=20` → [10, 50] |
| 标准差倍数 | `std`, `devfactor` | 0.5x ~ 2.0x | [0.5, 5.0] | `devfactor=2.0` → [1.0, 4.0] |
| 快速周期 | `fast`, `short` | 0.5x ~ 2.0x | [3, 50] | `fast_period=10` → [5, 20] |
| 慢速周期 | `slow`, `long` | 0.5x ~ 2.0x | [10, 200] | `slow_period=30` → [15, 60] |
| RSI阈值 | `rsi.*sold`, `rsi.*bought` | 0.7x ~ 1.3x | [10, 90] | `rsi_oversold=30` → [21, 39] |
| 阈值类 | `threshold`, `limit` | 0.5x ~ 2.0x | [0.01, 0.5] | `threshold=0.05` → [0.025, 0.1] |
| 未识别参数 | - | 0.5x ~ 2.0x | int: [1, ∞] / float: [0.0001, ∞] | `custom=100` → [50, 200] |

#### 🔍 参数归类机制

系统通过**正则表达式匹配参数名称**自动将参数归类到对应的规则：

**归类流程：**
```
提取参数名 → 遍历所有规则 → 正则匹配 → 应用第一个匹配的规则
```

**匹配算法：**
```python
# 对每个参数，遍历所有内置规则
for rule_name, rule in BUILTIN_RULES.items():
    if re.search(rule.param_pattern, param.name, re.IGNORECASE):
        # 找到匹配的规则，应用该规则
        matched_rule = rule
        break
```

**正则表达式模式说明：**

| 规则类型 | 正则模式 | 匹配逻辑 | 匹配示例 |
|---------|---------|---------|---------|
| 周期类 | `.*period.*\|.*window.*\|.*length.*` | 参数名包含 "period"、"window" 或 "length" | `period`, `ma_period`, `window_size`, `lookback_length` |
| 标准差 | `.*std.*\|.*dev.*factor.*` | 包含 "std" 或 "dev"+"factor" | `std_dev`, `devfactor`, `stddev`, `dev_factor` |
| 快速周期 | `.*fast.*\|.*short.*` | 包含 "fast" 或 "short" | `fast_period`, `short_ma`, `fast_window` |
| 慢速周期 | `.*slow.*\|.*long.*` | 包含 "slow" 或 "long" | `slow_period`, `long_ma`, `slow_window` |
| RSI阈值 | `.*rsi.*sold.*\|.*rsi.*bought.*` | 包含 "rsi"+"sold" 或 "rsi"+"bought" | `rsi_oversold`, `rsi_overbought` |
| 止损 | `.*stop.*loss.*\|.*sl.*` | 包含 "stop"+"loss" 或 "sl" | `stop_loss`, `stopLoss`, `sl_percent` |
| 止盈 | `.*take.*profit.*\|.*tp.*` | 包含 "take"+"profit" 或 "tp" | `take_profit`, `takeProfit`, `tp_percent` |

**关键特性：**
- ✅ **大小写不敏感**：使用 `re.IGNORECASE` 标志，`Period`、`PERIOD`、`period` 都能匹配
- ✅ **部分匹配**：只要参数名包含关键词即可，如 `ma_period` 会匹配周期类规则
- ✅ **第一个匹配优先**：如果参数名同时匹配多个规则，使用第一个匹配的规则
- ✅ **未匹配使用默认**：如果所有规则都不匹配，使用保守的默认规则（0.5x - 2.0x）

**归类示例：**

```
参数名: "period"
  → 匹配规则 "period" (.*period.*) ✓
  → 归类：周期类
  → 应用范围：[默认值×0.5, 默认值×2.5]，限制[5, 200]

参数名: "devfactor"
  → 匹配规则 "std_dev" (.*dev.*factor.*) ✓
  → 归类：标准差倍数
  → 应用范围：[默认值×0.5, 默认值×2.0]，限制[0.5, 5.0]

参数名: "fast_ma"
  → 匹配规则 "fast_period" (.*fast.*) ✓
  → 归类：快速周期
  → 应用范围：[默认值×0.5, 默认值×2.0]，限制[3, 50]

参数名: "custom_param"
  → 所有规则都不匹配 ✗
  → 归类：未识别（使用默认规则）
  → 应用范围：[默认值×0.5, 默认值×2.0]
```

**实际归类过程演示：**

假设策略有以下参数：
```python
策略参数:
  - period = 20
  - devfactor = 2.0
  - fast_ma = 10
  - slow_ma = 30
  - custom_param = 100
```

归类结果：
```
1. period
   → 匹配 "period" 规则 (.*period.*)
   → 归类：周期类
   → 范围：[10, 50]

2. devfactor
   → 匹配 "std_dev" 规则 (.*dev.*factor.*)
   → 归类：标准差倍数
   → 范围：[1.0, 4.0]

3. fast_ma
   → 匹配 "fast_period" 规则 (.*fast.*)
   → 归类：快速周期
   → 范围：[5, 20]

4. slow_ma
   → 匹配 "slow_period" 规则 (.*slow.*)
   → 归类：慢速周期
   → 范围：[15, 60]

5. custom_param
   → 所有规则都不匹配
   → 归类：未识别（默认规则）
   → 范围：[50, 200]
```

#### ⚙️ 双重约束机制

系统同时使用**相对倍数**和**绝对边界**来确保参数范围合理：

```
示例 1：周期参数 period=20
  相对范围：20 × [0.5, 2.5] = [10, 50]
  绝对边界：[5, 200]
  最终范围：max(10, 5) ~ min(50, 200) = [10, 50] ✓

示例 2：周期参数 period=150 (默认值过大)
  相对范围：150 × [0.5, 2.5] = [75, 375]
  绝对边界：[5, 200]
  最终范围：max(75, 5) ~ min(375, 200) = [75, 200] ✓
  → 绝对边界防止搜索范围过大
```

### 2. 智能搜索过程

系统使用 **TPE (Tree-structured Parzen Estimator)** 贝叶斯优化算法进行智能参数搜索。

#### 🎯 搜索流程

```
┌─────────────────────────────────────────────────┐
│  阶段 1：随机探索 (前 10 次试验)                  │
│  目的：获取初始数据，探索参数空间                 │
└─────────────────────────────────────────────────┘
        ↓
Trial 1-10: 在参数范围内均匀随机采样
  例：period ∈ [10,50], devfactor ∈ [1.0,4.0]
      Trial 1: {period: 15, devfactor: 1.8} → Sharpe: 0.52
      Trial 2: {period: 38, devfactor: 3.1} → Sharpe: 0.61
      ...

┌─────────────────────────────────────────────────┐
│  阶段 2：智能采样 (Trial 11+)                     │
│  目的：利用历史信息，集中搜索高价值区域           │
└─────────────────────────────────────────────────┘
        ↓
步骤 1：划分好坏结果
  - 按优化目标排序所有历史试验
  - 前 20% 标记为"好结果"
  - 后 80% 标记为"坏结果"

步骤 2：构建概率模型
  - l(x) = 产生"好结果"的参数分布 (高斯核密度估计)
  - g(x) = 产生"坏结果"的参数分布 (高斯核密度估计)

步骤 3：计算期望改进 (EI)
  - EI(x) = l(x) / g(x)
  - 高 EI 值 = 该参数组合更可能产生好结果

步骤 4：在高 EI 区域采样
  - 选择 EI 最高的参数组合进行下一次试验
  - 更新历史记录，重复步骤 1-4
```

#### 📈 概率模型示例

假设优化 `period` 参数，已完成 20 次试验：

```
好结果对应的 period 值：[35, 38, 40, 42]
→ 建立概率分布 l(period)
  期望: 38.75, 标准差: 3.0
  
  概率密度
     ↑
     |         ╱╲
     |        ╱  ╲      ← 集中在 35-42
     |       ╱    ╲
     └──────────────→ period
          30  40  50

坏结果对应的 period 值：[15, 22, 28, 45, 50, ...]
→ 建立概率分布 g(period)
  期望: 32.0, 标准差: 13.5
  
  概率密度
     ↑
     |      ╱─────╲    ← 分布很分散
     |     ╱       ╲
     └──────────────→ period
          20  40  60

计算 EI 并采样：
  period=39: EI = l(39)/g(39) = 0.13/0.03 = 4.3 ✓ 高优先级
  period=18: EI = l(18)/g(18) = 0.01/0.05 = 0.2   低优先级
  
→ 下一次试验选择 period≈39
```

#### 🔄 迭代优化

```
Trial 11: 根据概率模型建议 → 测试 → 更新历史
Trial 12: 根据新历史重新建模 → 测试 → 更新历史
Trial 13: ...
→ 持续改进，逐渐收敛到最优参数
```

### 3. 为什么这样做是智能的？

| 特性 | 随机搜索 | TPE 智能搜索 |
|------|---------|-------------|
| **利用历史** | ❌ 不考虑历史结果 | ✅ 基于历史构建概率模型 |
| **搜索效率** | 到处乱试，效率低 | 集中在高价值区域，效率高 |
| **收敛速度** | 慢，需要大量试验 | 快，20-50 次通常足够 |
| **适应性** | 固定策略 | 自适应调整搜索方向 |
| **结果质量** | 依赖运气 | 稳定找到较优解 |

**实际对比：**
```
随机搜索 50 次：
  最优 Sharpe: 0.68 (运气好才能找到)

TPE 搜索 50 次：
  最优 Sharpe: 0.89 (基于智能采样，稳定达到)
  
效率提升：约 30-50%
```

### 4. 参数约束处理

系统自动识别并处理参数间的约束关系：

```python
# 快速/慢速周期约束
fast_period < slow_period
→ 自动调整：fast_max < slow_min

# RSI 阈值约束
rsi_oversold < rsi_overbought
→ 自动调整：oversold_max=45, overbought_min=55
```

### 5. 优化结果分析

优化完成后，系统会自动分析参数空间使用情况：

```
参数空间分析：
  period = 48 (范围 [10, 50])
  → 接近上界，建议扩大范围到 [10, 80]
  
  devfactor = 2.1 (范围 [1.0, 4.0])
  → 位于中间区域，范围合理 ✓
```

**详细文档：** 参见 [参数空间优化指南.md](参数空间优化指南.md)

---

## 📜 许可证

本工具是量化交易研究项目的一部分，仅供学习和研究使用。

---

## 🆕 版本更新

### v2.1.0 (2026-01-25) 🆕

**新增功能：**
- ✨ **多数据源支持** - 支持多个CSV文件输入，适用于配对交易、跨市场套利等策略
- ✨ 通过 `-d` 参数多次指定不同数据源
- ✨ 可选的 `-n` 参数为每个数据源命名
- ✨ 自动处理多数据源的回测引擎
- ✨ **empyrical 集成** - 使用专业的 empyrical 库替代 backtrader 内置分析器计算性能指标

**改进：**
- 🚀 回测引擎支持同时加载多个数据feeds
- 📊 优化器支持多数据源的策略优化
- 📈 性能指标计算更准确、标准化（使用 empyrical）
- 💡 新增多数据源使用示例和文档

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

**更新时间**: 2026-01-26  
**版本**: 2.1.0  
**作者**: Peter
