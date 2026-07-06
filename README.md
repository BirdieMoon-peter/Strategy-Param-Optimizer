# 量化策略参数优化器 (Param Optimizer)

## 1. 简介

基于贝叶斯优化的量化交易策略参数调优工具。支持任意 Backtrader 策略与 CSV 数据，通过两阶段优化算法（正态分布探索 + TPE 贝叶斯利用）自动搜索最优参数组合，并提供 SPP 鲁棒性分析验证参数稳定性。

核心特性：

- 两阶段优化：正态分布采样探索 + 贝叶斯 TPE 利用
- 多目标支持：Sharpe、Sortino、Calmar、年化收益、总收益、最大回撤、做市商评分
- 做市商优化模式：在控制亏损和回撤的前提下最大化交易量
- 智能参数空间生成与自动边界扩展
- 专业指标计算（empyrical 框架）
- 期货合约支持（内置 SC/AG/AU/CU/RB/IF）
- 多数据源策略支持
- 批量优化与 SPP 鲁棒性分析
- 可选 LLM 辅助分析

---

## 2. 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 单标的优化
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py

# 批量优化
python run_optimizer.py -d data/*.csv -s strategy/Aberration.py

# 期货品种优化
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --asset-type futures --contract-code AG

# SPP 鲁棒性分析（基于优化结果）
python run_spp_analysis.py -r optimization_results/AG/result.json -d data/AG.csv -s strategy/Aberration.py
```

---

## 3. 环境要求

Python >= 3.8

### 核心依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| backtrader | >= 1.9.76.123 | 回测引擎 |
| optuna | >= 3.0.0 | 贝叶斯优化 |
| pandas | >= 1.5.0 | 数据处理 |
| numpy | >= 1.21.0 | 数值计算 |
| scipy | >= 1.9.0 | KDE 分布估计（SPP） |

### 可选依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| empyrical | >= 0.5.5 | 专业金融指标计算 |
| requests | >= 2.28.0 | LLM 通信 |
| matplotlib | >= 3.5.0 | 可视化（SPP 报告图） |
| tqdm | >= 4.64.0 | 进度条 |

```bash
pip install -r requirements.txt
```

---

## 4. 使用方法

### 4.1 run_optimizer.py — 参数优化

#### 必需参数

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--data` | `-d` | CSV 数据文件路径，支持多个文件或通配符 |
| `--strategy` | `-s` | 策略脚本路径（.py 文件，需包含 `bt.Strategy` 子类） |

#### 多数据源参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--multi-data` | 关闭 | 将多个 `--data` 文件作为同一策略的多数据源输入（顺序即数据源顺序） |
| `--data-names` | None | 多数据源名称列表，需与 `--data` 数量一致（例如 `QQQ TQQQ`） |

#### 优化参数

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--objective` | `-o` | `sharpe_ratio` | 优化目标，可选值见下表 |
| `--trials` | `-t` | `50` | 优化试验次数（启用动态试验时自动调整） |
| `--params-file` | `-p` | None | 指定要优化的参数列表文件（每行一个参数名），不指定则优化所有参数 |
| `--space-config` | `-S` | None | 参数空间配置文件（JSON 格式），手动指定参数搜索范围 |

优化目标可选值：

| 值 | 含义 |
|----|------|
| `sharpe_ratio` | 夏普比率（默认，推荐） |
| `annual_return` | 年化收益率 |
| `total_return` | 总收益率 |
| `max_drawdown` | 最大回撤（最小化） |
| `calmar_ratio` | 卡玛比率 |
| `sortino_ratio` | 索提诺比率 |
| `market_maker_score` | 做市商评分（量优先、风险门控） |

#### 数据频率

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-freq` | `-f` | `auto` | 数据频率。可选：`daily`, `1m`, `5m`, `15m`, `30m`, `hourly`, `auto` |

#### 资产类型 / 期货

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asset-type` | `stock` | 资产类型，可选 `stock` / `futures` |
| `--contract-code` | None | 内置期货合约代码：`SC`(原油), `AG`(白银), `AU`(黄金), `CU`(铜), `RB`(螺纹钢), `IF`(沪深300股指)。需配合 `--asset-type futures` |
| `--broker-config` | None | 自定义经纪商配置文件路径（JSON），优先级高于 `--contract-code` |

#### 做市商优化参数

当 `--objective market_maker_score` 时生效。做市商业务收益来自成交量返佣，优化目标是**在控制亏损和回撤的前提下最大化交易量**。

评分公式：`Score = log(1+V) + α×R - β×D - γ×max(0, -R)`

- V = 交易次数，R = 年化收益率（小数），D = 最大回撤（小数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mm-alpha` | `2.0` | 收益权重 α：鼓励正收益 |
| `--mm-beta` | `4.0` | 回撤惩罚权重 β：惩罚回撤 |
| `--mm-gamma` | `6.0` | 亏损额外惩罚权重 γ：额外惩罚亏损（R<0 时生效） |
| `--mm-max-dd` | `0.15` | 回撤容忍阈值 |
| `--mm-min-trades` | `10` | 最低交易次数门槛（低于此值直接判定为不合格） |

#### 增强采样器（v2.0+）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-enhanced-sampler` | 关闭 | 禁用增强采样器（正态分布采样），回退到传统均匀采样 |
| `--no-dynamic-trials` | 关闭 | 禁用动态试验次数，使用 `--trials` 指定的固定值 |
| `--no-boundary-search` | 关闭 | 禁用边界二次搜索 |
| `--max-boundary-rounds` | `2` | 边界二次搜索最大轮数 |
| `--boundary-threshold` | `0.1` | 最优参数落在搜索区间前/后多少比例内时触发边界拓展 |
| `--boundary-expansion-factor` | `1.5` | 每轮按初始区间宽度向外拓展的倍数，必须大于 `1` |

#### LLM 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-llm` | 关闭 | 启用 LLM 辅助优化 |
| `--llm-type` | `ollama` | LLM 类型，可选 `ollama` / `openai` / `custom` |
| `--llm-model` | `xuanyuan` | LLM 模型名称 |
| `--llm-url` | `http://localhost:11434` | LLM API URL |
| `--api-key` | 空 | API 密钥（OpenAI 需要） |
| `--timeout` | `180` | LLM 请求超时时间（秒） |

#### 输出参数

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-O` | `./optimization_results` | 输出目录 |
| `--quiet` | `-q` | 关闭 | 静默模式（减少输出） |

#### 使用示例

```bash
# 基本用法
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py

# 指定优化目标和试验次数
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py -o annual_return -t 100

# 批量优化多个标的
python run_optimizer.py -d data/AG.csv data/AU.csv data/CU.csv -s strategy/Aberration.py

# 通配符批量
python run_optimizer.py -d "data/*.csv" -s strategy/Aberration.py

# 多数据源策略（如配对交易）
python run_optimizer.py -d data/QQQ.csv data/TQQQ.csv -s strategy/PairTrading.py \
    --multi-data --data-names QQQ TQQQ

# 仅优化指定参数
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py -p params.txt

# 自定义参数空间
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py -S space_config.json

# 期货优化（内置品种）
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py \
    --asset-type futures --contract-code AG

# 期货优化（自定义配置）
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --broker-config my_broker.json

# 分钟级数据
python run_optimizer.py -d data/AG_5m.csv -s strategy/Aberration.py -f 5m

# 禁用增强采样器
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --no-enhanced-sampler

# 使用 LLM 辅助（本地 Ollama）
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --use-llm --llm-model xuanyuan

# 使用 OpenAI
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py \
    --use-llm --llm-type openai --api-key sk-xxx

# 做市商优化（基本用法）
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --objective market_maker_score

# 做市商优化（更保守：加大亏损惩罚）
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --objective market_maker_score \
    --mm-alpha 1.0 --mm-beta 5.0 --mm-gamma 8.0 --mm-max-dd 0.10

# 做市商优化（更激进：允许更大回撤换取更多交易量）
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --objective market_maker_score \
    --mm-alpha 2.0 --mm-beta 3.0 --mm-gamma 4.0 --mm-max-dd 0.20
```

---

### 4.2 run_spp_analysis.py — SPP 鲁棒性分析

以优化结果的最优参数为中心进行蒙特卡洛扰动采样，评估参数鲁棒性。

#### 必需参数

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--result` | `-r` | 优化结果 JSON 文件路径（`run_optimizer.py` 的输出） |
| `--data` | `-d` | CSV 数据文件路径 |
| `--strategy` | `-s` | 策略 .py 文件路径 |

#### 可选参数

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--samples` | `-n` | `300` | 蒙特卡洛采样次数 |
| `--perturbation` | `-p` | `0.20` | 扰动比例（0.20 = 参数 ±20%） |
| `--objective` | `-o` | None | 分析指标（默认从 JSON 读取） |
| `--use-llm` | — | 关闭 | 启用 LLM 识别敏感参数 |
| `--llm-model` | — | `xuanyuan` | LLM 模型名 |
| `--sensitive-params` | — | None | 手动指定敏感参数（逗号分隔，如 `period,devfactor`） |
| `--output` | — | `./spp_results` | 输出目录 |
| `--data-frequency` | — | None | 数据频率 |
| `--asset-type` | — | `stock` | 资产类型，可选 `stock` / `futures` |
| `--contract-code` | — | None | 期货合约代码 |
| `--quiet` | `-q` | 关闭 | 静默模式 |

#### 使用示例

```bash
# 基本用法
python run_spp_analysis.py \
    -r optimization_results/AG/optimization_AG_Aberration_20260224.json \
    -d data/AG.csv -s strategy/Aberration.py

# 增加采样次数和扰动范围
python run_spp_analysis.py -r result.json -d data/AG.csv -s strategy/Aberration.py \
    -n 500 -p 0.30

# 手动指定敏感参数
python run_spp_analysis.py -r result.json -d data/AG.csv -s strategy/Aberration.py \
    --sensitive-params period,devfactor

# 使用 LLM 识别敏感参数
python run_spp_analysis.py -r result.json -d data/AG.csv -s strategy/Aberration.py \
    --use-llm --llm-model xuanyuan

# 期货品种
python run_spp_analysis.py -r result.json -d data/AG.csv -s strategy/Aberration.py \
    --asset-type futures --contract-code AG
```

---

### 4.3 数据格式要求

CSV 文件需包含以下列：

| 列名 | 说明 | 必需 |
|------|------|------|
| `datetime` / `date` / `time_key` / `time` | 时间列（任选其一） | 是 |
| `open` | 开盘价 | 是 |
| `high` | 最高价 | 是 |
| `low` | 最低价 | 是 |
| `close` | 收盘价 | 是 |
| `volume` | 成交量 | 是 |

时间列支持格式：
- 日期字符串：`2025-01-15`、`2025-01-15 09:30:00`
- 秒级时间戳：`1705276800`
- 毫秒级时间戳：`1705276800000`

系统自动检测并转换时间格式。

---

### 4.4 策略脚本要求

策略必须满足以下条件：

1. 继承 `bt.Strategy`
2. 使用元组格式定义参数 `params`
3. 实现 `__init__` 和 `next` 方法
4. 可选：维护 `self.trade_log` 列表以启用精确指标计算；接受 `verbose` 参数控制日志

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('threshold', 1.5),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.trade_log = []  # 可选：启用 trade_log 精确指标模式

    def next(self):
        if not self.position:
            if self.fast_ma[0] > self.slow_ma[0] * (1 + self.params.threshold / 100):
                self.buy()
        else:
            if self.fast_ma[0] < self.slow_ma[0]:
                self.close()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_log.append({
                'entry_date': bt.num2date(trade.dtopen),
                'exit_date': bt.num2date(trade.dtclose),
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
            })
```

---

### 4.5 配置文件格式

#### 参数列表文件（`--params-file`）

纯文本，每行一个参数名，`#` 开头为注释：

```
# 要优化的参数
period
std_dev_upper
std_dev_lower
```

#### 参数空间配置文件（`--space-config`）

JSON 格式，手动指定各参数的搜索范围和分布类型：

```json
{
  "param_space": {
    "period": {
      "min": 10,
      "max": 50,
      "hard_min": 5,
      "hard_max": 80,
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

`min` / `max` 是本轮初始搜索范围；`hard_min` / `hard_max` 是可选硬性边界。启用自动边界二次搜索时，参数空间可以继续向外扩展，但不会突破对应的硬边界。未设置硬边界的参数保持原来的自动扩展行为。

边界拓展范围由命令行参数控制：`--boundary-threshold` 决定多接近边界才触发二次搜索，`--boundary-expansion-factor` 决定每轮向外扩展多少。扩展按初始区间宽度计算，例如 `[10, 50]` 使用 `--boundary-expansion-factor 1.5` 且最优值接近上界时，每轮向上增加 `20`，会优先尝试扩到 `[10, 70]`，再受 `hard_max` 裁剪。

##### 边界限制用法

使用 `hard_min` / `hard_max` 可以限制自动边界扩展的最大搜索范围，防止某些参数被扩到策略不接受或业务上无意义的值。

只限制上界：

```json
{
  "param_space": {
    "period": {
      "min": 10,
      "max": 50,
      "hard_max": 80,
      "step": 1,
      "distribution": "int_uniform"
    }
  }
}
```

只限制下界：

```json
{
  "param_space": {
    "threshold": {
      "min": 0.01,
      "max": 0.2,
      "hard_min": 0.0,
      "step": null,
      "distribution": "uniform"
    }
  }
}
```

锁定某个参数不允许继续拓宽：把 `hard_min` / `hard_max` 设成当前 `min` / `max`。

```json
{
  "param_space": {
    "period": {
      "min": 10,
      "max": 50,
      "hard_min": 10,
      "hard_max": 50,
      "step": 1,
      "distribution": "int_uniform"
    }
  }
}
```

配合命令行调整触发阈值和拓宽倍数：

```bash
python run_optimizer.py \
  --data data.csv \
  --strategy strategy.py \
  --space-config param_space.json \
  --boundary-threshold 0.2 \
  --boundary-expansion-factor 2.0
```

例如参数 `period` 初始范围为 `[10, 20]`，最优值接近上界：

| 设置 | 自动拓宽结果 |
|------|--------------|
| `--boundary-expansion-factor 2.0` | `[10, 30]` |
| `--boundary-expansion-factor 3.0` | `[10, 40]` |
| `--boundary-expansion-factor 5.0` 且 `hard_max=45` | `[10, 45]` |

`distribution` 可选值：

| 值 | 说明 |
|----|------|
| `int_uniform` | 整数均匀分布 |
| `uniform` | 连续均匀分布 |
| `log_uniform` | 对数均匀分布 |

#### 经纪商配置文件（`--broker-config`）

JSON 格式，用于自定义期货品种：

```json
{
  "asset_type": "futures",
  "mult": 15,
  "margin": 0.12,
  "comm_type": "PERC",
  "commission": 0.0005,
  "initial_cash": 100000,
  "contract_code": "AG",
  "contract_name": "白银"
}

```

`comm_type` 可选 `PERC`（百分比）或 `FIXED`（固定金额/手）。

---

### 4.6 输出文件说明

#### run_optimizer.py 输出

```
optimization_results/
├── {标的名}/
│   ├── optimization_{标的}_{策略}_{时间戳}.json    # 完整结果
│   └── optimization_{标的}_{策略}_{时间戳}.txt     # 可读摘要
├── batch_optimization_summary.txt                  # 批量汇总（文本）
└── batch_optimization_summary.json                 # 批量汇总（JSON）
```

JSON 结果包含：

| 字段 | 说明 |
|------|------|
| `best_parameters` | 最优参数集 |
| `performance_metrics` | 全部性能指标（Sharpe、Sortino、Calmar 等） |
| `optimization_metadata` | 试验次数、耗时、优化目标等元信息 |
| `parameter_space_analysis` | 参数边界使用情况与改进建议 |
| `daily_returns` | 每日收益率序列 |

#### run_spp_analysis.py 输出

```
spp_results/
└── {标的名}/
    ├── spp_{标的}_{策略}_{时间戳}.json     # 分析结果
    └── spp_{标的}_{策略}_{时间戳}.png      # 2x2 可视化报告
```

---

## 5. 算法原理

### 5.1 两阶段优化

优化过程分为探索（Exploration）和利用（Exploitation）两个阶段，平衡全局搜索与局部精细化：

```
┌─────────────────────────────────────────────────────────┐
│                    两阶段优化流程                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Trial 0: 策略默认参数（基线）                            │
│       ↓                                                 │
│  阶段一 [探索] ── 30% 试验次数                            │
│  │  采样方式: 正态分布（均值=默认值, σ=25%×范围）          │
│  │  目的: 广泛探索参数空间，发现有潜力的区域               │
│  │                                                      │
│  阶段二 [利用] ── 70% 试验次数                            │
│  │  采样方式: 贝叶斯 TPE (Tree-structured Parzen Estimator)│
│  │  目的: 在有潜力的区域精细搜索，收敛到最优解             │
│       ↓                                                 │
│  输出最优参数                                            │
│       ↓                                                 │
│  边界检测 → 如触及边界 → 自动扩展 → 二次搜索              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

- Trial 0 始终使用策略默认参数作为基线
- 探索阶段使用正态分布采样（相比均匀采样，更集中于默认值附近同时保留探索能力）
- 利用阶段使用 Optuna 的 TPE 采样器，基于探索阶段的结果建模，集中搜索高收益区域

### 5.2 参数空间智能生成

`ParamSpaceOptimizer` 根据参数名称自动匹配规则，生成合理的搜索范围：

| 参数类型 | 匹配规则（正则） | 相对范围 | 绝对范围 | 分布类型 |
|---------|-----------------|---------|---------|---------|
| 周期类 | `.*period.*\|.*window.*\|.*length.*` | 0.7x ~ 1.5x | [5, 200] | int_uniform |
| 标准差 | `.*std.*\|.*dev.*factor.*` | 0.7x ~ 1.5x | [0.5, 5.0] | uniform |
| 阈值 | `.*threshold.*\|.*limit.*` | 0.7x ~ 1.5x | [0.01, 0.5] | uniform |
| RSI 阈值 | `.*rsi.*sold.*\|.*rsi.*bought.*` | 0.85x ~ 1.15x | [10, 90] | int_uniform |
| 百分比 | `.*percent.*\|.*ratio.*` | 0.7x ~ 1.3x | [0.01, 1.0] | uniform |
| 快线周期 | `.*fast.*\|.*short.*` | 0.7x ~ 1.5x | [3, 50] | int_uniform |
| 慢线周期 | `.*slow.*\|.*long.*` | 0.7x ~ 1.5x | [10, 200] | int_uniform |
| 止损 | `.*stop.*loss.*\|.*sl.*` | 0.7x ~ 1.5x | [0.01, 0.2] | uniform |
| 止盈 | `.*take.*profit.*\|.*tp.*` | 0.7x ~ 1.5x | [0.02, 0.5] | uniform |
| 未匹配 | — | 0.7x ~ 1.3x | — | 自动推断 |

生成逻辑：
1. 以策略默认值为中心，按相对范围计算 min/max
2. 用绝对范围进行裁剪，防止出现不合理值
3. 自动处理参数约束（如 fast_period < slow_period）

### 5.3 自动边界扩展

当最优参数落在搜索空间边界时，自动扩展范围并进行二次搜索：

```
判定条件: 最优值位于 min/max 的边界阈值范围内即视为"触及边界"
默认阈值: 0.1，可通过 --boundary-threshold 调整

扩展策略:
  - 扩展因子: 默认 1.5x，可通过 --boundary-expansion-factor 调整
  - 扩展方式: 按初始区间宽度向触及边界的方向扩展，避免后续轮次越扩越快
  - 硬性边界: hard_min / hard_max 会裁剪最终范围
  - 最大扩展轮数: 默认 2 轮，可通过 --max-boundary-rounds 调整

示例:
  参数 period 搜索范围 [10, 50]，最优值 = 48（触及上界）
  → 第1轮扩展: 范围变为 [10, 70]
  → 若最优值仍触及边界 → 第2轮扩展: [10, 90]
  → 最多 2 轮后停止

如果配置 hard_max=80:
  → 第1轮扩展仍为 [10, 70]
  → 第2轮会被裁剪为 [10, 80]
```

### 5.4 动态试验次数

根据参数数量和特征自动计算合理的试验次数：

```
base_trials = 30 + n_params × 10

complexity_factor (0.7 ~ 1.5):
  - 参数数量多 → 因子增大
  - 参数类型复杂（混合整数/浮点） → 因子增大

final_trials = clamp(base_trials × complexity_factor, 20, 200)
```

| 参数数量 | 基础试验数 | 典型最终试验数 |
|---------|-----------|--------------|
| 2 | 50 | 35 ~ 75 |
| 5 | 80 | 56 ~ 120 |
| 8 | 110 | 77 ~ 165 |
| 10+ | 130+ | 最高 200 |

可通过 `--no-dynamic-trials` 禁用，使用 `--trials` 指定的固定值。

### 5.5 回测指标双模式计算

#### trade_log 模式（精确模式）

策略维护 `self.trade_log` 列表，记录每笔交易的精确盈亏，聚合为日收益率后计算指标。适用于需要精确逐笔交易统计的场景。

#### empyrical 模式（专业模式）

使用 Backtrader 的 `TimeReturn` 分析器获取权益曲线收益率，通过 empyrical 库计算专业金融指标：

| 指标 | 说明 |
|------|------|
| Sharpe Ratio | 夏普比率（风险调整收益） |
| Sortino Ratio | 索提诺比率（仅考虑下行风险） |
| Calmar Ratio | 卡玛比率（年化收益 / 最大回撤） |
| Omega Ratio | 欧米伽比率 |
| Annual Return | 年化收益率 |
| Annual Volatility | 年化波动率 |
| Max Drawdown | 最大回撤 |
| VaR (5%) | 在险价值 |
| Tail Ratio | 尾部比率 |

#### 年化因子

| 数据频率 | 年化因子 |
|---------|---------|
| daily | 252 |
| weekly | 52 |
| monthly | 12 |
| hourly | 252 × 6.5 = 1,638 |
| 1m | 252 × 390 = 98,280 |
| 5m | 252 × 78 = 19,656 |
| 15m | 252 × 26 = 6,552 |
| 30m | 252 × 13 = 3,276 |

日内数据会自动聚合为日收益率后再进行年化计算。

### 5.6 SPP 鲁棒性分析

SPP (System Parameter Permutation) 通过蒙特卡洛扰动评估最优参数的稳定性：

```
流程:
  1. 确定敏感参数（手动指定 > LLM 识别 > 全部参数）
  2. 以最优参数为中心，按 ±perturbation_ratio 生成扰动样本
  3. 对每组扰动参数运行回测，收集指标
  4. 计算衰减率、稳定性评分、参数相关性
  5. 输出判定结论与可视化报告

核心指标:
  - 衰减率 (Decay Rate): 扰动后指标相对最优值的平均下降比例
  - 稳定性评分 (Stability Score): 扰动后指标的变异系数的倒数
  - 参数相关性: 各参数变化与指标变化的相关系数

判定标准:
  - Strong (强鲁棒): 衰减率 < 15%
  - Medium (中等鲁棒): 衰减率 15% ~ 30%
  - Weak (弱鲁棒): 衰减率 > 30%

输出:
  - 2×2 PNG 可视化报告（分布图、衰减曲线、参数敏感性、相关矩阵）
  - JSON 格式完整分析结果
```

---

## 6. 版本更新

### v3.0 — SPP 鲁棒性分析

- 新增 `run_spp_analysis.py` CLI 入口
- 蒙特卡洛扰动采样评估参数鲁棒性
- 敏感参数识别（手动 / LLM / 全参数）
- 衰减率、稳定性评分、参数相关性分析
- 2×2 PNG 可视化报告 + JSON 结果输出
- Strong / Medium / Weak 三级判定

### v2.2 — 双模式指标计算

- 新增 trade_log 精确计算模式与 empyrical 专业指标模式
- 完善日内数据支持，自动聚合为日收益率
- 使用 pyfolio/empyrical 框架计算 Sharpe、Sortino、Calmar、Omega、VaR、Tail Ratio 等

### v2.1 (2026-02-02) — 两阶段优化

- 正态分布采样替代均匀采样（探索阶段）
- 两阶段优化：探索（30%）+ 利用（70%）
- 动态试验次数（基于参数数量与复杂度）
- 边界二次搜索（自动检测 + 扩展 + 重搜索）

### v2.0 (2026-01-29) — 自适应参数空间

- 自适应参数空间生成（默认值 ±30%）
- 自动边界扩展（默认 1.5x、最多 2 轮，可通过 CLI 与硬边界配置调整）
- 专业指标计算（empyrical 库）
- Trial 0 使用策略默认参数作为基线
- 批量处理支持（多 CSV 文件）
- 新增 Sortino / Calmar / Omega 指标

### v1.1 (2026-01-22) — 选择性优化

- 指定参数优化（`--params-file`）
- 智能参数空间自动生成
- 参数空间使用情况分析与改进建议
- 参数约束自动处理（如 fast < slow）
