# 通用策略优化器参数文档

本文档详细记录了 `run_optimizer.py` 脚本支持的所有命令行参数、用途及可选值。

## 基本用法

```bash
python run_optimizer.py [选项]
```

## 1. 核心参数 (必需)

| 参数 | 简写 | 说明 | 示例 |
|:---|:---|:---|:---|
| `--data` | `-d` | **[必需]** 标的数据 CSV 文件路径。<br>支持多个文件（空格分隔）或通配符（如 `*.csv`）。<br>数据必须包含 `datetime`/`date` 列。 | `--data data/AG.csv` <br> `--data data/*.csv` |
| `--strategy` | `-s` | **[必需]** 策略脚本文件路径（.py）。<br>必须包含继承自 `bt.Strategy` 的策略类。 | `--strategy strategies/MyStrategy.py` |

## 2. 优化目标与控制

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|:---|
| `--objective` | `-o` | string | `sharpe_ratio` | 优化目标函数。<br>**可选值:**<br>- `sharpe_ratio`: 夏普比率 (推荐)<br>- `annual_return`: 年化收益率<br>- `total_return`: 总收益率<br>- `max_drawdown`: 最大回撤 (最小化)<br>- `calmar_ratio`: 卡玛比率<br>- `sortino_ratio`: 索提诺比率 |
| `--trials` | `-t` | int | `50` | 优化试验次数 (迭代次数)。<br>如果开启了动态试验 (默认开启)，这只是初始/最小次数。 |
| `--params-file` | `-p` | string | None | 指定要优化的参数列表文件路径。<br>文件格式：每行一个参数名。<br>如果不指定，将优化策略中所有的 params。 |
| `--space-config` | `-S` | string | None | 参数空间配置文件 (JSON) 路径。<br>用于手动指定参数的搜索范围 (min, max)。 |

## 3. 数据处理与多数据源

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|:---|
| `--data-freq` | `-f` | string | `auto` | 数据频率设置。<br>**可选值:** `auto`, `daily` (日线), `1m` (1分钟), `5m`, `15m`, `30m`, `hourly` (小时线) |
| `--multi-data` | - | bool | False | **多数据源模式开关**。<br>开启后，所有通过 `--data` 传入的文件将作为一个策略的*多个*数据源输入 (例如股票组合或套利策略)。<br>关闭时 (默认)，会对每个数据文件分别运行独立的优化任务。 |
| `--data-names` | - | list | None | 多数据源的名称列表 (仅在 `--multi-data` 时有效)。<br>数量需与 `--data` 文件数一致。 |

## 4. 资产类型与期货设置

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--asset-type` | string | `stock` | 资产类型。<br>**可选值:** `stock` (股票/现货), `futures` (期货) |
| `--contract-code` | string | None | 内置期货合约代码 (如 `AG`, `RB`, `IF`)。<br>仅当 `--asset-type futures` 时有效。<br>使用内置配置自动设置合约乘数、保证金和手续费。 |
| `--broker-config` | string | None | 自定义经纪商/合约配置文件 (JSON) 路径。<br>用于非内置的期货品种或自定义费率。<br>优先级高于 `--contract-code`。 |

## 5. 高级采样与搜索控制 (v2.0)

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--no-enhanced-sampler` | bool | False | **禁用** 增强采样器。<br>默认开启：使用正态分布采样，更集中于参数中心。<br>开启此标志：使用传统的均匀分布采样。 |
| `--no-dynamic-trials` | bool | False | **禁用** 动态试验次数。<br>默认开启：根据参数空间大小自动调整试验次数。<br>开启此标志：严格使用 `--trials` 指定的次数。 |
| `--no-boundary-search` | bool | False | **禁用** 边界二次搜索。<br>默认开启：如果最优解在边界附近，会自动扩展搜索范围。 |
| `--max-boundary-rounds` | int | `2` | 边界搜索的最大扩展轮数。 |

## 6. LLM (大模型) 辅助优化

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--use-llm` | bool | False | 是否启用 LLM 来分析结果并建议参数范围。 |
| `--llm-type` | string | `ollama` | LLM 后端类型。<br>**可选值:** `ollama`, `openai`, `custom` |
| `--llm-model` | string | `xuanyuan` | 模型名称 (如 `llama3`, `gpt-4`, `xuanyuan`)。 |
| `--llm-url` | string | `http://localhost:11434` | LLM API 地址。 |
| `--api-key` | string | "" | API 密钥 (OpenAI 等云服务需要)。 |
| `--timeout` | int | `180` | LLM 请求超时时间 (秒)。 |

## 7. 输出与日志

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|:---|
| `--output` | `-O` | string | `./optimization_results` | 结果输出目录。 |
| `--quiet` | `-q` | bool | False | 静默模式，减少控制台输出。 |

## 常用命令组合示例

### 1. 基础单文件优化
```bash
python run_optimizer.py -d project_trend/data/AG.csv -s project_trend/src/Aberration.py
```

### 2. 期货配置优化 (使用内置合约)
```bash
python run_optimizer.py -d data/AG.csv -s strategy.py --asset-type futures --contract-code AG
```

### 3. 多数据文件批量跑测 (每个文件单独优化)
```bash
python run_optimizer.py -d data/*.csv -s strategy.py --objective annual_return
```

### 4. 多数据源策略优化 (如套利)
```bash
python run_optimizer.py -d data/QQQ.csv data/TQQQ.csv -s strategy_pair.py --multi-data --data-names QQQ TQQQ
```

### 5. 指定参数范围并使用 LLM
```bash
python run_optimizer.py -d data/BTC.csv -s strategy.py -S my_space.json --use-llm --llm-model gpt-4
```
