# SPP (System Parameter Permutation) 鲁棒性分析工具

以贝叶斯优化找到的最优参数为中心，在邻域内进行蒙特卡洛采样，评估参数鲁棒性。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 用法

支持两种参数输入方式（二选一）：

```bash
# 方式一：参数文件（推荐）
python run_spp_analysis.py \
  --params-file <参数文件路径> \
  -d <数据CSV路径> [<数据CSV路径2> ...] \
  -s <策略脚本路径>

# 方式二：优化结果 JSON
python run_spp_analysis.py \
  -r <优化结果JSON路径> \
  -d <数据CSV路径> [<数据CSV路径2> ...] \
  -s <策略脚本路径>
```

### 参数文件格式

简单的 key = value 文本文件，支持 `#` 注释：

```
# 最优参数
period = 35
std_dev_upper = 2.1
std_dev_lower = 1.8
```

使用参数文件时，SPP 会自动运行一次基准回测获取性能指标。

### 示例

```bash
# 参数文件模式
python run_spp_analysis.py \
  --params-file best_params.txt \
  -d ../project_trend/data/AG.csv \
  -s ../strategies/CatScan_Backtest.py

# JSON 模式
python run_spp_analysis.py \
  -r ../optimization_results/AG/optimization_AG_CatScan_20260223.json \
  -d ../project_trend/data/AG.csv \
  -s ../strategies/CatScan_Backtest.py

# 多数据源
python run_spp_analysis.py \
  --params-file best_params.txt \
  -d data/BTC.csv data/ETH.csv \
  -s strategy.py
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--params-file` | 参数文件路径（key=value 格式，与 -r 二选一） | — |
| `-r, --result` | 优化结果 JSON 文件路径（与 --params-file 二选一） | — |
| `-d, --data` | CSV 数据文件路径（支持多个，空格分隔） | 必填 |
| `-s, --strategy` | 策略脚本路径 | 必填 |
| `-n, --samples` | 蒙特卡洛采样次数 | 300 |
| `-p, --perturbation` | 扰动比例 | 0.20 |
| `-o, --objective` | 优化目标 (sharpe_ratio, annual_return 等) | sharpe_ratio |
| `--output` | 输出目录 | ./spp_results |
| `--data-frequency` | 数据频率 (daily, 1m, 5m, 15m, 30m, hourly) | 自动检测 |
| `--asset-type` | 资产类型 (stock, futures) | stock |
| `--contract-code` | 期货合约代码（期货模式时使用） | — |
| `--use-llm` | 启用 LLM 辅助识别敏感参数 | 否 |
| `--llm-model` | LLM 模型名称 | xuanyuan |
| `--sensitive-params` | 手动指定敏感参数 (逗号分隔) | — |
| `--seed` | 蒙特卡洛随机种子，用于复现采样 | 自动生成 |
| `--no-parallel` | 禁用多进程并行，使用串行回测 | 否 |
| `--n-workers` | 并行 worker 数 | CPU 核数 - 2 |
| `--batch-size` | 并行每批提交样本数 | 32 |
| `--sample-timeout` | 单个样本回测超时秒数 | 300 |
| `-q, --quiet` | 静默模式 | 否 |

默认启用 SPP 多进程并行。父进程负责生成 Monte Carlo 参数样本和写 JSON/PNG，worker 只执行回测；同一个 `--seed` 下，串行与并行使用相同的样本集合。

### 策略兼容性

工具会自动从策略脚本中提取：

- 自定义数据类（继承自 `bt.feeds.PandasData`）
- 自定义手续费类（继承自 `bt.CommInfoBase`）
- `trade_log` 检测（自动切换指标计算模式）

与主优化器 (`run_optimizer.py`) 的回测行为完全一致。

### 输出

- `spp_result_<资产>_<时间戳>.json` — 完整分析结果
- `spp_report_<资产>_<时间戳>.png` — 可视化报告
