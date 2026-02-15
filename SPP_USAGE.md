# SPP 鲁棒性分析使用指南

## 概述

SPP (System Parameter Permutation) 鲁棒性分析工具基于 Dave Walton 的论文《Know Your System!》实现，用于验证优化器找到的最优参数是否鲁棒，识别过拟合和数据挖掘偏差。

**核心思想**: 不只看单一最优参数，而是看参数空间的整体分布（中位数、概率密度），从而评估策略的真实 Edge。

## 三个分析维度

### 1. 全局分布 (Global Permutation)
- 在整个参数空间内均匀随机采样 500 组参数
- 对每组参数运行完整回测
- 构建 Sharpe/收益率的抽样分布
- **中位数** = 策略真实 Edge 的无偏估计
- **最优参数的分位数排名** = 衡量"运气成分"

### 2. 局部稳定性 (Local Stability)
- 以最优参数为中心，std=10% 范围内采样 200 组邻域参数
- 使用正态分布采样（复用 `NormalDistributionSampler`）
- **衰减率** = (best_sharpe - local_median) / |best_sharpe|
- 衰减率越低 = 参数越鲁棒

### 3. 短期最坏情况 (Short-Run Worst-Case)
- 按年切分数据，每年独立创建 BacktestEngine
- 每年均匀采样 200 组参数并回测
- 输出每年的 5th/25th/50th/75th/95th 分位数
- **5th 分位数** = 论文强调的"最坏情况应急"

## 基本用法

### 必需参数

```bash
python run_spp_analysis.py \
  -r <优化结果JSON路径> \
  -d <CSV数据文件路径> \
  -s <策略.py文件路径>
```

### 完整示例

```bash
# 标准分析（默认采样数）
python run_spp_analysis.py \
  -r optimization_results/AG/optimization_AG_Aberration_20260202.json \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py

# 输出到指定目录
python run_spp_analysis.py \
  -r optimization_results/BTC/optimization_BTC_Aberration.json \
  -d project_trend/data/BTC.csv \
  -s project_trend/src/Aberration.py \
  --output ./my_spp_results
```

## 可选参数

### 采样控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--global-samples` | 500 | 全局分布采样数 |
| `--local-samples` | 200 | 局部稳定性采样数 |
| `--local-noise` | 0.10 | 局部扰动比例（10% = std 为参数范围的 10%） |
| `--yearly-samples` | 200 | 每年采样数 |
| `--no-short-run` | - | 跳过逐年分析（加快速度） |

### 快速测试（小样本）

```bash
# 用于快速验证，大幅减少采样数
python run_spp_analysis.py \
  -r result.json -d data.csv -s strategy.py \
  --global-samples 50 \
  --local-samples 30 \
  --yearly-samples 30
```

### 分析指标

```bash
# 指定优化目标（默认从 JSON 读取）
python run_spp_analysis.py \
  -r result.json -d data.csv -s strategy.py \
  -o sharpe_ratio  # 或 annual_return, sortino_ratio 等
```

### 数据频率

```bash
# 明确指定数据频率（默认自动检测）
python run_spp_analysis.py \
  -r result.json -d data.csv -s strategy.py \
  --data-frequency daily  # 或 1m, 5m, 15m, 30m, hourly
```

### 期货品种

```bash
# 期货优化结果分析
python run_spp_analysis.py \
  -r optimization_results/AG/result.json \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py \
  --asset-type futures \
  --contract-code AG
```

### 其他选项

| 参数 | 说明 |
|------|------|
| `--output` | 输出目录（默认: ./spp_results） |
| `-q, --quiet` | 静默模式，减少输出 |

## 输出文件

分析完成后会在输出目录生成两个文件：

### 1. PNG 可视化报告

**文件名**: `spp_report_{资产名}_{时间戳}.png`

**布局**: 3x2 面板

- **(0,0)** 全局 Sharpe 分布直方图 + KDE + 最优值/中位数标线
- **(0,1)** 局部稳定性 KDE + 最优值标线 + 衰减率标注
- **(1,0)** 风险-收益散点图（回撤 vs 年化收益）
- **(1,1)** 逐年 Sharpe 箱线图 + 5th 分位线
- **(2,0)** 逐年分位数热力图
- **(2,1)** 文字总结面板（关键指标 + 过拟合判定）

### 2. JSON 结果文件

**文件名**: `spp_result_{资产名}_{时间戳}.json`

**结构**:

```json
{
  "spp_info": {
    "analysis_time": "2026-02-12 14:10:27",
    "elapsed_seconds": 17.0,
    "source_json": "...",
    "asset": "BTC",
    "strategy": "AberrationStrategy",
    "config": { ... }
  },
  "best_parameters": { ... },
  "best_metrics": { ... },
  "global_distribution": {
    "sample_count": 500,
    "median": 0.82,
    "mean": 0.85,
    "std": 0.08,
    "p5": 0.76, "p25": 0.80, "p75": 0.92, "p95": 0.96,
    "profitability_rate": 95.2,
    "best_percentile": 98.4
  },
  "local_stability": {
    "sample_count": 200,
    "median": 0.95,
    "mean": 0.95,
    "decay_rate": 0.11,
    "robustness_score": 89.0
  },
  "short_run": {
    "2018": {"p5": -0.50, "p25": -0.17, "median": -0.16, "p75": -0.10, "p95": 0.29},
    "2019": { ... },
    ...
  },
  "verdict": {
    "overfit_risk": "高 (>95th分位)",
    "global_edge": "有 (中位数>0)",
    "parameter_robust": "强 (衰减<15%)",
    "worst_year_5pct": -0.78,
    "summary": "过拟合风险较高"
  }
}
```

## 判定标准

### 过拟合风险

- **低**: 最优参数 < 80th 分位
- **中**: 最优参数 80-95th 分位
- **高**: 最优参数 > 95th 分位

### 全局 Edge

- **有**: 全局中位数 > 0
- **无**: 全局中位数 ≤ 0

### 参数鲁棒性

- **强**: 衰减率 < 15%
- **中**: 衰减率 15-30%
- **弱**: 衰减率 > 30%

## 典型工作流

```bash
# 1. 运行优化器
python run_optimizer.py \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py \
  --trials 100

# 2. 运行 SPP 分析（使用优化器输出的 JSON）
python run_spp_analysis.py \
  -r optimization_results/AG/optimization_AG_Aberration_20260212.json \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py

# 3. 查看结果
# - PNG 报告: spp_results/AG/spp_report_AG_*.png
# - JSON 结果: spp_results/AG/spp_result_AG_*.json
```

## 性能建议

### 标准分析（推荐）

- 全局: 500 组（覆盖参数空间）
- 局部: 200 组（充分评估邻域）
- 逐年: 200 组/年（统计显著性）
- **预计耗时**: 5-15 分钟（取决于策略复杂度和数据量）

### 快速验证

```bash
--global-samples 50 --local-samples 30 --yearly-samples 30
```

- **预计耗时**: 1-3 分钟
- 适用于快速检查或调试

### 跳过逐年分析

```bash
--no-short-run
```

- 只运行维度 1 和 2
- **预计耗时**: 减少 50-70%
- 适用于只关心全局和局部鲁棒性的场景

## 注意事项

1. **数据要求**: CSV 必须包含 `datetime/date`, `open`, `high`, `low`, `close`, `volume` 列
2. **策略要求**: 必须是 Backtrader 兼容策略，使用 `bt.params` 定义参数
3. **JSON 格式**: 必须是 `run_optimizer.py` 输出的标准格式
4. **内存占用**: 大样本数 + 长数据 + 复杂策略可能占用较多内存
5. **并行执行**: 当前版本串行执行回测，未来可能支持并行加速

## 常见问题

**Q: 为什么我的最优参数过拟合风险高？**

A: 这是正常现象。优化器总是找到历史数据上的最优解，但这不代表未来表现。SPP 分析通过全局分布揭示真实 Edge。如果全局中位数仍为正，说明策略本身有效，只是最优参数可能过度拟合。

**Q: 衰减率多少算合理？**

A: < 15% 为强鲁棒，15-30% 为中等，> 30% 为弱。但这取决于策略类型。趋势跟踪策略通常对参数不敏感（低衰减），而均值回归策略可能更敏感。

**Q: 逐年分析的 5th 分位数为负怎么办？**

A: 这表明在最坏年份，即使随机选择参数，也有 5% 概率亏损。这是策略固有风险，需要结合全局 Edge 综合判断。如果全局中位数为正，说明长期仍有优势。

**Q: 可以用 SPP 分析其他人的优化结果吗？**

A: 可以，只要有优化结果 JSON、原始数据和策略脚本即可。SPP 是独立的分析工具。

## 参考文献

- Dave Walton, "Know Your System! A Practical Guide to Avoiding Overfitting in Trading System Development"
- 论文核心观点：单一最优参数的表现不可靠，应关注参数空间的整体分布和中位数
