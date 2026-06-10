# Strategy-Param-Optimizer · 并行改造版


## 优化思路

核心流程替换为 **Batch 并行贝叶斯优化**：

```
[串行] TPE / 正态分布采样器批量生成 N 个参数组合
   ↓
[真多进程] ProcessPoolExecutor 并行跑所有回测
   ↓
[串行] 按原始顺序 tell 回 study，更新 TPE 模型
   ↓
循环直到 total trials 跑完
```

具体做了七件事：

1. **从多线程换到多进程** — `concurrent.futures.ProcessPoolExecutor`，每个子进程有独立 GIL，真并行。
2. **Batch 化 TPE** — 利用阶段每批一次性 `study.ask()` 32 个参数，全部并行评估完再按原始顺序 `study.tell()`，保证 TPE 模型拿到的是最新信息。
3. **Constant Liar 补偿** — `TPESampler(constant_liar=True)`：同一个 batch 内尚未返回的 trial 用悲观估计占位，防止一批 32 个 trial 扎堆采到同一片区域。
4. **数据预加载** — 用 `initializer=_init_worker` 让每个子进程在启动时**只读一次 CSV、只加载一次策略、只构建一次 `BacktestEngine`**，存到进程全局。128 个进程启动期只有一次磁盘读取。
5. **进程池生命周期** — 进程池在整个优化过程中只建一次、复用到底，不在 batch 循环里反复创建销毁。
6. **超时保护** — 每个 trial 用 `future.result(timeout=N)` 保护，僵死的回测不会拖垮整批；超时和异常都记为 `-inf` 并 `study.tell(FAIL)`，study 状态始终一致。
7. **分阶段 batch size** — 探索阶段 `batch=128`（TPE 不参与，可以全速吃核）；利用阶段 `batch=32`（batch 越小 TPE 信息越新，采样质量保持在 85%+）。

不变的部分（保持向后兼容）：

- `ParamSpaceOptimizer` 参数空间自动生成与边界扩展
- `EnhancedSampler` 正态分布探索采样器
- `DynamicTrialsCalculator` 动态试验数
- SPP 鲁棒性分析模块
- 所有指标计算（Sharpe / Sortino / Calmar / 做市商评分等）
- LLM 辅助分析模块
- 输出格式和 JSON 结果结构

---

## 3. 代码结构

新增 / 修改的文件：

```
Strategy-Param-Optimizer-parallel/
├── Optimizer/
│   ├── parallel_engine.py          ← 新增：批量并行贝叶斯优化引擎
│   ├── universal_optimizer.py      ← 修改：在 optimize() 中分流到并行引擎
│   ├── bayesian_optimizer.py       ← 未改动（保留作为 --no-parallel fallback）
│   ├── enhanced_sampler.py         ← 未改动
│   ├── backtest_engine.py          ← 未改动
│   ├── param_space_optimizer.py    ← 未改动
│   └── ...（其他模块全部保持原样）
├── run_optimizer.py                ← 修改：新增 --no-parallel / --n-workers 等 CLI 参数
├── README_PARALLEL.md              ← 本文件
├── 并行优化文档.md                 ← 原始技术规格
└── ...
```

### 3.1 `parallel_engine.py` 核心 API

```python
from parallel_engine import (
    BatchParallelOptimizer,   # 批量并行优化器
    WorkerInitArgs,           # 子进程初始化参数（可 pickle）
    create_parallel_study,    # 创建带 constant_liar 的 TPE study
)
```

对外的主入口是 `BatchParallelOptimizer`：

| 构造参数 | 说明 |
|---|---|
| `study` | 已创建的 optuna Study（推荐用 `create_parallel_study`） |
| `search_space` | `{param_name: SearchSpaceConfig}` 字典 |
| `worker_init_args` | `WorkerInitArgs` 数据类，打包数据路径 / 策略路径 / 经纪商配置 |
| `n_exploit_trials` | TPE 利用阶段的 trial 数 |
| `exploration_samples` | 可选：主进程预生成的正态分布探索样本 list |
| `n_workers` | 并行进程数（默认 120） |
| `explore_batch_size` | 探索阶段批大小（默认 128） |
| `exploit_batch_size` | 利用阶段批大小（默认 32） |
| `timeout_per_trial` | 单 trial 超时秒数（默认 300） |

方法：`run() -> (best_params, best_value, best_metrics)`。

### 3.2 子进程执行模型

```
 主进程                           子进程 × N
 ┌────────────────────┐           ┌────────────────────────────────┐
 │ study.ask() × B    │           │ _init_worker(payload)          │
 │  → B 个 params     │           │   加载 CSV                     │
 │                    │  submit   │   动态 import 策略模块         │
 │ executor.submit ───┼──────────▶│   构建 BacktestEngine          │
 │                    │           │   存入 _WORKER_STATE（全局）   │
 │ future.result()    │           │──────────────────────────────  │
 │  (timeout=300)     │           │ _worker_run((idx, params))     │
 │                    │◀──────────│   engine.run_backtest(params)  │
 │ study.tell(...)    │  result   │   engine.evaluate_objective()  │
 │   按原始顺序       │           │   return (idx, score, metrics) │
 └────────────────────┘           └────────────────────────────────┘
```

关键约束（已在代码中强制遵守）：

- `_worker_run` / `_init_worker` 都是模块顶层函数，可 pickle
- 子进程里**不持有任何 `optuna.Trial` 对象**，只传 `dict` 和 `float`
- `study.ask()` / `study.tell()` 只在主进程调用
- 进程池创建一次、复用到底
- 数据通过 `initializer` 预加载，不走函数参数
- batch 内结果按**原始提交顺序**收集并 tell，结果确定性
- 超时 / 异常都正常 `tell(FAIL)`，不让 study 内部状态错乱

---

## 4. 使用方式

### 4.1 快速开始（默认启用并行）


```bash
python run_optimizer.py \
    --data data/QQQ.csv \
    --strategy strategies/my_strategy.py \
    --objective sharpe_ratio \
    --trials 200
```

输出会看到一段 `[并行优化] 总 trials=... workers=120` 的提示，以及每个 batch 结束时的进度：

```
[Batch 4/12 | Exploit] 32 trials in 8.3s (3.9 tps) | best_in_batch=1.8240 | global_best=1.8472 | failed=0
```

### 4.2 新增命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--no-parallel` | False | 退回原始串行 / 多线程模式（方便调试） |
| `--n-workers` | 120 | 并行进程数（建议 `CPU 核数 - 8`） |
| `--explore-batch` | 128 | 探索阶段 batch size |
| `--exploit-batch` | 32 | 利用阶段 batch size（越小 TPE 质量越高） |
| `--trial-timeout` | 300 | 单 trial 超时秒数，超时记为 `-inf` |

### 4.3 常见场景

**128 核服务器，最大吞吐（默认最优）**

```bash
python run_optimizer.py \
    --data data/BTC.csv --strategy strategies/my_strategy.py \
    --trials 500 \
    --n-workers 120 --explore-batch 128 --exploit-batch 32
```

**16 核笔记本，适度并行**

```bash
python run_optimizer.py \
    --data data/BTC.csv --strategy strategies/my_strategy.py \
    --trials 200 \
    --n-workers 14 --explore-batch 14 --exploit-batch 14
```

**追求 TPE 质量（batch 小一点，串行感更强）**

```bash
python run_optimizer.py ... \
    --exploit-batch 8 --n-workers 64
```

**调试 / 复现问题（禁用并行）**

```bash
python run_optimizer.py ... --no-parallel
```

**期货 + 批量多 CSV + 并行**

```bash
python run_optimizer.py \
    --data "data/*.csv" \
    --strategy strategies/mm_strategy.py \
    --asset-type futures --contract-code AG \
    --objective market_maker_score \
    --trials 300 \
    --n-workers 100
```

### 4.4 Python API 直接调用

```python
from Optimizer.universal_optimizer import UniversalOptimizer

optimizer = UniversalOptimizer(
    data_path="data/BTC.csv",
    strategy_path="strategies/my_strategy.py",
    objective="sharpe_ratio",
)

result = optimizer.optimize(
    n_trials=200,
    use_parallel=True,          # 启用真多进程并行
    n_workers=120,
    explore_batch_size=128,
    exploit_batch_size=32,
    trial_timeout=300,
)
```

更底层的 API（直接用 `BatchParallelOptimizer`）见 `Optimizer/parallel_engine.py` 模块顶部 docstring 的示例。

---

## 5. 预期效果

| 指标 | 原项目（多线程） | 本版本（多进程） |
|---|---|---|
| 并行机制 | ThreadPoolExecutor，受 GIL 限制 | ProcessPoolExecutor，真并行 |
| 128 核 CPU 利用率 | 约 5–15% | 约 85–95% |
| 100 trials 耗时（回测 2s/次） | 约 200s | 约 12–18s |
| 数据加载 | 每次回测重复读 | 进程启动时一次读入 |
| TPE 采样质量 | 100% 基线 | 探索阶段无损，利用阶段约 -15%（batch=32） |
| 结果确定性 | 保持 | 保持（batch 内按原始顺序 tell） |
| 失败处理 | 中断 | 单 trial 失败只影响该 trial，study 状态一致 |

---

## 6. 故障排查

| 症状 | 可能原因 & 解法 |
|---|---|
| `PicklingError: Can't pickle strategy_class` | 策略脚本里有不可序列化的闭包或 lambda。把它们放到模块顶层。 |
| 子进程启动很慢 | 数据文件很大且多 workers 并发读。可以先把 CSV 转 parquet，或降低 `--n-workers`。 |
| 某个 trial 一直卡住 | `--trial-timeout` 缩短；超时后该 trial 自动记为 `-inf`，不会拖住整批。 |
| CPU 利用率还是上不去 | 单次回测太快（< 0.1s）时进程间调度开销占比变高，调大 `--exploit-batch` 或减少 trials。 |
| 想复现原项目行为 | `--no-parallel` 直接退回到 `study.optimize()` 老路径。 |

---

## 7. 与原项目的关系

- **输入/输出兼容**：命令行参数、JSON 结果结构、输出目录布局全部和原项目一致。
- **回退开关**：`--no-parallel` 任何时候都能切回原有的 `BayesianOptimizer.optimize_single_objective` 串行/线程路径。
- **未改动的模块**：`bayesian_optimizer.py`、`enhanced_sampler.py`、`backtest_engine.py`、`param_space_optimizer.py`、`spp_analyzer.py`、`strategy_analyzer.py`、`llm_client.py` 等都保持原样。
- **改动边界**：所有新增代码集中在 `Optimizer/parallel_engine.py`（新文件）和 `Optimizer/universal_optimizer.py` / `run_optimizer.py` 中少量的分流逻辑。
