# Log：记录内容、时机与保存路径

本文档约定 **run 级别**的日志：需要记录什么、在什么时机开始/写入、以及 log 的保存路径。与 [4-scripts](4-scripts.md)（脚本启动摘要进 log）、[6-Init_Path](6-Init_Path.md)（run 目录与 paths）、[10-Training](10-Training.md)（训练侧日志）一致。

---

## 1. 原则

- **写盘统一由 pipeline 完成**：脚本（Bash/Python）不直接写 `runs/` 下的日志文件；脚本可通过环境变量提供「启动摘要」，由 pipeline 在打开 log 时写入（见 [4-scripts](4-scripts.md) §4、§5.2）。
- **路径统一走 paths**：所有 log 文件路径来自 `init_run_paths(cfg)` 返回的 **paths**（[6-Init_Path](6-Init_Path.md) §6.7），不硬编码。
- **仅 rank0 写共享 log**：多卡时，`stdout_rank0.log`、`train.log`、`metrics/*`、`meta/*` 等仅由 rank0 写入；其他 rank 可选写 `logs/ranks/rank{N}.log`。

---

## 2. 保存路径（与 6-Init_Path 对齐）

run 目录为 `runs/<method_branch>/<exp_name>/<timestamp>/`，日志与相关产物路径如下。

| 用途 | 路径 | 说明 |
|------|------|------|
| 日志根目录 | **`paths.logs_dir`** | 即 `run_dir/logs/` |
| 合并 stdout/stderr（rank0） | **`paths.stdout_log_path`** 或 **`paths.logs_dir / "stdout_rank0.log"`** | 推荐：rank0 的终端输出重定向到此文件 |
| 应用层 logger 输出 | **`paths.logs_dir / "train.log"`** | 可选；Python logging 写入 |
| TensorBoard | **`paths.logs_dir / "events/"`** | 可选 |
| 各 rank 调试日志 | **`paths.logs_dir / "ranks/rank{N}.log"`** | 可选 |
| 训练步/epoch 指标 | **`paths.metrics_dir / "train.jsonl"`** | 见 [10-Training](10-Training.md) §5.1 |
| Epoch 时间与汇总指标 | **`paths.metrics_dir / "epochs.jsonl"`** | 由 callbacks（如 EpochTimeLogger）经 io 写入，见 §8 |
| 验证/测试指标 | **`paths.val_metrics_path`**, **`paths.test_metrics_path`** | `metrics/val.json`, `metrics/test.json` |
| 复现与元信息 | **`paths.meta_dir`** | `meta/resolved_config.yaml`, `meta/cmd.txt`, `meta/meta.json` |

**paths 对象**（§6.7）提供：`logs_dir`、`meta_dir`、`metrics_dir`；若实现里提供 `stdout_log_path` 则使用，否则用 `paths.logs_dir / "stdout_rank0.log"`。

---

## 3. 需要记录的信息

### 3.1 脚本层启动摘要（由 pipeline 写入 log）

当入口为 Bash（或 Python 脚本）时，脚本通过环境变量传递的摘要应由 **pipeline 在首次打开 run 的 log 文件时**写入该 log（[4-scripts](4-scripts.md) §5.2、§5.4）。建议记录：

- **mode**：`train` / `predict`
- **config 来源**：`--run <path>` 或 `--data/--model/--training` 路径
- **ngpu / world_size / rank**：卡数、全局 rank、world size（多卡时）
- **完整传入 Python 的命令行参数**：Bash 每次执行 `python scripts/train.py ...` 时传入的**完整参数列表**，包括：
  - 配置选择：`--run` 或 `--data` / `--model` / `--training`
  - 模式与 ckpt：`--mode`、`--ckpt`（predict 时）
  - 所有 **`--set key=value`**（预备覆盖的 config 参数，可多条）
  - 其他项目约定参数（如 `--nproc_per_node` 等）  
  这样不同 run 之间可以明确对比「究竟给 Python 传了哪些参数」，便于复现和排错。Bash 可把该完整命令行拼成字符串写入环境变量（如 `SCRIPT_PY_CMD` 或 `SCRIPT_FULL_ARGS`），pipeline 在写脚本摘要时一并写入 log（可单独一行）。
- **可选**：时间戳、主机名、可见 GPU 设备

格式示例：  
- 摘要一行：`[script] mode=train, run=configs/runs/main.yaml, ngpu=2, rank=0, world_size=2`  
- 完整命令一行（建议）：`[script] py_cmd: python scripts/train.py --run configs/runs/main.yaml --mode train --set training.epochs=10 --set data.batch_size=32`

### 3.2 Pipeline 启动与 run 元信息

在 **`init_run_paths(cfg)` 之后、开始数据/训练/预测之前**，建议：

- 创建 `paths.logs_dir`（及可选 `paths.meta_dir`）。
- 写入 **meta**（与复现强相关，见 6-Init_Path §6.5.1）：
  - **`paths.cmd_path`**（`meta/cmd.txt`）：完整 CLI 命令，便于复现。
  - **`paths.resolved_config_path`**（`meta/resolved_config.yaml`）：合并后的 cfg 快照。
- 打开 **run 的 log 文件**（如 `paths.logs_dir / "stdout_rank0.log"` 或 `train.log`）：
  - **先**读取环境变量 `SCRIPT_STARTUP_SUMMARY`（或分项 `SCRIPT_MODE` 等），若有则写一行到该 log；
  - **再**将 rank0 的 stdout/stderr 重定向到该文件（或通过 logging handler 写入），从而「从何时开始记录」= pipeline 拿到 paths 并打开 log 文件的那一刻。

### 3.3 训练与预测过程中的日志

- **训练**（[10-Training](10-Training.md) §5）：  
  - **train.jsonl**：按 `log_interval` 或每 epoch 写一行；建议字段：`time`, `epoch`, `global_step`, `loss`, `lr`, `throughput`, `gpu_mem_mb`, `metrics.*`。  
  - **val.json**：每次验证结束后写聚合结果。  
  - 可选：`train.log` 中写入应用层 logger（如 print 的替代、异常栈等）。
- **预测**：通常不需要单独的训练日志；若有 print/logger，可统一进入 rank0 的 stdout log 或 `train.log`。
- **评估**：指标写入 `metrics/val.json`、`metrics/test.json`；不单独定义「evaluation.log」，需要时可用 stdout 或 `train.log`。

### 3.4 结束或异常

- **meta.json**（`paths.meta_json_path`）：在 run 结束或关键节点写入/更新，记录例如：`mode`, `timestamp`, `ranks`, `source_ckpt`（predict 时）、`status`（ok / error）、early-stop 原因等（见 6-Init_Path §6.5.1、10-Training §5.3）。

---

## 4. 何时开始记录

| 阶段 | 时机 | 记录内容 |
|------|------|----------|
| 脚本启动 | 脚本内 | 仅 stdout 打印或设置环境变量；不写 run 目录下文件。 |
| Pipeline 拿到 paths | **init_run_paths(cfg) 之后** | 创建 `logs_dir`、打开 `stdout_rank0.log`（或 `train.log`）；**先写脚本摘要**（若存在 SCRIPT_*）；再重定向 rank0 的 stdout/stderr 到该文件。 |
| 复现快照 | 同一阶段内 | 写 `meta/cmd.txt`、`meta/resolved_config.yaml`。 |
| 训练/预测/评估 | 各模块运行中 | 写 `metrics/train.jsonl`、`val.json`、`test.json`；应用层 log 进 `train.log` 或 stdout log。 |
| Run 结束或异常 | 退出前 | 更新 `meta/meta.json`（status、source_ckpt 等）。 |

即：**「开始记录」= pipeline 在 R2（init_run_paths）之后、R3 之前（或与写 meta 同时）创建并打开本次 run 的 log 文件，并写入脚本摘要（若有）的那一刻。**

---

## 5. 实现建议（io/logging.py）

- **setup_run_log(paths, rank)**：  
  - 若 `rank == 0`：创建 `paths.logs_dir`，打开 `paths.stdout_log_path` 或 `paths.logs_dir / "stdout_rank0.log"`；读取 `os.environ.get("SCRIPT_STARTUP_SUMMARY")` 等，先写一行 `[script] ...`，再设置将 stdout/stderr 重定向到该文件（或 tee）；可选同时配置 Python logging 的 FileHandler 指向 `paths.logs_dir / "train.log"`。  
  - 若 `rank != 0` 且需要 rank 本地 log：可选打开 `paths.logs_dir / "ranks" / f"rank{rank}.log"`。
- **write_script_summary_if_present(stream)**：从环境变量读取脚本摘要并写入传入的 stream（即已打开的 log 文件 handle），供 `setup_run_log` 在重定向前调用。
- 写 meta（cmd.txt、resolved_config.yaml）可由 **io/artifacts.py** 在 pipeline 启动时调用，与 `setup_run_log` 同一阶段执行。

---

## 6. 与其它文档的对应

| 文档 | 相关约定 |
|------|----------|
| [4-scripts](4-scripts.md) | 脚本只提供启动摘要（stdout 或环境变量）；pipeline 将摘要写入 run 的 log 文件。 |
| [6-Init_Path](6-Init_Path.md) | log 保存路径：`logs_dir`、`stdout_rank0.log`、`train.log`、`meta/`、`metrics/`。 |
| [10-Training](10-Training.md) | 训练侧写 `train.jsonl`、`val.json`；路径用 paths；仅 rank0 写共享文件。 |
| [11-Prediction](11-Prediction.md) | 预测不写训练日志；若有输出走 stdout 或统一 log。 |
| [12-Evaluation](12-Evaluation.md) | 指标写 `metrics/*.json`；无单独 evaluation.log。 |

---

## 7. 小结

- **记录什么**：脚本启动摘要（mode、config、ngpu、rank、world_size）、复现快照（cmd、resolved_config）、训练/验证指标（train.jsonl、val.json、test.json）、run 元信息（meta.json）。
- **何时开始**：在 pipeline 执行 **init_run_paths(cfg) 之后**创建并打开本次 run 的 log 文件，先写脚本摘要（若有），再重定向 stdout/logger，从此开始对本次 run 的完整记录。
- **保存路径**：全部使用 **paths**（`paths.logs_dir`、`paths.metrics_dir`、`paths.meta_dir`），具体文件名见 §2 与 [6-Init_Path](6-Init_Path.md) §6.5。
