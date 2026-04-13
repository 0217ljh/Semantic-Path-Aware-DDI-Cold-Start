# Scripts: CLI Entry Points

This project uses thin CLI scripts as entry points. Scripts should only:
1) parse CLI arguments,
2) decide the run mode (train / predict),
3) support single-GPU and multi-GPU launch (prefer `torchrun`),
4) call `code/pipeline/run.py` to execute the unified pipeline.

All experiment artifacts (logs, checkpoints, metrics, resolved configs) are written by the pipeline (`io/*`, `train/*`, `eval/*`), not by the scripts.

---

## 1. `scripts/train.py`

### Purpose
Unified entry for:
- **Training** (produce checkpoint + metrics)
- **Direct prediction** (load checkpoint and evaluate, skipping training)

### Inputs (CLI signature)

#### Config selection (choose ONE)
- **Run config (recommended for reproducibility):**
  - `--run configs/runs/<exp>.yaml`
- **Three-part configs:**
  - `--data configs/data/<x>.yaml`
  - `--model configs/model/<x>.yaml`
  - `--training configs/training/<x>.yaml`

#### Mode
- `--mode train | predict`
- `--ckpt <path>` (required when `--mode predict`)

#### Overrides (optional)
- `--set key=value` (repeatable; highest priority)

#### Distributed / device (optional)
- Prefer launching multi-GPU via `torchrun`:
  - `torchrun --nproc_per_node=4 scripts/train.py --run ... --mode train`
- Script detects distributed environment from:
  - `RANK`, `WORLD_SIZE`, `LOCAL_RANK` (set by `torchrun`)
- Single GPU: run normally without `torchrun`.

### What it does (high-level)
1) Parse CLI args.
2) Detect whether running under distributed launch (via env vars).
3) Forward `args` (and distributed info) into `code/pipeline/run.py`.
4) Exit with pipeline status code.

### What it MUST NOT do
- Not responsible for:
  - writing experiment logs to files,
  - saving checkpoints,
  - computing/aggregating metrics,
  - managing `runs/` directory structure.
These are handled in the pipeline (`io/`, `train/`, `eval/`).

---

## 2. `scripts/eval.py` (optional)

### Purpose
A dedicated evaluation entry point for **checkpoint-only runs**:
- load checkpoint
- run evaluation / prediction
- save metrics and optional predictions

(If desired, this can be omitted and replaced by `scripts/train.py --mode predict`.)

### Inputs (CLI signature)
Same as `scripts/train.py`, but typically:
- `--mode predict`
- `--ckpt <path>`

### What it does (high-level)
1) Parse CLI args.
2) Forward into `code/pipeline/run.py` with `mode=predict`.
3) Exit with pipeline status code.

---

## 3. Multi-GPU Policy

### Recommended: `torchrun`
- Multi-GPU should be launched using `torchrun`, not custom Python multiprocessing in scripts.
- Scripts only **detect** distributed context using env vars:
  - `LOCAL_RANK`, `RANK`, `WORLD_SIZE`
- Distributed initialization (e.g., `init_process_group`) should live in:
  - `train/trainer.py` (preferred), or
  - `pipeline/run.py` if the entire pipeline needs rank-awareness.

---

## 4. Logging Policy

### Goal
Keep logging consistent and reproducible across all entry points.

### Rule
- Scripts may print a **startup summary** to stdout (mode, config paths, distributed world size).
- File logging, metrics logging, and artifact writing must be done by:
  - `io/logging.py` + `io/artifacts.py` (and optionally `train/callbacks.py`)

This ensures identical artifact structure whether running:
- training,
- prediction-only,
- single-GPU,
- multi-GPU.

---
**把 Bash 脚本的启动摘要写进 Python 的 log：** 若入口是 Bash 脚本，脚本不应直接写 `runs/` 下的日志文件（由 pipeline 的 `io/logging.py` 负责）。推荐做法：Bash 将启动摘要写入**环境变量**（如 `SCRIPT_STARTUP_SUMMARY` 或分项 `SCRIPT_MODE`、`SCRIPT_CONFIG`、`SCRIPT_NGPU`、`SCRIPT_RANK`、`SCRIPT_WORLD_SIZE`），再启动 Python；pipeline 在初始化日志时（例如在 `io/logging.py` 或 `pipeline/run.py` 中首次打开 run 的 log 文件时）读取这些环境变量，并**先写一行或几行**到该 log 文件，再继续正常日志。这样「脚本层」的摘要与「pipeline 层」的日志在同一文件中，且不破坏「只有 pipeline 写 run 日志」的约定。

---

## 5. 参考脚本（Bash 极简）

以下 Bash 脚本实现：**解析/转发参数**、**决定用 torchrun 还是单卡**、**把启动摘要通过环境变量交给 Python 写入 log**，最后调用 Python 入口（如 `scripts/train.py` 或直接 `python -m pipeline.run`）。

### 5.1 决定用 torchrun 还是单卡

逻辑建议：

1. **已在 torchrun 内**：若环境变量 `WORLD_SIZE` 已存在且 > 1，说明当前进程是由 `torchrun` 拉起的子进程，**不要再起一层 torchrun**，直接执行 `python ...` 即可。
2. **用户显式要求多卡**：若用户传入 `--nproc_per_node N`（或项目约定的如 `--multi-gpu`），则用 `torchrun --nproc_per_node=N` 启动 Python。
3. **其余情况**：单卡，直接 `python scripts/train.py "$@"`。

（可选：根据 `nvidia-smi` 或 `CUDA_VISIBLE_DEVICES` 自动推断卡数；这里保持极简，以「用户传参」或「已处于分布式」为准。）

### 5.2 启动摘要写入 Python log

- Bash 在调用 Python 前，把摘要拼成字符串或分项写入环境变量，例如：
  - `SCRIPT_STARTUP_SUMMARY="mode=$MODE, config=$CONFIG_SOURCE, ngpu=$NGPU, rank=${RANK:-0}, world_size=${WORLD_SIZE:-1}"`
  - 或分项：`SCRIPT_MODE`, `SCRIPT_CONFIG`, `SCRIPT_NGPU`, `SCRIPT_RANK`, `SCRIPT_WORLD_SIZE`
- Python 端（`io/logging.py` 或 `pipeline/run.py` 在创建/打开本次 run 的 log 文件时）：读取 `os.environ.get("SCRIPT_STARTUP_SUMMARY")` 或各 `SCRIPT_*`，先写一行（如 `[script] ...`）到该 log 文件，再继续正常日志。这样 Section 4 的 Logging Policy 仍满足：文件写盘只由 pipeline 做，脚本只提供摘要内容。

### 5.3 示例：`scripts/train.sh`

```bash
#!/usr/bin/env bash
# Unified entry: train or predict. Decides torchrun vs single-card, exports summary for Python log. See docs/4-scripts.md.

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- 1) 简单解析（仅示例：实际可用 getopts 或逐位解析）
MODE="train"
NPROC=""
RUN_CONFIG=""
DATA_CONFIG=""
MODEL_CONFIG=""
TRAINING_CONFIG=""
CKPT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2"; shift 2 ;;
    --run)         RUN_CONFIG="$2"; shift 2 ;;
    --data)        DATA_CONFIG="$2"; shift 2 ;;
    --model)       MODEL_CONFIG="$2"; shift 2 ;;
    --training)    TRAINING_CONFIG="$2"; shift 2 ;;
    --ckpt)        CKPT="$2"; shift 2 ;;
    --nproc_per_node) NPROC="$2"; shift 2 ;;
    --set)         EXTRA_ARGS+=(--set "$2"); shift 2 ;;
    *)             EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# --- 2) 检测是否已在 torchrun 内、GPU 信息
WORLD_SIZE="${WORLD_SIZE:-}"
RANK="${RANK:-0}"
LOCAL_RANK="${LOCAL_RANK:-0}"

if [[ -n "$WORLD_SIZE" && "$WORLD_SIZE" -gt 1 ]]; then
  IN_TORCHRUN=1
  NGPU="$WORLD_SIZE"
else
  IN_TORCHRUN=0
  NGPU=1
  if command -v nvidia-smi &>/dev/null; then
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi

# --- 3) 决定：torchrun 还是单卡
LAUNCH_CMD=()
if [[ "$IN_TORCHRUN" -eq 1 ]]; then
  # 已在 torchrun 子进程中，直接跑 Python
  LAUNCH_CMD=(python)
elif [[ -n "$NPROC" && "$NPROC" -gt 1 ]]; then
  # 用户指定多卡，用 torchrun 启动
  LAUNCH_CMD=(torchrun --nproc_per_node="$NPROC" python)
  WORLD_SIZE="$NPROC"
  NGPU="$NPROC"
else
  # 单卡
  LAUNCH_CMD=(python)
fi

# --- 4) 启动摘要：供 Python 写入 log（Section 4）
if [[ -n "$RUN_CONFIG" ]]; then
  CONFIG_SOURCE="run=$RUN_CONFIG"
else
  CONFIG_SOURCE="data=$DATA_CONFIG,model=$MODEL_CONFIG,training=$TRAINING_CONFIG"
fi
export SCRIPT_STARTUP_SUMMARY="mode=$MODE, $CONFIG_SOURCE, ngpu=$NGPU, rank=${RANK}, world_size=${WORLD_SIZE:-1}"
export SCRIPT_MODE="$MODE"
export SCRIPT_NGPU="$NGPU"
export SCRIPT_RANK="${RANK:-0}"
export SCRIPT_WORLD_SIZE="${WORLD_SIZE:-1}"

# 可选：打印到 stdout（终端可见）
echo "[script] $SCRIPT_STARTUP_SUMMARY"

# --- 5) 调用 Python 入口（由 pipeline 写 run 日志；pipeline 内读取 SCRIPT_* 写进 log 文件）
PY_ARGS=(scripts/train.py)
[[ -n "$RUN_CONFIG" ]] && PY_ARGS+=(--run "$RUN_CONFIG")
[[ -n "$DATA_CONFIG" ]] && PY_ARGS+=(--data "$DATA_CONFIG")
[[ -n "$MODEL_CONFIG" ]] && PY_ARGS+=(--model "$MODEL_CONFIG")
[[ -n "$TRAINING_CONFIG" ]] && PY_ARGS+=(--training "$TRAINING_CONFIG")
PY_ARGS+=(--mode "$MODE")
[[ -n "$CKPT" ]] && PY_ARGS+=(--ckpt "$CKPT")
PY_ARGS+=("${EXTRA_ARGS[@]}")

exec "${LAUNCH_CMD[@]}" "${PY_ARGS[@]}"
```

### 5.4 Python 端：把脚本摘要写进 run 的 log 文件

在 pipeline 首次打开本次 run 的日志文件时（例如在 `io/logging.py` 的 `setup_run_log` 或 `pipeline/run.py` 中 `init_run_paths` 之后打开 `paths.logs_dir / "stdout_rank0.log"` 或 `train.log` 时），先写一行脚本摘要，再继续：

```python
# 伪代码（在 io/logging.py 或 pipeline/run.py 中）
import os

def write_script_summary_if_present(log_file_handle):
    summary = os.environ.pop("SCRIPT_STARTUP_SUMMARY", None)
    if summary:
        log_file_handle.write(f"[script] {summary}\n")
        log_file_handle.flush()
```

这样 Bash 的启动信息就进入 Python 管理的同一 log 文件，符合 Section 4 的 Logging Policy。

### 5.5 与文档的对应

| 文档要求 | Bash 脚本中的实现 |
|----------|-------------------|
| 解析/转发 CLI（--run 或 --data/--model/--training；--mode；--ckpt；--set） | 5.3 中 while/case 解析，并组装 PY_ARGS 传给 Python |
| 决定 run mode（train / predict） | `--mode` 解析，传入 Python |
| 多 GPU 用 torchrun，脚本只负责决定是否启动 torchrun | 已在 torchrun 内则不再起 torchrun；否则若 `--nproc_per_node N` 则 `torchrun --nproc_per_node=N python ...`，否则单卡 `python ...` |
| 脚本启动摘要可打 stdout | `echo "[script] $SCRIPT_STARTUP_SUMMARY"` |
| 摘要进入 Python 的 log 文件 | 导出 `SCRIPT_STARTUP_SUMMARY`（及可选分项）；Python 在打开 run log 时读取并写入（5.4） |
| 不写 logs/ckpt/metrics | Bash 仅设置环境变量并 exec Python，所有写盘在 pipeline 内 |
| 退出码为 pipeline 状态 | `exec` Python 后，退出码即 Python 进程退出码 |

---
### 5.6 直接用 Python 启动

调试或不想经 Bash 时，可直接用 Python 调用同一套 CLI，参数与 §1、§2 一致。

**单卡：**
```bash
python scripts/train.py --run main --mode train
python scripts/train.py --run configs/runs/main.yaml --mode train --set training.epochs=10 --set data.batch_size=32
python scripts/train.py --run main --mode predict --ckpt /path/to/ckpt/best.pt
```

**多卡：** 需自行使用 `torchrun` 包裹，无 `--nproc_per_node` 参数（该参数仅由 Bash 脚本解析）：
```bash
torchrun --nproc_per_node=4 python scripts/train.py --run main --mode train
torchrun --nproc_per_node=4 python scripts/train.py --run main --mode train --set training.epochs=20
```

**说明：**
- 三块 config 同样支持：`--data configs/data/main.yaml --model configs/model/main.yaml --training configs/training/main.yaml --mode train`。
- 若希望 run 的 log 里也有 `[script]`、`py_cmd` 等脚本层摘要（与 Bash 启动一致），可在 `scripts/train.py` 里在调用 pipeline 前设置环境变量，例如 `os.environ["SCRIPT_STARTUP_SUMMARY"] = "..."`、`os.environ["SCRIPT_PY_CMD"] = "python scripts/train.py " + " ".join(sys.argv[1:])`（可选，见 [13-Log](13-Log.md) §3.1）。

---