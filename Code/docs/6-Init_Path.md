# 6. Init Paths: `init_run_paths(cfg)`

This document specifies how the pipeline initializes **all output paths** for a run, including:
- run identity (method + experiment label + timestamp),
- directories for **checkpoints**, **evaluation outputs**, and **logs**,
- a single `paths` object returned to downstream modules so every component writes to consistent locations.

`init_run_paths(cfg)` is called early in `pipeline/run.py` (step R2 in the pipeline diagram), after `load_cfg(args)` has produced a unified `cfg`.

---

## 6.1 Purpose and principles

### Purpose
- Create a **run directory** that uniquely identifies the experiment variant.
- Provide standardized sub-paths for:
  - training checkpoints,
  - evaluation metrics and predictions,
  - logs and run metadata.
- Return a single `paths` object so trainer/evaluator/io code never hard-codes file locations.

### Principles
- **Method-first organization:** group runs by `cfg["model"]["name"]`.
- **Run self-contained:** every run directory contains config snapshots + artifacts needed for reproduction.
- **Consistent names:** downstream writes to stable filenames (e.g., `metrics/test.json`, `ckpt/best.pt`).
- **DDP-safe:** only rank0 creates directories and writes shared files; other ranks write only rank-local logs if needed.

---

## 6.2 Inputs (from cfg / args)

`init_run_paths` consumes only information required to form run identity and layout:

- `cfg["model"]["name"]`: method identifier (e.g., `main`, `baseline_x`).
- `cfg.get("run", {}).get("exp_name")` (or top-level `cfg.get("exp_name")`): human-readable experiment label (optional).
- `cfg.get("mode")` (or provided separately from args): `train` or `predict`.
- `cfg.get("ckpt")` (or args.ckpt): checkpoint path when `mode == "predict"`.
- Optional runtime context:
  - timestamp (generated at runtime),
  - distributed rank info (RANK/WORLD_SIZE/LOCAL_RANK) if available.

> Note: `init_run_paths` does not compute `data_key` / `method_key` / `feature_key`. These keys are generated later; they may be appended into `meta/keys.json` after they become available.

---

## 6.3 Output branch layout (method → run root)

### 6.3.1 Method branch
The pipeline derives the top-level output branch from `cfg["model"]["name"]`:

- If `cfg["model"]["name"] == "main"`:
  - `runs/main/`
- Otherwise (baselines):
  - `runs/baselines/<method_name>/`

This ensures “which method” is encoded in the path and allows easy aggregation:
- scan `runs/main/**` for main method variants,
- scan `runs/baselines/**/**` for baseline runs.

---

## 6.4 Run identity: `<exp_name>/<timestamp>/...`

Within each method branch, each run is placed under:

`<method_branch>/<exp_name>/<timestamp>/`


Where:
- `<exp_name>` distinguishes **hyperparameter variants** within the same method.
- `<timestamp>` prevents collisions across repeated launches of the same variant.

### 6.4.1 How to set `exp_name`
Recommended options:

1. **Explicit in run config**  
   Set `run.exp_name` (or `exp_name` at top level) in the run YAML, e.g.:
   - `main_lr0.001`
   - `baseline_x_epoch50`
   - `main_batch32_seed42`

2. **Override via CLI**  
   When using a default method label and only tweaking a few hyperparameters:
   - `--set run.exp_name=main_lr0.001_batch32`

3. **Derived default (when missing)**  
   If `exp_name` is not provided, the pipeline **generates a short hash** (e.g. of the resolved cfg) and uses that as the effective `exp_name`. The full config is saved under `meta/resolved_config.yaml`, so the run remains fully identifiable.

Project may also allow other derivation (e.g. run config file name, or method + hash); choose one convention and document it.

### 6.4.2 Timestamp
`timestamp` is generated at runtime (e.g., `YYYYMMDD-HHMMSS`) and is not part of config merge.

**Multi-card sync:** The timestamp must be **identical on all ranks** so that every process writes to the same `run_dir`. Recommended: **rank0 generates the timestamp and broadcasts it** (or all ranks read it from the same env var / shared file). Otherwise each process may generate a different timestamp and write to different directories.

---

## 6.5 Standard run directory structure

Let: 

run_dir = `runs/<method_branch>/<exp_name>/<timestamp>/`

### 6.5.1 Metadata and reproducibility snapshots

run_dir/
meta/
meta.json # runtime info: mode, timestamp, ranks, ckpt source (if predict)
cmd.txt # exact CLI command line
resolved_config.yaml # merged cfg snapshot (final)
keys.json # (optional) data_key/method_key/feature_key (filled later)

**Minimum requirement:** every run must save `resolved_config.yaml` and `cmd.txt` under `meta/`.


### 6.5.2 Logs
run_dir/
logs/
stdout_rank0.log # rank0 combined stdout/stderr (recommended)
train.log # logger output (optional)
events/ # tensorboard events (optional)
ranks/ # optional rank-specific logs: ranks/rank1.log, ...


Guidelines:
- By default, write file logs only on **rank0**.
- If needed for debugging, allow rank-local logs under `logs/ranks/`.

### 6.5.3 Checkpoints (training outputs)
run_dir/
ckpt/
last.pt # last checkpoint
best.pt # best checkpoint by a monitored metric (if enabled)
epoch_0001.pt # optional epoch snapshots


Notes:
- Only `train` mode produces checkpoints.
- In `predict` mode, do not copy checkpoints into `run_dir` by default; record the source checkpoint path in `meta/meta.json`.

### 6.5.4 Evaluation outputs
Evaluation outputs should be stored regardless of mode (train or predict).

run_dir/
metrics/
train.jsonl # optional: per-step/epoch logs (easy to aggregate)
val.json # final validation metrics
test.json # final test metrics
preds/
val.parquet # optional prediction table (id, y, yhat, prob, ...)
test.parquet
figures/ # optional: curves, confusion matrix, etc.
artifacts/ # optional: explanations, error cases, attention maps, etc.


Guidelines:
- Metrics should use stable filenames so external summarizers can glob:
  - `runs/**/metrics/test.json`
- Predictions are optional but strongly recommended when you want error analysis.

---

## 6.6 Train vs Predict behavior

### 6.6.1 `mode == "train"`
- Writes:
  - `ckpt/` (last/best/optional epochs),
  - `metrics/train.jsonl` (optional),
  - `metrics/val.json`, `metrics/test.json`,
  - optional `preds/*` and `figures/*`.

### 6.6.2 `mode == "predict"` (no-train)
- Creates a new `run_dir` (recommended) and writes:
  - `metrics/*.json`,
  - optional `preds/*.parquet`,
  - `meta/meta.json` with `source_ckpt` path.
- Does not write `ckpt/` (or may create `ckpt/link.txt` that points to the source).

This keeps prediction-only evaluations isolated and reproducible.

---

## 6.7 Returned object: `paths`

`init_run_paths(cfg)` returns a single object (dict or dataclass) containing:

- Directories:
  - `run_dir`, `meta_dir`, `logs_dir`, `ckpt_dir`, `metrics_dir`, `preds_dir`,
    `figures_dir`, `artifacts_dir`
- Common file paths:
  - `resolved_config_path`
  - `meta_json_path`
  - `cmd_path`
  - `last_ckpt_path`, `best_ckpt_path`
  - `val_metrics_path`, `test_metrics_path`
  - `val_preds_path`, `test_preds_path`

All downstream modules (trainer/evaluator/io) must use `paths` to read/write artifacts.

---

## 6.8 Summary

- Method branch is determined by `cfg["model"]["name"]`:
  - `main → runs/main/`
  - `baseline_x → runs/baselines/baseline_x/`
- Each run is stored under:
  - `runs/<branch>/<exp_name>/<timestamp>/`
- Standard subdirectories:
  - `meta/` (config snapshot + cmd + runtime info)
  - `logs/` (stdout/logger/tensorboard)
  - `ckpt/` (training checkpoints)
  - `metrics/` and `preds/` (evaluation outputs)
- `init_run_paths` returns a unified `paths` object used by all downstream steps.
