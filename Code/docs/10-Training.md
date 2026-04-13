# Training (R8): Interfaces, Outputs, Logging, Callbacks (Model-Agnostic)

This document defines the pipeline **training step** (corresponding to step **T** in [3-Pipeline](3-Pipeline.md): `train(...)` in `train/trainer.py`): input/output interfaces, logging and artifact conventions, and callbacks. It does not define model structure or forward logic; the training module only interacts with the **method** via the agreed interface.

---

## 1. Scope and responsibilities

**Training module is responsible for:**

- Receiving **cfg**, **loaders**, **method**, **paths** (unified path object from [6-Init_Path](6-Init_Path.md)).
- Running the training loop (single-GPU or DDP).
- Recording train/val metrics and run info (machine-readable + human-readable).
- Saving checkpoints (last / best / optional epoch snapshots).
- Hooking extensible behavior via **callbacks** (e.g. early stop, extra logging, eval scheduling).

**Training module is not responsible for:**

- Building data (Dataset / collate / DataLoader): that is [9-DatasetAndCollate](9-DatasetAndCollate.md) (R4/R5).
- Building base processed / resources / feature caches: that is R3 / R3_6 / R7.
- Defining the model: that is the **method** (loaded via [pipeline/registry](2-Repo_Layout.md) from `cfg["model"]["name"]`).

---

## 2. Entry point

The training module exposes a single entry to the pipeline:

```text
train(cfg, loaders, method, paths, callbacks=None) -> TrainOutput
```

| Argument   | Source | Description |
|-----------|--------|-------------|
| `cfg`     | R1 [5-Config](5-Config.md) | Merged config from `load_cfg(args)`; trainer reads `cfg["training"]`, `cfg["data"]`, `cfg["model"]` as needed. |
| `loaders` | R5 [9-DatasetAndCollate](9-DatasetAndCollate.md) | Output of `build_loaders(...)`; dict with keys `"train"`, `"val"`, `"test"`. |
| `method`  | R6 (pipeline/registry) | Loaded via `load_method(cfg["model"]["name"])`; trainer only uses the agreed method interface. |
| `paths`  | R2 [6-Init_Path](6-Init_Path.md) §6.7 | Output of `init_run_paths(cfg)`; all writes go under these paths. |
| `callbacks` | Optional | List of callables; trainer invokes them at defined events. |

---

## 3. Inputs (contract)

### 3.1 cfg (training-side minimum)

Trainer reads from **`cfg["training"]`** (and optionally `cfg["data"]` / `cfg["model"]`). Align with [required_config_keys](required_config_keys.md) and [6-Init_Path](6-Init_Path.md) §6.5.3 (best.pt uses `cfg["training"]["monitor"]`).

**Required for trainer:**

| Key | Type | Description |
|-----|------|-------------|
| `cfg["training"]["epochs"]` | int | Number of epochs (see [required_config_keys](required_config_keys.md) training). |
| `cfg["training"]["batch_size"]` | int | Batch size (DataLoader already uses it; trainer may log it). |
| `cfg["training"]["log_interval"]` | int | Step-level log frequency (e.g. every N steps). |
| `cfg["training"]["save_strategy"]` | str or dict | Checkpoint strategy (e.g. `"last+best"`). |

**Optional:**

- `grad_accum_steps`, `max_steps`, `amp` (fp16/bf16/off), `clip_grad_norm`
- `eval_interval` (run val every N steps), `save_interval` (every N steps or epoch)
- **`monitor`**: metric name for best checkpoint and early stop (e.g. `val_auc`, `val_loss`). Must match [6-Init_Path](6-Init_Path.md) §6.5.3: **best.pt** is determined by **`cfg["training"]["monitor"]`**.
- `seed`: if the project uses only `data.global_seed`, this can be omitted.

Convention: trainer reads **`cfg["training"]["..."]`** / **`cfg["data"]["..."]`** / **`cfg["model"]["..."]`** consistently.

### 3.2 loaders

Trainer expects the **R5** output (see [9-DatasetAndCollate](9-DatasetAndCollate.md)):

```text
loaders = {
  "train": <DataLoader>,
  "val":   <DataLoader>,   # optional but strongly recommended for best/early-stop
  "test":  <DataLoader>    # optional (test usually in evaluator)
}
```

Minimum: **`"train"`** is required. For best/early-stop, **`"val"`** is required.

### 3.3 paths

Trainer must **not** hardcode any write paths; all artifacts go to **paths** from [6-Init_Path](6-Init_Path.md) §6.7. The following must be available (names aligned with doc 6):

**Directories:** `paths.run_dir`, `paths.meta_dir`, `paths.logs_dir`, `paths.ckpt_dir`, `paths.metrics_dir`, `paths.preds_dir` (optional), `paths.figures_dir`, `paths.artifacts_dir` (optional).

**Canonical file paths (doc 6 §6.7 and §6.5):**

| Use | Path / convention |
|-----|--------------------|
| Last checkpoint | `paths.last_ckpt_path` (e.g. `ckpt/last.pt`) |
| Best checkpoint | `paths.best_ckpt_path` (e.g. `ckpt/best.pt`) |
| Train step log | `paths.metrics_dir / "train.jsonl"` (doc 6 §6.5.4: `metrics/train.jsonl`) |
| Val metrics | `paths.val_metrics_path` (e.g. `metrics/val.json`) |
| Test metrics | `paths.test_metrics_path` (e.g. `metrics/test.json`) |
| Stdout log (rank0) | `paths.logs_dir / "stdout_rank0.log"` (doc 6 §6.5.2) |
| Meta / config | `paths.resolved_config_path`, `paths.meta_json_path`, `paths.cmd_path` (written at run start, not by trainer) |

If the **paths** object does not expose `train_metrics_path`, use **`paths.metrics_dir` + `"train.jsonl"`** and **`paths.logs_dir` + `"stdout_rank0.log"`** so filenames match [6-Init_Path](6-Init_Path.md) §6.5.

### 3.4 method (trainer ↔ method interface)

Trainer does not know the model internals; it only uses the following on the **method** object (loaded by R6 from [code/methods/](2-Repo_Layout.md)):

**Required:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `setup` | `method.setup(cfg, runtime) -> None` | Initialize device, DDP, optimizer, etc. (project-defined). |
| `train_step` | `method.train_step(batch, runtime) -> StepOutput` | One forward + loss + optional metrics. |
| `eval_step` | `method.eval_step(batch, runtime) -> StepOutput` | Validation batch → metrics/preds (if trainer runs val). |
| `state_dict` | `method.state_dict() -> dict` | For saving checkpoint. |
| `load_state_dict` | `method.load_state_dict(state: dict) -> None` | For resume or load weights. |

**Optional:** `method.on_epoch_end(runtime) -> dict` (epoch-level metrics); `method.zero_grad()` / `method.step()` if trainer does not manage optimizer.

**StepOutput (suggested):** `{"loss": float, "metrics": {str: number}, "preds": optional, "targets": optional}`. Trainer only requires **loss** (train) and serializable **metrics**.

### 3.5 Distributed / DDP

- **Only rank 0** writes shared files: checkpoints, aggregated metrics, meta.
- Other ranks: at most rank-local debug logs (optional).
- All ranks: same training behavior (gradients synchronized via DDP).

See [6-Init_Path](6-Init_Path.md) and [7-GetOrLoadData](7-GetOrLoadData.md) for the same “rank 0 only” rule.

---

## 4. Outputs (contract)

### 4.1 Return: TrainOutput

`train(...)` returns a small object for downstream (e.g. evaluator or reporting):

```text
TrainOutput = {
  "status": "ok" | "early_stopped" | "error",
  "global_step": int,
  "epochs_ran": int,
  "last_ckpt_path": str,
  "best_ckpt_path": str | None,
  "best_metric": float | None,
  "best_epoch": int | None
}
```

Train does **not** return large prediction tensors; those are written by the **evaluator** under `paths.preds_dir` (see evaluator doc).

### 4.2 Files written (standard artifacts)

**Checkpoints** ([6-Init_Path](6-Init_Path.md) §6.5.3): under `paths.ckpt_dir`:

- **last.pt**: last checkpoint (required).
- **best.pt**: best by **`cfg["training"]["monitor"]`** (optional).
- **epoch_xxxx.pt**: optional snapshots.

**Metrics** (doc 6 §6.5.4): under `paths.metrics_dir`:

- **train.jsonl**: step/epoch train log (strongly recommended).
- **val.json**: aggregated val metrics (if val is run).
- **best.json**: best record (optional).

**Logs** (doc 6 §6.5.2): under `paths.logs_dir`:

- **stdout_rank0.log**: stdout/stderr from rank 0.
- **train.log**: logger output (optional).
- **events/**: TensorBoard (optional).
- **ranks/**: rank-local logs (optional).

---

## 5. Logging (what to log)

### 5.1 metrics/train.jsonl

One line per record (per `log_interval` step or per epoch). Suggested fields: `time`, `epoch`, `global_step`, `loss`, `lr`, `throughput` (samples/sec), `gpu_mem_mb`, `metrics.*` (e.g. acc, f1).

### 5.2 metrics/val.json

Aggregated val metrics: e.g. `epoch`, `global_step`, `metrics`, `loss`, `n_samples`, `best_so_far`.

### 5.3 meta/

Trainer does not merge config; it may write **monitor** value, best ckpt path, early-stop reason to **`paths.meta_json_path`** or **`metrics/best.json`** for reproducibility (see [6-Init_Path](6-Init_Path.md) §6.5.1).

---

## 6. Callbacks

Callbacks extend the training loop without changing its core: save, log, early-stop, eval schedule, etc.

**Lifecycle events (suggested):** `on_train_start(ctx)`, `on_epoch_start(ctx)`, `on_step_end(ctx, step_output)`, `on_eval_end(ctx, eval_output)`, `on_checkpoint_end(ctx, ckpt_path)`, `on_train_end(ctx, train_output)`.

**Context `ctx`:** at least `cfg`, `paths`, `rank`, `world_size`, `epoch`, `global_step`, `mode`, optional `runtime` (device, optimizer, scheduler, timers, etc.).

**Built-in callbacks (roles only):** MetricLoggerCallback (write train.jsonl), CheckpointCallback (last/best), EarlyStoppingCallback (use `cfg["training"]["monitor"]`), EvalSchedulerCallback, TensorboardCallback. Trainer must run correctly with **no** callbacks (minimal loop: last checkpoint + basic log).

---

## 7. Boundary with evaluator

Trainer may run **val** (for best/early-stop) but does not have to run **test**. Final **test** metrics and **preds** should be written by the **evaluator** (R9), so formats are consistent and trainer/evaluator do not duplicate logic.

---

## 8. Summary (interface checklist)

- **Input:** cfg (R1), loaders (R5), method (R6), paths (R2), callbacks (optional).
- **Output:** TrainOutput + writes to **ckpt/**, **metrics/**, **logs/** per [6-Init_Path](6-Init_Path.md).
- **Logging:** train.jsonl (aggregatable) + val.json (aggregated) + stdout/train.log.
- **Callbacks:** Extend via events; do not hard-depend on any callback for correct training.
- **DDP:** Rank 0 writes shared files; others optional local logs.
- **best.pt:** Determined by **`cfg["training"]["monitor"]`** ([6-Init_Path](6-Init_Path.md) §6.5.3).

---

## 9. Cross-references and alignment

| Item | Doc / location |
|------|-----------------|
| cfg source | [5-Config](5-Config.md) (R1 load_cfg) |
| paths object | [6-Init_Path](6-Init_Path.md) §6.7 (R2 init_run_paths) |
| run dir layout | [6-Init_Path](6-Init_Path.md) §6.5 (ckpt/, metrics/, logs/, meta/) |
| best.pt / monitor | [6-Init_Path](6-Init_Path.md) §6.5.3; [required_config_keys](required_config_keys.md) training.monitor |
| loaders | [9-DatasetAndCollate](9-DatasetAndCollate.md) (R5 build_loaders) |
| method load | [2-Repo_Layout](2-Repo_Layout.md) code/methods/, pipeline/registry (R6) |
| Pipeline step | [3-Pipeline](3-Pipeline.md) step **T** (train) |
| training keys | [required_config_keys](required_config_keys.md) + optional log_interval, save_strategy |

**Gaps / suggestions:**

1. **paths.train_metrics_path:** [6-Init_Path](6-Init_Path.md) §6.7 does not list `train_metrics_path`. Trainer uses **`paths.metrics_dir / "train.jsonl"`** as the canonical path (see §3.3). Optionally add `train_metrics_path` to doc 6 for a single contract.
2. **paths.stdout_log_path:** Same: trainer uses **`paths.logs_dir / "stdout_rank0.log"`** (see §3.3). Optionally add `stdout_log_path` to doc 6.
3. **required_config_keys:** Add **monitor** (and optionally **log_interval**, **save_strategy**) under training in [required_config_keys](required_config_keys.md) if you want them in the shared schema.
4. **data_key / R2_5 in Pipeline:** 暂时不动；全文档完成后与 [3-Pipeline](3-Pipeline.md)、[7-GetOrLoadData](7-GetOrLoadData.md) 一并做一致性检查。
5. **Module naming (e.g. R8 vs T):** 留待全文档完成后统一做命名与步骤编号检查。
