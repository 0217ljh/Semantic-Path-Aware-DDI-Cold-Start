# Prediction: Interfaces and Outputs

This document defines the **predict** step: load a trained checkpoint, run inference on **selected splits** (e.g. val, test), and write predictions. It focuses on **input/output interfaces**.

**Pipeline 约定（[3-Pipeline](3-Pipeline.md)）：**
- **Train 路径：** 先 `train(...)` 得到 **TrainOutput**（含 `best_ckpt_path`），pipeline 用该路径调用 **predict** 在选定的 val/test 上跑预测 → **PredictOutput** → evaluate。
- **Predict-only 路径：** 不训练，直接用 `cfg["ckpt"]` 调用 **predict** 在选定的 val/test 上跑预测 → **PredictOutput** → evaluate。

**接口约定：** 与 Training **只对齐输入**（cfg, loaders, method, paths），**输出不同**：Training 返回最佳模型路径等（TrainOutput），Prediction 返回预测结果路径等（PredictOutput）。

---

## 1. Scope and responsibilities

**Prediction module is responsible for:**

- Receiving **cfg**, **loaders**, **method**, **paths** (same contract as Training; see [6-Init_Path](6-Init_Path.md)).
- Loading a **checkpoint** (from `cfg["ckpt"]` or equivalent) into the **method**.
- Running inference over the given **loaders** (e.g. `"test"` or `"val"`).
- Writing **predictions** under `paths.preds_dir` (e.g. `paths.test_preds_path`, `paths.val_preds_path`).

**Prediction module is not responsible for:**

- Building data or loaders: that is [9-DatasetAndCollate](9-DatasetAndCollate.md) (R5).
- Defining the model: that is the **method** (same as Training, loaded via [pipeline/registry](2-Repo_Layout.md)).
- Computing metrics from predictions: that is the **evaluator** ([12-Evaluation](12-Evaluation.md)).

---

## 2. Entry point (aligned with Training)

The prediction module exposes a single entry with the **same argument shape** as `train(...)` so the pipeline can call it consistently:

```text
predict(cfg, loaders, method, paths, callbacks=None) -> PredictOutput
```

| Argument   | Source | Description |
|-----------|--------|-------------|
| `cfg`     | R1 [5-Config](5-Config.md) | Merged config; predictor reads `cfg["model"]`, `cfg["data"]`, and **`cfg["ckpt"]`** (checkpoint path). |
| `loaders` | R5 [9-DatasetAndCollate](9-DatasetAndCollate.md) | Same as Training: dict with keys `"train"`, `"val"`, `"test"`. Predict typically uses only `"test"` (or `"val"`). |
| `method`  | R6 (pipeline/registry) | Same as Training: loaded via `load_method(cfg["model"]["name"])`; predictor uses **setup**, **load_state_dict**, **eval_step** only. |
| `paths`  | R2 [6-Init_Path](6-Init_Path.md) §6.7 | Same object as Training; all writes go under these paths (e.g. `paths.preds_dir`). |
| `callbacks` | Optional | List of callables; predictor invokes them at defined events (e.g. on_predict_start, on_batch_end, on_predict_end). |

---

## 3. Inputs (contract)

### 3.1 cfg (prediction-side)

Predictor reads **`cfg["model"]`**, **`cfg["data"]`** (e.g. batch size), and the **checkpoint path**:

| Key | Description |
|-----|-------------|
| `cfg["ckpt"]` | Path to the checkpoint file to load (e.g. `runs/main/exp/ts/ckpt/best.pt`). May be set from args or from a previous run. |
| `cfg["model"]["name"]` | Method name (same as Training); used to load the method and to initialize run paths. |
| `cfg["data"]` | Batch size, device, etc., as needed for inference. |

No **`cfg["training"]`** is required for prediction.

### 3.2 loaders (same as Training)

Same contract as [10-Training](10-Training.md) §3.2:

```text
loaders = {
  "train": <DataLoader>,   # optional for predict
  "val":   <DataLoader>,   # optional
  "test":  <DataLoader>    # typically used for predict
}
```

For predict-only, at least one loader must be provided (usually **`"test"`**). The predictor iterates over the requested split(s) and writes preds to the corresponding paths (e.g. `paths.test_preds_path`, `paths.val_preds_path`).

### 3.3 paths (same object as Training)

Predictor must **not** hardcode any write paths. All artifacts go to **paths** from [6-Init_Path](6-Init_Path.md) §6.7.

**Read:** Checkpoint is loaded from **`cfg["ckpt"]`**, not from `paths` (predict mode does not assume checkpoints live in the current run_dir). Optionally, `meta/meta.json` may record `source_ckpt` for reproducibility.

**Write:** Under `paths.preds_dir` (doc 6 §6.5.4, §6.6.2):

| Use | Path / convention |
|-----|--------------------|
| Test predictions | `paths.test_preds_path` (e.g. `preds/test.parquet`) |
| Val predictions | `paths.val_preds_path` (e.g. `preds/val.parquet`) |
| Meta (source ckpt) | `paths.meta_json_path` — record `source_ckpt` in meta (doc 6 §6.6.2). |

Stdout/logs: same as Training (e.g. `paths.logs_dir / "stdout_rank0.log"` when applicable).

### 3.4 method (predictor ↔ method interface)

Predictor uses **only** the following part of the **same** method interface as Training ([10-Training](10-Training.md) §3.4):

| Method | Purpose |
|--------|---------|
| `setup(cfg, runtime) -> None` | Initialize device, etc. (no optimizer needed for inference). |
| `load_state_dict(state: dict) -> None` | Load weights from the checkpoint. |
| `eval_step(batch, runtime) -> StepOutput` | One forward pass; predictor collects **preds** (and optionally **targets**) from `StepOutput`. |

**StepOutput (same as Training):** `{"loss": optional, "metrics": optional, "preds": ..., "targets": optional}`. Predictor only requires **preds** (and optionally **targets** for writing labeled outputs). No **train_step** is used.

---

## 4. Outputs (contract)

### 4.1 Return: PredictOutput (evaluator-friendly)

`predict(...)` returns a small object so the **evaluator** can consume it the same way whether predictions came from a training run or a predict-only run:

```text
PredictOutput = {
  "status": "ok" | "error",
  "ckpt_path": str,           # checkpoint that was loaded
  "preds_path": str | dict,   # primary preds path, or {"test": path, "val": path}
  "n_samples": int | dict,    # total samples predicted, or per-split
}
```

- **preds_path:** Either a single path (e.g. `paths.test_preds_path`) or a dict `{"test": paths.test_preds_path, "val": paths.val_preds_path}`. Evaluator reads from these paths to compute metrics.
- **n_samples:** Optional; useful for logging and for evaluator sanity checks.

This keeps the **output shape** compatible with what the evaluator expects: it always receives **paths to prediction files** (and optionally metadata), not raw tensors.

### 4.2 Files written (standard artifacts)

Per [6-Init_Path](6-Init_Path.md) §6.6.2 (predict mode):

- **preds/**  
  - `test.parquet` (or project convention) via `paths.test_preds_path`.  
  - `val.parquet` via `paths.val_preds_path` if val loader was run.
- **meta/meta.json**  
  - Include **source_ckpt** (the path passed as `cfg["ckpt"]`) for reproducibility.
- **logs/**  
  - Optional: `stdout_rank0.log`; no training logs.

Predictor does **not** write checkpoints or training metrics.

---

## 5. Alignment with Training and Evaluator

| Aspect | Training | Prediction |
|--------|----------|------------|
| Entry | `train(cfg, loaders, method, paths, callbacks)` | `predict(cfg, loaders, method, paths, callbacks)` |
| cfg | `cfg["training"]`, `cfg["model"]`, `cfg["data"]` | `cfg["ckpt"]`, `cfg["model"]`, `cfg["data"]` |
| loaders | Same dict (`"train"`, `"val"`, `"test"`) | Same dict; typically only `"test"` (or `"val"`) used |
| method | setup, train_step, eval_step, state_dict, load_state_dict | setup, load_state_dict, eval_step |
| paths | Same object; writes ckpt/, metrics/, logs/ | Same object; writes preds/, meta/, logs/ |
| Return | TrainOutput (ckpt paths, status, …) | PredictOutput (ckpt_path, preds_path, n_samples, status) |

**Feeding into Evaluator:**  
Evaluator can take either:
- After **train**: `TrainOutput` (and paths) and run evaluation on val/test using **method** and loaders, writing preds + metrics under **paths**, or  
- After **predict**: `PredictOutput` with **preds_path** already set; evaluator only needs to **read preds** from those paths and compute metrics (and optionally write to `paths.metrics_dir` / `paths.figures_dir`).

So the **interface is consistent**: same `(cfg, loaders, method, paths)` in, and in both cases the evaluator receives **paths to predictions and run metadata** (paths object + TrainOutput or PredictOutput).

---

## 6. Summary

- **Input：** 与 Training 对齐：**cfg**, **loaders**, **method**, **paths**（及可选 callbacks）。Predict 还需 **cfg["ckpt"]**（train 路径下由 pipeline 用 TrainOutput.best_ckpt_path 填入）。
- **Behavior：** 加载 checkpoint → 在选定 loaders（val/test）上跑 **eval_step** → **流式写入**推理记录到 **paths.preds_dir** 下的 **CSV**（不把全部结果压入内存）；多卡时正确 **聚合**各 rank 结果，最终得到一份完整 CSV。
- **Output：** **PredictOutput**（与 TrainOutput 不同）：仅含 **status**, **ckpt_path**, **preds_path**（推理记录 **CSV** 路径）, **n_samples**；**不返回**原始推理数据；evaluator 直接读 CSV 计算指标。
- **Paths：** 写盘一律走 **paths** ([6-Init_Path](6-Init_Path.md))；meta 记录 **source_ckpt**。
