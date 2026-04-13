# Evaluation: Interfaces and Outputs

This document defines the **evaluation** step: read prediction CSV(s) from [11-Prediction](11-Prediction.md), get metric names from **cfg**, compute metrics, and write results under **paths**. It is intentionally **simple**: from cfg get what to compute, then compute.

Pipeline: see [3-Pipeline](3-Pipeline.md) — after **predict** → **PredictOutput** → **evaluate(...)** → write metrics/figures under paths.

---

## 1. Scope and responsibilities

**Evaluation module is responsible for:**

- Receiving **cfg**, **PredictOutput** (with `preds_path` pointing to prediction CSV files), and **paths** ([6-Init_Path](6-Init_Path.md)).
- Reading **which metrics to compute** from **cfg** (e.g. `cfg["evaluation"]["metrics"]` or `cfg["metrics"]`).
- Reading prediction records from the **CSV** at `preds_path` (per split: test, val).
- **Computing** the requested metrics and writing them under `paths.metrics_dir` (e.g. `paths.test_metrics_path`, `paths.val_metrics_path` per [6-Init_Path](6-Init_Path.md) §6.5.4).
- Optionally writing figures under `paths.figures_dir`.

**Evaluation module is not responsible for:**

- Running inference or writing prediction CSV: that is [11-Prediction](11-Prediction.md).

---

## 2. Entry point

```text
evaluate(cfg, predict_output, paths) -> EvalOutput
```

| Argument | Description |
|----------|-------------|
| `cfg` | Merged config; evaluator reads **which metrics to compute** from e.g. `cfg["evaluation"]` or `cfg["metrics"]`. |
| `predict_output` | **PredictOutput** from the predict step: contains `preds_path` (str or dict of split → CSV path). |
| `paths` | Unified path object from [6-Init_Path](6-Init_Path.md) §6.7; evaluator writes to `paths.metrics_dir`, optionally `paths.figures_dir`. |

**Return:** **EvalOutput** (see §4).

---

## 3. Inputs (contract)

### 3.1 cfg (evaluation-side)

Evaluator reads **which metrics to compute** from cfg. Suggested key:

| Key | Type | Description |
|-----|------|-------------|
| `cfg["evaluation"]["metrics"]` | list[str] or dict | Metric names to compute (e.g. `["accuracy", "f1", "auc"]`) or per-metric config. |

Project may use `cfg["metrics"]` or another key; document the convention in the codebase.

### 3.2 predict_output

**PredictOutput** from [11-Prediction](11-Prediction.md):

- **preds_path:** Single path (e.g. `paths.test_preds_path`) or dict `{"test": path, "val": path}`. Each path points to a **CSV** of prediction records (e.g. columns `id`, `y_true`, `y_pred`, `prob`).
- Evaluator **reads these CSV files** and computes metrics; no raw tensors.

### 3.3 paths

Evaluator must **not** hardcode any write paths. All writes go to **paths** from [6-Init_Path](6-Init_Path.md) §6.7.

**Write:**

- **metrics/** — e.g. `paths.test_metrics_path` (e.g. `metrics/test.json`), `paths.val_metrics_path` (e.g. `metrics/val.json`). Content: computed metrics per split (e.g. `{"accuracy": 0.9, "f1": 0.85}`).
- **figures/** — optional (e.g. confusion matrix, ROC curve) under `paths.figures_dir`.

---

## 4. Outputs (contract)

### 4.1 Return: EvalOutput

```text
EvalOutput = {
  "status": "ok" | "error",
  "metrics": dict,       # e.g. {"test": {"accuracy": 0.9, "f1": 0.85}, "val": {...}}
  "metrics_path": dict,  # e.g. {"test": paths.test_metrics_path, "val": paths.val_metrics_path}
}
```

- **metrics:** Computed metric values per split.
- **metrics_path:** Paths where metrics were written (so downstream can read or log).

### 4.2 Files written

Per [6-Init_Path](6-Init_Path.md) §6.5.4:

- **metrics/test.json**, **metrics/val.json** — computed metrics (machine-readable).
- **figures/** — optional (e.g. `confusion_matrix.png`, `roc_curve.png`).

---

## 5. Generic evaluator class (interface only)

Below is a **minimal generic interface** for an evaluator. Only **methods to implement** and their **inputs/outputs** are specified; no full code.

### 5.1 Class: `BaseEvaluator` (or project-specific name)

**Purpose:** Encapsulate “read cfg → get metric list → read CSV → compute metrics → write results”.

**Methods to implement:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| **get_metrics_config** | `cfg: dict` | `list[str]` or `dict` | From cfg, return the list of metric names (or per-metric config) to compute. E.g. read `cfg["evaluation"]["metrics"]`. |
| **compute_metrics** | `preds_df: DataFrame`, `metrics_config: list[str] \| dict` | `dict[str, float]` | Given a DataFrame (from reading the prediction CSV) and the metrics config, return a dict `metric_name -> value`. |
| **evaluate** | `cfg: dict`, `predict_output: PredictOutput`, `paths: Paths` | `EvalOutput` | Main entry: resolve preds_path(s), read CSV(s), call get_metrics_config + compute_metrics per split, write metrics (and optionally figures) under paths, return EvalOutput. |

**Optional methods (if the project needs figures or custom I/O):**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| **plot_figures** | `preds_df: DataFrame`, `metrics: dict`, `paths: Paths` | `dict[str, str]` | Optionally produce figures (e.g. confusion matrix, ROC) and write under `paths.figures_dir`. Return figure_name -> path. |
| **write_metrics** | `metrics: dict`, `path: str` | `None` | Write metrics dict to a single file (e.g. JSON at `path`). |

**Input/output summary:**

- **get_metrics_config(cfg)** → list of metric names (or dict of config).  
- **compute_metrics(preds_df, metrics_config)** → dict of metric_name -> float.  
- **evaluate(cfg, predict_output, paths)** → EvalOutput (status, metrics per split, metrics_path).

Implementations may extend this class (e.g. add task-specific metrics or figure logic) as long as the **evaluate** entry and the above inputs/outputs are respected.

---

## 6. Summary

- **Input:** **cfg** (metric names/config), **PredictOutput** (preds_path → CSV), **paths** (write targets).
- **Behavior:** From cfg get metrics to compute → read prediction CSV(s) → compute metrics → write to paths.metrics_dir (and optionally figures_dir).
- **Output:** **EvalOutput** (status, metrics per split, metrics_path).
- **Generic class:** Implement **get_metrics_config**, **compute_metrics**, and **evaluate** (and optionally plot_figures / write_metrics) with the inputs/outputs above; no need to ship full code in this doc.
