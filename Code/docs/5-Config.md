# Config: load_cfg

After the script parses CLI arguments, the pipeline’s first step is to resolve the full experiment configuration. This is done by **`load_cfg(args)`** in `code/utils/config.py`. The result is a single, merged config object used by the rest of the pipeline.

---

## 1. Purpose

- **Merge** all config sources (run config **or** data + model + training) into one resolved config.
- **Apply** CLI overrides (`--set key=value`) with highest priority.
- **Expose** a unified `cfg` to `pipeline/run.py` and downstream steps (run_paths, data, methods, train, eval).

Config loading does **not**:
- Create directories or write files.
- Validate dataset or model implementation details (only structural/required-key checks if any).

---

## 2. Location and call site

- **Implementation:** `code/utils/config.py` (e.g. function `load_cfg(args)`).
- **Call site:** `pipeline/run.py` at the start of orchestration (step R1 in the pipeline diagram).

---

## 3. Inputs

Input is the **parsed `args`** object produced by `scripts/train.py` or `scripts/eval.py`. Relevant fields:

| Source | Field | Description |
|--------|--------|-------------|
| Config selection | `args.run` | Either a **path** to a run config file (e.g. `configs/runs/exp_main_branch1.yaml`), or a **method label** (e.g. `main`, `baseline_x`). When a label is given, the loader resolves it under `configs/` to the corresponding data/model/training configs (see §3.1). |
| Config selection | `args.data` | Path to data config (used only when **not** using `--run`). |
| Config selection | `args.model` | Path to model config (used only when **not** using `--run`). |
| Config selection | `args.training` | Path to training config (used only when **not** using `--run`). |
| Overrides | `args.set` | List of `"key=value"` strings (repeatable). Highest priority. |
| Mode | `args.mode` | `"train"` or `"predict"`. |
| Checkpoint | `args.ckpt` | Path to checkpoint (required when `mode == "predict"`). |
| Distributed | (env) | Scripts may pass through `RANK`, `WORLD_SIZE`, `LOCAL_RANK`; config layer may attach these to `cfg` or leave them to trainer. |

**Constraint:** Exactly one of the following must be used:

- `--run <path_or_label>`, **or**
- `--data` + `--model` + `--training` (all three).

Scripts are responsible for enforcing this before calling the pipeline.

### 3.1. `--run` as method label

Instead of passing a full path, you may pass a **method label** (e.g. `main`, `baseline_x`). The loader then **resolves** this label under the `configs/` folder to obtain the three configs (data, model, training) for that method. Two common conventions:

- **Run file per method:** The loader treats the label as a run config name and loads `configs/runs/<label>.yaml`. That file must have the same structure as any run config (i.e. provide or include `data`, `model`, `training`). Example: `--run main` → load `configs/runs/main.yaml`.
- **Direct lookup by label:** The loader maps the label to three paths under `configs/`: e.g. `configs/data/<label>.yaml`, `configs/model/<label>.yaml`, `configs/training/<label>.yaml`, or a project-defined mapping (e.g. a small registry that says `main` → `default` data + `main` model + `default` training). The project chooses one convention and documents it.

In both cases, the **result** is the same as loading a single run file or three separate files: one merged cfg with `data`, `model`, `training`. Using a label keeps the CLI short and avoids typing full paths when you only want “run this method with its default configs.”

**Config selection vs overrides:** `--data`, `--model`, and `--training` each take **exactly one path** (the YAML file to load). They do not accept multiple key-value parameters. To override **multiple** parameters from the command line—e.g. many fields in the data config—use **multiple** `--set` options. You can pass as many `--set key=value` as you need, and mix keys from different sections (`data.*`, `model.*`, `training.*`). Example:

```bash
# Use default data config but override several data-related fields from CLI
python scripts/train.py \
  --data configs/data/default.yaml \
  --model configs/model/main.yaml \
  --training configs/training/default.yaml \
  --set data.split_seed=42 \
  --set data.preprocess_version=v1 \
  --set data.batch_size=32 \
  --set training.epochs=10 \
  --mode train
```

Same idea with `--run`: the run file provides the base, and any number of `--set` overrides can tweak data, model, or training fields.

---

## 4. Config sources and merge order

Merge order (later overrides earlier):

1. **Base configs** (depending on mode):
   - **Run mode:** If `args.run` ends with `.yaml`, load that file (e.g. under **configs/runs/**). If `args.run` is a **label** (no `.yaml`), resolve to **configs/runs/<label>.yaml** and load it. Result: single dict with `data`, `model`, `training`.
   - **Three-part mode:** Load the three files from **configs/data/**, **configs/model/**, **configs/training/** (paths from `args.data`, `args.model`, `args.training`), then merge into one with top-level keys `data`, `model`, `training`.
2. **Defaults (optional):** For three-part mode, project may merge in `configs/data/default.yaml` and `configs/training/default.yaml` as the lowest priority (before the chosen data/training files). Model typically has no global default.
3. **CLI overrides:** Apply each `args.set` entry as `key=value`. Keys may use dot notation for nested overrides (e.g. `training.batch_size=32`, `data.split_seed=42`). These have **highest** priority.

No other source (e.g. env vars for config values) is required by this spec; projects may add optional env-based overrides as long as they remain below `--set` in priority.

---

## 5. Run config file format

When `--run` is a path (or when a method label is resolved to a run file, e.g. `configs/runs/<label>.yaml`), that file must produce (after any includes or inlining) a structure equivalent to:

```yaml
data:       { ... }   # same shape as a configs/data/*.yaml
model:      { ... }   # same shape as a configs/model/*.yaml
training:   { ... }   # same shape as a configs/training/*.yaml
```

So the pipeline always sees a **single** cfg with top-level keys `data`, `model`, `training`. Optional: run config may also set run-level keys (e.g. `exp_name`, `seed`) if the project agrees on their names; those can be merged into the same object or under a `run` key.

---

## 6. Output: resolved `cfg`

- **Type:** One unified config object used by the rest of the pipeline. Implementation can be a nested dict, or an OmegaConf-style object; the important part is that downstream code accesses it in a consistent way via the three top-level sections: `data`, `model`, `training`.

- **Standard access pattern (downstream):**

  In general, downstream code should follow:
  - `cfg["data"][...]` for dataset paths, split specs, preprocess versions, feature settings, etc.
  - `cfg["model"][...]` for method selection and model-specific options.
  - `cfg["training"][...]` for all optimization / training loop options.

- **Required top-level keys:**
  - `data`: dataset name, paths, split spec, preprocess version, and any data-related options.
  - `model`: at least `name` (used by `pipeline/registry.py` to load the method); plus any model-specific options.
  - `training`: batch size, epochs, optimizer, learning rate, etc., and any training-specific options.

- **Run-level / mode:** The pipeline also needs `mode` and, when `mode == "predict"`, checkpoint path. These may be:
  - attached onto `cfg` by `load_cfg` (e.g. `cfg["mode"]`, `cfg["ckpt"]`), or
  - passed separately from `args` into `run.py` and only the “experiment config” part is `cfg`.

  Either approach is fine as long as it is consistent and documented; the rest of the pipeline doc will refer to “cfg (and optionally args for mode/ckpt)”.


---

## 7. Override key syntax

- **Nested keys:** Use dot notation, e.g. `--set training.batch_size=32`, `--set data.split_seed=42`. This applies to any depth under `data`, `model`, or `training`.
- **Multiple overrides:** You can pass **any number** of `--set` options. Each overrides one key in the merged config. Later `--set` overrides earlier for the same key; order is defined by the order of `args.set`.
- **Mixing sections:** Overrides can target different sections in one command, e.g. several `data.*` and several `training.*` in the same run. There is no limit on how many parameters you override from the CLI.
- **Types:** Parser may treat values as strings and let downstream coerce to int/float/bool, or implement simple type inference; this is project-specific. Booleans are often expressed as `true`/`false` or `1`/`0`.

---

## 8. 完整 config 的落盘（log）

合并后的 **resolved config** 必须在 run 开始时写入 run 目录，便于复现与排查。不由 `load_cfg` 写文件，而由 **pipeline** 在拿到 `paths` 后写入（[6-Init_Path](6-Init_Path.md) §6.5.1）：

- **路径：** `paths.resolved_config_path`，即 `run_dir/meta/resolved_config.yaml`。
- **内容：** `load_cfg(args)` 返回的完整 `cfg`（含合并后的 `data`、`model`、`training` 及 run-level 字段），序列化为 YAML（或项目约定的格式）。
- **时机：** 在 `init_run_paths(cfg)` 之后、开始数据/训练/预测之前，由 `io/artifacts.py` 或等价模块写入。这样每次 run 的 meta 里都有一份与本次执行完全一致的 config 快照。

---

## 9. 读取 config 的示例函数

以下为 **`load_cfg(args)`** 的极简示例逻辑（仅说明合并顺序与结构，实际实现可选用 PyYAML、OmegaConf 等）：

```python
def load_cfg(args):
    """
    Merge config from run file OR (data + model + training), then apply --set overrides.
    Returns a single dict with top-level keys: data, model, training; optional: mode, ckpt, run.
    """
    import yaml

    if args.run:
        if args.run.endswith(".yaml") or args.run.endswith(".yml"):
            with open(args.run) as f:
                base = yaml.safe_load(f)
        else:
            root = "configs"
            with open(f"{root}/data/{args.run}.yaml") as fd:
                data = yaml.safe_load(fd)
            with open(f"{root}/model/{args.run}.yaml") as fm:
                model = yaml.safe_load(fm)
            with open(f"{root}/training/{args.run}.yaml") as ft:
                training = yaml.safe_load(ft)
            base = {"data": data, "model": model, "training": training}
    else:
        with open(args.data) as f:
            data = yaml.safe_load(f)
        with open(args.model) as f:
            model = yaml.safe_load(f)
        with open(args.training) as f:
            training = yaml.safe_load(f)
        base = {"data": data, "model": model, "training": training}

    cfg = base
    for s in getattr(args, "set", []) or []:
        key, _, val = s.partition("=")
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = _coerce(val)

    cfg["mode"] = getattr(args, "mode", "train")
    if cfg["mode"] == "predict":
        cfg["ckpt"] = getattr(args, "ckpt", None)
    return cfg
```

项目可实现 `_coerce(val)` 做简单类型推断（如 `"42"` → int，`"0.1"` → float），并统一用 `load_cfg(args)` 作为唯一入口。

---

## 10. 示例：三块 config 文件（data / model / training）

以下示例使用 **三块独立文件**（configs/data、configs/model、configs/training），包含当前文档流程所需的必要参数，可直接用三参模式启动：

```bash
python scripts/train.py \
  --data configs/data/test_full.yaml \
  --model configs/model/test_full.yaml \
  --training configs/training/test_full.yaml \
  --mode train
```

或使用 label（若 loader 将 `--run test_full` 解析为上述三文件）：`--run test_full --mode train`。

**文件路径与必要字段**（见 [required_config_keys](required_config_keys.md)、[7-GetOrLoadData](7-get_or_load_data.md)、[10-Training](10-Training.md)、[12-Evaluation](12-Evaluation.md)）：

| 文件 | 路径 | 必要 / 常用字段 |
|------|------|------------------|
| Data | `configs/data/test_full.yaml` | `dataset`, `global_seed`, `split_spec`, `preprocess_version`；可选 `batch_size` |
| Model | `configs/model/test_full.yaml` | `name`（方法标识，用于 registry 与 run 路径） |
| Training | `configs/training/test_full.yaml` | `batch_size`, `epochs`, `optimizer`, `learning_rate`；可选 `monitor`, `log_interval`, `save_strategy` |

### 10.1 `configs/data/test_full.yaml`

```yaml
# Data config: required for data_key and pipeline R3. See docs/7-GetOrLoadData, required_config_keys.
dataset: dataset_a
global_seed: 42
split_spec: random
preprocess_version: v1
batch_size: 32
```

### 10.2 `configs/model/test_full.yaml`

```yaml
# Model config: required name for registry and run path. See docs/required_config_keys.
name: main
```

### 10.3 `configs/training/test_full.yaml`

```yaml
# Training config: required for trainer. See docs/10-Training, required_config_keys.
batch_size: 32
epochs: 10
optimizer: adam
learning_rate: 0.001
monitor: val_loss
log_interval: 10
save_strategy: last+best
```

### 10.4 Evaluation（可选）

若 evaluator 从 cfg 读指标列表（[12-Evaluation](12-Evaluation.md)），可在 training 或 run 级增加 `evaluation.metrics`，例如在 training 文件中：`evaluation: { metrics: [accuracy, f1, auc] }`。

---

## 11. Summary

| Item | Convention |
|------|------------|
| Entry | `load_cfg(args)` in `utils/config.py`（示例见 §9） |
| Input | Parsed `args` (run **or** data+model+training, plus `--set`, mode, ckpt) |
| Merge order | Run YAML or (data + model + training) → optional defaults → `--set` (highest) |
| Output | Single `cfg` with `data`, `model`, `training` (+ optional run-level, mode, ckpt) |
| **Full config 落盘** | Pipeline 将 resolved cfg 写入 **meta/resolved_config.yaml**（§8） |
| Example | 三块示例见 §10 及 `configs/data/test_full.yaml` 等 |
| Downstream | `init_run_paths(cfg)`, `compute_data_key(cfg)` 及后续步骤均使用该 cfg |


This keeps config resolution in one place and ensures the pipeline always runs with a single, reproducible resolved config.
