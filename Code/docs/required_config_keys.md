# Required config keys (by section)

Per current docs (5-Config, 3-Pipeline, 6-Init_Path, 2-Repo_Layout), the merged `cfg` must have the following **required** keys. Anything else is project-specific or optional.

---

## data

| Key | Type | Why required |
|-----|------|--------------|
| `dataset` | string | Name of the dataset; used for path `data/<dataset>/` and for `data_key` (Pipeline R2_5, R3). |
| `split_spec` | string | Split policy (e.g. `random`, `time`); part of `data_key` for processed dir. |
| `split_seed` | int | Seed for split; part of `data_key`. |
| `preprocess_version` | string | Version of preprocessing; part of `data_key` (e.g. `v1`). |

`data_key = f(dataset, split_spec, split_seed, preprocess_version)` is used to resolve `data/<dataset>/processed/<data_key>/`.

---

## model

| Key | Type | Why required |
|-----|------|--------------|
| `name` | string | Method identifier; used by `pipeline/registry.py` to load the model, and by `init_run_paths` to choose path: `main` → `runs/main/<branch>/`, else → `runs/baselines/<name>/`. |

---

## training

| Key | Type | Why required |
|-----|------|--------------|
| `batch_size` | int | Used by dataloaders and trainer. |
| `epochs` | int | Training length. |
| `optimizer` | string or dict | Optimizer type (e.g. `adam`); trainer uses it to build the optimizer. |
| `learning_rate` | float | Learning rate (or use key `lr` if the project agrees). |

---

## Optional but referenced in docs

- **data:** paths overrides, resource configs (fewshot, kgpaths), etc.
- **run:** `exp_name` (for init_run_paths run identity); `seed` (for reproducibility).
- **mode / ckpt:** may live on `cfg` or on `args`; required at runtime for train vs predict and for loading checkpoint.

---

## Minimal test config

See [configs/runs/test_required.yaml](../configs/runs/test_required.yaml) for a minimal run config containing only the required keys above.
