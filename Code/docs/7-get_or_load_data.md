# Base Data Processed : `data/<dataset>/processed/<data_key>/base.pkl`

This document specifies the **base processed** caching step in the pipeline:

> **Base processed exists?**  
> check: `data/<dataset>/processed/<data_key>/`

If the cache exists, the pipeline loads it directly. Otherwise, it builds the base processed package and saves it.

The base processed package is intentionally **minimal**:
- one serialized file (`base.pkl`) that bundles **data + splits + extra**,
- one marker file (`DONE`) that indicates the cache is complete.

---

## 1. Goal

- Provide a **single, reusable** processed dataset bundle for training and evaluation.
- Ensure the training server can load all required dataset-side information (including KG or other auxiliary data) **without re-reading external raw assets**.
- Keep naming short and stable by using a **single global seed** in `cfg["data"]["global_seed"]`.

---

## 2. Global seed rule

All randomness in the data pipeline uses **one global seed**:

- `cfg["data"]["global_seed"] = <int>`

**Default:** Every operation (split, subset sampling, augmentation, etc.) uses **the same** `global_seed`. Do not derive different seeds per operation unless necessary.

**Exception:** Only when **a single operation** internally needs multiple random seeds (e.g. one subroutine that uses several RNGs), derive them deterministically from `global_seed` **within that operation only** (e.g. `global_seed`, `global_seed + 1`, `global_seed + 2`). Other operations still use `global_seed` unchanged.

Reproducibility is ensured by using one global seed everywhere; derivation is only for the rare case where one step needs several seeds.

---

## 3. `data_key` (short, stable)

`data_key` identifies one base processed cache version and is typically tied to:
- the dataset name,
- split policy,
- global seed,
- optional subset specification,
%- optional preprocess version tag.

A recommended short format:

`<data>__sp-<split_tag>__sd<seed>__sub-<subset_tag>`

Examples:
- `datasetB__sp-rand801010__sd42__sub-none`
- `datasetB__sp-time182022__sd42__sub-r10`
- `datasetB__sp-rand801010__sd42__sub-k5000`

Notes:
- `<split_tag>` should be a **short label** (project-defined).
- `<subset_tag>` can be `none`, `r10` (10%), `k5000`, etc.


---

## 4. Storage layout (minimal)

Base processed cache is stored under:

data/<dataset>/
processed/
<data_key>/
base.pkl
DONE


### Files
- `base.pkl`  
  A single pickle file containing a Python dict that bundles:
  - base data
  - split indices
  - auxiliary/extra assets (KG, mappings, etc.)

- `DONE`  
  An empty marker file created **only after** `base.pkl` is successfully written.  
  If `DONE` exists, the cache is considered ready.

---

## 5. Content of `base.pkl` (minimal schema)

`base.pkl` must be a dict with the following keys:

- `data`: base examples / records (project-defined structure)
- `splits`: a dict of split indices or ids:
  - `{"train": ..., "val": ..., "test": ...}`
- `extra`: auxiliary information bundled into the cache (e.g. KG, entity maps, text resources, metadata tables). Keep it simple: put any auxiliary info that may be needed into `extra` so one load gives everything.

Optional but recommended:
- `meta`: minimal provenance info for debugging/repro:
  - `dataset`, `data_key`, `global_seed`, `split_tag`, `subset_tag`, `preprocess_version`, `created_at`

Example structure:

```python
{
  "data": ...,
  "splits": {"train": ..., "val": ..., "test": ...},
  "extra": {...},
  "meta": {...}  # optional
}
```

## 6. Existence check: "exists → load"

The cache is treated as valid if:

1. directory data/<dataset>/processed/<data_key>/ exists, and

2. file DONE exists.

If valid:

* load base.pkl via pickle.load

If invalid or missing:

* rebuild and overwrite the directory.

Recommended robustness:

* if pickle.load(base.pkl) fails (corruption / incompatibility), delete the directory and rebuild.


## 7. Write rule (avoid partial caches)

To prevent partially-written caches from being mistakenly loaded:

* write base.pkl first

* create DONE last

Optionally, implement a safer write:

* write into a temporary directory

* rename into the final <data_key>/ directory

* ensure DONE is written only when all required data is present


## 8. Pipeline integration

**Where it lives:** `compute_data_key(cfg["data"])` and `build_base_processed(cfg["data"])` are implemented in **`my_code/data/preprocess/build.py`**. (A separate `process/` folder for method-specific data processing may be added later; not covered here.)

At runtime, the pipeline does:

1. `data_key = compute_data_key(cfg["data"])`
2. `cache_dir = data/<dataset>/processed/<data_key>/`
3. If **DONE** exists: `processed = load(base.pkl)`.
4. Else: `processed = build_base_processed(cfg["data"])`, save to `base.pkl`, then write **DONE**.
5. Return `processed` downstream for dataset construction and method-specific processing.

**Multi-card:** Base processed build and save are done by **rank 0 only**; other ranks wait or read after rank 0 has written. This avoids multiple processes writing the same cache.

**Evaluation:** When running evaluation, the pipeline should save **full input and output** (e.g. per-sample inputs and predictions) so that results can be inspected and reproduced. Where to store them (e.g. under `run_dir/metrics/` or `run_dir/preds/`) is defined in the evaluation / run-path docs.

This design keeps the base cache simple: one pickle bundle per dataset version, with auxiliary info in `extra`.
