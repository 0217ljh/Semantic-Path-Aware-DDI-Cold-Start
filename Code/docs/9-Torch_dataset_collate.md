# Dataset and collate (R4, R5)

This document specifies how the pipeline turns **processed + features** into **torch Dataset(s)** for train / val / test, and how to build **collate_fn** from the global info prepared in the feature cache. The same Dataset construction and collate logic apply to **all three splits** (train, val, test).

---

## 1. Purpose

- **Convert data to torch Dataset:** Build `torch.utils.data.Dataset` (or `IterableDataset`) for each split using `processed["data"]`, `processed["splits"]`, and `resources["by_index"]`, so that `__getitem__(i)` returns a single sample (base fields + per-sample extras from resources).
- **Create collate from global info:** Use `resources["collate"]` (padding rules, vocab, max_len, etc.) to build **collate_fn**, which turns a list of samples into a batch (e.g. a dict of tensors).
- **Apply to all splits:** Train, val, and test all use the same Dataset class; only the **indices** (from `processed["splits"]["train"]`, `["val"]`, `["test"]`) differ. **Collate:** Train, val, and test are **allowed to use different collate_fn** (e.g. train with augmentation-related collate, val/test with a different padding or no shuffle of fields). The pipeline can provide one shared collate_fn or one per split.

---

## 2. Where it lives

- **`my_code/data/dataset.py`:** `build_dataset(cfg, processed, resources)` → returns Dataset(s) for train/val/test.
- **`code/data/collate.py`:** `build_collate_fn(cfg, resources)` (or equivalent) → returns one callable or a dict of callables per split (`{"train": ..., "val": ..., "test": ...}`).
- **`code/data/dataloaders.py`:** `build_loaders(cfg, datasets, collate_fn)` → returns DataLoader(s) for each split. `collate_fn` can be a single callable (used for all) or a dict of callables per split.

---

## 3. Input

| Input | Type | Description |
|-------|------|-------------|
| `cfg` | dict | Config; at least `cfg["data"]` (e.g. batch_size, num_workers) and `cfg["training"]` if needed for loader options. |
| `processed` | dict | From base.pkl: `data`, `splits` (e.g. `{"train": indices, "val": indices, "test": indices}`), `extra`, optional `meta`. |
| `resources` | dict | From resource cache: `by_index`, `collate`, `meta`. See [8-ResourceCache](8-ResourceCache.md) (or 8-EnsureResourceCache). |

---

## 4. Dataset: one per split

For each split name in `["train", "val", "test"]`:

1. Get indices (or IDs) from `processed["splits"][split_name]`.
2. Build a Dataset that:
   - Exposes **only** the samples for that split (via the indices).
   - In `__getitem__(i)`, uses the **global** index into the full dataset (or the split-local index, depending on how you store indices); then:
     - Gets the base item from `processed["data"]` (project-defined).
     - Merges per-sample extras from `resources["by_index"]`. **If** the resource cache used the **offsets+values** convention (see [8-ResourceCache](8-ResourceCache.md) §5.1: `<name>_offsets` + `<name>_values`), then for those fields `__getitem__` must slice with `offsets[i]:offsets[i+1]` instead of a simple `v[i]`; otherwise use `v[i]` for indexable columns. One simple check (e.g. presence of `f"{k}_offsets"` in `by_index`) is enough to branch the two cases.
   - Implements `__len__` if map-style (length = len(indices for this split)).

So you get three datasets: `dataset_train`, `dataset_val`, `dataset_test`, all built from the same `processed` + `resources`, differing only by which indices they expose.

**Contract:** Each `__getitem__(i)` returns a single sample (e.g. a dict) with at least the keys the model and collate expect (e.g. `x`, optional `y`, optional `meta`). The exact shape is project-defined; the important part is that the collate_fn(s) can batch these samples.

---

## 5. Collate: built from `resources["collate"]`

The **collate_fn** is a callable that takes a **list of samples** (as returned by `Dataset.__getitem__`) and returns a **batch** (e.g. a dict of tensors or lists, suitable for the model forward).

- **Input to collate_fn:** `list[dict]` (or list of whatever one sample is).
- **Output:** One batch structure (e.g. `{"x": tensor, "y": tensor, "meta": ...}` or project-defined).

**How to build it:** Use `resources["collate"]` as the **global context** when constructing the collate function(s). For example:

- `resources["collate"]` may contain: `padding_side`, `pad_token_id`, `max_length`, `vocab`, etc.
- The project implements `build_collate_fn(cfg, resources)` and may return either:
  - **One callable** — used for all three splits (train, val, test); or
  - **A dict per split** — e.g. `{"train": collate_train, "val": collate_val, "test": collate_test}`, so train/val/test can use different collate logic (e.g. train with augmentation or different padding, val/test with eval-style batching).

**Allowed:** Train, val, and test are **allowed to use different collate_fn**. When they do, each DataLoader gets its own collate callable; when they don't, a single collate_fn is reused for all.

---

## 6. DataLoaders: train, val, test

After building the three Datasets and collate_fn(s):

- If **one** collate_fn: use it for all three loaders.
- If **per-split** collate_fn (dict): use `collate_fn["train"]`, `collate_fn["val"]`, `collate_fn["test"]` for the corresponding DataLoader.

Example:
- **train_loader** = `DataLoader(dataset_train, batch_size=..., shuffle=True, collate_fn=collate_fn_or_dict["train"], ...)`
- **val_loader**  = `DataLoader(dataset_val,  batch_size=..., shuffle=False, collate_fn=collate_fn_or_dict["val"], ...)`
- **test_loader** = `DataLoader(dataset_test, batch_size=..., shuffle=False, collate_fn=collate_fn_or_dict["test"], ...)`

Optional: `num_workers`, `pin_memory`, etc. from `cfg["data"]` or `cfg["training"]`.

---

## 7. Pipeline integration (R4, R5)

After resources are ready (R3_6):

1. **R4:** `datasets = build_dataset(cfg, processed, resources)`  
   Returns a dict (or tuple) with keys `"train"`, `"val"`, `"test"` (or equivalent), each a `torch.utils.data.Dataset`.

2. **Collate:** `collate_fn = build_collate_fn(cfg, resources)`  
   Uses `resources["collate"]`; returns either one callable (shared by all splits) or a dict `{"train": ..., "val": ..., "test": ...}` so train/val/test can use different collate_fn.

3. **R5:** `loaders = build_loaders(cfg, datasets, collate_fn)`  
   Returns a dict with `"train"`, `"val"`, `"test"` DataLoaders. Each loader uses the corresponding collate (one shared or per-split).

4. Downstream (trainer, evaluator) receives `loaders` and runs training/validation/test on the corresponding loader.

---

## 8. Summary

- **Purpose:** Turn processed + resources into torch **Dataset** per split (train/val/test) and **collate_fn** from `resources["collate"]`; use the same logic for all splits. **Train, val, and test are allowed to use different collate_fn** (one shared or a dict per split).
- **Input:** `cfg`, `processed`, `resources`.
- **Dataset:** One Dataset per split; each uses `processed["data"]`, `processed["splits"][split]`, and `resources["by_index"]` in `__getitem__`. If resources use **offsets+values**, `__getitem__` branches on e.g. `f"{k}_offsets"` in `by_index` and uses slice vs simple index.
- **Collate:** Built from `resources["collate"]`; train/val/test may share one or use a dict per split.
- **Loaders:** `build_loaders(cfg, datasets, collate_fn)` returns train/val/test DataLoaders.
- **Code:** `data/dataset.py`, `data/collate.py`, `data/dataloaders.py`.
