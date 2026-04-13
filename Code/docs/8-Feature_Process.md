# Ensure resource caches (R3_6)

This document specifies the **resource cache** step in the pipeline (R3_6): per-method supplementary data (e.g. KG paths, negatives, image/text indices) that are built from the **base processed** output and cached so they can be loaded directly on later runs.

---

## 1. Purpose

- **Per-method support:** Different methods need different auxiliary data (KG paths, negative samples, image features, few-shot indices, etc.). This step builds and caches that data **per method** (and per `data_key`).
- **Input:** The base processed bundle (the dict loaded from `base.pkl`).
- **Output:** A resource cache (method-specific) that downstream (**Dataset**, **collate**) can use directly.
- **Existence check:** If the cache for this `data_key` + method already exists, **load it**; otherwise build and save. This saves time on repeated runs.

---

## 2. Where the cache is stored

By the time we reach R3_6, R3 has already run: **base processed** is at `data/<dataset>/processed/<data_key>/` (base.pkl + DONE). We only **append** a per-method subdirectory under that existing path:

```
data/<dataset>/processed/<data_key>/   # already exists (R3)
  base.pkl
  DONE
  resources/
    <method_key>/
      resource.pkl
      DONE
```

- `resource_dir = <existing_processed_dir>/resources/<method_key>/`, with `method_key = cfg["model"]["name"]` (e.g. `main`, `baseline_x`).

---

## 3. Existence check

Before building, the pipeline checks:

1. `resource_dir` exists, and
2. **DONE** exists under `resource_dir`.

- **If both true:** Load the resource cache (e.g. `resource.pkl`) and skip building.
- **If missing:** Call the method's resource builder, save under `resource_dir`, then write **DONE**.

Same idea as base processed: **DONE** is written only after the cache is fully written, so partial writes are not treated as valid.

---

## 4. Input interface

The resource builder receives:

| Input | Type | Description |
|-------|------|-------------|
| `processed` | dict | Base processed bundle loaded from `base.pkl` (keys: `data`, `splits`, `extra`, optional `meta`). See [7-GetOrLoadData](7-GetOrLoadData.md). |
| `cfg` | dict | Full config; at least `cfg["data"]` and `cfg["model"]` (including `cfg["model"]["name"]`) so the builder knows which method's resources to build. |
| `resource_dir` | str | Output path: `data/<dataset>/processed/<data_key>/resources/<method_key>/`. |

Raw data paths (e.g. `data/<dataset>/raw/...`) can be derived from `cfg["data"]` and project layout.

---

## 5. Output interface: ResourceDict (for Torch Dataset / collate)

The output of this step is a single dict called `resources`, saved/loaded as `resource.pkl`.

To make `resources` **plug-and-play** with `torch.utils.data.Dataset` and `collate_fn`, the schema is **fixed** as:

```python
resources = {
  "by_index": {
    # Per-sample extras (columnar, indexable): each value supports v[i].
    # e.g. "kg_paths": <list/np.ndarray/torch.Tensor>, len == N
    # e.g. "image_path": <list[str]>, len == N
    #
    # Variable-length fields use offsets + values (CSR style), e.g.:
    # "neg_offsets": <int array, shape [N+1]>
    # "neg_ids":     <int array, shape [M]>
  },
  "collate": {
    # Global batching context for collate_fn: padding rules, vocab, max_len, etc.
  },
  "meta": {
    "method_key": "<method_key>",
    "data_key": "<data_key>",
    "n": N   # must match base dataset length (aligned indexing)
  }
}
```

### 5.1 Variable-length (per-sample) data: offsets + values

For per-sample variable-length resources (e.g. negatives, KG paths, neighbor lists), store them as:

- `<name>_offsets`: int array of length **N+1**
- `<name>_values` (or multiple value arrays): total length **M**

Access sample `i` via slice: `s = offsets[i]`, `e = offsets[i+1]`, values are `values[s:e]`. This keeps storage columnar and indexing fast.

---

## 6. Torch integration: Dataset.__getitem__ (recommended pattern)

A concrete pattern for `Dataset.__getitem__(i)` so that each sample merges base data with per-sample extras from `resources["by_index"]`:

```python
def __getitem__(self, i):
    # Base item from processed["data"] (project-defined)
    item = self.get_base_item(i)

    # Merge per-sample extras from resources["by_index"]
    by_index = self.resources["by_index"]
    for k, v in by_index.items():
        # Skip offset arrays; we use them to slice the value array
        if k.endswith("_offsets"):
            continue
        if f"{k}_offsets" in by_index:
            offsets = by_index[f"{k}_offsets"]
            s, e = offsets[i], offsets[i + 1]
            item[k] = v[s:e]   # v is the flattened values array
        else:
            item[k] = v[i]     # simple indexable column

    return item
```

- **collate_fn** reads `resources["collate"]` for padding, vocab, max_len, etc.
- **`resources["meta"]["n"]`** must equal the base dataset length **N** so indices are aligned.

---

## 7. Methods (build / load)

Recommended minimal API (implementations can live in the method module, e.g. `my_code/models/<name>/resources.py`):

| Method | Purpose |
|--------|---------|
| `build_resource_cache(processed, cfg, resource_dir) -> resources` | Build resources from `processed` and cfg, write `resource.pkl` and **DONE** under `resource_dir`, return the same `resources` dict. |
| `load_resource_cache(resource_dir) -> resources` | Load `resource.pkl`. Used by the pipeline when DONE exists. |

Pipeline requirements:

1. If cache missing: build writes `resource.pkl`, then writes **DONE**.
2. If cache exists: load returns `resources` (same schema as above).
3. `build_dataset(cfg, processed, resources)` and `collate_fn` receive the same `resources` object.

---

## 8. Pipeline integration (R3_6)

After base processed is ready (R3):

1. `data_key = compute_data_key(cfg["data"])`
2. `method_key = cfg["model"]["name"]`
3. `resource_dir = data/<dataset>/processed/<data_key>/resources/<method_key>/`
4. If **DONE** exists under `resource_dir`: `resources = load_resource_cache(resource_dir)`.
5. Else: `resources = build_resource_cache(processed, cfg, resource_dir)` (writes `resource.pkl` + **DONE**).
6. Pass `processed` and `resources` to `build_dataset(cfg, processed, resources)` (R4).

**Multi-card:** Building and writing the resource cache should be done by **rank 0 only**; other ranks wait or read after the cache is written.

---

## 9. Summary

- **Purpose:** Per-method supplementary data (KG, negatives, images, etc.); input = base processed (pkl); output = resource cache; **existence check** to load when present and skip rebuild.
- **Where:** `data/<dataset>/processed/<data_key>/resources/<method_key>/` with `resource.pkl` + **DONE** (under existing R3 path).
- **Input:** `processed` (from base.pkl), `cfg`, `resource_dir`.
- **Output:** `resources` dict with **fixed schema** for Torch: `by_index` (per-sample, indexable; variable-length via offsets+values), `collate` (batching context), `meta` (method_key, data_key, n).
- **Torch:** `Dataset.__getitem__` merges base item with `resources["by_index"]` using the offsets convention; `collate_fn` uses `resources["collate"]`; `meta["n"]` must match base length.
- **Methods:** `build_resource_cache`, `load_resource_cache`.
