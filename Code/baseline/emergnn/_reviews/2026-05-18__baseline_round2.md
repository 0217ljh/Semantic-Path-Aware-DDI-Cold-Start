# EmerGNN Baseline Review — Round 2 (re-review against current spec)

- **Date**: 2026-05-18
- **Primary reviewer**: Claude (sonnet-4.5) — drafted this report,
  inventory audit, cross-folder dep check
- **Independent reviewer**: codex (gpt-5-codex) — verbatim verdict §6
- **Triggered by**: 用户 "好，现在开始根据 CLAUDE.md 的约束，对
  `/baseline/Emergnn` 进行 review"
- **Scope**: `Code/baseline/emergnn/` — all .py + previous PASS doc state
- **Relationship to prior review**: `_reviews/2026-05-17__baseline.md`
  was a PASS verdict against the **older** spec (before HDN/TIGER
  added `_data/necessary/` + `_shared.py` detect-and-build requirements
  to CLAUDE.md §"Baseline 规范"). That PASS is now stale; this round
  re-reviews against the CURRENT spec.

---

## 1. Read inventory

### Files (100% read this round)

| File | Status |
|---|---|
| `__init__.py` | ✓ re-exports both `EmerGNNBaseline` + `EmerGNNMulticlassBaseline` |
| `binary_cls/__init__.py` + `baseline.py` | ✓ `@register("emergnn")` |
| `multi_cls/__init__.py` + `baseline.py` + `model.py` | ✓ `@register("emergnn_mc")` + `EmerGNN_MC` subclass |
| `model.py` | ✓ shared `EmerGNN` base class |
| `kg_builder.py` | ✓ legacy 5-bucket KG (drugbank source) |
| `kg_builder_merged.py` | ✓ NEW project KG (merged parquet source) |
| `morgan_features.py` | ✓ `compute_morgan_matrix` |
| `shuffle_utils.py` | ✓ paper-faithful `shuffle_train` + `build_edge_lists_from_triplets` |
| `_smoke_save_load.py` | ✓ round-trip smoke test |
| `_reviews/2026-05-17__baseline.md` | ✓ prior PASS verdict (stale relative to current spec) |

### Cross-folder dep check

| | Status |
|---|---|
| `grep "from reproductions"` in baseline code | ✅ zero hits (clean) |
| `grep "import reproductions"` in baseline code | ✅ zero hits (clean) |

Independence verified. Note: prior session's reproduction migration
(2026-05-18 round 4 PASS) made reproductions/EmerGNN independent of
baseline; the converse already held, so the two sides are now mutually
independent — both keep their own copies of `EmerGNN`, `kg_builder`,
`morgan_features`, `shuffle_utils`.

---

## 2. Layout audit vs CLAUDE.md §"Baseline 规范" §1 (lines 450-485)

| Spec requirement | Actual | Status |
|---|---|---|
| `_results/` exists | ❌ does not exist | ❌ |
| `_reviews/` exists | ✅ exists (this file + prior PASS doc) | ✅ |
| `_data/` exists | ❌ does not exist | ❌ |
| `_data/necessary/` exists | ❌ does not exist | ❌ |
| `_data/necessary/<name>__mine.<ext>` for required artefacts | ❌ no `__mine` artefacts (everything in-memory) | ❌ |
| `_data/necessary/build_<name>.py` builders | ❌ none | ❌ |
| Task subfolders `binary_cls/`, `multi_cls/` | ✅ both exist | ✅ |
| Shared code at top level | ✅ `model.py`, `kg_builder*.py`, `morgan_features.py`, `shuffle_utils.py` all flat at top | ✅ |
| Top-level `__init__.py` re-exports all task variants | ✅ | ✅ |
| Sub-`__init__.py` only re-exports own task | ✅ | ✅ |
| `_shared.py` detect-and-build pipeline | ❌ does not exist | ❌ |

**Verdict on layout**: 5 of 11 spec items fail. All 5 are the
**same set** TIGER baseline had before its 5-round migration (now PASS).
Pattern: pre-existing baselines designed before the spec's `_data/necessary/`
+ `_shared.py` rules were added need a migration round.

---

## 3. Required-artefacts detect-and-build audit (§"Baseline 规范" §2)

The spec requires that any required baseline-side artefact (per §1 line 460
"KG / 分子图 cache / ...") be built via subprocess from a local
`_data/necessary/build_*.py` script and consumed via
`_shared.py:ensure_*` helpers. EmerGNN's required artefacts:

| Artefact | Currently built | Spec wants | Status |
|---|---|---|---|
| Morgan FP matrix `[n_ent, 1024]` | in-memory each `fit()` via `morgan_features.compute_morgan_matrix(drug_id_list, smiles)` (binary_cls/baseline.py:259) | `_data/necessary/morgan_features__mine.pkl` + `build_morgan_features.py` + `ensure_morgan_features` | ❌ |
| KG triplets (legacy drugbank 5-bucket OR merged parquet) | in-memory each fit() via `build_kg_from_kb` (line 215) or `build_kg_from_merged_parquet` (line 218) | `_data/necessary/kg__<source>__<hash>__mine.pkl` + `build_kg.py` + `ensure_kg_cache` | ❌ |
| Edge tensors (forward + reverse + self-loop) | in-memory via `build_sparse_adj` + `edges_as_dense_lists` (line 250-253) | `_data/necessary/kg_edges__<hash>__mine.pkl` + `ensure_kg_edges` | ❌ |

Each fit() rebuilds all 3 → no incremental gain, plus baseline-vs-spec
non-conformance.

---

## 4. Cross-folder independence (§"Baseline 规范" §6 line 606)

Spec: "baseline 跨目录 import / subprocess 调用 `reproductions/` 下的
代码 / builder (复制独立维护, 不是借用; 否则 silent drift)".

Result: ✅ verified clean. baseline-side has its OWN copies of:
- `model.py` (`EmerGNN`)
- `multi_cls/model.py` (`EmerGNN_MC`)
- `kg_builder.py`, `kg_builder_merged.py`
- `morgan_features.py`
- `shuffle_utils.py`

Same files now exist on reproduction side too (different physical files,
independently maintained). No cross-folder import.

---

## 5. Filename-suffix + paper-extension audit

- ✅ no `baseline_multiclass.py` flat anti-pattern (was migrated in
  prior round; spec line 514 compliant)
- ✅ `shuffle_train` (paper inductive core) preserved per §4 spec line 569
- ✅ `weight_decay=1e-8` + `ReduceLROnPlateau(patience=10, factor=0.1)` +
  loss `reduction='sum'` + best ckpt by `val_macro_f1` — all paper-faithful
  per §4 (verified in prior PASS review)
- ✅ Project extension `kg_source="merged"` (with `merged_kg_path` requirement)
  is explicitly opt-in; default `kg_source="drugbank"` preserves paper behaviour

---

## 6. Codex independent verdict (verbatim)

> **VERDICT: FAIL**
>
> 当前 `Code/baseline/emergnn/` 仍符合旧 PASS 的算法/任务拆分检查，但
> 不符合新版 `CLAUDE.md` §"Baseline 规范" 对 baseline-side artefact
> pipeline 的硬性要求。
>
> 主要结论：
>
> - Layout：缺失 `_results/`、`_data/`、`_data/necessary/`、`_shared.py`。
> - `__mine` artefact convention：未发现 baseline-built `__mine` artefacts。
> - Detect-and-build：缺失 `_shared.py:ensure_*` + `_data/necessary/build_*.py`。
> - Builder consumption：`binary_cls/baseline.py._setup_graph` 仍在
>   `fit()` 路径内现场构建（`build_kg_from_kb`/`build_kg_from_merged_parquet`/
>   `build_sparse_adj`/`edges_as_dense_lists`/`compute_morgan_matrix` 全部
>   in-memory）。`multi_cls` 复用同一 `_setup_graph`。
> - Cross-folder independence：PASS。
> - Filename-suffix anti-pattern：PASS。
> - Paper extension：`shuffle_train` 保留；`merged KG` 有 opt-in flag。
>
> 优先修复清单：
> 1. 新增 `_results/`、`_data/necessary/`、`_shared.py`。
> 2. 从 reproduction 端复制 builder（但需独立维护）。
> 3. Builders 产出 `_data/necessary/*__mine.*` for Morgan/KG/edge tensor。
> 4. `_shared.py` 实现 `ensure_*` subprocess pattern (HDN/TIGER 同款)。
> 5. `_setup_graph` 切换到消费 builder products，不再现场构建。

---

## 7. Three-bucket discrepancy classification

### paper-vs-impl (paper algorithm vs current baseline code)
- ✅ all 4 paper contributions preserved (verified in prior review §3):
  shuffle_train + bidirectional propagation + attention-weighted aggregation +
  Morgan features
- ✅ paper hyperparams faithfully wired

### repo-vs-impl (reproduction-side canonical vs baseline copy)
- ⚪ baseline keeps independent copies of `model.py`, `kg_builder.py`,
  `morgan_features.py`, `shuffle_utils.py` (intentional per §文件级独立性)
- ⚪ no drift detection between baseline and reproduction copies; risk
  of silent algorithm drift if either side is edited without sync

### baseline-vs-spec (CLAUDE.md §"Baseline 规范" violations)
- ❌ Missing `_data/necessary/` directory (§1)
- ❌ Missing `_shared.py` detect-and-build (§2 step 3)
- ❌ No `__mine` artefact convention applied (§1 line 482-484)
- ❌ No subprocess builders (§2 step 3)
- ❌ `_setup_graph` still in-memory build for 3 required artefacts (§4 anti-pattern)
- ❌ Missing `_results/` directory (§1)

---

## 8. Final verdict: ❌ FAIL (against current spec; prior PASS was relative to older spec)

| Category | Status |
|---|---|
| Paper algorithm preservation | ✅ all 4 contributions wired (carried over from prior PASS) |
| Task subfolder layout | ✅ correct |
| Cross-folder independence | ✅ verified clean |
| `_data/necessary/` + builders | ❌ ENTIRELY MISSING |
| `_shared.py` detect-and-build | ❌ entirely missing |
| `__mine` artefact convention | ❌ no artefacts in cache |
| `_results/` | ❌ missing |
| `_setup_graph` consumer pattern | ❌ still in-memory build |

The baseline is **functionally working** (per `_results/...partial.md`
attached to reproduction side, single S2_1 val_f1 ~ paper) but **fails
the current spec** on 5 structural items added during HDN/TIGER work.

---

## 9. Migration plan (mirrors TIGER baseline's 5-round path)

| Round | Task | Notes |
|---|---|---|
| R1 | Create `_results/` + `_data/necessary/` + `_shared.py` skeleton; copy 2-3 builders from reproduction-side `_Original-Dataset/necessary/` (mol features + KG vocab). Update `data_loader`-style consumer to prefer cached pkl. | mirrors HDN+TIGER initial round |
| R2 | Write builder CLI scripts: `build_morgan_cache.py`, `build_kg_cache.py`, `build_edge_tensor_cache.py`. Each takes a manifest JSON (drug pool + KG source) → outputs `_data/necessary/<name>__<hash>__mine.pkl`. | manifest pattern (TIGER round-4) |
| R3 | `_shared.py:ensure_morgan_cache`, `ensure_kg_cache`, `ensure_edge_tensor_cache` — HDN/TIGER subprocess pattern with `__official` > `__mine` priority + cache key derivation + corrupt-pkl guard. | |
| R4 | Switch `binary_cls/baseline.py._setup_graph` from inline `compute_morgan_matrix` + `build_kg_from_*` + `build_sparse_adj`/`edges_as_dense_lists` to `_ensure_*` consumer calls. `multi_cls` inherits unchanged. | |
| R5+ | Codex iterate to PASS (clean dead imports, fix any cache-key bugs, README polish). | |

Estimated 4-5 rounds × 30-60 min each, mirroring TIGER baseline timing.

---

## 10. Future work / unresolved (carryover from prior review)

1. End-to-end smoke train on project PairDataset for `"emergnn"` +
   `"emergnn_mc"` — still not run on real project data.
2. Numerical sanity vs paper Table 1 on project data — N/A since
   project data differs from paper data; comparable numbers would
   come from cross-baseline comparison, not paper Table 1.
3. After this round's migration, the `_reviews/2026-05-17__baseline.md`
   verdict needs an update note ("PASS relative to spec dated
   2026-05-17; superseded by `_reviews/2026-05-18__baseline_round2.md`
   under new spec").
