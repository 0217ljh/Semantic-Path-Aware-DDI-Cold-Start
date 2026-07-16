# EmerGNN Baseline Review — Round 5 (FINAL PASS)

- **Date**: 2026-05-18 (round 5, supersedes round-2 FAIL)
- **Primary reviewer**: Claude (sonnet-4.5)
- **Independent reviewer**: codex (gpt-5-codex), 5 rounds (round-1 was
  the original 2026-05-17 audit, round-2 re-audit, rounds 3-5 iteration)
- **Triggered by**: 用户 "进入循环"
- **Outcome**: ✅ codex round-5 verdict = **PASS confirmed**

---

## 5-round progress timeline

| Round | Date | Verdict | Change |
|---|---|---|---|
| 1 | 2026-05-17 | PASS (vs older spec) | initial baseline review (pre-`_data/necessary/` spec) |
| 2 | 2026-05-18 | FAIL (vs current spec) | re-audit against newer spec; flagged missing `_data/necessary/`, `_shared.py`, builders, `__mine` artefacts, `_results/` |
| 3 | 2026-05-18 | PASS_WITH_NITS | scaffold + `_shared.py` + builder + consumer refactor; nit on missing STALE/BUILT log labels |
| 4 | 2026-05-18 | NOT_PASS | STALE+BUILT labels added but STALE filter buggy (would mislabel CORRUPT/force-rebuild) |
| 5 | 2026-05-18 | ✅ **PASS** | STALE sibling filter fixed (exclude current key's path) |

---

## Final state vs CLAUDE.md §"Baseline 规范" checklist

| Spec requirement | Status |
|---|---|
| §1 layout: `_results/`, `_reviews/`, `_data/necessary/` | ✅ all present (created round-3) |
| §1 layout: task subfolders `binary_cls/`, `multi_cls/` | ✅ existed since 2026-05-17 |
| §1 layout: shared code at top + `_shared.py` | ✅ |
| §1 `__mine` convention | ✅ `kg_setup__<hash>__mine.pkl` |
| §2 step 2: builder under `_data/necessary/build_<name>.py` | ✅ `build_kg_setup_cache.py` |
| §2 step 3: detect-and-build via subprocess + cache HIT/STALE/CORRUPT/MISS/BUILT logs | ✅ all 5 labels present + STALE filter correct |
| §3 filename-suffix anti-pattern | ✅ removed (binary_cls/multi_cls subfolders) |
| §4 case A binary task: paper algorithm preservation | ✅ all 4 contributions wired (shuffle_train + bidirectional + attention + Morgan); carried over from 2026-05-17 PASS |
| §4 builder consumption | ✅ `_setup_graph` (merged path) routes through `_ensure_kg_setup_cache`; `drugbank` legacy stays in-memory (documented limitation) |
| §6 cross-folder independence | ✅ zero `from reproductions` imports |
| §"必需文件 detection pipeline" line 635 log labels | ✅ HIT / STALE / CORRUPT / MISS first-time / BUILT |

---

## Files created / modified (5 rounds total)

### New files (round-3+)
| Path | Purpose |
|---|---|
| `_data/necessary/build_kg_setup_cache.py` | CLI builder (manifest-based) for combined kg_setup cache |
| `_shared.py` | `_kg_setup_cache_key`, `load_kg_setup_cache`, `ensure_kg_setup_cache` (HDN/TIGER-style subprocess pattern) |
| `_results/` | scaffold dir for future result archives |
| `_reviews/2026-05-18__baseline_round2.md` | round-2 re-audit (FAIL) |
| `_reviews/2026-05-18__baseline_round5_PASS.md` | THIS file (PASS) |

### Modified (round-2+)
| Path | Change |
|---|---|
| `binary_cls/baseline.py` | `_setup_graph` refactored: `kg_source="merged"` branch consumes cache; `kg_source="drugbank"` legacy in-memory fallback; added `_ensure_kg_setup_cache` import |
| `multi_cls/baseline.py` | inherits refactored `_setup_graph` unchanged; automatically gets cache consumer pattern |

### Files NOT modified (preserved per CLAUDE.md "代码修改规则")
- `model.py`, `kg_builder.py`, `kg_builder_merged.py`, `morgan_features.py`, `shuffle_utils.py`, `multi_cls/model.py`
- `_smoke_save_load.py`, `_reviews/2026-05-17__baseline.md`

---

## Codex round-5 verdict (verbatim)

> PASS confirmed。
>
> 核对 `Code/baseline/emergnn/_shared.py`：
>
> - `CORRUPT` 路径：当前 key 的 `pkl_path` 存在但反序列化失败后，会进入
>   rebuild；`sibling_stale` 已用 `if p.name != pkl_path.name` 排除当前
>   文件，所以不会再误报 `STALE`。
> - `force_rebuild=True`：即使当前 key 文件存在，也跳过 HIT 并进入 rebuild；
>   同样因当前文件被排除，不会误报 `STALE`。
> - 真正 `STALE`：只有存在其他 `kg_setup__*__mine.pkl` 且文件名不同于当前
>   key 时，才打印 `kg_setup STALE`，语义正确。
> - 调用面只看到 `binary_cls/baseline.py` 使用 `_ensure_kg_setup_cache`，
>   builder/consumer 相关接口未见回归迹象。
>
> 结论：Round-5 修复有效，确认 PASS。未运行测试，仅做代码审查核验。

---

## Cache key derivation (codex verified correct invalidation)

Cache key = sha256 of:
- `kg_source` ("merged" only — drugbank not cached)
- sorted `drug_id_list`
- sorted `(drug_id, smiles)` pairs from `train.drugs`
- merged_kg_path string + file (size, mtime_ns) [for in-place rewrite detection]
- sorted `blocklist`

Invalidation:
- Drug pool change → key changes → STALE rebuild
- SMILES change → key changes → STALE rebuild
- merged KG parquet rewrite (different mtime) → key changes → STALE rebuild
- Same inputs → same key → cache HIT (reproducibility guaranteed)

---

## Reproduction command (post-PASS canonical usage)

```bash
# First run on a new project dataset → MISS first-time → subprocess build
# → BUILT log → cache written
python Code/scripts/run_baseline.py --baseline emergnn \
    --kg-source merged --epochs 100 --tag emergnn_binary_e100

# Subsequent runs on same dataset → cache HIT (no subprocess)
python Code/scripts/run_baseline.py --baseline emergnn \
    --kg-source merged --epochs 100 --tag emergnn_binary_e100_v2

# Change drug pool / SMILES / KG file → STALE rebuild
# (auto-detected via cache key change)

# Multi-class variant inherits cache consumer pattern automatically
python Code/scripts/run_baseline.py --baseline emergnn_mc \
    --kg-source merged --epochs 100 --tag emergnn_mc_e100

# Legacy in-memory path (drugbank 5-bucket schema; no caching)
python Code/scripts/run_baseline.py --baseline emergnn \
    --kg-source drugbank --epochs 100 --tag emergnn_legacy_e100
```

---

## Acceptable deferred items (documented)

1. **`kg_source="drugbank"` legacy path stays in-memory** — input is a
   transient Python object with 5 DataFrames; caching key derivation
   would require hashing DataFrame content which is complex. The merged
   path covers production runs.
2. **No smoke train on real project data this round** — algorithm
   preservation (4 paper contributions) verified statically + in prior
   PASS review. Numerical validation deferred.
3. **No `__official` artefact convention** — baseline has no
   upstream-shipped artefact to mirror as `__official`; reproduction-side
   has its own `__official` for paper data (per
   `reproductions/EmerGNN/_reviews/2026-05-18__paper_faithful_round4_PASS.md`).
4. **Old `_reviews/2026-05-17__baseline.md` PASS verdict** — relative to
   older spec; not retracted (history-preserving per CLAUDE.md
   §"_reviews/ 目录约定"). This file (round-5) is the current verdict.

---

## ✅ FINAL VERDICT: PASS

After 5 codex rounds (1 audit + 1 re-audit + 1 scaffold/refactor + 1
log-labels + 1 STALE-filter bug fix), `Code/baseline/emergnn/` fully
conforms to CLAUDE.md §"Baseline 规范" (lines 443-637).

All paper-algorithm contributions preserved from prior PASS. New
spec requirements (`_data/necessary/`, `_shared.py`, `__mine` artefacts,
detect-and-build subprocess, log labels) all implemented. Cross-folder
independence from `reproductions/EmerGNN/` maintained (both sides keep
independent copies of `EmerGNN`, `kg_builder`, `morgan_features`,
`shuffle_utils`).

Codex round-5 explicitly confirms "Round-5 修复有效，确认 PASS"
with no further nits or regressions.
