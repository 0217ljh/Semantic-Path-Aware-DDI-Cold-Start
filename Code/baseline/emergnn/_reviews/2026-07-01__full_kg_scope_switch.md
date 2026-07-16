# EmerGNN — full-merged-KG scope switch (EMERGNN_KG_SCOPE) review

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex, thread 019f1f17)
- **Trigger**: user decision — all KG-based baselines must use the FULL merged KG,
  not the drug-incident subgraph EmerGNN was silently using (4.4% of edges).

## Problem
`build_kg_from_merged_parquet` (kg_builder_merged.py:89) hard-filters the merged KG to
drug-incident edges (`edges[src_is_drug | dst_is_drug]`). Measured on `ddi_full` (1900
drugs): **311,328 / 7,099,528 edges kept = 4.4%**; 21,122 / 174,944 nodes. Pointing
`run_emergnn_kgswap.py --kg-dir` at the full KG does NOT help — the builder re-filters.
So no run path exercised the full KG. This deviates from the paper's multi-hop biomedical
flow and is inconsistent across KG baselines.

## Change (opt-in, default preserves old behavior)
Env var `EMERGNN_KG_SCOPE` ∈ {`drug_incident` (default), `full`}.

1. **kg_builder_merged.py** — ADDED `build_full_kg_from_merged_parquet()` (copy of the
   drug-incident builder minus the filter; keeps all edges; drug-first vocab; same
   direction-normalization + output schema). Existing `build_kg_from_merged_parquet()`
   UNTOUCHED.
2. **_data/necessary/build_kg_setup_cache.py** — `build_kg_setup()` gained keyword-only
   `kg_scope="drug_incident"`; branches to the full builder when `"full"`; `main()` reads
   manifest field `kg_scope` (default drug_incident).
3. **_shared.py** — `ensure_kg_setup_cache()` reads `EMERGNN_KG_SCOPE`, validates it, folds
   it into the cache key ONLY for non-default scope (`key = _stable_hash({"base": key,
   "kg_scope": scope})`), and writes `kg_scope` into the subprocess manifest. `import os`
   added. `_kg_setup_cache_key()` signature UNCHANGED.

## Functional test (ran)
`build_full_kg_from_merged_parquet` on ddi_full drug set:
- n_ent **175,009** (1900 drugs type0 + 173,109 non-drug type1), n_rel **59**,
  n_triplets **7,099,343** (7.1M full minus 185 dups), build 5.7s. Confirms full graph
  consumed, not re-filtered; drugs first in vocab. (drug-incident baseline: ~21k ent /
  311k triplets.)

## CLAUDE.md compliance
- ✅ New functionality via NEW function; existing drug-incident builder untouched.
- ✅ No existing function signature changed (`build_kg_setup` gained an additive
  keyword-only param with a behavior-preserving default; `_kg_setup_cache_key` unchanged).
- ✅ Default path byte-identical → existing caches + in-flight kgswap runs unaffected.
- ✅ No file deleted/moved/renamed; no new global mutable state (env-var read at call time).

## Codex independent verdict (verbatim excerpt)
> Verdict: **APPROVED**. No blocking or nit-level correctness issues found.
> (a) full builder consistent with original contract; flip rule correct; vocab/schema match.
> (b) default-path cache key + pkl filename identical to pre-change (re-hash only for
> non-default scope). (c) env propagates via manifest, not child env inheritance. (d)
> signature compatibility safe. (e) no silent-corruption path; `n_base_rel` from
> `kg_artifacts["n_rel"]`, model sizes `2*n_base_rel+1` (model.py:59); real risk is
> resource/latency at 7.1M triplets, not wrong tensors.

## Open risk (the reason we smoke-test next)
Full KG = 23× edges, 8.3× nodes vs drug-incident. Avg degree ≈ 81. EmerGNN `length=3`
flow propagation over 175k entities may OOM or be very slow (drug-incident 311k-edge run =
~4.5h/20ep). NOT a code-correctness issue — a feasibility question. Smoke on `ddi800`
(partial) 1-epoch first before any full-scale run.

## Reproduce (smoke, after this change)
```
EMERGNN_KG_SCOPE=full python Code/scripts/run_baseline_unified.py \
    --baseline emergnn --task binary --dataset ddi800 --split cold_s2 --fold fold0 --epochs 1
```
