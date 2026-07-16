# TIGER unified wrappers — simplified acceptance (core-idea preservation)

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex; design thread 019f2066, review thread 019f2076)
- **Trigger**: user — do TIGER next per the full baseline regime (paper+repo+implement+check,
  discuss with codex throughout). SIMPLIFIED acceptance = codex confirms core idea preserved;
  no numeric run. TIGER core already codex-PASS (2026-05-18__baseline_round5_PASS.md).
- **Scope**: NEW `binary_cls/baseline_unified.py` + `multi_cls/baseline_unified.py`; full-KG
  BKG scope in `kg_builder.py` + `_shared.py` + `_data/necessary/build_bkg_cache.py`.

## Paper core idea (TIGER, Su et al. AAAI 2024)
Dual-channel: drug MOLECULAR-graph GraphTransformer + biomedical-KG (BKG) subgraph channel;
relation-aware self-attention heterogeneous graph transformer; 3 subgraph extractors
(randomWalk / khop-subtree / probability-PPR); mutual-information (MI) loss.

## What was built
- Wrappers reuse the PASS-reviewed cores UNCHANGED via `data_utils.leaf_adapter.make_dataset`;
  register_unified("tiger") for binary + multiclass. Paper-faithful defaults: cold_start_patch
  =False, extractor="randomWalk". multiclass predict scatters dense train-vocab -> global y_cls
  via `_idx_to_ddi_type` (OOV->0), like EmerGNN/HDN-DDI/SumGNN.
- **BKG defaults to the FULL merged KG** (standing decision 2026-07-01) via `TIGER_KG_SCOPE`
  env (default "full"), read in `_shared._bkg_cache_key` (so it AND `bkg_cache_key_for` stay
  consistent -> subgraph-cache key matches bkg-cache key) + `ensure_bkg_cache` manifest +
  `build_bkg_cache.py` -> `kg_builder.build_bkg_from_merged_parquet(kg_scope=...)`. NEW
  `_full_kg_edges` (keeps all edges); `_drug_incident_edges` untouched (selectable).
- **test_s2 shim** (TIGER-local): core reads `splits.test_s2` in a DEAD all_drugs union
  (binary_cls/baseline.py:308); adapter has none. Wrapper sets `ds.splits.test_s2 = empty
  pair-frame` — faithful because g1∪g2 + items() all_drugs already cover the universe.

## Verification (ran, CPU)
- compile OK; both wrappers import + `get_unified("tiger","binary"/"multiclass")` register.
- full-KG BKG builder on ddi800: **174,976 nodes / 60 rel** (59 KG + rel0 DDI/self-loop) vs
  drug_incident 15412/19. drugs-first vocab, rel0 reserved.
- (No training smoke: full-KG BKG build + subgraph extraction is slow + GPU busy w/ EmerGNN;
  per the deferred-run decision the fit-path run is left for the real-run phase.)

## Codex verdict: APPROVED (no findings)
(1) wrappers reuse cores, no forbidden hyperparam/loss override, dual-channel + MI +
relation-aware attention preserved, cold_start_patch=False. (2) test_s2 shim faithful (only
dead-code read; universe from items()/g1∪g2). (3) full-KG scope plumbing consistent across
both key paths + manifest + builder; num_rel/vocab/rel0 correct at 174,976/60. (4) multiclass
scatter correct + OOV->0. (5) no reproductions/ import.

## Decision: ACCEPTED (simplified gate) — core idea preserved
TIGER binary + multiclass wrappers accepted. Runtime note (not a faithfulness issue): full-KG
subgraph extraction is heavy; probability/PPR extractor is the risky one on full KG, so the
default extractor stays randomWalk (bounded per-drug subgraphs). Acceleration = deferred.
