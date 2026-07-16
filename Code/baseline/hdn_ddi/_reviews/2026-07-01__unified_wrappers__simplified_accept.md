# HDN-DDI unified wrappers — simplified acceptance (core-idea preservation)

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex, thread 019f1f44)
- **Trigger**: user directive 2026-07-01 — migrate remaining baselines to unified
  benchmark; SIMPLIFIED acceptance = codex confirms the port preserves the paper +
  original-code core idea; NO paper-gate run required.
- **Scope**: `binary_cls/baseline_unified.py` + `multi_cls/baseline_unified.py`
  (reused cores `binary_cls/baseline.py`, `multi_cls/baseline.py`, shared `models.py`,
  `mol_features.py`, `_shared.py`; adapter `data_utils/leaf_adapter.py`).

## Paper core idea (HDN-DDI, Sun & Zheng 2025)
Hierarchical molecular graph via refined BRICS decomposition (3-level) + y==1
substructure bipartite interaction + co-attention + RESCAL readout. KG-free.

## Smoke (done)
Import + registry OK: `get_unified('hdn_ddi','binary')` -> HDNDDIUnifiedBinary,
`get_unified('hdn_ddi','multiclass')` -> HDNDDIUnifiedMulticlass.

## Codex verdict: APPROVED_WITH_NITS
Core idea preserved end-to-end.
- (a) Routing intact: fit() -> make_dataset(task) -> core.fit(ds,ds); predict() ->
  core.predict_proba. BRICS 3-level (mol_features.py:231,283-317) + y==1 bipartite
  (_shared.py:53-77) + co-attention/RESCAL (models.py:53-75,98-115) still exercised on
  the leaf's drugs[[drugbank_id, smiles]].
- (b) Multiclass scatter correct: train dense idx -> global id via `_idx_to_ddi_type`
  (multi_cls/baseline_unified.py:73-76); OOV-in-train gold -> ~0 mass (counted wrong,
  not crash). No off-by-one/wrong-axis.
- (d) No forbidden hyperparam override: wrapper passes only n_epochs/batch_size/device/
  log_step_every/run_dir; core keeps Adam/LambdaLR/BCE-or-CE/best-metric.
- (e) File-independent: no `reproductions/**` import.

### Nit (low, ACCEPTED as benchmark-level decision)
Binary training negatives are FROZEN from the leaf's materialized `y_bin==0` rows
(leaf_adapter.py:36-39,84-85) instead of the core's intended per-epoch fresh negative
resample (binary_cls/baseline.py:337). **Accepted**: fixed materialized negatives is a
DELIBERATE unified-benchmark design (deterministic, structure-matched, identical
negatives across ALL baselines for fair comparison — EmerGNN's unified wrapper does the
same via the same adapter). It does NOT touch HDN-DDI's headline hierarchical-graph
contribution. Documented here as a known, uniform, intentional deviation.

## Decision: ACCEPTED (simplified gate)
Core idea preserved through the unified path; the single nit is an intentional
benchmark-wide choice, not a method-specific corruption. Paper-gate run WAIVED per the
2026-07-01 simplified acceptance amendment (_ACCEPTANCE.md §2).
