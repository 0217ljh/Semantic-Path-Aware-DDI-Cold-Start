# EmerGNN rspmm backend — binary training core review

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex), thread 019f1f67
- **Trigger**: user — build the rspmm binary training core (Milestone 2).

## Built
- `_per_mode_rspmm.py` — `_PerModeEmerGNN_RSPMM(_PerModeEmerGNN)`, overrides `fit()` +
  `predict_proba()` + `save()`/`load()`. Builds sparse KG per epoch from TRIPLETS via
  `build_sparse_kg_from_triplets`; instantiates `EmerGNN_RSPMM`; model called `(head, tail,
  kg_sparse)`. All other training behavior inherited/copied verbatim.
- `_rspmm_utils.build_sparse_kg_from_triplets` — mirrors original `load_graph`+`double_triple`
  byte-for-byte (verified: (0,1,0),(2,3,1) → nnz {(1,0,0),(0,1,2),(3,2,1),(2,3,3),self-loops}).
- `binary_cls/baseline_unified.py` — `EMERGNN_BACKEND` dispatch (chunk default / rspmm),
  backend printed for provenance. Same constructor for both cores.

## Codex verdict (thread 019f1f67): CHANGES_REQUIRED → fixed → OK
- **APPROVED**: (a) fit() behaviorally identical to parent except {model class, sparse KG,
  KG-from-triplets} — same shuffle_train, negatives, pairs shuffle, sum BCE, Adam +
  ReduceLROnPlateau(patience=50, factor=0.1, mode=max), best-val deepcopy + load_best_at_end,
  eval/save hooks. (b) eval KG = train_ddi + base_kg, same triplet set + convention. (c)
  per-epoch KG = same epoch_kg triplets as parent. (d) n_rel bookkeeping correct
  (n_base_rel_with_ddi to both model + builder; forward slots < n_base_rel_with_ddi, reverse
  from n_rel, self-loop isolated at 2*n_rel). (f) no faithfulness drift; only new global state
  is the benign lazy `_RSPMM_FN` memo.
- **HIGH (fixed)**: inherited `save/load` was wrong for rspmm (parent save persists
  `_eval_edges` which rspmm lacks; parent load hardcodes chunk `EmerGNN`). **Fix**: overrode
  `save`/`load` in the subclass — persist `_eval_kg_triplets` + rebuild `EmerGNN_RSPMM` on
  load, rebuild `_eval_kg` lazily in `predict_proba`.

## Post-fix verification (Claude, self-run)
- save/load round-trip: loaded model is `EmerGNN_RSPMM`, `_eval_kg_triplets` preserved,
  `predict_proba` scores identical before/after **diff 0.00**.

## Status
Binary rspmm path COMPLETE + verified (model core 0.00 E/M, training core codex-APPROVED
post-fix, save/load 0.00). NEXT: real full-KG smoke (ddi800 S2) for epoch time + memory;
then multiclass + multilabel rspmm cores.
