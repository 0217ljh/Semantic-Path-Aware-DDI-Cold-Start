# EmerGNN rspmm backend — multiclass + multilabel cores review

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex), thread 019f1f82
- **Trigger**: user — complete rspmm backend for all 3 tasks.

## Built
**Multiclass**
- `multi_cls/model_rspmm.py` — `EmerGNN_MC_RSPMM(EmerGNN_RSPMM)`: inherits rspmm forward,
  swaps `Wr` -> n_classes (mirrors `EmerGNN_MC(EmerGNN)`; squeeze(-1) is a no-op for K>1).
- `multi_cls/baseline_rspmm.py` — `EmerGNNMulticlassBaseline_RSPMM`: overrides fit/predict/
  save/load; `n_base_rel_with_ddi = n_kg_rel + n_classes`; sparse KG from triplets per epoch;
  positives-only sum-CE; macro-F1 best-ckpt.
- `multi_cls/baseline_unified.py` — backend dispatch.

**Multilabel (TWOSIDES, separate model)**
- `multi_label_cls/model_rspmm.py` — `EmerGNN_ML_RSPMM(EmerGNN_ML)`: inherits __init__
  (all_rel_slots = 2*all_rel-eval_rel+1), rspmm forward, ML concat order + Wr->eval_rel.
- `multi_label_cls/_core_twoside_rspmm.py` — `BaseModelTwoside_RSPMM`: overrides fit/
  _score_pairs; sparse KG via build_sparse_kg from the SAME `_build_edges` output; cumulative
  vKG/tKG resident + epoch KG per epoch; paired pos/neg BCE; valid-PR best-ckpt.
- `multi_label_cls/baseline_unified.py` — backend dispatch.

## Numerical verification (Claude, self-run)
- `EmerGNN_MC_RSPMM` -> (B, n_classes), finite.
- `EmerGNN_ML_RSPMM` == dense replay of its forward, **diff 0.00** (feat='M', all_rel_slots
  = 8 for all_rel=5/eval_rel=3). Confirms reshape/attention/bidirectional/ML-concat wiring.
- Multilabel `_build_edges` output = original TWOSIDES load_graph `(first, second, rel)` layout
  ({(t,h,r),(h,t,r_inv),self-loop 2*all_rel-eval_rel}); build_sparse_kg reconstructs it exactly.

## Codex verdict (thread 019f1f82): APPROVED_WITH_NITS
- APPROVED: (a) multiclass fit aligned, n_rel bookkeeping correct (n_base_rel_with_ddi to
  builder + model). (b) multiclass save/load correct (no Wr shape mismatch). (c) multilabel
  fit aligned (load_twoside_kg / _shuffle_train / paired BCE / valid-PR best-ckpt / scheduler).
  (d) _score_pairs override consistent (parent _eval_paired + predict_proba route through it).
  (e) no latent multilabel save/load bug (no save/load surface; in-process fit+predict).
  (f) no all_rel_slots mismatch; DDI facts slot 0, self-loop 2*all_rel-eval_rel. (g)
  faithfulness good; only benign lazy `_RSPMM_FN` global.
- NIT (Low, fixed): multilabel rspmm dropped parent's `_edge_counts` diagnostic that
  `smoke_emergnn_multilabel.py:87` reads. **Fixed**: re-added `self._edge_counts`
  (train_graph_edges from _build_edges, valid/test from sparse ._nnz()).

## Status
All 3 tasks' rspmm backend BUILT + numerically verified + codex-reviewed. Binary full-KG
smoke PASSED (6.4 min/epoch, 385.8s). NEXT: multiclass + multilabel full-KG smokes; then
acceptance gates + real benchmark runs.
