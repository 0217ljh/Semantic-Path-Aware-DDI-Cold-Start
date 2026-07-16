# HDN-DDI multilabel (TWOSIDES) — simplified acceptance (Case-B task backfill)

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex; design 019f209d, review 019f20a7)
- **Trigger**: user — every baseline must have all 3 tasks (bin/mc/ml) available; backfill
  HDN-DDI's missing multilabel. SIMPLIFIED acceptance = codex confirms core idea preserved.
- **Scope**: NEW `multi_label_cls/{baseline.py, baseline_unified.py, __init__.py}`; additive
  runner `_MODULES` + top `__init__` re-export. Reused HDN-DDI cores UNTOUCHED.

## Task framing (Case B)
HDN-DDI's paper task = BINARY molecular DDI. Multilabel (TWOSIDES, 200 side-effect labels)
is a Case-B adaptation: reuse the paper ALGORITHM CORE unchanged (BRICS 3-level hierarchical
mol-graph encoder + y==1 substructure bipartite + co-attention + RESCAL all-relations head),
swap only head-activation/loss/targets/metric. codex 019f209d: multilabel is actually MORE
natural than the existing multiclass softmax (RESCAL restores independent per-relation scoring).

## Design (codex-approved) + implementation
- `HDNDDIMultilabelBaseline(HDNDDIMulticlassBaseline)`: reuses _build_graphs, _make_pair_batch,
  y==1 bipartite, co-attention, RESCAL head (HDN_DDI_MC, rel_total=n_labels). Overrides: fixed
  n_labels=200 (no re-vocab, _ddi_type_to_idx=None), sigmoid + MASKED BCE, predict_proba->(n,200)
  sigmoid, macro-AUPRC selection, save/load persist n_labels.
- **Masked BCE (codex correction, verified)**: bundle's neg_y == pos multihot is used ONLY as
  the active-label mask; loss = mask=(pos_y>0); BCE(pos_logits[mask],1)+BCE(neg_logits[mask],0)
  (baseline.py:215/223/226). Not fed as neg target; no softmax/CE.
- Wrapper mirrors EmerGNN ml: task="multilabel", register_unified("hdn_ddi"), loads
  train/val_pair_links.parquet, make_multilabel_bundle(n_labels=meta labels), predict->(n,200).
- Twoside has SMILES for all 604 drugs -> HDN-DDI molecular encoder applies.

## Verification
- import + registry smoke PASS: get_unified("hdn_ddi","multilabel") -> HDNDDIUnifiedMultilabel.
- (No training run: GPU busy w/ EmerGNN; deferred-run.)

## Codex verdict: APPROVED_WITH_NITS
Core idea preserved (encoder/RESCAL/graphs reused unchanged); masked-BCE correct; fixed-200
sigmoid; macro-AUPRC; file-independent (no emergnn/reproductions import); no softmax/double-
sigmoid bug.
- **NIT (medium, latent — documented for real-run)**: predict_proba leaves missing-mol-graph
  pairs at 0.5 (baseline.py:346,383); runner _multilabel_metrics scores all linked rows with
  no keep-mask, so a SMILES-unbuildable drug's pairs would score 0.5 placeholders. On twoside
  all 604 drugs have SMILES -> does not trigger; consistent with other baselines' dropped-pair
  fallback. Fix (drop-aware runner mask) is a shared-runner concern, out of scope here.

## Decision: ACCEPTED (simplified gate) — core idea preserved
HDN-DDI now has all 3 tasks (binary + multiclass + multilabel).
