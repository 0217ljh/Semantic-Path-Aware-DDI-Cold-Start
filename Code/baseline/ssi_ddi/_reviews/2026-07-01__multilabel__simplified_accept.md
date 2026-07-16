# SSI-DDI multilabel (TWOSIDES) — simplified acceptance (Case-B task backfill)

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex, review thread 019f20bd; design = the HDN-DDI
  ml design 019f209d, same Case-B pattern)
- **Trigger**: user — all baselines need all 3 tasks; backfill SSI-DDI's missing multilabel.
- **Scope**: NEW `multi_label_cls/{baseline.py, baseline_unified.py, __init__.py}` + additive
  runner `_MODULES` registration. SSI-DDI cores UNTOUCHED.

## Task framing (Case B)
SSI-DDI paper task = molecular substructure-substructure DDI (its multiclass core restores the
paper's multi-relational RESCAL). Multilabel (TWOSIDES 200 side-effect labels) reuses the SSI-DDI
ALGORITHM CORE unchanged; swap only head-activation/loss/targets/metric. Same pattern as the
codex-approved HDN-DDI multilabel.

## Design + implementation
- `SSIDDIMultilabelBaseline(SSIDDIMulticlassBaseline)`: REUSES UNCHANGED the SSI-DDI GAT
  substructure encoder + co-attention + RESCAL all-relation head (`SSI_DDI_MC.forward_all`,
  multi_cls/model.py:48), `build_drug_graphs` (SMILES), `_make_pair_batch`, Adam/scheduler.
  OVERRIDES: fixed n_labels=200 (rel_total=n_labels, no re-vocab, _idx_to_ddi_type=None),
  SIGMOID scoring, masked BCE, macro-AUPRC selection, predict_proba->(n,200), save/load n_labels.
- **Masked BCE** (multi_label_cls/baseline.py:180-192): mask=(pos_y>0); BCE(pos[mask],1)+BCE(neg[mask],0);
  bundle neg_y (== pos multihot) used ONLY as active-label mask, never as neg target.
- **Strict pair alignment** (fit :163-176 + _eval_paired :274-289): keep=pos_kept&neg_kept, remapped
  onto kept-row order (keep[pos_kept]/keep[neg_kept]).
- Wrapper mirrors HDN-DDI/EmerGNN ml: task="multilabel", register_unified("ssi_ddi"),
  make_multilabel_bundle(n_labels=meta), predict->(n,200). KG-free (SMILES only). twoside all 604 have SMILES.
- Note: ssi_ddi top __init__ left unchanged (it only re-exports the binary core by existing convention).

## Verification
- import + registry smoke PASS: get_unified("ssi_ddi","multilabel") -> SSIDDIUnifiedMultilabel.
- (No training run: GPU busy w/ EmerGNN; deferred-run.)

## Codex verdict: APPROVED (no findings)
Core idea preserved (GAT+co-attn+RESCAL reused unchanged); masked-BCE correct (neg_y mask-only,
no softmax/double-sigmoid); fixed-200 sigmoid; pair alignment correct both fit+val; macro-AUPRC;
n_labels persisted; file-independent.

## Decision: ACCEPTED (simplified gate) — core idea preserved
SSI-DDI now has all 3 tasks (binary + multiclass + multilabel).
