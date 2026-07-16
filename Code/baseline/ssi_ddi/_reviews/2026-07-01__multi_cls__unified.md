# SSI-DDI multiclass — unified wrapper (simplified acceptance)

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex), thread 019f1fd0
- **Trigger**: user — implement SSI-DDI multiclass (non-KG baseline). Simplified
  acceptance (core-idea + correctness; paper-gate waived per 2026-07-01 amendment).

## Paper core idea (SSI-DDI, Nyamabo et al. 2021)
Substructure-aware GAT blocks + co-attention + RESCAL multi-relational scoring. The
binary baseline collapsed RESCAL to rel_total=1; multiclass RESTORES the paper's
multi-relational head (rel_total = n_classes, all-relation scoring).

## Built (new files only; SSI_DDI/RESCAL/SSIDDIBaseline untouched)
- `multi_cls/model.py` — `SSI_DDI_MC(SSI_DDI)`: inherits GAT+co-attention+RESCAL rel_emb;
  adds `forward_all((h,t)) -> (B, n_rels)` via `einsum('bij,bip,rpq,bjq->br', alpha, Hn,
  Mn, Tn)` with RESCAL's normalization.
- `multi_cls/baseline.py` — `SSIDDIMulticlassBaseline(SSIDDIBaseline)`: rel_total=n_classes,
  ddi_type vocab, positives-only sum-CE via forward_all, macro-F1 best-ckpt, predict (n,K)
  softmax. Inherits _build_graphs/_make_pair_batch.
- `multi_cls/baseline_unified.py` — `SSIDDIUnifiedMulticlass`: scatter train-vocab -> global
  (unseen global classes -> 0 mass, argmax-unreachable; mirrors EmerGNN mc wrapper).
- registered `("ssi_ddi","multiclass")` in run_baseline_unified._MODULES.

## Numerical verification (Claude, self-run)
`SSI_DDI_MC.forward_all((h,t))[:, r] == SSI_DDI.forward((h,t,full(r)))` for every r,
**max abs diff 3.7e-08** (real mol graphs, K=5). The all-K head is numerically identical
to the original per-relation RESCAL scoring.

## Codex verdict (thread 019f1fd0): APPROVED_WITH_NITS
- APPROVED: (a) forward_all faithful vectorization of RESCAL.forward (normalization + alpha
  + sum match). (b) _pair_reprs replicates SSI_DDI.forward's GAT loop + co-attention exactly
  minus RESCAL. (c) multiclass contract consistent with EmerGNN mc (positives-only, sum-CE,
  macro-F1 best-ckpt, scatter); all-K softmax CE is a sound case-B adaptation preserving the
  multi-relational core. (d) no masking/label-alignment bug. (e) save/load correct (rel_total
  rebuilt from persisted n_classes). (f) additive, file-independent, no signature changes.
- NIT (Low, fixed): baseline_unified.py docstring said unseen global classes get ~1/K mass;
  actually zero-init -> 0 mass. **Fixed** the wording (behavior was already correct + matches
  EmerGNN).
- Residual: codex did static review only; runtime smoke run separately (this session).

## Status
SSI-DDI multiclass BUILT + numerically verified + codex-approved. Runtime smoke (ddi800 S2)
launched. SSI-DDI now: binary (accepted) + multiclass (this). Non-KG baseline complete.
