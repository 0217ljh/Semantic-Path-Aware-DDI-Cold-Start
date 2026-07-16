# EmerGNN rspmm backend — design + model-core review

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex) — design thread 019f1f51, model-core thread 019f1f5c
- **Trigger**: user — integrate torchdrug `generalized_rspmm` into our EmerGNN baseline
  (all 3 tasks) so the FULL merged KG (7.1M edges) is computationally feasible; reference
  the original EmerGNN repo; collaborate with codex throughout.

## Why
Our pure-torch chunk-loop EmerGNN (`model.py`) on the FULL merged KG = ~14-27 s/step
(~8-17 h/epoch) and OOM-prone. The ORIGINAL EmerGNN uses torchdrug's fused
`generalized_rspmm` CUDA kernel (`LARS-research/EmerGNN/DrugBank/models.py:70,87`). After a
1-line header patch it compiles on our torch 2.7.1+cu128 / RTX 5090 (sm_120). Measured
(self, independent): at full-KG scale (n=178k, E=10M, feat-width 2048) rspmm is ~20x faster
than a torch_scatter path AND the scatter path OOMs at E>=2M while rspmm stays ~6GB.
Numerics exact (diff 0.00), convention `out[first] += rel * in[second]`.

## Design agreed with codex (thread 019f1f51)
- **Faithfulness anchor** = the ORIGINAL `enc_ht`, NOT our chunk model. Confirmed: chunk
  loop scatters `out[edge_dst] += rel*h[edge_src]` = `out[second]+=rel*h[first]`, the
  OPPOSITE of rspmm, so it swaps the forward/reverse relation-slot labels — functionally
  learnable, but NOT numerically identical to paper. rspmm path reproduces paper exactly.
- **Backend seam** = env var `EMERGNN_BACKEND=chunk|rspmm` (default chunk, existing path
  untouched); record chosen backend in manifest/log for provenance.
- **File plan** (all NEW, existing chunk path untouched): `_backend.py`, `_rspmm_utils.py`,
  `model_rspmm.py`, `_per_mode_rspmm.py` (binary core parallel to `_per_mode.py`),
  `multi_cls/model_rspmm.py` + `multi_cls/baseline_rspmm.py`, `multi_label_cls/model_rspmm.py`
  (separate, TWOSIDES relation-slot scheme) + `_core_twoside_rspmm.py`; minimal dispatch in
  the 3 `*/baseline_unified.py`.
- Codex flags to honor: relation-slot dims, `relation_input` -> `(all_rel, B*n_dim)`,
  static `ht_embed` reused both directions, independent hiddens per direction, self-loop in
  attention space, coalesced unit-value sparse; multilabel don't keep 3 KGs resident (build
  train per-epoch, tKG at predict); dynamic KG from TRIPLETS not cached edges.

## Model core built + verified (thread 019f1f5c: APPROVED, no findings)
- `_backend.py` — `EMERGNN_BACKEND` selector (additive, no signature change).
- `_rspmm_utils.py` — lazy `get_generalized_rspmm()` (chunk path never imports torchdrug) +
  `build_sparse_kg(row,col,rel,n_ent,all_rel)` (coalesced unit-value 3D COO; convention
  `out[row]+=rel*in[col]`, reconstructs original KG from our edge_src/dst/rel = idx0/1/2).
- `model_rspmm.py` — `EmerGNN_RSPMM(EmerGNN)`: inherits ALL layers unchanged; `forward(head,
  tail, kg)` takes the pre-built sparse KG (mirrors original `enc_ht(head,tail,KG)`);
  `_propagate_rspmm` replicates the original per-layer loop exactly.
- **Numerical verification**: `EmerGNN_RSPMM.forward` == dense replay of the same forward
  (rspmm replaced by explicit `out[row]+=rel*in[col]`, same weights) at **diff 0.00 for BOTH
  feat='E' and feat='M'** (n_ent=8, n_base_rel=2, L=2). So the reshape/attention/bidirectional/
  head-concat wiring is correct end-to-end, and combined with the kernel (also 0.00) the
  model faithfully implements the original enc_ht.

## Codex verdict (verbatim excerpt, 019f1f5c)
> No correctness findings. Verdict: APPROVED. (a) faithful to original enc_ht line-by-line;
> (b) build_sparse_kg convention correct; (c) inherited __init__ compatible (rel_kg
> 2*n_base_rel+1, Wr widths match, only intended diff is binary Wr out_dim=1); (d) feat='M'
> path consistent; (e) sparse handling correct (unit, coalesced, not in state_dict, same
> device, rebuild on KG change); (f) constraints respected (additive, lazy torchdrug, new
> subclass API). Residual gap: feat='M' dense check (SINCE CLOSED: 0.00), AMP not tested
> (original is fp32).

## Next (pending)
- `_per_mode_rspmm.py` — binary training core parallel to `_per_mode.py`, builds sparse KG
  per epoch (from triplets), instantiates `EmerGNN_RSPMM`; training behavior IDENTICAL
  (shuffle_train / negatives / batch / scheduler / sum-loss / best-ckpt metric).
- binary `baseline_unified.py` dispatch on `EMERGNN_BACKEND`.
- Smoke on real full-KG (ddi800 S2, EMERGNN_BACKEND=rspmm EMERGNN_KG_SCOPE=full) -> confirm
  real epoch time + numerics sane.
- Then multiclass + multilabel cores; codex review each.
- Freeze torchdrug patch into a reproducible setup entry (patch handled by another agent;
  ensure project has a setup script).
