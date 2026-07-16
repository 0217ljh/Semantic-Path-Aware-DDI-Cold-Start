# MRCGNN binary + multilabel extensions — review

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8, coordinator) + general-purpose subagents (porters)
- **Independent reviewer**: codex (gpt-5.2-codex) — binary design 019f20ab, binary impl 019f20b5,
  multilabel design 019f20bc, multilabel impl 019f20c6
- **Trigger**: user — MRCGNN must cover 3 tasks. binary = normal case-B extension;
  multilabel = reliable implementation, partial-non-faithful allowed.

## Task coverage (all 3 now)
- multiclass ✅ (2026-07-01, prior review, real smoke PASS).
- **binary** ✅ (this) — code + codex APPROVED_WITH_NITS.
- **multilabel** ✅ (this) — code + codex APPROVED_WITH_NITS.

## BINARY (binary_cls/{__init__,baseline,baseline_unified}.py; register ("mrcgnn","binary"))
Case-B: MRCGNN(n_classes=1) → single-relation RGCN + 1-logit head + BCE + per-epoch negatives.
codex-decided (019f20ab) + verified (019f20b5):
- **Q1 View B DROPPED**: with num_relations=1, edge_type all-zeros → view B (relation-shuffle) is
  a no-op (2nd pass differing only by dropout) → loss3 is noise. Binary loss = 1.0*BCE_cls +
  0.05*BCE_A (2-loss). Documented deviation.
- **Q2 best-ckpt val AUROC** (matches ssi_ddi/emergnn binary; report both AUROC+AUPRC).
- **Q3 TrimNet from SIBLING MULTICLASS leaf**: binary model n_classes=1, but TrimNet features
  built from the same (dataset/split/fold) multi_cls leaf's train (y_cls→ddi_type, K=K_mc) —
  event-supervised features, faithful, cold-start safe, binary/MC share the cache. Missing MC
  sibling → FileNotFoundError (no binary-supervised fallback).
- Verified: single-relation graph (edge_type=[0]), 2-loss finite, view B not in loss, predict
  (n,) sigmoid, per-epoch negs differ, TrimNet source = sibling MC leaf.
- Nits (non-blocking): fold=split_code in cache key (fold-distinct via train-triples anyway;
  consistent w/ MC wrapper); wasted view-B forward (compute only); save/load provenance minor.

## MULTILABEL (multi_label_cls/{__init__,baseline,baseline_unified}.py; register ("mrcgnn","multilabel"))
TWOSIDES. codex-designed (019f20bc) + verified (019f20c6). Partial-non-faithful (allowed).
- **Q1 MULTI-RELATIONAL explosion**: num_relations=200; each pos pair exploded into one
  bidirectional edge-pair per active side-effect label r → (a,b,r)+(b,a,r), edge_type=r. Reuses
  MRCGNN(n_classes=200) UNCHANGED. Faithful to MRCGNN's multi-relational RGCN (mirrors HDN-DDI ml).
- **Q2 masked BCE**: 200-logit head; paired pos/neg; mask = positive pair's active labels applied
  to BOTH pos + paired-neg logits (matches emergnn/_core_twoside recipe); sigmoid at inference.
- **Q3 both views + 3-loss KEPT**: 1.0*masked_BCE_cls + 0.05*BCE_A + 0.1*BCE_B; view B shuffles
  the [r,r] pairs (meaningful under 200 relations).
- **Q4 TrimNet**: builder unchanged, fed exploded (a,b,r) triples + n_classes=200 (CE, one relation
  per example). Cache key distinct by triples+K.
- **Q5 cold-start**: graph/views/TrimNet from TRAIN positives only; all-drug universe; unseen drugs
  get TrimNet features + no incident edges. TWOSIDES ships SMILES for all 604 drugs.
- best-ckpt val macro-AUPRC; predict (n,200) sigmoid, 0.5 fallback for missing-graph drugs.
- Verified: explosion (2,12) edges edge_type∈[0,K), 3-loss finite, predict (3,5) sigmoid, unseen
  drug works, registry OK.
- Non-faithful (allowed): sigmoid head, masked BCE, macro-AUPRC, label→relation-incidence explosion.
- **Risk (codex)**: incidence explosion over-weights polypharmacologic pairs (one pair → many
  parallel edges) → could distort message passing + noisier view B. Fallback: drop view B, then
  single-interaction graph. Watch S2 macro-AUPRC.
- Nit: fold=split_code (same as binary/MC, cosmetic).
- Impl note: 2 full-graph forwards/step (pos + paired neg) → ~2× RGCN forwards vs multiclass
  (timing note for the GPU smoke).

## Status
MRCGNN binary + multilabel code-complete + codex-approved. Real GPU smokes PENDING (deferred while
the EmerGNN ddi800 run occupies the GPU — no contention). Run when GPU frees:
- binary: `run_baseline_unified.py --baseline mrcgnn --task binary --dataset deng --split cold_s2 --fold fold0 --epochs 5`
- multilabel: `--baseline mrcgnn --task multilabel --dataset twosides --split cold_s2 --fold fold0 --epochs 5` (note: first run builds TrimNet ~300ep + 2 forwards/step).
