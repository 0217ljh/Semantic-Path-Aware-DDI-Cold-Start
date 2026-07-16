# v4 PLAN — Pair-Conditional Meeting-Node Selector (i3 × i2 × i4)

**Status**: DRAFT for codex review (round 21). Do NOT code until PASS.
**Split**: seed42 800-drug (fast iteration). **Target**: S2 AUC > 0.80 (MNAH ~0.77).
**Why this (not multimodal)**: gate proved bag-of-molecular is redundant with MNAH
(oracle ensemble gain = 0). Pivot to KG-side per codex r20. This is insight i3.

## Motivation
MNAH Stage 1 aggregates shared mediators by COUNT per kind — treats all mediators of a
kind equally. i3 says: not all shared meeting nodes matter equally for a given pair;
which specific mediator matters is PAIR-CONDITIONAL. A pair-conditional SELECTOR over
individual mediators should beat static counts. This synthesizes i2 (meeting-node anchor)
+ i3 (pair-conditional selection) + i4 (PubMedBERT node semantics).

## Architecture (builds on MNAH, replaces count-aux with a selector)
Per pair (u,v):
- Enumerate shared non-drug mediators M = N(u) ∩ N(v) (same set as Stage-1 counts, but
  KEEP node identities). Cap at top-K by degree if |M| large (e.g. K=64).
- Per mediator m, feature f_m = [ kind-onehot(11) ; PubMedBERT(name)→PCA64 ; log1p(deg_m) ;
  hop-indicator(1/2) ].  (PubMedBERT reuses cached d_name_only.pt — i4.)
- Drug-context c_u, c_v: reuse the 22-dim shared-mediator-count vector of the pair? No —
  use per-drug context: pooled f over each drug's own mediators, OR Morgan FP proj. Decide w/ codex.
- Pair-conditional attention: α_m = softmax_m( MLP([c_u ; c_v ; f_m ; c_u⊙f_m ; c_v⊙f_m]) ).
- aux_repr = Σ_m α_m · f_m  (attention-weighted meeting-node representation, per pair).
- aux_logit = MLP(aux_repr).
- combined = emergnn_logit + softplus(β)·aux_logit. BCE on combined. (Same fusion as MNAH.)

## Why it could break 0.80 where counts plateau
- Counts can't distinguish "shared CYP3A4 (mechanistically critical)" from "shared generic
  GO term"; the selector can, using pair-conditional + PubMedBERT semantics.
- This is exactly the i3 "perception" the count aux lacks.

## Controls (anti-artifact, per established discipline)
- vs MNAH Stage-1 counts (same data) — does selection beat counting?
- uniform-attention ablation (α=1/|M|) — isolates the SELECTION value vs just per-node features.
- shuffled-mediator control — scramble m→feature → should collapse.
- 3 repeats, within-run Δ.

## Open questions for codex (round 21)
1. Per-mediator enumeration is expensive (precompute per-pair top-K mediator lists +
   features). Feasible at ~50k train pairs × K=64? Or subsample mediators?
2. Drug-context c_u, c_v: 22-dim count vector of drug's own neighborhood, Morgan FP proj,
   or pooled mediator features? Which gives the cleanest pair-conditioning signal?
3. Is softmax attention right, or top-k hard selection (sparser, more interpretable i3)?
4. PubMedBERT per-mediator features reuse the redundancy-with-flow risk we saw in Stage 2
   (text redundant at combined). Will per-mediator PubMedBERT help here where pooled didn't?
5. Risk this is just a higher-capacity count-aux that overfits 800-drug — how to guard?
6. Realistic: can pair-conditional selection plausibly add +3pt over counts to hit 0.80,
   or is 0.77→0.80 just hard? Kill criterion?
