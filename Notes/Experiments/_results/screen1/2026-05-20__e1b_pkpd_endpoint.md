# E1b PK/PD Path Endpoint Asymmetry — Motivation for i1

**Date**: 2026-05-20
**Script**: `Code/my_code/exp_e1b_pkpd_endpoint/run_e1b.py`

## Claim
PK-pair intermediates concentrate in molecular layer (Gene/Protein, Pathway); PD-pair intermediates concentrate in effect-system layer (SideEffect, Disease, Anatomy, Phenotype).

## Inputs
- pkpd labels: 215 ddi_types ({'PD': 184, 'PK': 30, 'Mixed': 1})
- positive pairs sampled: 6,000 (PK=3,000 PD=3,000)
- KG: merged DrugBank+Hetionet+PrimeKG (178,029 nodes, 7,099,528 edges)
- path enumeration: ≤3 hops, 1 shortest path per pair (tractability)

## 2x6 contingency

| class | PK_layer | PD_layer | BioProcess | Other | Direct | NoPath |
|---|---|---|---|---|---|---|
| PK | 1,611 | 1,353 | 0 | 112 | 11 | 197 |
| PD | 1,071 | 1,823 | 0 | 383 | 30 | 328 |

**Chi-square omnibus (excl Direct/NoPath/zero-cols)**: chi2=320.60, dof=2, **p=2.41e-70**

## Effect-size measures (post codex round 5)

- **Cramer's V = 0.2246** (rule of thumb: 0.1=small, 0.3=medium, 0.5=large)
  → small-to-medium effect
- **PK pairs PK_share − PD_share margin**: +0.084 [+0.050, +0.118] (significant but modest)
- **PD pairs PK_share − PD_share margin**: −0.230 [−0.262, −0.199] (moderate)

## Per-class semantic-layer share (with 95% bootstrap CI)

- **PK** (n=3,076): PK_layer 52.4% [50.6%, 54.1%], PD_layer 44.0% [42.2%, 45.8%]
- **PD** (n=3,277): PK_layer 32.7% [31.2%, 34.4%], PD_layer 55.6% [54.0%, 57.3%]

## Reframed interpretation (per codex round 5)

The asymmetry is **statistically robust but moderate in magnitude**: PD pairs
clearly favor PD-layer intermediates (23pt margin), while PK pairs only
modestly favor PK-layer intermediates (8pt margin). Avoid causal language
like "PK pairs use molecular intermediates"; prefer "PK pairs are
ENRICHED for molecular intermediates in their shortest paths".

PD-side asymmetry is robust enough to motivate Screen 5 ⑤a (PK/PD subgraph
split). PK-side asymmetry is real but small — expect moderate gain from
Screen 5 on PK pairs.

## Methodology limitations (codex round 5)

- **1-path-per-pair BFS bias**: sampled one shortest path per pair; if many
  ≤3-hop equivalents exist, path-selection may reflect graph order rather
  than true distribution. Run with random-tied-path sampling to verify.
- **Sample size n=6000 makes chi-square trivially significant** even at
  small effects — Cramer's V is the more reliable metric here.

## Top 5 intermediate node kinds per class

- **PK**: Gene/Protein=1,611, SideEffect=799, Phenotype=299, Disease=255, Drug=112
- **PD**: Gene/Protein=1,070, SideEffect=916, Disease=550, Drug=383, Phenotype=357