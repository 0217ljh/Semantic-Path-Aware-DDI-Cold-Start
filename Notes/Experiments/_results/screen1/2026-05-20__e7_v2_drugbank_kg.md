# E7 v2 — LR over Meeting-Node Features (DRUGBANK-only KG)

**Date**: 2026-05-20
**Motivation**: Codex round-5 caveat that E7 v1 used merged KG (178K nodes)
while EmerGNN trains on drugbank KG (~10K nodes), creating possible
KG access asymmetry. v2 recomputes with matched-access drugbank KG.

## Results

| Method | KG source | test_s2 AUC | 95% CI |
|---|---|---|---|
| LR-binary | drugbank-only | 0.7182 | [0.7030, 0.7330] |
| LR-count | drugbank-only | 0.7311 | [0.7153, 0.7468] |
| LR-count | merged (E7 v1) | 0.7573 | [0.7411, 0.7726] |
| LR-binary | merged (E7 v1) | 0.7384 | [0.7217, 0.7543] |
| EmerGNN multimode | drugbank-only | 0.7462 | — |

Non-zero feature columns: 3/24 (SideEffect/Disease/Anatomy/etc. NOT in drugbank KG so zero by construction)

## Interpretation
- Gap between LR-count drugbank-only and LR-count merged: -0.0262pt
- Gap between LR-count drugbank-only and EmerGNN multimode: -0.0151pt

If LR-count drugbank-only stays close to EmerGNN: architecture insight
is real, shared-mediator features capture most of EmerGNN's signal.
If LR-count drops significantly: most of E7 v1's gain comes from richer
merged-KG context (SE / Disease / Phenotype / etc. that drugbank lacks).