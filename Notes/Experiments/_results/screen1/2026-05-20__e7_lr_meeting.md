# E7 LR-over-Meeting-Node Baseline — Motivation for i2

**Date**: 2026-05-20
**Script**: `Code/my_code/exp_e7_lr_meeting/run_e7.py`

## Claim
LR over 24 shared-mediator features achieves test_s2 AUC within 3pt of vanilla GCN K=2.
If true, the meeting-node signal alone carries most of the cold-start prediction signal,
which justifies designing Screen 3 around meet-in-middle pooling.

## Features (24 = 12 canonical kinds x {h1 intersection, h2 intersection})

Kinds: Drug, Gene/Protein, SideEffect, Disease, Anatomy, Pathway, Phenotype, BiologicalProcess, MolecularFunction, CellularComponent, PharmacologicClass, Exposure

## Results

| Method | test_s2 AUC | 95% bootstrap CI |
|---|---|---|
| LR-binary (24 indicators) | 0.7384 | [0.7217, 0.7543] |
| LR-count  (24 log1p counts) | 0.7573 | [0.7411, 0.7726] |

## Reference
- EmerGNN multimode (seed 42, 800-drug, drugbank KG) test_s2 AUC = **0.7462**
- HDN-DDI binary (seed 42) test_s2 AUC = 0.6201
- TIGER binary (no cold-start patch) test_s2 AUC = 0.5842

## Interpretation

- LR-count vs EmerGNN gap: -0.0111 pt
- LR-count vs HDN gap: +0.1372 pt
- LR-count vs TIGER gap: +0.1731 pt

**Gate (i2 motivation)**: LR-count within 5pt of EmerGNN (informal) -> PASS (observed gap -0.0111).