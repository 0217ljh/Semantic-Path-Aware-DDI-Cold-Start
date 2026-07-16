# MKG-FENN multi-cls (task3, both-unseen) on Set_6 — single fold, seed 1

**Dataset: OUR Set_6 (1994-drug / 215-class DrugBank), using the ORIGINAL
MKG-FENN architecture ported to dynamic drug count + GPU.** Archived under
`baseline/` because the dataset is our own (not the upstream 65-event event.db).

## Run identity
- script: `Code/baseline/mkg_fenn/multi_cls/driver_task3_set6.py`
- model: `Code/baseline/mkg_fenn/multi_cls/model_task3_set6.py` (orig arch, 572→1994, GPU)
- KGs: `kg_builder.build_all_kgs` (KG1 entities / KG2 Morgan-FP / KG3 drug-drug from TRAIN pairs / KG4 properties)
- log: `_mkgfenn_task3_set6_out/train_e15b.log`, metrics: `_mkgfenn_task3_set6_out/metrics.json`
- wall time: ~20 min (15 epochs, RTX 5090)

## Config
| epoches | batch | lr | weight_decay | embedding_num | neighbor_sample | seed | device |
|---|---|---|---|---|---|---|---|
| 15 | 1024 | 5e-3 | 1e-8 | 256 | 6 | 1 | cuda |

## Data stats
- 1994 drugs, 215 ddi_type classes, 565,978 DDI pairs (valid).
- Single fold (fold 0 of the original j%5 drug split): 1600 train / 394 test drugs.
- both-unseen pairs: 364,607 train / 21,912 test.
- KG3 built from TRAIN pairs only (no cold-start leakage; codex-confirmed).

## Final metrics (best-epoch via max test macro-F1)
| ACC | micro-AUPR | micro-AUC | macro-F1 | macro-PRE | macro-REC |
|---|---|---|---|---|---|
| 0.2657 | 0.2003 | 0.9443 | **0.1195** | 0.2190 | 0.1057 |

macro-F1 trajectory (test_score per epoch): 0.101, 0.104, 0.092, 0.087, 0.101,
0.097, 0.098, 0.106, 0.096, 0.101, 0.098, 0.100, 0.109, 0.114, **0.120**.

## Verdict: PASS-with-caveats
- Pipeline runs end-to-end on GPU; both-unseen split + KG3-from-train correct (codex review thread 019ef228).
- **CAVEAT 1 — NOT converged**: macro-F1 still climbing at epoch 15 (peak = last epoch). 0.1195 is a LOWER BOUND; more epochs needed for the converged value.
- **CAVEAT 2 — single fold, single seed** (per user instruction; no std).
- **CAVEAT 3 — micro-AUC 0.944 is inflated** (215-class one-vs-rest micro averaging); macro-F1 is the honest metric.

## Context / comparison
- MKG-FENN on upstream event.db (65-class, both-unseen): macro-F1 0.2186 (paper Table 4, reproduced ≈0.20 fold-0).
- Set_6 (215-class) lower macro-F1 is consistent with more classes + non-convergence; supports the "our dataset is harder" read.

## Reproduce
```
python Code/baseline/mkg_fenn/multi_cls/driver_task3_set6.py --epoches 15 --seed 1
```
(For the converged number, raise --epoches to ~40.)
