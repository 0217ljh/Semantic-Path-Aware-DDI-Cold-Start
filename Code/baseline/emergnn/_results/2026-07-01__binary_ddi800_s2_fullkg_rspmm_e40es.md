# EmerGNN binary — ddi800 S2 fold0, full merged KG + rspmm, early-stopped

- **Date**: 2026-07-01
- **Type**: baseline result on OUR data (drugbank_latest_partial = ddi800), for the cold-start
  binary DDI benchmark. NOT a paper-gate (EmerGNN paper has no binary-DrugBank number).
- **Run**: Code/runs/emergnn__ddi800__cold_s2__fold0/results.json

## Config
- baseline=emergnn, task=binary, dataset=ddi800 (drugbank_latest_partial, 800 drugs), split=cold_s2
  (S2 both-unseen), fold0. backend=rspmm, KG scope=full merged KG (7.1M edges), n_dim=64 length=3
  (S2 feat=M batch=32 wd=1e-8). EMERGNN_EARLY_STOP_PATIENCE=5.
- epochs: cap 40, **EARLY-STOPPED at epoch 21** (val_auc no improvement for 5 epochs; best ~ep16).
- fit_time 7971.7s (~2.2h), single seed, GPU (RTX 5090), ~6.3 min/epoch.

## Final metrics (test, best-val-AUROC checkpoint)
| metric | value |
|---|---|
| **AUPRC (PRIMARY)** | **0.7406** |
| **AUROC** | **0.7302** |
| F1 | 0.6667 |
| Accuracy | 0.5000 |
| MCC | 0.0 |
| n_test | 6084 (50% pos) |
| score dispersion | min 0.657 / mean 0.926 / max 1.0 / std 0.070 |

## Interpretation
- **AUROC 0.73 / AUPRC 0.74 (threshold-free) are the meaningful numbers** — the model's ranking
  separates pos/neg reasonably for S2 cold-start binary.
- **Accuracy 0.5 / MCC 0.0 / F1 0.667 are a THRESHOLD-0.5 ARTIFACT, not a ranking failure**: sigmoid
  scores are all skewed high (min 0.657, mean 0.926), so at threshold 0.5 nearly everything is
  predicted positive → on the balanced test set accuracy=0.5, f1=0.667, mcc=0. The sigmoid is
  uncalibrated; a validation-tuned threshold would fix acc/f1/mcc. Report AUPRC/AUROC, not acc/f1/mcc.
  (Same threshold degeneracy noted for the multilabel gate.)
- Early stopping (patience=5) worked: stopped at ep21, saving ~19 epochs.

## Role
This is the **full-KG arm** of the pending KG-scope ablation (does bigger KG help cold-start?).
Comparison arms still to run on the SAME ddi800 S2 fold0 (same epochs/early-stop):
drug-incident KG (EMERGNN_KG_SCOPE unset) + molecular-only (SSI-DDI/HDN-DDI). AUROC 0.73 is the
full-KG reference.

## Caveat
Single fold, single seed. NOT apples-to-apples with the earlier A1 reference (0.7579 val_auc) which
was ddi_full + drug-incident + 20ep — different dataset/scope/epochs, do not compare directly.

## Reproduce
EMERGNN_BACKEND=rspmm EMERGNN_KG_SCOPE=full EMERGNN_EARLY_STOP_PATIENCE=5 python Code/scripts/run_baseline_unified.py \
    --baseline emergnn --task binary --dataset ddi800 --split cold_s2 --fold fold0 --epochs 40
