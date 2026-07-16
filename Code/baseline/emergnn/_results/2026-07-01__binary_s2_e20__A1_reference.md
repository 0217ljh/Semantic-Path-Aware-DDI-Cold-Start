# EmerGNN binary S2 (drugbank_latest_full / ddi_full) — 20-epoch converged (A1 reference)

**Date**: 2026-07-01
**Run**: `Code/runs/emergnn__ddi_full__cold_s2__fold0/` (run_baseline_unified, e20)
**Log**: `Code/runs/_idea2_emergnn_binary_s2_e20.log`

## Config
- baseline=emergnn, task=binary, dataset=ddi_full (drugbank_latest_full), split=cold_s2 (S2), fold0
- epochs=20, KG = old flat `_merged_kg` (drug-incident only, per kg_builder_merged), n_dim=64 length=3 (S2 feat=M batch=32 wd=1e-8)
- fit_time 16034.4s (~4.5h), single seed (run_baseline_unified default)

## Final metrics (test, best-val checkpoint)
- **test AUROC 0.7465** | AUPRC 0.7516 | F1 0.6667 | MCC 0.013 | n_test 50296 (50% pos)
- **best val_auc 0.7579** (loaded best-epoch state; ep20 raw val 0.7326)

## Role in idea2 lever-A KG-swap
This is the **A1 reference** (old flat KG) for the KG-swap experiment (codex 019f1dd2).
It is the run_baseline_unified path; the controlled A2/A3/B arms go through
`run_emergnn_kgswap.py` (identical model/config, only KG swapped + explicit seed).
Compare arms on **best val_auc** (codex go/no-go metric) and report test AUROC.
A1 val reference = **0.7579**. Prior 3-epoch run was 0.7377 (do not use — undertrained).

**Honest ceiling note**: hand-crafted micro+meso feature ceiling (GBT) was 0.729 < this
0.746/0.758, so the KG-swap lift is expected to be incremental, not a clear-margin jump.
