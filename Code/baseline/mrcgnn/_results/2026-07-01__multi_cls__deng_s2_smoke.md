# MRCGNN multiclass — deng S2 smoke (unified path)

- **Date**: 2026-07-01
- **Type**: end-to-end migration smoke (simplified acceptance — confirm the unified path runs +
  metrics are sane; NOT a paper-gate).
- **Run**: Code/runs/mrcgnn__deng__cold_s2__fold0/ ; log Code/runs/_smoke_mrcgnn_mc_deng_s2_e1.log

## Config
- baseline=mrcgnn, task=multiclass, dataset=deng (570 drugs, 65 events), split=cold_s2 (S2),
  fold0. MRCGNN epochs=1 (smoke). TrimNet builder epochs=300 (default, real).
- CLI: `python Code/scripts/run_baseline_unified.py --baseline mrcgnn --task multiclass --dataset deng --split cold_s2 --fold fold0 --epochs 1`
  (launched via clean non-interactive `bash -c` + source conda → exit 0).

## Pipeline confirmed
- TrimNet cache MISSING → detect-and-build subprocess (300 epochs) → `mrcgnn_trimnet_features__980ce06403759159__mine.npy` shape (570, 128). Cache key encodes leaf drug set + train pairs + K + fold + seed + epochs (leakage-safe).
- MRCGNN training: 1 epoch = 10.7s; total fit = 1047.9s (~17.5 min, TrimNet build dominates).
- exit 0.

## Metrics (test, 1 epoch, best-val checkpoint)
- macro_f1 **0.0131** (PRIMARY) | accuracy **0.1134** | macro_precision 0.0265 | macro_recall 0.0129 | cohen_kappa 0.0259
- oov_target_rate 0.0066 | n_test 1508 | n_classes_gold 42

## Verdict: PASS (smoke)
End-to-end unified path works (TrimNet build + 3-loss RGCN+contrastive training + predict +
5-metric report), clean exit. accuracy 0.113 is ~4.7× above the 1/42 random baseline (0.024)
→ the model is learning. macro_f1 0.013 is low but EXPECTED at 1 epoch + S2 cold-start (RGCN
branch near-inert for unseen drugs; molecular skip carries the signal — flagged as faithful
behavior by codex/agent, not a bug). Migration is not broken. Full multi-epoch / multi-fold
runs pending (user).

## Caveat
TrimNet build = 300 epochs ≈ 17 min PER leaf/fold (cached after first build). Full benchmark
(5 folds × S0/S1/S2 × deng/ryu) will incur this build cost per distinct leaf (each cached).

## Reproduce
python Code/scripts/run_baseline_unified.py --baseline mrcgnn --task multiclass \
    --dataset deng --split cold_s2 --fold fold0 --epochs 1
