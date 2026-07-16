# EmerGNN Reproduction Results

## Run identity

- **run_id**: `2026-05-17_15-40-53__emergnn_reproduce__s2_1_full__seed0`
- **run_dir**: `Code/runs/2026-05-17_15-40-53__emergnn_reproduce__s2_1_full__seed0/`
- **train log**: same dir, `train.log` (58 lines)
- **setting**: `S2_1` (paper's 1st of 5 seeds for the S2 cold-start protocol, both drugs unseen)
- **status**: user-interrupted at ep 38/100; final test eval NOT run because best_state is in-memory only and Ctrl+C lost it
- **wall time**: ~3.9 hours (38 epochs × avg 370 s)

## Config (verified from train.log line 4)

| param | value |
|---|---|
| n_epoch (target) | 100 |
| n_batch | 32 |
| n_dim | 64 |
| length | 3 |
| feat | M (Morgan) |
| lr | 1e-3 |
| weight_decay | 1e-8 |
| shuffle_ratio | 0.8 |
| epoch_per_test | 5 |
| seed | 0 |
| chunk_size | 500000 |
| use_checkpoint | True |
| device | cuda (RTX 5090) |

## Data stats (verified from train.log lines 7-9)

| field | value |
|---|---|
| n_drug | 1,710 |
| n_entity | 34,124 |
| n_relation | 109 (= 86 DDI + 23 KG) |
| train_ddi (S2_1) | 137,864 |
| valid_ddi (S2_1) | 536 |
| test_ddi (S2_1) | 1,901 |
| vKG edges (= train_ddi + valid_kg) | 3,642,486 |
| tKG edges (= train_ddi + valid_ddi + test_kg) | 3,692,310 |

## Validation trajectory (verified from train.log lines 17, 22, 27, 32, 37, 42, 47)

| epoch | val_macro_f1 | val_acc | val_kappa |
|---|---|---|---|
| 5 | 0.0297 | 0.2743 | 0.0170 |
| 10 | 0.0440 | 0.3470 | 0.1122 |
| 15 | 0.1741 | 0.3918 | 0.2253 |
| 20 | 0.2044 | 0.4291 | 0.2678 |
| 25 | 0.2295 | 0.4291 | 0.2804 |
| **30** | **0.2487** | **0.4440** | **0.2942** |
| 35 | 0.2487 | 0.4440 | 0.2907 |

**Best so far**: epoch 30, val_macro_f1 = **0.2487**

## Comparison vs paper Table 1 (verified from `paper-and-github/_paper_text.txt` line 760)

Paper EmerGNN* DrugBank **S2** mean±std across 5 paper seeds (S2_1, S2_12, S2_123, S2_1234, S2_12345):

| metric | paper (mean ± std) | ours (S2_1 seed 0, val ep30) | distance |
|---|---|---|---|
| Macro F1 | 0.250 ± 0.028 | 0.2487 | ≈ 0.05σ (essentially exact) |
| Accuracy | 0.463 ± 0.036 | 0.4440 | ≈ -0.53σ |
| Kappa | 0.319 ± 0.038 | 0.2942 | ≈ -0.65σ |

All three metrics fall within 1σ of paper means.

## Caveats

1. Paper reports **test** metrics; we report **val** metrics (ep 30 best so far). Training was killed before the final test eval block, so test number was never computed.
2. Paper averages over 5 S2 seeds (S2_1...S2_12345); we ran only S2_1, so our value is a single-sample estimate.
3. best_state ckpt was held in memory only (per upstream `train_one_setting`), so resuming for final test eval is not possible without re-training.
4. PageRank extractor is NOT applicable here (EmerGNN is k-hop-flow-based, not subgraph-based); this caveat is for TIGER.

## Verdict

**EmerGNN reproduction partial-validation signal** for DrugBank S2_1.

Caveat: this is a **single seed**, **val-only**, **early-stopped at
ep 38/100** result. The full PASS evidence required by CLAUDE.md
§"复现代码规范" §3 step 4 (5-seed test mean±std comparison to paper
Table 1) has not been produced. The matching of val_macro_f1 = 0.2487
against paper's 0.250 ± 0.028 (within 0.05σ) is a strong **early
indicator** that the ported algorithm tracks paper, but is not a
complete reproduction. Running the full 5-seed × 100-epoch test sweep
remains future work (see `_reviews/2026-05-18__paper_faithful.md` §10).

## Reproducing this result

```bash
python Code/reproductions/EmerGNN/run_reproduction.py \
    --setting S2_1 --n-epoch 100 --tag s2_1_full
```

Wall time on RTX 5090 with chunk_size=500000 + gradient checkpointing: ~6 min/epoch for the first 30 epochs, then ~7-11 min/epoch as graph sparsity changes. Full 100 epoch run estimate ~12 hours.
