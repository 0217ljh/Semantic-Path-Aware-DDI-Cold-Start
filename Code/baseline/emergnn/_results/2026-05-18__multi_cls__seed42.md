# EmerGNN Baseline multi_cls — seed 42, 100 epochs (K-way DDI type prediction)

> **Why this lives under `baseline/emergnn/_results/`**: this run uses our
> project's 800-drug DrugBank cold-start dataset
> (`Code/data/coldddi_legacy/800drug/seed42.pkl`) + `baseline/emergnn/multi_cls/`.
> Sister file to `2026-05-18__binary_cls__seed42.md` (same data, different head).

## 1. Run identity

- **run_id**: `2026-05-18_14-43-49__run_baseline__emergnn_mc_seed42_aligned__seed42`
- **run_dir**: `Code/runs/2026-05-18_14-43-49__run_baseline__emergnn_mc_seed42_aligned__seed42/`
- **artifacts**: `results.json`, `train.log`, mirror at `Code/runs/_logs/<run_id>.log`
- **dataset**: 800-drug DrugBank cold-start seed42 PKL bundle
- **KG**: DrugBank 5-bucket
- **status**: completed full 100 epochs; loaded best val_macro_f1=0.0626 ckpt
- **wall time**: fit=1835 s + eval ~55 s = **~31 min total**

## 2. Config

| param | value | source |
|---|---|---|
| baseline | `emergnn_mc` (legacy form; new canonical `emergnn-mcc`) | CLI `--baseline` |
| kg_source | drugbank | CLI |
| epochs | 100 | CLI |
| seed | 42 | CLI |
| n_classes | **152** (observed train ddi_types) | runtime |
| batch_size | 32 | inherited (paper) |
| n_dim | 64 | inherited (paper) |
| optimizer | Adam(lr=1e-3, wd=1e-8) | paper |
| scheduler | ReduceLROnPlateau | paper |
| shuffle_train_mode | "S2" | paper inductive simulation |
| loss | `F.cross_entropy` | K-way task |
| best ckpt metric | `val_macro_f1` | spec §4 case A |
| neg sampling | N/A (positives-only K-way training) | task-specific |

## 3. Data stats

| split | n_pos | n_oov_excluded | eval n |
|---|---|---|---|
| train | 53,743 | 0 | — |
| test_s0 (warm) | 2,986 | 1 | 2,985 |
| test_s1 (1 cold) | 15,258 | 119 | 15,139 |
| test_s2 (2 cold) | 1,919 | 32 | 1,887 |

## 4. Final metrics (results.json `eval`, best-val-macro_f1 ckpt)

| split | n | top1_acc | top3_acc | top5_acc | macro_f1 | macro_auc |
|---|---|---|---|---|---|---|
| test_s0 (warm) | 2,985 | **0.5591** | 0.7843 | 0.8616 | 0.1478 | **0.9488** |
| test_s1 (1 cold) | 15,139 | **0.3673** | 0.5966 | 0.6996 | 0.0604 | **0.8637** |
| test_s2 (2 cold) | 1,887 | **0.2284** | 0.4128 | 0.5257 | 0.0410 | **0.6675** |

**Trend**: monotonic degradation s0 > s1 > s2 across every metric (expected
cold-start ranking).

## 5. vs hdn_ddi multi_cls (same seed, same data)

| metric | split | hdn_ddi | emergnn | who wins |
|---|---|---|---|---|
| top1_acc | s0 | 0.8044 | 0.5591 | hdn_ddi (+24.5pt) |
| top1_acc | s1 | 0.4330 | 0.3673 | hdn_ddi (+6.6pt) |
| top1_acc | s2 | 0.2109 | 0.2284 | **emergnn (+1.8pt)** |
| macro_auc | s0 | 0.9717 | 0.9488 | hdn_ddi (+2.3pt) |
| macro_auc | s1 | 0.8616 | 0.8637 | ≈ tie |
| macro_auc | s2 | 0.6775 | 0.6675 | hdn_ddi (+1.0pt) |
| macro_f1 | s0 | 0.6722 | 0.1478 | hdn_ddi (+52.4pt) |
| macro_f1 | s1 | 0.2648 | 0.0604 | hdn_ddi (+20.4pt) |
| macro_f1 | s2 | 0.0619 | 0.0410 | hdn_ddi (+2.1pt) |

**Reading**:
- On **warm-start** (s0): hdn_ddi crushes emergnn — hdn_ddi's mol-graph encoder
  excels when both drugs are seen (memorize pair-type association)
- On **1-cold** (s1): hdn_ddi still ahead but gap narrows
- On **2-cold** (s2): they're tied on top1, emergnn slightly worse on macro metrics
  — both models fail similarly when drugs are unseen

The emergnn macro_f1 numbers (0.15 / 0.06 / 0.04) are notably lower than hdn_ddi's
(0.67 / 0.26 / 0.06). Even on warm, emergnn doesn't dominate per-class F1, only
macro_auc (which is ranking-only). Suggests emergnn's predictions are smoother /
less peaked than hdn_ddi's.

## 6. vs binary (same seed, same data, sister run)

| split | binary AUC | MC macro_auc | gap |
|---|---|---|---|
| s0 (warm) | 0.5000 (bug) | 0.9488 | N/A (binary broken) |
| s1 (1 cold) | 0.4963 (bug) | 0.8637 | N/A (binary broken) |
| s2 (2 cold) | 0.7445 | 0.6675 | **binary > MC by 7.7pt** |

Note: binary's s0/s1 numbers are degenerate (see binary _results md for diagnosis).
Only s2 can be meaningfully compared. On s2, binary (0.74) > MC (0.67) — but apples
to oranges since binary metric is over pos+neg pairs while MC is over pos only.

## 7. Verdict + justification

**PASS-as-baseline** (no implementation bug; numbers reflect honest model
generalization on our cold-start setting).

Justification:
1. **Pipeline correctness**: best ckpt selection on `val_macro_f1` per fixed spec;
   training healthy (loss decreasing); all 3 splits produce reasonable outputs
2. **Trend correctness**: s0 > s1 > s2 monotonic
3. **Reasonable degradation**: warm-start has strong macro_auc 0.95 (memorization),
   2-cold drops to 0.67 (similar to hdn_ddi's 0.68) — both models struggle equally
   when drugs are truly novel
4. **Aligned with cross-baseline negative sampling change** (regenerate=True),
   though MC doesn't use negatives directly

## 8. Caveats

1. **Single seed (42)** — no std deviation
2. **n_classes=152 vs paper's 86** — our taxonomy is finer
3. **macro_f1 very low** across all splits (0.15 / 0.06 / 0.04) — model can rank
   but not precision/recall classify. Cold-start exposes this strongly.
4. **n_oov_excluded** silently drops 1+119+32 = 152 cold-start pairs from eval
5. Cross-baseline comparison (hdn_ddi vs emergnn) is fair only on s2; warm splits
   reflect very different inductive biases (mol vs KG)

## 9. Reproduce command

```bash
python Code/scripts/run_baseline.py \
    --baseline emergnn-mcc \
    --kg-source drugbank \
    --seed 42 \
    --epochs 100 \
    --tag emergnn_mc_seed42_aligned
```

(This run used legacy `--baseline emergnn_mc`. New canonical form per
keyword-rename today: `--baseline emergnn-mcc`. Same behaviour.)

Wall time on this machine: ~31 min for 100 epochs (faster than binary because no
shuffle_train negatives + smaller per-step compute).
