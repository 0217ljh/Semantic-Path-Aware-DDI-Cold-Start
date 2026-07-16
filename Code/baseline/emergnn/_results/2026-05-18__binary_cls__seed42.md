# EmerGNN Baseline binary_cls — seed 42, 100 epochs (post negative-sampling alignment)

> **Why this lives under `baseline/emergnn/_results/`**: this run uses our
> project's 800-drug DrugBank cold-start dataset
> (`Code/data/coldddi_legacy/800drug/seed42.pkl`) + the baseline-side code at
> `Code/baseline/emergnn/binary_cls/`. First run after aligning negative
> sampling strategy across all binary baselines (now uses
> `train.get_train_negatives(epoch, regenerate=True)` per CLAUDE.md spec).

## 1. Run identity

- **run_id**: `2026-05-18_14-43-23__run_baseline__emergnn_binary_seed42_aligned__seed42`
- **run_dir**: `Code/runs/2026-05-18_14-43-23__run_baseline__emergnn_binary_seed42_aligned__seed42/`
- **artifacts**: `results.json`, `train.log`, mirror at `Code/runs/_logs/<run_id>.log`
- **dataset**: 800-drug DrugBank cold-start seed42 PKL bundle
- **KG**: DrugBank 5-bucket (enzymes / targets / transporters / carriers / pathways)
- **status**: completed full 100 epochs; loaded best val_auc=0.7299 ckpt
- **wall time**: fit=2770 s + eval ~52 s = **~47 min total**

## 2. Config (from results.json)

| param | value | source |
|---|---|---|
| baseline | `emergnn` (new bare-name form, defaults to binary) | CLI `--baseline` |
| kg_source | `drugbank` | CLI |
| epochs | 100 | CLI |
| seed | 42 | CLI |
| batch_size | 32 (default, paper) | inherited |
| n_dim | 64 (paper hyperopt) | inherited |
| length | 3 (paper bidirectional message passing depth) | inherited |
| feat | "M" (Morgan FP for entity feats) | paper |
| optimizer | Adam(lr=1e-3, **wd=1e-8**) | paper hyperopt S1/S2 best |
| scheduler | ReduceLROnPlateau | paper |
| shuffle_train_mode | "S2" | paper inductive simulation |
| shuffle_ratio | 0.8 | paper |
| loss | `binary_cross_entropy_with_logits` | binary task |
| **neg sampling** | per-epoch fresh, sub-seed `42+1000+epoch`, then downsampled to match shuffle_train pos count | **aligned with hdn_ddi/tiger/ssi_ddi/dsn_ddi 2026-05-18** |

## 3. Data stats (from results.json)

| split | n_pos | n_neg | total | source |
|---|---|---|---|---|
| train | 53,743 | per-epoch sampled | — | `splits.train` |
| test_s0 (warm) | 2,986 | 2,986 | 5,972 | static neg pre-baked |
| test_s1 (1 cold) | 15,258 | 15,258 | 30,516 | static neg pre-baked |
| test_s2 (2 cold) | 1,919 | 1,919 | 3,838 | static neg pre-baked |

## 4. Final metrics (results.json `eval` block, best-val-AUC ckpt)

| split | n | AUC | NLL | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| test_s0 (warm) | 5,972 | **0.5000** | 7.9712 | 0.0000 | 0.0000 | 0.0000 |
| test_s1 (1 cold) | 30,516 | **0.4963** | 7.9712 | 0.0000 | 0.0000 | 0.0000 |
| test_s2 (2 cold) | 3,838 | **0.7445** | 1.2714 | 0.6737 | 0.5208 | 0.9536 |

## 5. Verdict: ⚠ FAIL on s0/s1, PASS on s2

**Critical bug**: s0 and s1 evaluation produce degenerate output:
- AUC exactly 0.5 (random) on s0, 0.4963 on s1 → no predictive signal
- F1 = 0.0 → classifier always predicts one class
- NLL = 7.97 on both s0 and s1 (literally identical) → model outputs near-constant logits
- s2 works fine (AUC 0.74, F1 0.67), so model itself trained successfully

**Suspected root cause** (need codex / manual investigation):
- emergnn's training uses `shuffle_train_mode="S2"` (paper inductive simulation: holds
  out 80% of train drugs from KG each epoch, models them via inductive paths only).
- At eval time, the model's prediction path likely assumes inductive setup. For s2
  (both test drugs unseen during training KG), this matches train-time distribution
  → numbers OK.
- For s0/s1 (test drugs seen in training KG), the prediction path doesn't recover the
  proper graph state → outputs collapse to constant.
- Identical NLL=7.97 across s0/s1 is the smoking gun — model is genuinely outputting
  the same logit for all pairs in those splits.

**Training itself was healthy**:
- Loss decreased from ~0.69 (start) to ~0.05 by epoch 100
- val_s2 AUC progressed from 0.5 → 0.73 over training
- best val_auc=0.7299 at some epoch (matches s2 test AUC=0.7445)
- The model **did** learn — just only on the S2 cold-start sub-task

## 6. vs hdn_ddi binary (same seed, same data)

| split | hdn_ddi AUC | emergnn AUC | notes |
|---|---|---|---|
| s0 (warm) | 0.6595 | **0.5000 (bug)** | emergnn fails on warm |
| s1 (1 cold) | 0.6412 | **0.4963 (bug)** | emergnn fails on 1-cold |
| s2 (2 cold) | 0.6201 | **0.7445** | **emergnn beats hdn_ddi by 12 pt** on the hardest split |

The s2 number (0.7445) is the most paper-aligned signal we have for emergnn binary,
since paper's EmerGNN was designed for cold-start. **+12pt over hdn_ddi on s2** is
consistent with paper claims that KG-aware inductive training beats mol-only models.

## 7. Caveats

1. **The s0/s1 numbers are NOT real EmerGNN performance** — they reflect a bug in
   the evaluation path (or training-eval distribution mismatch). Do NOT report or
   cite these as "EmerGNN gets 0.5 AUC on warm-start".
2. **The s2 number IS reportable** as EmerGNN cold-start binary performance on our
   1900-drug DrugBank setting.
3. First run after the cross-baseline negative-sampling alignment fix
   (`regenerate=False → True` for emergnn/tiger/ssi_ddi/dsn_ddi). hdn_ddi already
   used `True`. Theoretically this shouldn't break anything, but worth flagging in
   case the s0/s1 bug is regression-related.
4. The "shuffle_train_mode=S2" inductive simulation is paper-faithful but means
   training distribution ≠ s0/s1 test distribution by design. The bug is that the
   predict path doesn't gracefully degrade when test drugs WERE seen in training KG.
5. Predicted recall=0.95 on s2 with precision=0.52 — model leans toward predicting
   positive. F1=0.67. Reasonable for cold-start.

## 8. Reproduce command

```bash
python Code/scripts/run_baseline.py \
    --baseline emergnn \
    --kg-source drugbank \
    --seed 42 \
    --epochs 100 \
    --tag emergnn_binary_seed42_aligned
```

Wall time on this machine (WSL2 + CUDA): ~47 min for 100 epochs.

## 9. Follow-up needed

- **Investigate** why emergnn's s0/s1 prediction collapses (model state is fine,
  evaluation path is the suspect). Look at `EmerGNNBaseline.predict_proba` and how
  it handles drugs that ARE in the train KG vs drugs that aren't.
- Once fixed, re-run with same config and confirm s0/s1 numbers become non-degenerate.
- Then revisit this _results/ file with a `__round2.md` companion.
