# EmerGNN multiclass — unified-benchmark wrapper review

- Date: 2026-06-30
- Primary reviewer: Claude (opus-4.8)
- Independent reviewer: codex (gpt-5-codex), thread 019f1a6c
- Trigger: finish EmerGNN on the unified benchmark (binary done; this is multiclass).

## Scope
- `baseline/emergnn/multi_cls/baseline_unified.py` — `EmerGNNUnifiedMulticlass(UnifiedBaseline)`,
  registered ("emergnn","multiclass"). Reuses core `EmerGNNMulticlassBaseline` UNCHANGED.
- Shared adapter `data_utils/leaf_adapter.py` task="multiclass" path (ddi_type=y_cls).
- Runner: registry keyed on (name,task); multiclass metric = accuracy + macro_f1 + oov_target_rate.

## Design
- One leaf = one regime → single EmerGNN_MC model, shuffle_train_mode = split_code, no routing.
- Core builds its dense class vocab from TRAIN uniques (auto-adjusts n_classes) — codex: correct,
  forcing K_global would add dead CE classes. predict scatters train-vocab probs back to the
  GLOBAL class axis (via core._idx_to_ddi_type) → runner argmaxes over global, compares to y_cls.
- OOV gold (class unseen-in-train, y_cls_train==-1) → counted WRONG (can't be predicted); reported
  via oov_target_rate.

## codex verdict (thread 019f1a6c): no blocker
- global-scatter + argmax fine for acc/macro_f1 (it's a score matrix, NOT a distribution over
  K_global; if probability metrics like log-loss/global-AUROC are added later, renormalize over
  train-seen classes or mark unsupported when any y_cls_train==-1).
- re-vocab to train classes is the right choice; keep _idx_to_ddi_type scatter.
- For multilabel (next): do NOT copy this OOV/re-vocab pattern — TWOSIDES is fixed-head BCE over
  global 200 labels; keep K=200; check one-row-per-pair dense multi-hot + paired pos/neg preserved
  + add per-label support metrics (n_zero_pos_labels_train, test_positive_on_unseen_label_rate).

## Verification
Smoke (ddi800 multiclass cold_s2 fold0, subsample train 1500 / test 400, 1 epoch, GPU, KG cache
reused): n_global=165; train had 82 distinct classes → core adjusted 165→82; predict shape
(400,165) ✓; non-degenerate (12 distinct predicted classes); macro_f1=0.009 acc=0.060 (1-ep tiny,
wiring only); test OOV=24/400, OOV predicted-correct=0 ✓. PASS (wiring).
Full run: PENDING (user).

## Reproduce (full)
python Code/scripts/run_baseline_unified.py --baseline emergnn --task multiclass \
    --dataset ddi_full --split cold_s2 --fold fold0 --epochs 100
