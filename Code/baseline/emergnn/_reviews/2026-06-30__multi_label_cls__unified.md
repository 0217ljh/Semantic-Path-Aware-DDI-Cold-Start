# EmerGNN multilabel (TWOSIDES) — unified-benchmark port review

- Date: 2026-06-30
- Primary reviewer: Claude (opus-4.8) + porting subagent
- Independent reviewer: codex (gpt-5-codex), threads 019f1b12 (round 1) + 019f1b1c (re-review)
- Trigger: finish EmerGNN on the unified benchmark (binary + multiclass done; this is multilabel).

## Scope (ported, independent COPY+adapt of the TWOSIDES reproduction — no import of reproductions/)
- `Code/baseline/emergnn/multi_label_cls/model.py` — `EmerGNN_ML`: fixed 200-output sigmoid head
  (Wr=Linear(2*n_dim, eval_rel=200)), bidirectional L-layer message passing (chunked
  index_select+scatter_add, same primitive codex-accepted for binary/multiclass since env lacks
  torchdrug generalized_rspmm).
- `Code/baseline/emergnn/multi_label_cls/_core_twoside.py` — ports load_data.py + base_model.py:
  cumulative KG (train_kg / valid_kg=train+valid / test_kg=train+valid+test), kg_entity_set over
  all 3 KG files, shuffle_train (S0/S1/S2), paired pos/neg BCE, Adam+ReduceLROnPlateau, best ckpt
  on valid PR-AUC. Fold→reproduction-dir via S2/fold0→S2_1, fold1→S2_12, ... (verified by exact
  drug-pool match, all 5 folds × both regimes).
- `Code/baseline/emergnn/multi_label_cls/baseline_unified.py` — `EmerGNNUnifiedMultilabel`,
  register_unified("emergnn"), task="multilabel", no routing, shuffle_train_mode=split_code,
  per-regime hyperparams. predict→(n,200) sigmoid.
- `Code/data_utils/leaf_adapter.py` — make_multilabel_bundle + _dense_multihot (one dense
  200-multihot per pair from y_label_ids; paired pos/neg via <split>_pair_links.parquet; neg
  carries the pos multihot, matching upstream). binary/multiclass paths untouched.
- Runner: ("emergnn","multilabel") + _multilabel_metrics (macro AUROC/AUPRC over labels with both
  classes in the paired test + n_zero_pos_labels_train + test_positive_on_unseen_label_rate).

## codex review (round 1, thread 019f1b12): 6 deviations, 5 accepted, 1 must-fix
- ACCEPTED: (1) rspmm→scatter_add primitive substitution; (2) fold→dir mapping (verified by pool
  match); (3) vectorized _build_edges order (sum-aggregation is order-invariant); (5) fixed paired
  negatives (matches base_model.py); (6) smoke-only is a verification gap not a faithfulness bug.
- MUST-FIX (#4): the port used only train_KG for BOTH validation and test predict. Upstream builds
  valid_kg=train+valid, test_kg=train+valid+test, uses vKG for validation + tKG for test, and
  computes ddi_in_kg against the union of all 3 KG files.

## Fix applied + codex re-review (thread 019f1b1c): APPROVED
Core now carries three cumulative KG payloads; ddi_in_kg over the full union; vKG (train facts +
valid_kg) for validation model-selection; tKG (train+val facts + test_kg) for predict_proba (stored
on the instance). Independent smoke re-run (twoside multilabel cold_s2 fold0, 300-pair subsample,
1 epoch, real KG): built-graph edges train=3,380,096 < valid=3,394,154 < test=3,419,100 (strictly
cumulative, deltas match valid/test KG line counts); validation on vKG (val_pr≈0.68); predict on
tKG → (60,200), non-degenerate (std 0.118); multihot exact; metrics compute. PASS.

codex re-review verdict (verbatim): "Final verdict: approved. On this re-review, I do not see a
remaining faithfulness bug in the EmerGNN multilabel TWOSIDES port, so EmerGNN binary + multiclass
+ multilabel can be considered complete."

## Status: EmerGNN COMPLETE (binary + multiclass + multilabel), all codex-approved. Full-scale runs
(all pairs, 100 epochs) PENDING (user runs). Reproduce (full):
  python Code/scripts/run_baseline_unified.py --baseline emergnn --task multilabel \
      --dataset twosides --split cold_s2 --fold fold0 --epochs 100
