# SumGNN binary (DDI-existence) — simplified acceptance (Case-B task backfill)

- **Date**: 2026-07-02
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex; design 019f20e9, review 019f20f8)
- **Trigger**: user — all baselines need all 3 tasks; backfill SumGNN's missing binary.
- **Scope**: NEW `binary_cls/{_binary_core.py, baseline_unified.py, __init__.py}` +
  `_data/necessary/build_sumgnn_binary_data.py` + additive runner registration. `_core/**` UNTOUCHED.

## Task framing (Case B, GraIL-native)
SumGNN paper = DrugBank multiclass + TWOSIDES multilabel; binary DDI-existence is a Case-B
adaptation. Its base method GraIL IS binary inductive link prediction, so binary is natural:
reuse SumGNN's subgraph-summarization encoder UNCHANGED, only head train_rels=1 + BCE + score
pos/neg subgraphs.

## Design (codex Option A) + implementation
- `_binary_core.py`: REUSES from _core (imports, never edits) — GraphClassifier, RGCN,
  SubgraphDataset (subclassed), extraction/labeling helpers, process_files_ddi. NEW binary-only:
  generate_binary_subgraph_datasets/_links2subgraphs_binary (g_label=1 pos / 0 neg), BinarySubgraph
  Dataset (reads BOTH pos+neg LMDBs), BinaryTrainer (BCEWithLogitsLoss, per-epoch loss log),
  BinaryEvaluator (AUPRC/AUROC).
- Wrapper mirrors mc: task="binary", register_unified("sumgnn"), full-KG default + materialize,
  predict->(n,) sigmoid aligned to test rows via test_manifest.
- Head train_rels=1; num_neg_samples_per_link=0 (no extra negs beyond the leaf's set).

## 5 faithfulness traps — all AVOIDED (codex-verified)
A. negatives NEVER in train.txt/adjacency (train.txt = y_bin==1 only; process_files_ddi adjacency
   from train.txt only); negatives -> separate *_neg.txt query files.
B. binary extraction sets g_label=1 pos / 0 neg (not _core's all-ones).
C. BinarySubgraphDataset reads BOTH LMDBs (len = pos+neg), not _core's pos-only yield.
D. BinaryTrainer BCEWithLogitsLoss on 1-logit vs binary g_label, checkpoint by val AUPRC.
E. num_neg_samples_per_link=0.
Plus: predict (n,) sigmoid aligned to full test row order (known_mask/is_pos manifest), dropped
-> 0.5; no row scramble / double sigmoid.

## Verification
- import + registry smoke PASS: get_unified("sumgnn","binary") -> SumGNNUnifiedBinary; builder
  pos/neg separation + manifest alignment verified on synthetic input.
- (No training run: GPU busy w/ EmerGNN; deferred-run.)

## Codex verdict: APPROVED_WITH_NITS
Core preserved (only train_rels=1 changes the model; R-GCN/attention/fusion via _core intact);
all 5 traps avoided; predict alignment correct; file-independent.
- **NIT (latent, non-triggering on real data)**: build_sumgnn_binary_data writes positive-only
  train/dev/test.txt into _core process_files_ddi's bare np.loadtxt; a split with EXACTLY 1
  positive row would be shape-fragile. Real binary leaves (ddi800/ddi_full) have thousands of
  positives per split -> never triggers. Fix (atleast_2d guard) is a real-run TODO if a 1-row
  positive split ever occurs.

## Decision: ACCEPTED (simplified gate) — core idea preserved
SumGNN now has all 3 tasks (binary + multiclass + multilabel).
