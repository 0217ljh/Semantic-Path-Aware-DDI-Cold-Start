# SumGNN unified wrappers — simplified acceptance (core-idea preservation)

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex, thread 019f1f88)
- **Trigger**: user "完成sumgnn" — finish the paused SumGNN port under the 2026-07-01
  SIMPLIFIED acceptance (codex confirms paper+code core idea preserved; NO paper-gate run).
- **Scope**: `multi_cls/baseline_unified.py` + `multi_label_cls/baseline_unified.py`,
  baseline-local ported `_core/**`, builder `_data/necessary/build_sumgnn_data.py`.

## Paper core idea (SumGNN, Bioinformatics'21)
Per-drug-pair ENCLOSING SUBGRAPH extraction from the KG + LAYER-INDEPENDENT self-attention
KG SUMMARIZATION (relation/BR attention, the key efficiency contribution) + Morgan
multi-channel features; CE over DDI-event types (DrugBank) / BCE multi-label (TWOSIDES/BioSNAP).

## Evidence already gathered
- Reproduction gate (SumGNN's OWN DrugBank data): BALLPARK PASS, ported core learns
  monotonically (acc 0.602 / κ 0.488 @6ep on 2% data). reproductions/SumGNN/_results/2026-07-01__drugbank_f1_gate.md.
- Unified smokes: mc predict (120,165), ml predict (120,200), both non-degenerate.
  baseline/sumgnn/_results/2026-07-01__unified_smokes.md.

## Codex verdict: APPROVED_WITH_NITS (core idea preserved)
Traced + confirmed EXERCISED end-to-end:
- Enclosing subgraph extraction: _core/subgraph_extraction/graph_sampler.py:193, invoked via
  datasets.py:18 + wrappers (mc:173 / ml:166).
- Layer-independent KG-summarization attention (paper key): shared A1/A2 created once
  rgcn_model.py:40, passed to every layer :77/:98; attention + threshold prune layers.py:157-171. NOT bypassed.
- Basis-decomposed relation message passing: layers.py:118/136/146, aggregators.py:13.
- Morgan multi-channel fusion: graph_classifier.py:25/58/71 (feats loaded mc:239 / ml:216).
- CE (DrugBank) / BCE (TWOSIDES): trainer.py:38/68, selected by wrapper dataset choice.
- Multiclass scatter (dense train vocab -> global via train_class_to_global.json) +
  _align_to_test unknown-drug handling: correct, row-stable (shuffle=False), no off-by-one.
- Caveats RESOLVED as correct: keeptrainone=False = paper-faithful TWOSIDES BCE
  (data_utils.py:144); max_id+1 = correct non-dense-id adaptation (data_utils.py:119/229).
- File independence: wrappers add only baseline-local _core to sys.path; no reproductions/ import.

## Nits (documented; do NOT block simplified acceptance; FIX BEFORE REAL RUNS)
1. **Medium — eval leakage**: both wrappers pass the TEST SubgraphDataset to the train-time
   Evaluator (mc:180/202, ml:173/187) instead of a separate valid (dev.txt) evaluator
   (original: reproductions/SumGNN/train.py:34/90). Currently COSMETIC — early_stop disabled,
   predict uses the final in-memory model (no best-ckpt reload) — so it does not affect the
   accepted port's behavior, but MUST be fixed to a dev-based evaluator before enabling
   early-stop / best-ckpt selection.
2. **Low — default config drift**: wrapper __init__ defaults n_epochs=100/batch_size=256/
   num_bases=10 vs original 50/128/4 (optimizer/lr/l2 match). Irrelevant to no-run acceptance
   but must be set to paper values for a faithful real run (batch_size especially — else lr
   drift per CLAUDE.md §4). NOTE: verify paper num_bases (gate doc said 10; codex read train.py
   default 4) before a real run.

## Decision: ACCEPTED (simplified gate) — core idea preserved
SumGNN multiclass + multilabel wrappers are accepted under the 2026-07-01 simplified gate.
Two nits recorded above are pre-real-run TODOs, not core-idea breaks.

## Update 2026-07-01 — BOTH nits FIXED (user "先把nit修复掉")
- Nit 1 (eval leakage): both wrappers now build a separate `valid` SubgraphDataset from
  `valid_pos` (dev) and wire `Trainer(train, valid_ev=VALID, valid_ev, test_ev=TEST)` — best
  checkpoint is selected on VALIDATION 'auc' (paper macro-F1 / PR-AUC), test is logging-only.
  Added `load_best_model_at_end`: reload `best_graph_classifier.pth` (map_location + weights_only
  =False for torch 2.7) after train, fallback to final model if no eval fired. multilabel also
  mkdir's `experiments/<exp>/` so a best-improvement's result.json write can't crash.
- Nit 2 (config drift): __init__ defaults restored to paper (train.py argparse): n_epochs 100->50,
  batch_size 256->128, num_bases 10->4; params.early_stop 100000->100. (gate doc's "num_bases=10"
  was wrong; verified train.py:179 default=4.)
- Re-smoke after fixes: mc predict (120,165) + ml predict (120,200) both PASS non-degenerate;
  valid dataset builds without error. No behavior regression.
