# KnowDDI Phase 1 (shared core + MULTICLASS) — from-scratch port acceptance

- **Date**: 2026-07-02
- **Primary reviewer**: Claude (opus-4.8) — verified core integrity + DGL migration on disk; spot-checked GSL/digraph vs official
- **Independent reviewer**: codex (gpt-5.2-codex; design 019f23fe? no — design read separate; port review + fix re-review thread 019f26c3)
- **Trigger**: user — add KnowDDI to the 10-baseline benchmark (NEED-PORT, no prior code/reproduction).
- **Scope**: NEW `Code/baseline/knowddi/` — shared core (model/, data_processor/, utils/, manager/,
  _data/necessary/build_knowddi_data.py) + `multi_cls/{baseline_unified.py,__init__.py}` + top `__init__.py`
  + `_MODULES` registration. Multilabel + binary = Phase 2 (TODO). Design spec produced by a read-only agent;
  port done across 2 agents (core landed after a mid-run process crash — core verified intact, then wrapper
  finished) + a fix agent for the 3 codex findings.

## Method (paper-cited)
KnowDDI (Wang, Yang, Yao, "Accurate and interpretable DDI prediction enabled by knowledge subgraph
learning", arXiv 2311.15056). GraIL/SumGNN-family enclosing-subgraph DDI + the paper's CRUX = a
Graph-Structure-Learning (GSL) block that learns per-edge connection strengths over a full-connect version
of each drug-pair subgraph and adds "resemble" edges between similar-but-unconnected node pairs, iteratively
alternating structure-estimation and embedding-refinement. Tasks: DrugBank multiclass (86-rel) + BioSNAP
multilabel. readme: "framework based on SumGNN".

## CRITICAL framing (user-required): borrow migration, not design
Every KnowDDI file ported from KnowDDI's OWN official code (Paper/Reference/Original-Code/KnowDDI/pytorch/).
Our SumGNN port used ONLY as a DGL-0.6->2.x migration reference. NO `import baseline.sumgnn` (verified clean).
Preserved KnowDDI's 3 divergences from SumGNN: (1) extract_r_digraph directional pruning; (2) GraphSAGE run
ONCE on the global graph (not per-subgraph R-GCN); (3) subgraph contract feeding the GSL block.

## Faithfulness (codex-confirmed line-by-line vs official)
- GSL block (model/gsl_model.py): MLP(exp(-|hu-hv|)‖rel_emb), lamda blend with original structure,
  edge_softmax, threshold sparsification, resemble edge id = num_rels+1, weighted update_all message-pass,
  per-layer concat repr — all preserved (official gsl_model.py:40+).
- extract_r_digraph (data_processor/datasets.py:147): directional tail-selection pruning matches official.
- GraphSAGE-once-on-global + head [mean_nodes(repr)‖head‖tail]->Linear(3*score_dim, num_rels)
  (model/Classifier_model.py) — faithful; num_nodes/pre_embed sized from ACTUAL entity count (not the
  35000 hardcode).
- DGL-0.6->2.x migration: codex found NO silent model-logic simplification; dgl.graph/from_networkx,
  dgl.node_subgraph + NID/EID relabel, edge_softmax, batched full-connect graph, update_all UDF all
  semantically preserved.
- Hyperparams LOCKED to DrugBank official LOG (experiments/Drugbank/log_train.txt): num_dig_layers=4,
  num_infer_layers=3, lamda=0.5, threshold=0.05, hop=2, emb_dim=32, num_gcn_layers=2, MLP_hidden_dim=32,
  lr=0.005, weight_decay=1e-5, lr_decay=0.93, batch_size=256, num_epochs=50, early_stop=10,
  eval_every_iter=526, num_workers=32.

## Wrapper (multi_cls/baseline_unified.py)
KnowDDIUnifiedMulticlass (register_unified("knowddi"), task="multiclass"), kg_scope="full" default
(standing decision) merged KG. Mirrors the SumGNN mc wrapper SHAPE (copied, not imported): _leaf_hash cache,
_ensure_data detect+build via build_knowddi_data, chdir/symlink sandbox for LMDB, best-ckpt reload,
head = train-observed ddi_type vocab (K_train), predict scatters to global class axis, _align_to_test row
alignment; OOV-in-train gold counted wrong (runner oov_target_rate). CrossEntropyLoss, best-ckpt VAL macro-F1.

## codex verdict: CHANGES_REQUIRED -> 3 fixes -> APPROVED_WITH_NITS (re-review 019f26c3)
Round 1 (CHANGES_REQUIRED): (1) val OOV gold silently remapped to dense class 0 (false credit, corrupt
best-ckpt); (2) eval/LR-scheduler cadence drifted to per-epoch (official = per eval_every_iter updates);
(3) num_workers 8 vs log 32.
Fixes (re-review APPROVED_WITH_NITS): (1) OOV valid/test gold -> sentinel = K_train (unpredictable; head
width stays K_train; evaluator counts wrong on plain lists); to stop the sentinel inflating the head,
utils/data_utils.py made ONLY train register DDI relations — codex confirmed faithful-equivalent (checked
official drugbank data files: no valid/test relation outside train's 0..85; official adjacency is train-only;
sentinel r_label not used to index adjacency -> no subgraph corruption). (2) trainer.py restored per-iter
eval + scheduler.step per-eval + mid-epoch early-stop break, matching official trainer.py:103/:127-133;
eval_every_iter=526 (log + argparse agree). (3) num_workers=32.
- **NIT (accepted, non-blocking)**: landed keeps an epoch-level early-stop halt (trainer.py:191-194) while
  official leaves the epoch-level break commented out (official trainer.py:160-162). The critical mid-epoch
  eval/scheduler/early-stop semantics ARE aligned; the halt is arguably more sensible. Documented, not churned.

## Verification (no training)
- py_compile all files OK; import+registry smoke: get_unified("knowddi","multiclass") -> KnowDDIUnifiedMulticlass.
- Offline plumbing (synthetic, no training, no full-KG extraction): builder -> dense vocab + sentinel;
  enclosing subgraph -> LMDB; extract_r_digraph applied; one forward GraphSAGE->GSL->classifier -> logits
  (n, K_train); scatter to global axis; sentinel row per-class F1 = 0.0 (counted wrong). num_rels not inflated.
- (No training / no full-KG extraction: GPU busy; deferred. Full-KG subgraph extraction cost + GSL O(n^2)
  densification memory are the two run-time levers to settle before the first real run — same as SumGNN.)

## Cold-start caveat (property, not bug)
KnowDDI is transductive-leaning (GraphSAGE learns a per-node embedding table; unseen drugs lack a learned
embedding; S2 drug-pairs often have no h->t path -> empty digraph). Expect weak S1/S2 numbers — a property
to MEASURE, not a defect.

## Decision: Phase 1 ACCEPTED (simplified gate) — GSL core + multiclass faithful.
Run cmd (when GPU free): python Code/scripts/run_baseline_unified.py --baseline knowddi --task multiclass
  --dataset ddi_full --split cold_s2 --fold fold0 --epochs 50
Next: Phase 2 = multilabel (native BioSNAP->TWOSIDES) + binary (Case-B), reusing this core.
