# KnowDDI Phase 2 (MULTILABEL + BINARY) — task-wrapper acceptance

- **Date**: 2026-07-02
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex; design 019f2930/019f2938, acceptance review 019f293f)
- **Trigger**: user — complete KnowDDI's 3-task mandate after Phase-1 (core + multiclass).
- **Scope**: NEW `multi_label_cls/{baseline_unified.py,__init__.py}`, `binary_cls/{_binary_core.py,
  baseline_unified.py,__init__.py}`, `_data/necessary/{build_knowddi_twoside.py,build_knowddi_binary_data.py}`
  + additive runner registration + top __init__ re-export. Phase-1 core UNEDITED.

## Task framing
KnowDDI paper = DrugBank multiclass + BioSNAP multilabel. multilabel (BioSNAP→our TWOSIDES) is NATIVE
(the core already ships the BioSNAP branch); binary (DDI-existence) is Case-B (GraIL-native — the base
method IS binary inductive link prediction). Both REUSE the Phase-1 GSL/GraphSAGE/subgraph core UNCHANGED
and swap only head/loss/label/metric.

## MULTILABEL (native BioSNAP branch, mostly a wrapper)
- Head = fixed L=200 via params.num_rels forced to n_labels (baseline_unified.py:246-251); process_files_
  decagon otherwise learns rel from all splits (data_utils.py:143-175), so the override is necessary +
  present. predict = direct sigmoid (n,200), NO scatter (head axis == label axis, :282-322).
- Loss = core's BioSNAP masked BCE + polarity: nn.BCELoss(reduce=False), sigmoid(scores),
  target=multihot*polarity, then mask *= multihot (trainer.py:77-78,110-123) — matches official
  trainer.py:45-46,77-88. Neg positions masked out; one sigmoid at inference.
- Best-ckpt = VAL ROC-AUC (Evaluator_multilabel 'auc', evaluator.py:114-118; trainer best-ckpt :140-155).
- KG = leaf's OWN TWOSIDES KG (resources.kg.source train_KG.txt, int namespace disjoint from DBxxxxx),
  NOT the merged DrugBank KG (baseline_unified.py:131-149; builder writes BKG_file.txt h t r ints).
- build_knowddi_twoside.py emits decagon `h t multihot polarity` faithfully (:103-125); OOV-drug rows
  skipped + counted. Hyperparams locked to BioSNAP log (eval_every_iter=452, threshold=0.1, lamda=0.5,
  num_infer_layers=1, num_dig_layers=3, gsl_rel_emb_dim=24, MLP_hidden_dim=24, MLP_num_layers=3, dropout=0.2).

## BINARY (Case-B, GraIL-native)
- Adjacency from train POSITIVES only (build_knowddi_binary_data.py:117-125 writes y_bin==1;
  process_files_ddi builds DDI adjacency from triplets['train'] only, data_utils.py:46-78). Negatives are
  extraction QUERIES only via separate *_neg.txt (_binary_core.py:135-147) — NEVER in the adjacency.
- Core REUSED UNCHANGED: BinaryKnowDDISubgraphDataset subclasses SubgraphDataset + calls inherited
  _prepare_subgraphs (_binary_core.py:218-276) -> inherited directional extract_r_digraph
  (datasets.py:116-182) -> same Classifier_model/GraphSAGE/GSL. ONLY change = params.num_rels=1 ->
  Linear(...,1) (baseline_unified.py:227-238, Classifier_model.py:43-45).
- Loss = BCEWithLogitsLoss on subgraph g_label (1 pos/0 neg) (_binary_core.py:337,369-376);
  BinaryEvaluator sigmoid-once + AUPRC/AUROC (:295-314); best-ckpt VAL AUPRC (:392-405). predict (n,)
  aligned via test_manifest.json [pos,neg] order, dropped/OOV -> 0.5 (baseline_unified.py:262-309). No
  double-sigmoid. Negatives = leaf's FIXED materialized y_bin==0 (mirrors SumGNN binary). Full-KG default.
- LMDB env: binary needs 6 named dbs but core _open_env uses max_dbs=3; fix pre-seeds
  datasets._ENV_CACHE[db_path] with a max_dbs=6 env on a DEDICATED binary db_path (`..._bin`,
  baseline_unified.py:196-199) before super().__init__ (codex Opt-C 019f2938). No collision with mc/ml
  cache (different db_path key).

## codex verdict: APPROVED_WITH_NITS (acceptance review 019f293f)
All 11 axes confirmed: multilabel fixed-200 head + masked-BCE+polarity + ROC-AUC + leaf-own KG + decagon
format; binary GraIL adjacency-from-positives + g_label + core-reused-unchanged + BCEWithLogits + AUPRC +
manifest alignment + no double-sigmoid; LMDB env non-colliding; Phase-1 core UNEDITED (the tiny-synthetic
map_size underflow is a PRE-EXISTING core/official artifact, not introduced here — only bites tiny synthetic
data); no baseline.sumgnn import; TrainProgress present.
- **NIT 1 (docstring) — FIXED**: _binary_core.py:79-93 _get_or_install_binary_env docstring claimed a
  fail-fast compatibility check it didn't perform. Rewritten to accurately describe the dedicated-path,
  always-max_dbs=6, reuse-if-cached behavior (logic unchanged; safe because only this fn seeds that path).
- **NIT 2 (accepted, non-blocking)**: multilabel has no explicit assert that train.aug_num_rels is large
  enough for the GSL resemble edge id num_rels+1 (baseline_unified.py:250-251, gsl_model.py:241-243).
  Implicit invariant; holds on the real TWOSIDES KG. Documented, not guarded.

## Verification (no training)
- py_compile all new files OK; import+registry smoke: get_unified("knowddi","multilabel") +
  ("knowddi","binary") resolve; multiclass still resolves (no regression).
- Offline plumbing (agent, no training, tiny slice): ML build emits decagon + BKG + OOV-drop; BIN emits
  train-POS-only + pos/neg query + test_manifest; tiny CPU forward BIN (3,) sigmoid in [0.477,0.551],
  ML (3,6) sigmoid in [0.362,0.578]. (No training / full-KG extraction: GPU busy; deferred.)

## Decision: ACCEPTED (simplified gate).
KnowDDI now has ALL 3 tasks (binary + multiclass + multilabel), all reusing the faithful GSL core.
Cold-start S2 expected weak (transductive subgraph method) — property to measure, not a defect.
