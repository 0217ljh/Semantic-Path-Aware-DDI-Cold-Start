# Baseline migration to unified benchmark — status tracker

Goal: wire all 10 baselines to the unified per-leaf interface (decision B), smoke-test each
task, verify task settings (transductive/inductive, binary negatives, multiclass classes,
multilabel labels). Design + results to codex each step.

## Shared infrastructure (DONE, codex-approved)
- `Code/data_utils/unified_loader.py` — load_leaf -> Leaf{train,val,test, resources}. Tested 135 fold-loads.
- `Code/baseline/unified_base.py` — `UnifiedBaseline` ABC: fit(train_df,val_df,*,resources)+predict(test_df); fit_leaf(leaf); register_unified.
- `Code/data_utils/leaf_adapter.py` — SHARED leaf->legacy-PairDataset adapter, task-aware (binary/multiclass/multilabel). All legacy cores reuse it.
- `Code/scripts/run_baseline_unified.py` — leaf runner (+ score dispersion metric).
- codex notes: keep adapter as minimal shim + per-baseline conformance (fail-hard on missing reads). Multiclass: feed y_cls (global), core re-vocabs from train; OOV-in-train eval rows = WRONG not dropped (report oov_target_rate).

## Applicability (modality -> tasks). KG-based need a KG leaf (drugbank_latest_*, twoside); molecular/text apply broadly.
| # | baseline | modality | tasks | core status |
|---|---|---|---|---|
| 1 | EmerGNN | KG | binary✅ / multi / multilabel | core in baseline/emergnn (binary+multi) |
| 2 | SumGNN | KG | multi / multilabel | NEED PORT (Original-Code only) |
| 3 | KnowDDI | KG | **ALL 3 ✅** | **DONE 2026-07-02**: from-scratch port, GSL core + multiclass (Phase1) + multilabel+binary (Phase2). codex APPROVED_WITH_NITS ×2 |
| 4 | HDN-DDI | molecular | binary / multi | core in baseline/hdn_ddi (binary+multi) |
| 5 | TIGER | subgraph+mol | binary / multi | core in baseline/tiger (binary+multi) |
| 6 | MKG-FENN | KG+mol | multi | core in baseline/mkg_fenn |
| 7 | SSI-DDI | molecular | binary✅ / multi | core in baseline/ssi_ddi (binary) |
| 8 | TextDDI | text | binary/multi | NEED PORT |
| 9 | MRCGNN | molecular | multi | NEED PORT |
| 10| DDIPrompt | KG prompt | multi | NO official code |

## Done
- [x] EmerGNN binary — full wrapper + end-to-end run (ddi800 cold_s2, AUROC 0.67@3ep), codex PASS. Review: baseline/emergnn/_reviews/2026-06-30__binary_unified.md
- [x] EmerGNN multiclass — wrapper + GPU smoke (predict (n,165), OOV=wrong, KG cache reused), codex PASS. Review: .../2026-06-30__multi_cls__unified.md. MULTICLASS TEMPLATE established (adapter ddi_type=y_cls, core re-vocabs from train, predict scatters to global axis, runner oov_target_rate).
- [x] SSI-DDI binary — wrapper + CPU smoke (AUROC 0.558@1ep tiny), codex PASS. MOLECULAR TEMPLATE.
- [x] **HDN-DDI binary + multiclass** (2026-07-01, SIMPLIFIED acceptance) — unified wrappers
  existed; import+registry smoke OK; codex APPROVED_WITH_NITS (core idea = BRICS 3-level
  hierarchical mol graph + y==1 bipartite + RESCAL preserved end-to-end; nit = frozen
  benchmark negatives, intentional+uniform). NO paper-gate run (simplified gate). Review:
  baseline/hdn_ddi/_reviews/2026-07-01__unified_wrappers__simplified_accept.md.
- Registry now keyed on (name, task) so a baseline can have one class per task.

## SIMPLIFIED ACCEPTANCE (user directive 2026-07-01)
Remaining baselines: acceptance = codex confirms port preserves paper+code CORE IDEA;
NO paper-gate run / numeric result required. Still do: review+codex, import/shape smoke,
tracker update. See baseline/_ACCEPTANCE.md §2 amendment. **Report to user after EACH
baseline acceptance, then pause.**

## Per-epoch train negatives restored (2026-07-01, design B, codex APPROVED)
Unified binary train negatives are now DETERMINISTIC PER-EPOCH (not frozen): adapter
`get_train_negatives(epoch, regenerate=True)` -> reconstructable draw via new
`data_utils/leaf_negatives.py` (reuses build_train_negatives; blake2b leaf seed; excludes
global positives + fixed val/test negs; g1 pool = no cold-start leakage). regenerate=False
still returns the fixed materialized anchor. Resolves the HDN-DDI "frozen negatives" nit for
ALL binary baselines. Detail: Notes/Log/per_epoch_train_negatives.md.

## KG-scope note (2026-07-01, blocks nothing here)
Full merged KG in EmerGNN is 600-1200x slower (8-17h/epoch) + torchdrug rspmm kernel
won't compile on torch 2.7. EMERGNN_KG_SCOPE=full switch added (codex OK, review
emergnn/_reviews/2026-07-01__full_kg_scope_switch.md). KG-scope decision (drug_incident
vs full vs k-hop) STILL OPEN but DECOUPLED from porting (ports faithful regardless of
runtime KG scope). See memory project_ddi_fullkg_infeasible.md.

- [x] **SumGNN multiclass + multilabel** (2026-07-01, SIMPLIFIED acceptance) — from-scratch
  port (baseline/sumgnn/_core, DGL-2.4). Reproduction gate BALLPARK PASS (learns on SumGNN's
  own DrugBank data). Unified smokes PASS (mc predict (n,165), ml predict (n,200)). codex
  APPROVED_WITH_NITS (core preserved: enclosing subgraph + layer-independent KG-summarization
  attention + Morgan multi-channel + CE/BCE all traced+exercised; 2 flagged caveats judged
  CORRECT). 2 nits BOTH FIXED 2026-07-01 (valid-dev evaluator + best-ckpt reload; defaults ->
  paper 50/128/4 + early_stop 100; re-smoke PASS). Original nits were: (1) eval wired to test not dev
  (cosmetic now, no best-ckpt reload); (2) __init__ default hyperparam drift (100/256/10 vs
  paper 50/128/4). Review: baseline/sumgnn/_reviews/2026-07-01__unified_wrappers__simplified_accept.md.

## SumGNN now DEFAULTS to FULL merged KG (2026-07-01, codex APPROVED_WITH_NITS)
build_sumgnn_data gained `build_entity_index_full` + `kg_scope="full"` default (verified
174,976 ent / 59 rel / 7.1M triplets vs drug_incident 15412/18/128042); multi_cls wrapper
kg_scope="full", scope in cache dir (mc__<hash>__<scope>). drug_incident kept selectable.
multilabel/TWOSIDES uses its own KG (unaffected). NOTE: full-KG SumGNN is CORRECT but SLOW
(~13.6h extraction unoptimized, measured) — acceleration deferred (materialize + incidence-
once + BFS per-drug-cache; profiling in Notes/Log; memory project_ddi_fullkg_infeasible).
SSI-DDI is FULLY done (bin + mc, mc reviewed 2026-07-01) — earlier "SSI-DDI multi pending"
was wrong. EmerGNN default still drug_incident (has EMERGNN_KG_SCOPE=full opt-in) — flipping
to full-default needs backend=rspmm default + rspmm mc/ml cores confirmed complete.

## ALL-3-TASKS MANDATE (user 2026-07-01): every baseline must have binary+multiclass+multilabel
Even if not all used — must exist so they're ready when needed. Non-paper tasks = Case-B
adaptations (binary=BCE+negs; multilabel=TWOSIDES 200-label BCE head). Current 3-task matrix:
| baseline | binary | multiclass | multilabel |
|---|---|---|---|
| EmerGNN | done | done | done |
| HDN-DDI | done | done | **done 2026-07-01** (multi_label_cls, codex APPROVED_WITH_NITS; masked-BCE, fixed-200 RESCAL sigmoid, macro-AUPRC; review _reviews/2026-07-01__multilabel__simplified_accept.md) |
| SSI-DDI | done | done | **done 2026-07-01** (multi_label_cls, codex APPROVED; reuses GAT+co-attn+RESCAL forward_all, masked-BCE fixed-200 sigmoid, macro-AUPRC; review _reviews/2026-07-01__multilabel__simplified_accept.md) |
| SumGNN | **done 2026-07-02** (binary_cls, GraIL-style, codex APPROVED_WITH_NITS; reuses subgraph encoder, train_rels=1 BCE, adjacency-from-pos-only, sep pos/neg query files, binary dataset reads both LMDBs, AUPRC; review _reviews/2026-07-02__binary__simplified_accept.md) | done | done |
| TIGER | done | done | **done 2026-07-02** (multi_label_cls, codex APPROVED_WITH_NITS; dual-channel core preserved via ADDITIVE model.forward_logits (raw logits, existing softmax forward UNTOUCHED); fixed-200 sigmoid + masked-BCE + macro-AUPRC; TIGER-id remap (DrugBank id else synthetic twoside:<id>, ~45% unmapped → isolated self-loop BKG, user-approved data limitation); full merged KG; review _reviews/2026-07-02__multilabel__simplified_accept.md) |
| MRCGNN | registered? | done | registered? | (runner _MODULES now has mrcgnn binary+multiclass+multilabel — done by user/parallel; NOT reviewed/accepted by me — verify)
| KnowDDI | **done 2026-07-02 Phase 2** (Case-B GraIL-native: adjacency-from-train-positives, subgraph g_label 1/0, single-logit BCEWithLogits, val-AUPRC, test_manifest align; GSL/digraph/GraphSAGE core reused UNCHANGED; codex APPROVED_WITH_NITS 019f293f) | **done 2026-07-02 Phase 1** (from-scratch port; GSL knowledge-subgraph-learning core + enclosing subgraph + extract_r_digraph directional pruning + GraphSAGE-once; DGL-0.6→2.x migrated from KnowDDI's OWN code (SumGNN only a migration ref, no import); full-KG default; hyperparams locked to DrugBank log; codex CHANGES_REQUIRED→3 fixes→APPROVED_WITH_NITS thread 019f26c3; review _reviews/2026-07-02__phase1_core_multiclass__from_scratch_port.md; spec Notes/Log — cold-start S2 expected weak (transductive method)) | **done 2026-07-02 Phase 2** (native BioSNAP branch→our TWOSIDES: fixed-200 sigmoid + masked BCE+polarity, val-ROC-AUC, leaf-own KG not merged; codex APPROVED_WITH_NITS 019f293f) |
| MKG-FENN | **done 2026-07-02** (binary_cls, Case-B, codex APPROVED; regime-aware reuse of the multiclass cold infra: S0=4ch / S1S2=3ch+impute w/ event_num=2; design-B per-epoch deterministic negs; CE{pos=1,neg=0}; val-AUPRC best-ckpt w/ OOV NaN-drop, test 0.5-fill; softmax[:,1]; review _reviews/2026-07-02__binary__faithful_cold_port.md) | **done 2026-07-02** (multi_label_cls, Case-B TWOSIDES 200-label, codex APPROVED_WITH_NITS; regime-aware reuse S0=4ch/S1S2=3ch+impute event_num=200; KG1 re-keyed drugbank_id→pool-id (312/604 cover, rest ghost — data limitation); fixed-200 sigmoid + masked-BCE + macro-AUPRC; review _reviews/2026-07-02__multilabel__faithful_cold_port.md) — **MKG-FENN ALL 3 TASKS DONE** | **done 2026-07-02** (multi_cls, FAITHFUL cold port, codex CHANGES_REQUIRED→fixed→APPROVED_WITH_NITS; S0=4ch model.py / S1S2=3ch+nearest-seen-impute new model_cold.py ported from official modeltask3; native KG1 from data/KG/drugbank/filtered; re-vocab ddi_type + scatter; CE + sym-aug + val-macro-F1; fixed CUDA device bug + determinism flags; review _reviews/2026-07-02__multiclass__faithful_cold_port.md; spec Notes/Log/mkg_fenn_faithful_cold_port.md) | TODO |
Backfill order: HDN-DDI ml ✅ -> SSI-DDI ml ✅ -> SumGNN bin ✅ -> TIGER ml ✅ -> MRCGNN (bin+ml registered — verify) -> MKG-FENN (all 3).
nit (TIGER ml): (1) forward_logits docstring says sub_coeff, code uses mol_coeff/mi_coeff (wording); (2) no explicit assert real DrugBank id can't collide with twoside:* namespace (safe: real ids are DB...).
nit (HDN-DDI ml): predict leaves missing-mol-graph pairs at 0.5; harmless on twoside (all 604 have SMILES); real-run TODO if SMILES-incomplete.

## Queue (existing-core -> wrapper; then full ports)
- [x] **TIGER binary + multiclass** (2026-07-01, SIMPLIFIED acceptance, codex APPROVED
  thread 019f2076). Unified wrappers reuse PASS-reviewed cores (dual-channel mol GraphTransformer
  + BKG subgraph + relation-aware attn + MI loss + randomWalk, cold_start_patch=False).
  BKG defaults to FULL merged KG via TIGER_KG_SCOPE env (default "full"; kg_builder._full_kg_edges
  + _shared cache-key + build_bkg_cache manifest; verified 174,976 nodes/60 rel). test_s2 dead-code
  shim (empty frame, faithful). Registered in runner. Review:
  baseline/tiger/_reviews/2026-07-01__unified_wrappers__simplified_accept.md. Runtime: full-KG
  subgraph extraction heavy (accel deferred); no training smoke (GPU busy + deferred-run).
- [ ] MKG-FENN (multi core flat; needs multi_cls subdir + wrapper)
- [ ] SSI-DDI multiclass (binary done; needs multi core check + wrapper)
- [ ] KnowDDI / MRCGNN / TextDDI (full ports); DDIPrompt (no official code)

## EmerGNN COMPLETE (all 3 tasks, codex-approved) ✅
- [x] EmerGNN MULTILABEL (twoside) — ported TWOSIDES core to baseline/emergnn/multi_label_cls/
  (fixed 200-head BCE, cumulative train/valid/test KG = vKG/tKG, paired pos/neg dense multihot).
  codex 2-round review (019f1b12 must-fix KG background → fixed → 019f1b1c APPROVED). Review:
  baseline/emergnn/_reviews/2026-06-30__multi_label_cls__unified.md. Smoke PASS.
EmerGNN = binary + multiclass + multilabel all wired + smoked + codex-approved. Full-scale runs pending (user).

## EmerGNN acceptance: multilabel paper-gate PASS (ballpark)
- twoside S2 fold0, 10ep: our macro_auprc 0.774 ∈ paper S2 PR-AUC band [0.740,0.888] (paper 81.4±7.4).
  Confirms the migrated core preserves EmerGNN's key idea. Archived
  baseline/emergnn/_results/2026-06-30__multilabel__paper_gate_S2.md. Runner now reports 5 metrics/task.

## In progress: #2 SumGNN (from-scratch port; agent-drafting)
- SumGNN (Bioinformatics'21): KG hop-k subgraph extraction + summarization (DGL). Tasks: drugbank
  MULTICLASS (86 types, primary Macro-F1) + BioSNAP/TWOSIDES MULTILABEL (primary PR-AUC). WARM/transductive.
- DGL 2.4.0 installed. Original-Code + data/{drugbank,BioSNAP} present at Paper/Reference/Original-Code/SumGNN.
- PAPER Table 1 gate targets (paper_text.txt:590): DrugBank F1 86.85±0.44 ; TWOSIDES PR-AUC 93.35±0.14.
- reproductions/SumGNN/_paper-and-GitHub/ set up (paper text + github.txt). Port + minimal drugbank-F1 gate
  + unified wrappers (multi_cls + multi_label_cls) delegated to a subagent; codex-review + gate-verify pending.
- Risk flagged to agent: subgraph extraction on OUR merged KG for the unified wrapper (Hetionet-specific
  preprocessing our leaves may lack).

## Next
- [ ] codex-review + verify SumGNN port + gate when agent returns.
- [ ] Remaining: KnowDDI, HDN-DDI, TIGER, MKG-FENN, SSI-DDI(finish mc), TextDDI, MRCGNN, DDIPrompt(no code).
- [ ] Wire remaining existing-core: hdn_ddi (bin+multi), tiger (bin+multi), mkg_fenn (multi), emergnn multi+multilabel. (parallelizable via agents once templates proven)
- [ ] Port NEED-PORT: sumgnn, knowddi, mrcgnn, textddi (read paper+Original-Code, codex each).
- [ ] DDIPrompt: find/implement (no official code).
