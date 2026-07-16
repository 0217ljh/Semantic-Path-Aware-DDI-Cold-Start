# MKG-FENN multilabel (TWOSIDES 200-label) — faithful cold-start port acceptance (Case-B)

- **Date**: 2026-07-02
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex; review thread 019f248c; KG1-remap design thread 019f2485)
- **Trigger**: user — every baseline needs all 3 tasks; MKG-FENN multilabel (last of the three).
- **Scope**: NEW `multi_label_cls/{baseline.py, baseline_unified.py, __init__.py}` + additive runner
  registration. Reuses the multiclass/binary cold infra. multi_cls/*, model.py, kg_builder.py, dead
  top baseline.py UNTOUCHED.

## Task framing (Case B)
MKG-FENN paper = 65-event multiclass. Multilabel (TWOSIDES 200 side-effect labels) is a Case-B
adaptation: reuse the paper ALGORITHM CORE (4ch warm / 3ch cold + nearest-seen imputation) UNCHANGED;
swap only the task surface (fixed-200 sigmoid, masked BCE, macro-AUPRC, paired pos/neg bundle).

## The KG1-on-TWOSIDES re-key (the one novel piece)
- TWOSIDES pairs use integer pool drug_id (0..603). KG2 (Morgan/SMILES), KG3 (train DDI), KG4
  (property/SMILES) build natively in pool-id space (all 604 have SMILES). KG1 entity tables are keyed
  by DrugBank id; only 335/604 twoside drugs carry a drugbank_id.
- Fix (codex 019f2485 confirmed): keep dict1 in POOL-ID space (KG2/3/4 native); re-key the KG1 TABLES'
  drugbank_id column -> pool-id via a crosswalk from resources.drugs BEFORE the unchanged
  build_all_kgs. Crosswalk = drugbank_id -> [pool_id...] (one-to-many safe, dup rows on collision,
  0 collisions measured). Unmapped drugs (no drugbank_id / absent from filtered tables) are KEPT in
  dict1 and get a ghost KG1 slot via _ghost_pad_kg1 (cold path). All 4 KGs in ONE pool-id space.
- KG1 source = Code/data/KG/drugbank/filtered (drugbank entity tables); Code/data/KG/twosides holds
  only raw decagon DDI (no entity tables) — codex confirmed drugbank filtered is correct.
- **KG1 coverage (agent-measured, codex-confirmed code path; not independently recomputed by me —
  codex's sandbox blocked ad-hoc parquet reads): 312/604 (~52%) twoside drugs get >=1 KG1 edge; the
  other ~292 get a ghost KG1 slot (data limitation, analogous to TIGER's ~45% isolated-BKG).** KG2/3/4
  cover all 604.

## Design + implementation
- `MKGFENNMultilabelBaseline`: REGIME-AWARE — S0 -> MKGFENN(event_num=200) 4ch warm (precompute_adj,
  no impute); S1/S2 -> MKGFENNCold(event_num=200) 3ch + drug_sim1..4 + test_adj nearest-seen impute
  (baseline.py:255-287). Reuse (import) the multi_cls pure helpers + MKGFENNCold + MKGFENN +
  kg_builder, same pattern as the binary sibling.
- Task surface: fixed n_labels=200 from meta; per-label sigmoid; MASKED BCE (baseline.py:329/334-343):
  mask=(pos_y>0); BCE(pos_logits[mask],1)+BCE(neg_logits[mask],0); bundle neg_y is the POS multihot
  reused ONLY as active-label mask, never a target. Exactly one sigmoid at inference (_score_pairs
  :435-436, .detach().cpu().numpy() after sigmoid -> device-safe). predict_proba -> (n,200).
- Best-ckpt by VAL macro-AUPRC over active labels (paired pos/neg). Paired bundle
  (make_multilabel_bundle-style) with endpoints mapped through the SAME dict1 pool-id index space
  (baseline_unified.py:245-280); predict maps test rows through the same dict1 (:322-330).
- Deterministic seeding + determinism flags + symmetric pair aug + per-epoch shuffle + TrainProgress.
  save/load persist n_labels=200.

## Codex verdict: APPROVED_WITH_NITS
codex independently verified: KG1 re-key on the TABLES not dict1 (baseline_unified.py:155-191,
dict1 pool-id :217-221); crosswalk drugbank_id->[pool_id] one-to-many safe (:129-151, dup on collision
:176-183); unmapped kept + ghost (kg_builder.py:43-80 + _ghost_pad_kg1 multi_cls/baseline.py:181-207);
KG1 source = drugbank filtered (:72,193-200); fixed-200 head, raw logits in training + one sigmoid
(baseline.py:275-286, :435-436); neg_y NOT a target (mask from pos_y>0, :329/334, pos->1/neg->0 same
cols :342-343); endpoint/index alignment consistent (:245-280, predict :322-330); regime switch
(:212-215 + baseline.py:255-287); device safety (reused fixed MKGFENNCold + .detach().cpu().numpy());
independence holds.
- **NIT (doc-only, non-bug)**: baseline_unified.py:20-31 docstring states ~335/604 and ~312/604 KG1
  coverage but the code doesn't assert those exact numbers (data-dependent). No code bug.
- codex limitation: could not independently recompute 312/604 (sandbox blocked parquet reads); the
  code path that produces it is correct.

## Verification
- py_compile OK; import+registry smoke: get_unified("mkg_fenn","multilabel") -> MKGFENNUnifiedMultilabel.
- Offline (agent, no training) on twosides/multilabel/cold_s2/fold0: KG1 coverage 312/604; bundle
  train_pos_y (30134,200), endpoints all <604; one COLD forward -> (8,200) finite in [0,1]. S0 warm
  4ch forward -> (8,200) finite in [0,1]. (No training: GPU busy; deferred.)

## Decision: ACCEPTED (simplified gate) — core idea + cold mechanism preserved
MKG-FENN now has ALL 3 tasks (binary + multiclass + multilabel), all regime-aware faithful cold ports.
