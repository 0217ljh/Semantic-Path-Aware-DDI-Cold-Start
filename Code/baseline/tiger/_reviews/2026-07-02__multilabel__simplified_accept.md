# TIGER multilabel (TWOSIDES) — simplified acceptance (Case-B task backfill)

- **Date**: 2026-07-02
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex; review thread 019f21e6; design thread 019f2138)
- **Trigger**: user — all baselines need all 3 tasks (bin/mc/ml); backfill TIGER's missing
  multilabel. User approved design (a) + the TIGER-scoped model modification
  ("可以修改，只在tiger上这么改").
- **Scope**: NEW `multi_label_cls/{baseline.py, baseline_unified.py, __init__.py}` + an ADDITIVE
  `TIGER.forward_logits` in `model.py` + additive runner `_MODULES` + top `__init__` re-export.
  TIGER binary/multiclass cores + existing softmax `forward` UNTOUCHED.

## Task framing (Case B, dual-channel)
TIGER paper (Su et al., AAAI 2024) = BINARY (pos-pair vs random neg, softmax CE on a (B,2) head).
TIGER is dual-channel: molecular GraphTransformer + biomedical-KG (BKG) subgraph GraphTransformer
+ relation-aware attention + mutual-information (MI) aux loss. Multilabel (TWOSIDES 200 side-effect
labels) is a Case-B adaptation: reuse the ALGORITHM CORE unchanged, swap only head-activation
(sigmoid), loss (masked BCE), targets (fixed-200 multihot), metric (macro-AUPRC).

## The two points that make TIGER ml harder than HDN/SSI ml (both surfaced + resolved)
1. **Model forward is softmax/CE-hardwired.** `TIGER.forward` writes log_softmax→nll_loss and
   returns softmax probs (model.py:344 mol_only, :401 dual-channel). So a wrapper-side sigmoid is
   NOT enough. Resolved by an ADDITIVE `forward_logits` (model.py:407-503): same encoder + fc1
   fusion + fc2 + cold-start patch + MI aux (mol_coeff*loss_s_m + mi_coeff*loss_s_d, matching
   forward :396-400), returns RAW logits, no log_softmax/nll. Existing forward byte-intact.
2. **BKG channel on TWOSIDES.** Only 335/604 twoside drugs carry a real DrugBank id (all unique,
   0 collisions); 269 (~45%) have drugbank_id=='<NA>'. User-approved option (a): dual-channel +
   FULL merged KG, unmapped drugs get an ISOLATED self-loop BKG node (kg_builder.py:201-213), NOT
   dropped (dropping → _make_pair_batch drops every pair touching them). Molecular channel covers
   all 604. Documented as a data limitation, not hidden.

## Design + implementation
- `TIGERMultilabelBaseline(TIGERBaseline)` (@register("tiger_ml")): subclasses the BINARY core
  (NOT multiclass, whose fit re-vocabs). Inherited UNCHANGED: `_build_mol_graphs`,
  `_build_bkg_and_subgraphs` (full merged KG), `_make_pair_batch`. Overrides: fixed n_labels=200
  (fc2→n_labels via n_classes, no re-vocab), `_forward_logits`/`_batch_logits` use the additive
  raw-logits path, sigmoid + MASKED BCE, macro-AUPRC selection, predict_proba→(n,200), save/load
  persist n_labels.
- **Masked BCE** (baseline.py:307-323): mask=(pos_y>0); BCE(pos_sel,1)+BCE(neg_sel,0) on the SAME
  active columns; bundle neg_y is the POS multihot reused ONLY as an active-label mask, never a
  target. MI aux (pos_aux+neg_aux) added each step (:328), mirroring binary's two forward passes.
- **Strict pair alignment**: fit :295-306 (keep=pos_kept&neg_kept, pos_take=keep[pos_kept],
  neg_take=keep[neg_kept]); _score_pairs :425-428 (kept_local=np.where(mask)[0] → probs[i]).
  codex verified _make_pair_batch preserves kept rows in ASCENDING input order
  (binary_cls/baseline.py:433-435), so both mappings are correct.
- **Unified wrapper** `TIGERUnifiedMultilabel` (@register_unified("tiger"), task="multilabel"):
  - id-remap (_build_tiger_id_map :119-166): TIGER id = real DrugBank id else synthetic
    `twoside:<int_id>`; 1-to-1 collision check on real ids with synthetic fallback+log.
  - _build_remapped_ds (:176-207): does NOT use generic make_dataset (which renames
    drug_id→drugbank_id, breaking the merged-KG id space); drugs keyed by TIGER id with SMILES for
    all 604; g1/g2/all_drugs remapped; unmapped drugs kept; test_s2 empty shim.
  - _make_bundle (:212-248): string-endpoint mirror of make_multilabel_bundle (generic casts
    int64, breaks on 'DB00813'); _dense_multihot label vectors.
  - merged_kg_path (:94-101) = canonical Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet
    (NOT resources.kg.source, TWOSIDES's own KG). cold_start_patch=False, extractor=randomWalk.

## Verification
- import + registry smoke PASS: get_unified("tiger","multilabel") → TIGERUnifiedMultilabel.
- id-remap verified on real twoside leaf: 335 real DrugBank ids all unique (0 collisions), 269 <NA>.
- BKG build smoke (full scope): 604 drugs, 174612 entities, 30134 DDI edges kept, 332 real ids
  resolve to KG entities.
- Primary reviewer (Claude) independently read model.py forward/forward_logits + both ml files.
- (No training run: GPU busy w/ EmerGNN; deferred-run.)

## Codex verdict: APPROVED_WITH_NITS (verbatim key points)
"No correctness or faithfulness bugs found in the reviewed TIGER multilabel backfill. This is a
static source review only; I did not run training." Core preserved (forward unchanged, forward_logits
reuses same encoders/fc1/fc2/patch/MI, raw logits only); multilabel surface correct (fc2 via
n_classes, BCEWithLogits on raw logits, exactly one sigmoid at inference, neg_y not a target, mask
from y_pos>0); both alignment concerns check out (_make_pair_batch ascending input order verified at
binary_cls/baseline.py:433-435); wrapper faithful (canonical merged-KG path, no generic make_dataset,
remap into TIGER-id space, unmapped drugs kept, kg_builder self-loops at kg_builder.py:201-213,
predict returns (n,200) aligned); file independence OK (no reproductions/ / hdn_ddi / ssi_ddi).
- **NIT 1 (wording)**: forward_logits docstring (model.py:430) says aux = "sub_coeff*MI"; code
  uses mol_coeff/mi_coeff (:453, :502). Functionally fine, docstring wording only.
- **NIT 2 (latent, non-triggering)**: no explicit assert that a real DrugBank id can't collide with
  the `twoside:*` synthetic namespace. Safe in practice (real ids are `DB...`).

## Decision: ACCEPTED (simplified gate) — core idea preserved
TIGER now has all 3 tasks (binary + multiclass + multilabel).
