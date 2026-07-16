# Codex review — unified benchmark 5-fold rebuild (transductive + inductive)

- Date: 2026-06-30
- Primary reviewer: Claude (opus-4.8) — protocol + implementation
- Independent reviewer: codex (gpt-5-codex), thread 019f19b6, 2 rounds (protocol + final code)
- Trigger: user required every dataset to have BOTH transductive and inductive splits,
  all as 5-fold CV (cold-start = drug-disjoint 5-fold, 20% held out per fold).

## Outcome
`Code/data/ddi_unified/<task>/<dataset>/<regime>/<split>/<fold>/` — 21 leaves, all 5-fold:
- {binary_cls, multi_cls} × {drugbank_latest, drugbank_deng, drugbank_ryu} × {S0, S1, S2}
- multi_label_cls × twoside × {S0, S1, S2}

## Protocol (codex round-1 approved)
- transductive/S0: deng/ryu use the source's 5 official warm CV folds; drugbank_latest
  uses pair-level 5-fold (`folds.warm_cv_pairs`); twoside re-splits official S0 into 5.
- inductive/S1,S2: drug-disjoint 5-fold (`folds.drug_cv_partition`). fold j: test=G[j],
  val=G[(j+1)%5], train=rest. S2 = both held-out; S1 = one held-out + one SEEN
  (= actual train-positive drug). twoside uses EmerGNN official S1_1..12345 / S2_1..12345.
- Negatives (binary): `UniformNegativeSampler`, pools matched to each split's geometry,
  excluding the GLOBAL positive set AND negatives already used in earlier splits of the
  fold (running exclude, no cross-split overlap).
- multiclass: positives only, closed-set y_cls_train. multilabel: official negatives.

## Code
`Code/data_utils/folds.py` (partition/carve/validators), `Code/data_utils/mrcgnn_cv.py`
(shared deng/ryu builder), `Code/scripts/prepare_ddi800.py` (drugbank_latest),
`prepare_deng.py`, `prepare_ryu.py`, `prepare_twosides.py`.

## codex round-2 findings (all fixed)
1. drugbank_latest multiclass `pair_format` was `ordered_source` but data is canonical
   (one unordered row per pair) -> changed to `canonical`. deng/ryu mc stay
   `ordered_source` (directional source rows).
2. cold semantic validators (`check_cold_s1/s2`) now called INSIDE the builders, not
   only in ad hoc verification (fast-fail on future refactor).
3. `assert_cv_coverage` strengthened to simulate the rotation and assert every drug is
   test-once and val-once (not just a disjoint partition).
4. warm train-coverage assertion added for drugbank_latest (generated warm). NOT applied
   to deng/ryu warm — those are official source folds where a drug may be eval-only in a
   given fold (appears in other folds' train); imposing coverage would corrupt the split.

## Extra correctness fixes found during implementation
- Negative cross-split collision (2250 train/val overlaps) -> running-exclude per leaf.
- ryu warm binary: source directional/multi-type rows put a canonical pair in >1 split
  -> binary assigns each canonical pair to ONE split, priority test>val>train. mc keeps
  directional rows (official).
- S1 "seen" endpoint must be an ACTUAL train-positive drug (not just a nominal train-group
  member); an isolated train-group drug otherwise made 3 deng S1 pairs both-unseen and
  failed check_cold_s1. `carve_cold` now carves S1 against the train-positive drug set and
  returns it so S1 negatives draw their seen endpoint from the same set.
- Deterministic seeds only (replaced a `hash(str)` seed offset with a fixed map).

## Verified invariants (all pass)
- S2: test-drug leak into train = 0; train/test canonical pair overlap = 0.
- CV coverage: each drug is the test group exactly once across 5 folds (570/1700/800).
- S1 full-frame (pos+neg) exactly-one-seen = 1.000 (deng/ryu/latest).
- multiclass closed-set: train y_cls_train>=0; eval unseen-class rows = -1.
- warm S0: train/test & train/val canonical pair overlap = 0; latest train covers all 800.
- drugbank_latest types = 165 (present among the 800-drug positives, not global 215).

## codex verdict
"I do not see a remaining benchmark-leak path in the negative scheme or the warm binary
dedup logic." Remaining notes are documentation-only:
- S2 = "same held-out group only" (cross-group both-unseen pairs dropped: per-dataset
  totals deng 15014 / ryu 76721 / latest 37673). Protocol-defining, documented here.
- Mixed sources (deng/ryu warm = official source folds; cold = generated; twoside = official
  EmerGNN folds with NON-partition CV geometry). Fair within dataset×regime; cross-method
  comparability claims only where the official protocol is preserved.

Verdict: PASS. Benchmark locked.

## 2026-06-30 addendum — drugbank_latest split into partial + full
User clarified the real drugbank_latest is the FULL DrugBank (1900 drugs), and the
800-drug version was only a fast-test subset. So:
- `drugbank_latest_full` (group `ddi_full`): 1900 drugs, 215 types, 565731 canonical
  positives (from ddi_edges.csv) — the MAIN corpus.
- `drugbank_latest_partial` (group `ddi800`): the 800-drug fast-test subset, 165 types.
`prepare_ddi800.py` now builds BOTH via a parametrized `build_latest(group, drugs_set,...)`.
Tree now has 27 leaves. Full audit (`Code/scripts/analyze_unified_audit.py`, strengthened
with codex's 7 extra checks): **2088 PASS, 0 FAIL**. drugbank_latest_full warm S0 =
strict 1.000 transductive; full S2 CV coverage = each of 1900 drugs test-once + val-once.
Audit confirms: labels complete, transductive/inductive known-unknown partition correct,
binary negatives correct (1:1, ∉ global pos, structure-matched, dedup, no cross-split
overlap). Only documented note: deng/ryu warm = official source folds, ~99% (not strict).
