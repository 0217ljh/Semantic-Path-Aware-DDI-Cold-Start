# Baseline acceptance process (standard) — unified benchmark

Authoritative acceptance checklist for every baseline wired to the unified benchmark
(`Code/baseline/<method>/<task>_cls/baseline_unified.py`). A baseline task is **done**
only when ALL steps below PASS. Designed + codex-reviewed 2026-06-30 (thread 019f1bad).
Complements the CLAUDE.md "Baseline 规范" (that governs layout/faithfulness; this governs
acceptance).

## 1. Metrics — exactly 5 per task (report all; PRIMARY = model-selection metric)

Computed by `Code/scripts/run_baseline_unified.py`. OOV / unseen-label diagnostics are
REQUIRED context fields, NOT headline metrics.

| task | 5 metrics (primary first) | diagnostics (required, not scored) |
|---|---|---|
| binary | **AUPRC** (primary), AUROC, F1, Accuracy, MCC | score_min/max/mean/std, n_pos |
| multiclass | **Macro-F1** (primary), Accuracy, Macro-Precision, Macro-Recall, Cohen's Kappa | oov_target_rate, n_classes_gold |
| multilabel | **Macro-AUPRC** (primary), Macro-AUROC, Micro-AUPRC, Micro-F1, Label-micro-Accuracy | n_zero_pos_labels_train, test_positive_on_unseen_label_rate, n_labels_scored |

Rules (codex):
- multiclass OOV-in-train gold classes are counted WRONG in all 5 metrics (the
  global-scatter predict can't select them); `oov_target_rate` reports how many.
- multilabel F1 / accuracy are thresholded at 0.5 on the sigmoid probs (a
  validation-tuned threshold is a future refinement).
- macro vs micro is explicit in the metric name.

## 2. Key-idea preservation gate (did the migration keep the paper's core idea?)

> **AMENDMENT 2026-07-01 (user directive — simplified acceptance for the remaining
> baselines):** for baselines migrated AFTER EmerGNN, the gate is **codex confirming
> the port preserves the paper's + original code's core idea** (contribution-first
> review per CLAUDE.md §"Baseline 规范"), archived to `_reviews/`. **No paper-gate run
> / observable metric required** ("不用跑出具体结果"). Still done: §3.1 review+codex,
> and an import/shape smoke so a run is POSSIBLE later. WAIVED unless user asks: §3.3
> paper-gate run, §3.5 our-benchmark report. All other rules (file independence,
> faithful hyperparams, 5-metric wiring, tracker update) unchanged. The §2 text below
> is the original fuller gate, kept for reference.

REQUIRED for every baseline. Reusing/porting a core is NOT proof the migrated
unified-wrapper path is faithful — the adapter/label-mapping/split-plumbing can silently
corrupt it. So:

**Gate = MINIMAL observable run** (user directive 2026-06-30: demonstrate the baseline,
don't fully reproduce — full reproduction is too costly). Run the baseline **through the
unified wrapper path** on the PAPER's OWN dataset + task + ONE representative setting, at
REDUCED cost (single fold, few epochs, or a subsample), and confirm the primary metric is
**in the paper's ballpark** (same order / roughly matches the paper's reported value for
that setting, not an exact ±0.02 match, not a 5-fold mean).
- PASS if the primary metric lands **near the paper band** (roughly within the paper's
  reported ±std, or clearly the same order of magnitude and trending toward it if
  under-trained). The point is to catch a BROKEN migration (metric near chance / wildly
  off), not to re-certify the paper number.
- One representative setting is enough (prefer the paper's main/hardest, e.g. S2). Other
  settings/folds optional.
- Reused cores (reproduction already PASSED): a minimal wrapper-on-paper run suffices.
- Newly ported cores: still run a minimal paper-data run (the point is to prove the port
  isn't silently broken).
- Record the observed metric + the paper target + a ballpark PASS/FAIL note.

Archive the gate result to `baseline/<method>/_results/<date>__paper_gate__<task>.md`
with the paper Table/line citation for the target number (NEVER cite a paper number from
memory — pull it from the PDF/paper_text).

## 3. Acceptance checklist (ordered; PASS all or BLOCKED)

1. **Code review + codex** — implementation, task contract, data plumbing, label
   semantics reviewed; codex independent verdict; report in `_reviews/`.
2. **Smoke on our interface** — every supported task runs via `run_baseline_unified.py`
   (or a smoke): sane shapes, non-degenerate predictions, all 5 metric keys present.
3. **Paper-setting reproduction gate** (§2) — primary metric within tolerance on the
   paper dataset/task/setting via the unified wrapper; archive.
4. **Unified-wrapper equivalence** — for reused cores, wrapper-path result matches the
   validated core / paper number on paper data (catches adapter bugs).
5. **Our-benchmark 5-metric report** — the 5 metrics + diagnostics on our data
   (per split/fold), archived to `_results/`.
6. **Robustness sanity** — ≥1 rerun/seed for stochastic methods; no metric instability
   or leakage.
7. **Artifact archive** — frozen config, dataset split IDs, exact CLI, commit SHA, env.
8. **Acceptance decision** — PASS only if 1–7 complete; else BLOCKED with the failing step.

## 4. Per-baseline acceptance status

| baseline | task | 1.review | 2.smoke | 3.paper-gate | 4.wrapper-equiv | 5.our-report | decision |
|---|---|---|---|---|---|---|---|
| emergnn | binary | ✅ codex | ✅ (AUROC 0.67@3ep) | ⏳ PENDING (wrapper on paper DrugBank binary; note: EmerGNN paper task is multiclass, so binary has no direct paper number — anchor on core + adapter-equivalence) | ⏳ | ⏳ | BLOCKED (gate) |
| emergnn | multiclass | ✅ codex | ✅ (predict(n,K), OOV=wrong) | ⏳ PENDING (paper DrugBank multiclass; primary Macro-F1; reused core has reproduction PASS → minimal wrapper-on-paper run) | ⏳ | ⏳ | BLOCKED (gate) |
| emergnn | multilabel | ✅ codex (2-round) | ✅ (predict(n,200), micro/macro) | ✅ **PASS (ballpark)** — PR-AUC 0.774 ∈ paper S2 [0.740,0.888] @10ep (_results/2026-06-30__multilabel__paper_gate_S2.md) | ✅ (gate = wrapper path on paper data) | ⏳ (our benchmark) | PASS-core (gate ✅; our-data report pending) |

## 5. EmerGNN paper targets (metric TYPES; NUMBERS to be pulled from the paper at gate time)
EmerGNN (Zhang et al. 2023, arXiv:2311.09261; PDF at Paper/Reference/EmerGNN.pdf,
reproduction at Code/reproductions/EmerGNN):
- DrugBank (multiclass DDI-event, S0/S1/S2 cold-start): primary **F1-score**
  (paper_text.txt:143, verified). Pull exact Table 1 numbers from the paper PDF before recording.
- TWOSIDES (multilabel side-effect, S0/S1/S2): primary **PR-AUC**, also ROC-AUC + Accuracy
  (paper_text.txt:146-147, verified). Table 1 = S1 & S2 (main, "emerging drugs"), S0 in
  Supplementary Table 3. Pull exact Table 1 numbers from the paper PDF before recording.
- Our `twoside` unified leaves are built FROM EmerGNN's official TWOSIDES S0/S1/S2 splits,
  so running the unified multilabel wrapper on `twoside` IS the paper-gate (paper data +
  setting). `run_baseline_unified.py --baseline emergnn --task multilabel --dataset twosides
  --split cold_s2 --fold fold0 --epochs 100` → compare macro/PR-AUC to paper Table 1 S2.
EmerGNN does NOT define a binary DDI-existence benchmark — the binary variant is OUR
task; its faithfulness is anchored on the shared core + adapter-equivalence, not a paper
number.

### EmerGNN paper Table 1 targets (VERIFIED from paper_text.txt, mean±std over 5 folds, %)
Columns confirmed (paper_text.txt:736-737): DrugBank [F1 / Accuracy / Kappa] |
TWOSIDES [PR-AUC / ROC-AUC / Accuracy]. Our unified metric `macro_auprc` == the paper's
TWOSIDES PR-AUC (base_model.py:106 `np.mean(prc_auc)` = macro over per-label AP).

| setting | DrugBank F1 (primary) | TWOSIDES PR-AUC (primary) | TWOSIDES ROC-AUC | TWOSIDES Acc |
|---|---|---|---|---|
| S1 (one-unseen) | 62.0±2.0 (:751) | **90.6±0.7** (:751) | 91.5±1.0 | 84.6±0.7 |
| S2 (both-unseen) | 25.0±2.8 (:760) | **81.4±7.4** (:760) | 79.6±7.9 | 73.0±8.2 |

Gate bands (primary ±std): TWOSIDES S2 PR-AUC ∈ [74.0, 88.8]; S1 ∈ [89.9, 91.3];
DrugBank S2 F1 ∈ [22.2, 27.8]; S1 ∈ [60.0, 64.0]. Note S2 std is huge (single-fold points
scatter widely); rigorous gate = 5-fold mean.

> NEXT for EmerGNN: multilabel S2/S1 paper-gate running on twoside (= paper TWOSIDES);
> then multiclass/binary wrapper-on-paper checks; archive to _results/ + flip decisions.
