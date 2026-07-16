# EmerGNN multilabel — paper-faithfulness gate (TWOSIDES S2)

- Date: 2026-06-30
- Type: key-idea preservation gate (minimal observable run) — does the MIGRATED unified
  multilabel wrapper still reproduce EmerGNN's paper TWOSIDES behavior?
- Data: `twosides` unified leaf `multi_label_cls/twoside/inductive/S2/fold0` — built FROM
  EmerGNN's official TWOSIDES S2 split (S2_1), so this IS the paper's data + setting run
  through our unified wrapper path.

## Run identity
- run_id: emergnn__twosides__cold_s2__fold0; run_dir: Code/runs/emergnn__twosides__cold_s2__fold0/
- CLI: `python Code/scripts/run_baseline_unified.py --baseline emergnn --task multilabel --dataset twosides --split cold_s2 --fold fold0 --epochs 10`
- epochs: 10 (MINIMAL — under-trained; paper uses more); fit_time 804.9s; GPU.
- KG graphs: train 3,380,096 < valid 3,394,154 < test 3,419,100 edges (cumulative vKG/tKG).

## Result vs paper
Our metric `macro_auprc` == paper's TWOSIDES PR-AUC (base_model.py:106 = macro over per-label AP).

| metric | ours (10 ep, fold0) | paper Table 1 S2 (5-fold mean±std) | in band? |
|---|---|---|---|
| **PR-AUC (macro_auprc)** PRIMARY | **0.7744** | **0.814±0.074** → [0.740, 0.888] (paper_text.txt:760) | **YES ✅** |
| ROC-AUC (macro_auroc) | 0.7641 | 0.796±0.079 → [0.717, 0.875] | YES |
| micro_auprc | 0.7040 | (paper reports macro) | — |

Other 5-metric fields: micro_f1 0.667, label_micro_accuracy 0.50 — DEGENERATE at the fixed
0.5 threshold because the sigmoid scores are skewed high (score_mean 0.918); threshold not
calibrated. Not a migration bug (the paper's primary PR-AUC is threshold-free and passes);
validation-tuned threshold is a known future refinement.

## Verdict: PASS (ballpark)
Our PR-AUC 0.7744 lands INSIDE the paper's S2 ±std band [0.740, 0.888] after only 10
under-trained epochs (val PR-AUC was still climbing). This confirms the migrated multilabel
core preserves EmerGNN's key idea end-to-end (adapter + wrapper + eval did not corrupt it).
Caveat: single fold (paper is 5-fold mean), minimal epochs. Per the minimal-observable
acceptance rule (user directive), this is sufficient to accept the migration.

## Reproduce
python Code/scripts/run_baseline_unified.py --baseline emergnn --task multilabel \
    --dataset twosides --split cold_s2 --fold fold0 --epochs 10
