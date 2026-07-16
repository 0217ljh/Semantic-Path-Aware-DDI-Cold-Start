# Screen 1 + Motivation — Final Synthesis (autonomous research run 2026-05-20)

This document is written incrementally during the autonomous research run.
Variant D and E full training results will be filled in when their
respective runs complete (D ETA ~00:45, E ETA ~08:30 next morning).

## Executive summary (preliminary, mid-run)

| Component | Result | Status |
|---|---|---|
| Screen 1 code | EmerGNN_TAG with feat='X' external init, full multimode wrapper | ✅ DONE |
| PubMedBERT cache (D, D-name, E, F variants) | 178K nodes × 768d cached, 0.1% truncation | ✅ DONE |
| Node2Vec variant C cache | ⏳ generating (random walks on 178K-node graph) | IN PROGRESS |
| **Variant D full training** (PRIMARY claim) | val_s0 plateau ~0.97 at ep 15, ETA ~00:45 | ⏳ IN PROGRESS |
| Variant E full training (critical control) | Auto-launches after D completes | ⏸️ QUEUED |
| Variant A (random anchor) | 1-epoch smoke only: test_s0=0.870 / s1=0.802 / s2=0.658 | (full skipped to prioritize D) |
| E1c node readability | 81.8% biomedical-relevant nodes readable | ✅ PASS |
| E1b PK/PD path endpoint | χ²=320.6, p=2.4e-70 (PD-strong, PK-moderate) | ✅ PASS (asymmetric) |
| E7 LR-meeting baseline | E7 v2 matched access: LR-count 0.7311 vs EmerGNN 0.7462 (CI overlap, statistical tie); E7 v1 merged-KG 0.7573 was 2.6pt KG-access inflated | ✅ PASS-with-caveat, MODERATE signal for Screen 3 |
| Screen 3 code | full trainer + CPU smoke 3/3 PASS | ✅ READY (untrained) |
| Screen 5 code | skeleton + VME generator (sandbox-blocked smoke) | ✅ SKELETON (NotImplementedError trainer) |

## Top finding (preliminary, with codex caveats)

**E7 v2 (matched drugbank-only KG)**: LR-count AUC = **0.7311** vs EmerGNN
multimode **0.7462** on test_s2. EmerGNN has a modest 1.5pt point-estimate
advantage with overlapping bootstrap CIs (EmerGNN inside LR's CI
[0.7153, 0.7468]) — **statistical tie**.

A 24-feature LR (12 canonical-kind × {1-hop, 2-hop} intersections) approaches
but does not exceed EmerGNN's flow under matched KG access. The earlier E7
v1 result (LR-count 0.7573 on merged KG) was partly a KG-access artifact —
2.6pt of LR's apparent advantage came from access to SE / Disease /
Phenotype / Anatomy node types that drugbank lacks.

### ⚠️ Codex round-5 critical caveat — KG access asymmetry

E7 v1 used the **merged KG (178K nodes)** for LR features but EmerGNN
multimode trains on the **drugbank KG only (~10K nodes)**. KG access
asymmetry could confound the "LR ties EmerGNN" claim. **E7 v2 was
launched to test this** (drugbank-only KG features, matched access):

#### E7 v2 result (matched drugbank-only KG access)

| Method | KG | test_s2 AUC | 95% CI |
|---|---|---|---|
| LR-binary | drugbank-only | 0.7182 | [0.7030, 0.7330] |
| **LR-count** | **drugbank-only** | **0.7311** | **[0.7153, 0.7468]** |
| LR-count | merged | 0.7573 | [0.7411, 0.7726] |
| LR-binary | merged | 0.7384 | [0.7217, 0.7543] |
| **EmerGNN multimode** | **drugbank-only** | **0.7462** | — |

**Critical observation**: Under matched-access drugbank-only KG:
- LR-count drops to 0.7311 (CI [0.7153, 0.7468])
- EmerGNN at 0.7462 falls JUST inside LR's CI upper edge → technically tied,
  but the point-estimate gap widens from +1.1pt (E7 v1) to **−1.5pt (E7 v2)**
- The merged-KG gain (∼2.6pt over drugbank-only) reflects access to SE /
  Disease / Anatomy / Phenotype annotations that drugbank lacks

**Revised interpretation**:
- E7 v1's "LR ties EmerGNN" was partly a data-access artifact, NOT a pure
  architecture insight
- Under matched KG: EmerGNN's flow propagation extracts more signal than
  a 24-feature LR, but the gap is small (1.5pt)
- Shared-mediator structure IS a real source of cold-start signal but
  is NOT sufficient to fully replace flow propagation

#### Implications for screen prioritization

1. **Screen 1 (TAG init)**: NOW HIGHEST PRIORITY. Variants D/E/F give the
   model access to merged-KG semantic content (via PubMedBERT) without
   needing to flow through merged-KG topology. This may capture the ~2.6pt
   gain that E7 v1 vs E7 v2 reveals.
2. **Screen 3 (meet-in-middle)**: motivation is MODERATE not strong. The
   architectural change (junction readout) may add a few pt but is not
   enough alone. Run Screen 3 with both kg_source options to verify.
3. **Screen 5 (PK/PD subgraph)**: motivation moderate-strong if it leverages
   merged-KG diversity. Worth running ⑤a (S3 variant) with merged KG.

## Variant D vs A early-epoch comparison

Both ran on seed 42, drugbank KG, multimode 3-submodel orchestration.

| Sub-model | Variant A (random init) | Variant D (PubMedBERT P1) |
|---|---|---|
| s0 ep 1 val_auc | 0.8697 | 0.8967 (+2.7pt) |
| s0 ep 5 val_auc | (smoke only, no ep 5) | 0.9684 |
| s0 ep 9 val_auc | — | 0.9744 (peak so far) |
| s0 ep 15 val_auc | — | 0.9616 |

→ D shows faster early convergence than A. Final-state comparison pending.

## Per-experiment details

### E1c — Node Readability

- 178,029 total merged-KG nodes
- 174,882 (98.2%) with non-empty text
- 105,123 (81.8% of biomedical-relevant) readable by regex check
- **Gate PASS** (≥80%)

### E1b — PK/PD Path Endpoint Asymmetry

- 6,000 pairs sampled (3,000 PK + 3,000 PD)
- BFS shortest path ≤3 hops, intermediate kind classified by canonical layer
- 2×6 contingency, chi-square excl. Direct/NoPath/zero-cols

Results:
```
              PK_layer  PD_layer  BioProcess  Other  Direct  NoPath
PK (n=3000)    1,611     1,353         0       112      11     197
PD (n=3000)    1,071     1,823         0       383      30     328
```

- Chi-square: χ²=320.60, dof=2, **p=2.41e-70**
- PD pairs: PD-layer 55.6% vs PK-layer 32.7% (margin 22.9pt) → STRONG
- PK pairs: PK-layer 52.4% vs PD-layer 44.0% (margin 8.4pt) → MODERATE

**Verdict**: i1 paradigm asymmetry IS real and statistically robust, but
effect-size asymmetric. Screen 5 ⑤a motivated for PD-side; ⑤b VME
exploratory.

### E7 — LR-over-Meeting-Node Baseline

- 24 features = 12 canonical kinds × {1-hop intersect, 2-hop intersect}
- LR-binary: AUC = 0.7384 [0.7217, 0.7543]
- LR-count: AUC = 0.7573 [0.7411, 0.7726]  (E7 v1, merged KG)
- **LR-count (matched access): AUC = 0.7311 [0.7153, 0.7468]**  (E7 v2, drugbank-only KG)
- EmerGNN multimode anchor: 0.7462 (within both LR CIs)
- The 2.6pt gap between v1 and v2 reveals KG-richness contribution; the
  matched-access result is the publication-grade comparison.

## Code deliverables

```
Code/my_code/models/
├── screen1_tag_init/          (9 files, all PASS codex round 2)
├── screen3_meet_in_middle/    (5 files, CPU smoke 3/3 PASS)
└── screen5_pkpd_subgraph/     (5 files, skeleton — trainer NotImplementedError)

Code/my_code/exp_e1b_pkpd_endpoint/run_e1b.py
Code/my_code/exp_e7_lr_meeting/run_e7.py
```

## Codex review history

| Round | Subject | Verdict |
|---|---|---|
| #1 | Screen 1 skeleton (post-cache) | NEEDS_FIX (2 CRITICAL + 7 WARN) |
| #2 | Screen 1 post-fix | PASS (proceed to training) |
| #3 | Screen 3 + 5 code | NEEDS_FIX (3 CRITICAL for Screen 3 / 5) |
| #4 | Screen 3 + 5 post-fix | PASS (WARN remaining) |

## Outstanding items for user when they return

### Required immediate review
1. **Variant D results** (will be in `Code/runs/.../screen1_D_P1_seed42_full.../results.json` after ~00:45)
2. **Variant E results** (after ~08:30 next morning)
3. **Aggregate report**: run `python Code/my_code/models/screen1_tag_init/aggregate_results.py` to get comparison table

### Optional (deferred)
1. **VME generation API call** (sandbox blocked autonomously, ~$0.005 smoke):
   `python Code/my_code/models/screen5_pkpd_subgraph/smoke_vme.py`
2. **Variant H (Qwen-72B)**: requires downloading Qwen-72B weights to G: drive first
3. **Screen 5 trainer rel-vocab audit** + **VME gap re-definition**: see `Code/my_code/models/screen5_pkpd_subgraph/README.md` §"Outstanding work"

### Screen-by-screen priority (revised post codex round 5)

| Priority | Screen | Justification |
|---|---|---|
| **HIGH** | **1 (TAG init)** | E1c PASS; E7 v1 vs v2 (2.6pt merged-KG-only gain) implies semantic node features carry significant cold-start signal. D early convergence > A. |
| MED-HIGH | 5 ⑤a (PK/PD subgraph + merged KG) | E1b PASS Cramer's V=0.22; PD-side strong, PK-side modest. Run with merged KG to leverage SE/Disease/etc annotations. |
| MED | 3 (meet-in-middle) | E7 v2 (matched access): LR ties EmerGNN within CI, but point-estimate EmerGNN advantage 1.5pt. Architecture change adds value but is not the dominant lever. |
| LOW | 5 ⑤b (VME) | Most fragile design; defer until ⑤a result |
| LOW | 2 (PK/PD path prior) | Likely dominated by Screen 5 ⑤a |
| DROP | 4 (LLM decoder verifier) | Already deferred per first_step_plan.md §4.9 |

---

*This document will be updated when variant D + E results are in.*
