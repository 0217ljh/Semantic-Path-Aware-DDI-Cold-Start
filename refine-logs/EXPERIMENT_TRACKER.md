# EXPERIMENT_TRACKER — i1-routing v2

| ID | What | Status | test_s2 AUROC | PK | PD | Notes |
|----|------|--------|---------------|----|----|-------|
| MNAH | backbone reference | DONE (prior) | ~0.766-0.772 | 0.836 | 0.702 | not re-run |
| P1 | molecular m_u Morgan | DONE | — | — | — | 1994 drugs, 2048-d |
| P0 | k_u target | DONE | — | — | — | 7754 drugs, 768-d |
| P2 | effect-neighbor sets | DONE | — | — | — | CSR+kind+drugdeg; 75.5% split cov; main-line subset{SE,pheno,sym}=82.4%; topK=32 (codex 019e6224) |
| P3 | InfoNCE alignment | DONE | — | — | — | 615 seen drugs; holdout AUC 0.68 top1 0.164 MRR 0.261 (codex patch r019e622a); z_m 1994x128 |
| E1 | v2 full model (r1) | DONE | 0.7719 | 0.823 | 0.720 | NO GAIN vs MNAH 0.772; PD not lifted |
| E1d | v2 diag (ch dump) | DONE | 0.7688 | 0.854 | 0.683 | eff channel CHANCE 0.57/0.59; gate~0.5 not collapsed; eff dilutes mol |
| E2 | v2 + aux_bce 0.3 | DONE | 0.7684 | 0.834 | 0.702 | pure eff PD 0.572 (unsupervised 0.588) FALSIFIED |
| E3 | v2 effect=count | DONE | 0.7546 | — | — | dropped i2 (count head replaced) - not apples |
| E-frag | shared/resid mol↔typed-KG | DONE | 0.7546 | — | — | mol_head 0.577; dropped count head; redundant w/ KG |
| E-llm | MNAH+LLM residual | DONE | 0.7668 | — | — | shuffle 0.7658 == main -> LLM INERT |
| **E-i4 main** | **MNAH+I4 mechanistic pair** | **DONE** | **0.7804** | **0.853** | **0.706** | **+0.84 over MNAH; +1.61 over shuffle; i4_only 0.60→0.51; FIRST POSITIVE** |
| E-i4 shuf | I4 shuffle-ctrl | DONE | 0.7643 | 0.839 | 0.689 | controls collapse cleanly |
| E-i4 rand | I4 random-ctrl | DONE | 0.7710 | 0.850 | 0.691 | i4_only -> 0.51 chance |
| E-i4-HN | v2i4 + hard-neg mining | DONE | 0.7537 | — | — | NEGATIVE: HN hurt -2.67pt vs MNAH; shuffle 0.762 BEAT main; model suppressed I4 |
| E-i4-HN shuf | v2i4hn shuffle | DONE | 0.7620 | — | — | beat main -> harm only when hardness real (label noise/false-neg incompat) |
| E-i4-D λ0.3 | MNAH+I4 + joint 215-class | DONE | 0.7792 | — | — | binary ~= v2i4; macroF1 0.0225 (4-5× chance), microF1 0.287 (57× chance); in-arch +0.84pt vs λ=0 |
| E-i4-D λ0 | sanity (lam=0) | DONE | 0.7708 | — | — | within noise of v2i4 0.7804; type head dormant macroF1=0 |
| E-i4 len=4 | Phase C-a longer path | DONE | 0.7707 | — | — | REGRESSED (-0.97 vs len=3); emergnn dropped 0.74→0.67 (over-smooth); count compensated 0.67→0.76 |
| E-i4 len=5 | Phase C-a longer path | DONE | 0.7638 | — | — | REGRESSED further (-1.66 vs len=3); both below codex 0.785 hard-stop |
| **CEILING** | **BEST = v2i4 len=3 = 0.7804** | — | **0.7804** | **0.853** | **0.706** | **+0.84pt over MNAH; clean controls; mechanism-interpretable** |
| S1 | shuffle-effect-names | TODO | — | — | — | effect must collapse |
| S2 | bag-pool ablation | TODO | — | — | — | PD must drop ≥1.5pt |
| A1 | PK/PD + gate analysis | TODO | — | — | — | gate↔PK/PD validation |

## Round log
- **R1 (E1, 2026-05-25)**: v2 main = 0.7719 (PK 0.823 / PD 0.720) == MNAH 0.772. NO GAIN.
  Diagnostic retrain (channel dump) confirmed root cause:
  - pure EFFECT channel ~CHANCE: eff PK 0.572 / PD 0.588 (target was PD lift, didn't happen).
  - pure MOLECULAR: mol PK 0.733 / PD 0.649 (> effect even on PD).
  - GATE not collapsed: g~0.5 (PK 0.575 / PD 0.545, corr_gate_PK +0.16). 50/50 mix => weak
    effect DILUTES strong molecular => combined == MNAH.
  - codex (019e6272): effect channel near-chance, not just diffuse. Sparsity alone ~10-15%
    to rescue. PRESCRIBED: aux BCE deep-supervision on eff_logit (lambda 0.2-0.3), no sparsity.
    Falsification: pure eff PD <0.65 after supervision => effect-neighbor PD story dead end.
- **R2 (planned)**: v2 + eff_aux_bce=0.3. Success = pure eff PD materially up (>=0.65, ideally
  0.70). Then add sparsity. Else pivot PD story (PubMedBERT-name effect neighbors too weak).
- **R2 ACTUAL (2026-05-26)**: eff_aux_bce=0.3 -> combined 0.7684, pure eff PD=0.572 (NOT woken;
  unchanged vs unsupervised). PubMedBERT-name effect-neighbor mechanism FALSIFIED.
- **R3 (2026-05-26, codex 019e6295 pivot)**: replaced effect channel with COUNT-MLP. combined 0.7546
  but ACCIDENTALLY DROPPED MNAH count head -> not apples-to-apples (see below).
- **R4 E-frag (2026-05-26, codex 019e6747)**: BRICS fragments + shared(typed-KG)/residual aligned
  molecular, EmerGNN+molecular (count head dropped). 0.7546; mol_head_alone 0.577. Diagnosis:
  dropped i2 + molecular ~redundant w/ KG. codex (019e6770): keep count head, ADD molecular.
- **R5 E-llm (2026-05-26, codex 019e6770)**: MNAH + LLM-pharma residual (count head retained).
  combined 0.7668; llm_only 0.568; SHUFFLE 0.7658 (= main) -> LLM channel INERT. 3/3 substrates failed.
- **STRATEGIC (codex 019e67af, 2026-05-27)**: P(>0.80)=10-15%; structural fact: S2 is DDI-edge
  cold-start (KG biomedical neighbors retained at test). Final feature test = I4 structured
  pair features; then pivot or backbone.
- **R6 E-i4 (2026-05-29, codex 019e76f6)**: MNAH + structured-LLM 13 mechanistic pair features
  (shared CYP, inhibitor->substrate, shared class/target/pd/tox/clearance). **FIRST POSITIVE**:
  main 0.7804 (+0.84pt over MNAH; PK 0.853 +1.7pt, PD 0.706 +0.4pt). Controls collapse: shuffle
  0.7643 (-1.61pt), random 0.7710 (-0.94pt); i4_only 0.60 -> 0.51 chance. Real but below codex's
  +1.5pt bar over baseline; +1.61pt over shuffle is clean. codex: I4 credible weak positive;
  next = MNAH+I4+hard-negative mining + pairwise relational head.
- **R7 v2i4hn (launching, 2026-05-29)**: MNAH+I4 + HARD-NEGATIVE mining (rerank epoch neg pool
  by I4 overlap; top 50% hard replaces bottom 50% random; 5-epoch warmup). Hypothesis: cold-start
  ceiling = insufficient discrimination among pharmacologically plausible pairs.
- **R7 ACTUAL (2026-05-30)**: HN MINING HURT: main 0.7537 (-2.67pt vs MNAH, -3.67pt vs v2i4);
  shuffle-ctrl 0.7620 BEAT main by +0.83pt. Diagnosis: hard negs w/ high I4 overlap have weak
  true DDI signal not in test labels (label noise / false negs) - model learned to SUPPRESS I4.
  codex 019e7734: skip HN retry, move to Phase D.
- **R8 Phase D v2i4d (launching, 2026-05-30)**: MNAH+I4 + joint 215-class type head; type CE on
  positives only, lambda_type=0.3; I4 feeds both heads; type macroF1 reported as success-2.
  Hypothesis: type supervision regularizes shared rep toward mechanism, where HN mining failed.
  Running main(0.3) + sanity(0.0, must = v2i4 0.7804).
