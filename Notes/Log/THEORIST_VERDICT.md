# Theorist Verdict — Round 4 / Phase A-2

**Date.** 2026-06-02
**Theorist.** general-purpose Agent (codex MCP unavailable for ChatGPT account, 2026-06-02 verified). Independent reviewer: NOT INVOKED (fallback per CLAUDE.md review archive规范).
**Companion doc.** Full derivations in `ANALYSIS_REPORT.md §Theorist Assessment`.

---

## 1. F1 ceiling under current softplus-gated additive fusion

- **Optimistic (rank-aligned branches):** 0.7405 (degenerate — would equal max-branch AUC; refuted by observed 0.7804).
- **Independence ceiling (Hanley-McNeil normal margin model):** **≈ 0.794** given branch AUCs e=0.7405, c=0.6688, i=0.6003.
- **Best estimate honoring documented Stage-2 transfer 11.4% (paper_writeup.md:241-248):** combined ≤ **0.785 ± 0.003 single seed** without fusion reform.
- **Conclusion.** The +0.79 target lives at the very edge of the additive-fusion envelope. To hit 0.79 with the current architecture you would need ≥ +1.5 pp on the emergnn branch AND favorable transfer — neither D1 nor D2 delivered. **Fusion reform is required**, not optional.

## 2. Recommended execution order

| Rank | Direction | E[combined] | P(≥0.79) single seed | Why |
|------|-----------|:-----------:|:--------------------:|------|
| 1 (paper-priority) | **R3 PMP-style readout** | 0.788 | **28 %** | paper Section 4 commitment; R3 ⊇ v2i4 under input augmentation `[e, m_pool, pair_feat, feats22, m_present]`; formally cleanest superset. |
| 2 (novelty payoff) | **R2 Hypernet message function** | 0.787 | 23 % | strongest architectural diff; requires zero-init of gate-MLP last layer + 5-epoch grad-norm smoke test. r=8 default, sweep {4,8,16}. |
| 3 (enabler) | **R1 Gated replacement fusion** | 0.785 | 5 % | cheap unblock of F1; useful as warmup; alone unlikely to reach 0.79. |

**Reordered vs Research Agent (R1→R3→R2):** I keep R1 as a 30-min warmup smoke test, but the **main 0.79-hunting attempt is R3**, because (a) paper-promised and (b) highest single-seed P(≥0.79).

**Concrete recommendation.** Run R1 first as fusion-reform pipeline validation (verify gate entropy > 0.05 nats); then R3 as the main attempt with 3-seed protocol; R2 only if both miss 0.785.

## 3. Seed plan

Per-seed std σ ≈ 0.3 pp (K4 observation, task-brief assumption). For a 95 % two-sided CI claim of "method M beats anchor 0.7804 by δ pp", n ≥ (1.96σ/δ)²:

| Claim | Seeds needed |
|---|---|
| δ = 0.3 pp (combined ≥ 0.7834) | 4 |
| δ = 0.5 pp (combined ≥ 0.7854) | 2 statistically, **3 for stability** |
| δ = 1.0 pp (combined ≥ 0.7904) | 1 statistically, **3 for review robustness** |

**Plan:** R1 and R3 each on seeds {42, 7, 1234}; R2 on {42, 7, 1234, 2026, 2027} (5 seeds due to higher variance) plus 5-epoch grad-norm smoke per seed.

## 4. Top risk per direction

- **R1 — gate collapse.** Optimizer drives `g_ab → (1, 0, 0)` and R1 degenerates to emergnn-only. **Detect:** log batch-mean entropy of `g_ab` on val every epoch; if < 0.05 nats by ep 10 → gate is dead. **Mitigate:** entropy regularizer on g_ab during first 20 epochs; do NOT initialize the gate MLP at random — initialize last layer at zero so g starts uniform (1/3,1/3,1/3).
- **R2 — gradient explosion in the U·diag(g)·V chain.** Effective gradient gain through g scales as O(√r); at r=16+ without normalization, training diverges. **Detect:** print grad-norm every step for first 5 epochs. **Mitigate:** zero-init last layer of g(p) MLP so initial `W_r^t(p) = A_r^t` (vanilla EmerGNN); apply layer-norm on g(p) output.
- **R3 — empty-mediator mode collapse.** 70% of Path B pairs have m_pool=0; MLP_score risks learning two disjoint sub-functions and over-fitting one. **Detect:** evaluate combined AUROC separately on q1 (empty) and q2 (present) subsets; if q1 AUC drops below emergnn-only baseline 0.7405 by >1 pp, mode collapse confirmed. **Mitigate:** explicit `m_present_ab ∈ {0,1}` bit appended to pair_feat (1 extra dim, ~0 cost).

## 5. Falsifiable summary claims

1. **F1 ceiling claim.** No hyperparameter sweep over (β_c init, β_i init, count/i4 hidden width, dropout) within the v2i4 fusion class exceeds combined 0.788 on test_s2 seed42 across 5 seeds. Violation refutes the 0.794 independence ceiling estimate.
2. **R1 claim.** Single-seed combined ∈ [0.783, 0.787]. If < 0.778, gate collapsed → diagnose via entropy log.
3. **R2 claim.** With r=8, zero-init g, layer-norm: single-seed combined ∈ [0.783, 0.790]. If grad-norm exceeds 10× baseline at ep 1 → hypernet unstable.
4. **R3 claim.** With augmented input `[e, m_pool, pair_feat, feats22, m_present]` and h=64: single-seed combined ∈ [0.785, 0.792]. If < anchor 0.7804 → either over-parameterized (try h=32) or m_pool informationally redundant with pair_feat.

---

**Hard-stop rule for downstream ML Engineer.** If R1 + R3 both miss combined 0.785 across 3 seeds → escalate, do not run R2 blindly. Re-open whether the underlying branch AUCs themselves are saturated (need new features, not new fusion).
