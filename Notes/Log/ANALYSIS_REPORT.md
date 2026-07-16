# Round 4 — ANALYSIS_REPORT (ARIS analyze-iterate Phase A-1)

**Date.** 2026-06-02
**Round.** 1 (analyze-iterate skill cycle)
**Architecture under iteration.** D2 (meet-mask propagation), attempt 2 of 5
**Research Agent.** general-purpose subagent (codex MCP unavailable for ChatGPT account)
**Inputs verified.** `round4_backbone_diff_plan.md`, `round4_d1_seed42_observation.md`, `round4_d2_seed42_observation.md`, `_reviews/2026-06-01__d1_results__round1.md`, `_reviews/2026-06-01__d2_implementation__round1.md`, `round4_d2_vocab_mismatch_discovery.md`, `paper_writeup.md §3.X/§3.Y/§3.Z/§3.W1`, `v2i4_trainer.py:99-103`, `ITERATE_STATE.json`, plus run results.json files for K1/K3.

---

## Failure Mode Analysis

### Shared root cause (operationalized)

`v2i4_trainer.py:99-103` confirms softplus-gated additive fusion:
```
combined_logit = emergnn_logit + softplus(β_count)·MLP_count(c_ab) + softplus(β_i4)·MLP_i4(p_ab)
```
When a backbone-side change pushes new signal into `emergnn_logit`, the optimizer re-balances the three additive terms so what the readout already captured is suppressed at the head end. Combined ranking moves by only a fraction of branch lift — sometimes inverted.

### Branch-lift / combined-lift translation ratios (verified)

**Anchor v2i4** (`2026-05-29_22-02-27`, paper_writeup.md:296): combined 0.7804, emergnn 0.7405, count 0.6688, i4 0.6003, NLL 1.4723.

**D1 main** (`2026-05-30_19-05-21`, paper_writeup.md:297): combined 0.7770 (**−0.33 pp**), emergnn 0.7533 (**+1.28 pp**), count 0.6818 (+1.30), i4 0.5301 (**−7.01**). Backbone lift +1.28 pp → combined −0.33 pp. **Transfer ratio = −0.26 (inverted)**.

**D2 main** (`2026-06-01_15-46-08`, ITERATE_STATE.json:41-49): combined 0.7780 (**−0.24 pp**), emergnn 0.7344 (**−0.61 pp**), count 0.7172 (+4.84), i4 0.5506 (−4.97). Backbone itself REGRESSED; count head absorbed +4.84 pp.

**D2 K3 freeze-α_meet** (`2026-06-02_00-52-02`, results.json line 28-31, verified): combined **0.7843** (+0.39 pp over anchor), emergnn 0.7497, count 0.7229, i4 0.6023. **Turning the D2 mechanism OFF beats D2 main by +0.63 pp AND beats v2i4 by +0.39 pp.**

**D2 K1 shuf-mediators** (`2026-06-02_00-51-59`, results.json verified line 27-30): combined 0.7784 (essentially identical to D2 main). Shuffle-invariant — same falsification pattern as D1 K1.

Cross-reference: paper_writeup.md:241-248 documented v2i4 Stage 2 had +4.55 pp branch → +0.52 pp combined (transfer 11.4%). D1 and D2 are both well below even that.

### Three failure mechanisms (jointly responsible)

**F1. Additive-fusion saturation (load-bearing).** The softplus-gated additive sum is a hard ceiling on backbone→combined translation. Documented at Stage 2 (+4.55/+0.52), recurs in D1 (+1.28/−0.33) and D2 (−0.61/−0.24).

**F2. Shuffle-invariant capacity confound.** D1 K1 (combined +0.57 pp under drug→token shuffle) and D2 K1 (combined +0.04 pp under mediator shuffle) both show backbone mechanisms are not pair-specific. EmerGNN length-3 sum-aggregation is fundamentally a degree/density operator on the affected axes once readouts absorb identity.

**F3. Pair-conditional reach limited by KG bipartite schema.** D2 specific: Path B mediator universe has 70% zero-mediator pairs (round4_d2_vocab_mismatch_discovery.md:43-46). D2 q2 AUC 0.7218 vs q1 0.6767 — mechanism cannot lift the empty-mask majority.

D1 and D2 both hit F1 and F2; F3 is D2-specific.

---

## Literature References

(WebSearch on 2026-06-02; arxiv IDs / URLs verified by search; full-text verification deferred to Theorist Agent.)

**Pair-conditional / cold-start DDI:**
- **EmerGNN** (Zhang et al., Nat. Comput. Sci. 2023, arXiv:2311.09261) — our backbone.
- **KnowDDI** (Wang et al., 2024, arXiv:2311.15056) — subgraph-conditional reasoning; beats SumGNN for low-data DDI types.
- **Rethinking DDI as Generalizable Relation Learning** (arXiv:2601.15771, Jan 2026) — argues against generic GNN density augmentation for cold-start.
- **NASNet-DTI** (Zhong & Du, arXiv:2510.04126v1, 2025) — node-adaptive depth for cold-start DTI; per-pair adaptive routing beats fixed-depth.
- **SumGNN** (Yu et al., Bioinformatics 2021, DOI:10.1093/bioinformatics/btab207) — layer-independent self-attention per edge.

**Gated multimodal fusion replacing additive heads:**
- **Gated Multimodal Units** (Arevalo et al., arXiv:1702.01992) — classical sigmoid-gated convex combination.
- **IMF: Interactive Multimodal Fusion** (Li et al., WWW 2023, arXiv:2303.10816) — pair-wise interactive fusion; demonstrates additive-fusion bottleneck is removable.
- **IMKGA-SM** (arXiv:2301.02445) — sequence-modeling multimodal KG completion replacing parallel-head fusion.

**Molecular / KG init pretraining (relevant if D3 revived):**
- **Knowledge-aware contrastive heterogeneous molecular graph learning** (PLOS Comput. Biol. 2025, arXiv:2502.11711).
- **MMSA: Multi-Modal Molecular Representation Learning via Structure Awareness** (arXiv:2505.05877, 2025).

**Warming Up Cold-Start CTR with hypernetworks** (Wang et al., KDD 2024, arXiv:2407.10112) — hypernetwork-conditioned feature interaction precedent.

---

## Improvement Directions (3 ranked)

### R1 (top pick) — Gated REPLACEMENT fusion

**Mechanism.** Replace additive fusion with pair-conditional softmax/sigmoid gate:
```
g_ab = softmax(MLP(pair_feat))  ∈ Δ²
combined_logit = g_ab[0]·emergnn + g_ab[1]·count + g_ab[2]·i4
```

**Why it works.** Breaks F1: optimizer cannot freely redistribute mass across heads — gate is supervised by pair_feat. Breaks F2: under shuffle, gate input is corrupted, K1/K2 should now drop (real falsifier). Avoids F3 (orthogonal to mediator sparsity).

**Effort.** ~150 LOC. New `v3_gated_fusion_trainer.py` subclassing `_PerModeEmerGNN_V2I4`, override only `_combined_logit`. ~35 min single-seed run.

**Architectural-diff claim.** Weak alone (still readout-side), but **unblocks** subsequent backbone changes by removing the F1 ceiling. Treat as enabler.

**Citations.** arXiv:1702.01992 (GMU); arXiv:2303.10816 (IMF); arXiv:2301.02445 (IMKGA-SM).

### R2 — Hypernetwork-Conditioned Message Function (HCMF)

**Mechanism.** Pair-conditional per-relation weights at every propagation layer:
```
m_{u→v}^t = W_r^t(pair_feat) · h_u^t,   W_r^t(p) = A_r^t + U_r^t·diag(g(p))·V_r^t  (low-rank)
```
Path-flow propagation becomes intrinsically pair-specific.

**Why it works.** Breaks F1 (signal must flow through backbone; can't be routed around). Breaks F2 strongest of all options (shuf-pair-feat breaks every layer). Avoids F3 (no mediator-mask dependency).

**Effort.** ~250-300 LOC. New `emergnn_hcmf.py` + `v3_hcmf_trainer.py`. Run ~50-60 min (1.5-2× v2i4). Risk: param explosion → low-rank factorization mandatory.

**Architectural-diff claim.** STRONGEST. Backbone propagation fundamentally different from EmerGNN.

**Citations.** arXiv:2510.04126 (NASNet-DTI); arXiv:2407.10112 (hypernet for cold-start CTR); SumGNN.

### R3 — PMP-style REPLACEMENT readout (mediator-pooled, drop count + i4 heads)

**Mechanism.**
```
m_pool_ab = Σ_{u ∈ mediator(a,b)} attn(u | a,b) · h_u^T
combined_logit = MLP_score([emergnn_logit, m_pool_ab, pair_feat])
```
Drop β_count and β_i4 entirely. MLP_score replaces additive sum.

**Why it works.** Breaks F1 (no parallel additive paths). Partial F2 (still risks shuffle-invariance, but readouts gone so signal must be in m_pool). Mitigates F3 (falls back to emergnn alone on empty mask, no damage).

**Effort.** ~200 LOC. New `v3_pmp_trainer.py`. Reuses Path B mediator cache (sha16 `ac2b6b345f4264dd`). ~40 min run.

**Architectural-diff claim.** Medium. **Already promised in paper_writeup.md:269-273 as Section 4 PMP.**

**Citations.** arXiv:2303.10816 (IMF); arXiv:2311.15056 (KnowDDI); existing paper Section 4 commitment.

### Ranking summary

| Rank | Direction | Fixes F1 | Breaks F2 | Avoids F3 | Effort | Architectural diff |
|------|-----------|:--------:|:---------:|:---------:|-------:|:-------------------:|
| R1   | Gated replacement fusion | ✓ | ✓ pair-cond gate | ✓ | low (150 LOC, 35min) | weak alone, enabler |
| R2   | Hypernetwork message fn | ✓ | ✓ strongest | ✓ | high (300 LOC, 60min) | strongest |
| R3   | PMP replacement readout | ✓ | partial | partial | medium (200 LOC, 40min) | medium (paper-promised) |

Recommended order: R1 first (cheapest, unblocks F1), then R3 (paper-committed), then R2 (highest novelty payoff).

---

## Pre-Screen of Candidate Adjustments (D2.2–D2.6)

| ID | Verdict | Promote to | Reason |
|----|---------|------------|--------|
| **D2.2** drop-i4-head | **Reject (standalone)** | subsumed into R1 | D1 K3 already ran combined 0.7779; D2 K3 already 0.7843. F1 not addressed. |
| **D2.3** layer-0 only α | **Reject** | cheap ablation only | Optimizer already concentrated α at layer 0 (`ITERATE_STATE.json:73-77` α_ep100 = [0.147, −0.033, 0.0]); change is ~0. |
| **D2.4** gated pair-conditional α | **Conditional** | combine with R1 | Strong literature (GMU, IMF, NASNet-DTI). Fixes F2 only; F1 still ceiling. Expect 0.78-0.785 solo. |
| **D2.5** mediator-pool replace count | **PROMOTE → R3** | this IS R3 (PMP) | Aligned with paper Section 4 commitment; high-priority next attempt. |
| **D2.6** expand mediator universe | **Defer** | later if needed | F3-only fix; F1/F2 still ceiling. Risk: leakage guards needed (per Compound-leakage history). |

---

## Flags

### For Theorist Agent (Phase A-2)
1. **F1 quantitative ceiling.** Given current branch AUCs (emer 0.7405, count 0.6688, i4 0.6003, combined 0.7804), what is max combined AUC under softplus-gated additive fusion? Use Stage-2 documented 11.4% transfer ratio (paper:241-248) as anchor. Suggested ceiling estimate: ~0.79 absent fusion reform; ~0.81-0.82 with R1/R3.
2. **R1 dominance proof.** Does learned 3-way gate provably dominate softplus-gated additive sum when branch logits are not rank-aligned?
3. **R2 param budget.** Smallest rank factorization that keeps hypernet-generated `W_r^t` expressive enough for length-3 path flow on drugbank 5-bucket KG; target ≤ 0.5× v2i4 total params.
4. **R3 identifiability.** Does concat+MLP strictly dominate additive sum, or only equivalent under certain init? Need separation argument for paper Section 4.

### For ML Engineer (Phase A-3)
1. **R1 P0 (150 LOC, 35 min).** `v3_gated_fusion_trainer.py` subclassing `_PerModeEmerGNN_V2I4`. Override only `_combined_logit`. Verify gate input does not leak labels; K1/K2/K3 controls now meaningful because gate is pair-conditional.
2. **R3 P0/P1 (200 LOC, 40 min).** `v3_pmp_trainer.py`. Drop `_build_aux_head`; mediator-pool + MLP_score. Reuse Path B parquet sha16 `ac2b6b345f4264dd`. Decide pooling (mean / softmax-attn / PMA).
3. **R2 P1 (300 LOC, 60 min, higher risk).** `emergnn_hcmf.py` + `v3_hcmf_trainer.py`. Low-rank factorization. 5-epoch gradient-norm smoke test before 100-epoch.
4. **All three ship with `--deterministic`** by default (D1 K4 / D2 CP-2 R1 process lesson).

### For Reviewer Agent (Phase A-4)
- CP-3 PASS gate for R1: combined ≥ 0.785 single seed AND K1/K2 drop ≥ 1.5 pp from main (now a real falsifier because gate is pair-conditional).
- R3 needs new control template: "drop mediator pool" ablation analogous to D1 K3.

---

## Summary

Failure analysis confirms shared root cause: D1 and D2 both hit (F1) additive-fusion saturation + (F2) shuffle-invariant capacity confound. D2 additionally hits (F3) Path B mediator sparsity. The K3 freeze-α run at combined 0.7843 is direct evidence that the v2i4 fusion structure itself — not D2's mechanism — is the binding constraint right now.

Three ranked directions: **R1 gated replacement fusion** (cheap enabler, ~150 LOC), **R2 hypernetwork message function** (strongest architectural-novelty, ~300 LOC), **R3 PMP-style replacement readout** (matches paper Section 4 promise, ~200 LOC).

Pre-screen of D2.2-D2.6: promote D2.5 to R3; reject D2.2 and D2.3; combine D2.4 with R1; defer D2.6. Recommended execution: R1 → R3 → R2.

## Theorist Assessment

**Date.** 2026-06-02
**Theorist agent.** general-purpose Agent (codex MCP unavailable for ChatGPT account, 2026-06-02 verified; per CLAUDE.md "Code review 报告归档规范" Reviewer field rule — Independent reviewer: NOT INVOKED, fallback agent acting as Theorist).
**Inputs verified.** `v2i4_trainer.py:99-103,208-217` (additive fusion + softplus gates), `Code/baseline/emergnn/model.py:60-95,116-180` (EmerGNN backbone: n_dim=64, L=3, all_rel=2·n_base_rel+1; for drugbank 5-bucket n_base_rel=5 → all_rel=11), `paper_writeup.md:222-248` (Stage 1/2 branch vs combined transfer), `ITERATE_STATE.json:41-77` (D2 runs + α_meet trajectory), `Code/runs/2026-05-29_22-02-27__run_v2i4__.../train.log` (v2i4 anchor config: epochs=100, batch=32, v2i4_hidden=32).

**Method.** Closed-form / order-statistic arguments where possible; small-h MLP universal-approximation arguments where not. No empirical fitting. All numbers labelled either "verified", "computed from verified inputs", or "estimate (assumption noted)".

---

### T1. F1 quantitative ceiling under softplus-gated additive fusion

**Setup (assumptions).** Let `s = e + β_c·c + β_i·i` with `β_c, β_i ≥ 0` (softplus-positive). Holding the three branch AUCs at their verified anchor levels e≈0.7405, c≈0.6688, i≈0.6003, treat the per-pair branch scores as random variables on positives (P) and negatives (N). AUC is `P(s_P > s_N)` over the product distribution.

**Sub-question (a). Optimistic ceiling — branches perfectly rank-aligned.**
If on every pair the three branch logits induce the SAME pairwise ordering (full Kendall τ = 1 between any two branches restricted to {pos, neg} pair comparisons), then `1{s_P > s_N} = 1{e_P > e_N} = 1{c_P > c_N} = 1{i_P > i_N}` and any positive linear combination yields AUC = max(e, c, i) = 0.7405. This is a HARD lower-than-anchor ceiling and contradicts the observed combined 0.7804. → Empirical refutation: the branches are NOT in perfect rank agreement; the gain comes from disagreement on a non-empty subset.

**Sub-question (b). Pessimistic ceiling — branches independent.**
Treat branch errors as conditionally independent given (pos/neg) label. Each branch's AUC defines a margin distribution; sum-of-independent-margins is a convolution. By Hanley-McNeil-type approximation, for three independent ROC-equivalent normal margin models with AUCs a₁, a₂, a₃, AUC of optimal weighted sum is bounded by `Φ(√(d₁² + d₂² + d₃²))` where dₖ = √2·Φ⁻¹(aₖ).
- d_emer = √2·Φ⁻¹(0.7405) = √2·0.645 = 0.912
- d_count = √2·Φ⁻¹(0.6688) = √2·0.437 = 0.618
- d_i4 = √2·Φ⁻¹(0.6003) = √2·0.254 = 0.359
- combined d* = √(0.832 + 0.382 + 0.129) = √1.343 = 1.159
- AUC* = Φ(1.159/√2) = Φ(0.820) ≈ **0.794**

**Sub-question (c). Best estimate given Stage-2 transfer 11.4%.**
Verified anchor combined 0.7804 (D1 Round-4 line 23) sits 1.0 pp below the independence ceiling 0.794 — so v2i4 has ALREADY consumed most of the independence headroom. The Stage-2 transfer (+4.55 pp branch → +0.52 pp combined = 11.4%) was measured holding emergnn fixed at 0.7363. Apply that same transfer to the residual (0.794 − 0.7804) = 1.36 pp of headroom: even an oracle branch improvement of +X pp on count or i4 returns only ≈0.114·X pp on combined, so to gain 1 pp combined you'd need ≈9 pp on a branch. Branches are already at their data-attainable levels under fixed Morgan / 22-d count input — branch ceiling itself is plausibly ≤ 0.74 (i4) / 0.72 (count) without new feature engineering.

**Theorist verdict T1.**
- Optimistic ceiling under THIS fusion: 0.7405 (rank-aligned case, useless lower envelope).
- Pessimistic/independence ceiling: **≈ 0.794** with current branch AUCs.
- Realistic ceiling honoring the 11.4% transfer: **0.785 ± 0.003 single seed** absent fusion reform. The target 0.79 single-seed is at the very edge of the additive-fusion physical envelope, and only reachable if a backbone change ALSO lifts emergnn branch ≥ +1.5 pp; D1/D2 evidence shows backbone gains do NOT translate at 1:1.
- **Falsifiable claim.** Under softplus-gated additive fusion with branches frozen at current levels, no hyperparameter sweep over (β_c init, β_i init, count/i4 hidden width, dropout) will exceed combined 0.788 on test_s2 seed42 across 5 seeds. If this is violated, my ceiling estimate is wrong.

---

### T2. R1 gated-fusion dominance over softplus-gated additive

**R1 mechanism (from ANALYSIS_REPORT §R1 lines 75-79).** `g_ab = softmax(MLP(pair_feat))`, `combined = g_ab[0]·e + g_ab[1]·c + g_ab[2]·i`.

**Claim.** R1 weakly dominates softplus-gated additive as a function class, and STRICTLY dominates whenever the optimal per-pair weight vector is not constant across pairs.

**Proof sketch (dominance).** Define the function class
- F_add = { s(e,c,i) = e + β_c·c + β_i·i : β_c, β_i ≥ 0 }
- F_R1 = { s(e,c,i; p) = g_0(p)·e + g_1(p)·c + g_2(p)·i : g ∈ Δ², MLP-realizable }

After non-degenerate rescaling (Δ² simplex weights vs unconstrained nonneg β), F_add ⊂ F_R1 iff for every (β_c, β_i) there is a constant g̃ ∈ Δ² and a positive scalar λ such that λ·(g̃₀, g̃₁, g̃₂) = (1, β_c, β_i). Take g̃ = (1, β_c, β_i) / (1+β_c+β_i) and λ = 1+β_c+β_i. Then a constant-MLP (zero hidden layers, bias-only) on the simplex realizes F_add up to a per-pair-constant positive rescaling which does not change ranking → does not change AUC. So **F_add ⊆_rank F_R1**.

**Strict dominance criterion.** Let p* (a, b) ↦ (g₀*, g₁*, g₂*) be the Bayes-optimal pair-conditional weighting. R1 strictly beats additive iff p* is non-constant on the population of test pairs. A sufficient condition: there exists a pair (a,b) where Var(e | label, pair-feat) ≪ Var(c | label, pair-feat) (e is confident, c is noisy) AND another pair (a', b') with the opposite, and pair_feat distinguishes them. Empirically: D2 K3 showed emergnn 0.7497 on the FULL 1919 test pairs but Path B q2-mediator-present subset (30%) showed emergnn 0.7218 vs q1-empty 0.6767 (round4_d2_seed42_observation flagged this). This is direct evidence that per-pair branch confidence varies — p* is provably non-constant → R1 strictly dominates.

**Collapse condition.** R1 collapses back to F_add iff MLP(pair_feat) is constant on the input distribution. Sufficient prevention: ensure pair_feat has rank ≥ 2 across train pairs (verified — 13 features, std-floored at 1e-3, normalizer fit log shows nonzero variance on all 13). Additional safety: do NOT freeze MLP at random init (let optimizer move it).

**Expected lift over 0.7804.** Lower bound 0 (F_add ⊆ F_R1, AUC is a sup over the class so cannot decrease in expectation given enough data). Realistic estimate: the K3 finding (+0.39 pp from determinism alone) plus the per-pair confidence variation we documented suggests R1 captures **+0.3 to +0.7 pp = 0.783–0.787 single seed**. Hitting 0.79 requires the MLP gate to ALSO correctly route on q1-empty pairs (where i and c provide no signal — gate must put g₀ ≈ 1 there). This is the binding skill.

**Theorist verdict T2.** YES strictly dominates under stated condition (non-constant Bayes weights, which empirically holds). Expected combined ≈ **0.785 ± 0.004 single seed**. Hitting 0.79 is plausible but not guaranteed. **Falsifiable claim.** If R1 single-seed lands below 0.778 (i.e. worse than additive), then either (a) the gate MLP is over-regularized into a constant, or (b) the optimizer collapses to F_add by setting g₀ ≈ 1 always. Diagnosis: log entropy of g_ab over a held-out batch; entropy < 0.05 nats → gate collapsed.

---

### T3. R2 hypernet parameter budget

**Verified base (file:line).** `Code/baseline/emergnn/model.py:78` defines `self.rel_kg = nn.ModuleList([nn.Embedding(self.all_rel, n_dim) for _ in range(self.L)])`. For drugbank 5-bucket: n_base_rel=5 → all_rel=2·5+1=11 (model.py:59), n_dim=64 (default), L=3.

**v2i4 trainable param count (computed from verified architecture):**
- EmerGNN Went (1024×64 + 64) = 65,600
- EmerGNN Wr (2·64×1 + 1) = 129
- Per-layer × 3:
  - rel_kg embedding: 11 × 64 = 704
  - linear (64×64 + 64) = 4,160
  - relation_linear (128×5 + 5) = 645
  - attn_relation (5×11 + 11) = 66
  - Per-layer total = 5,575; × 3 = 16,725
- CountPlusI4Head: count_mlp (22→32→1) ≈ 769; i4_mlp (13→32→1) ≈ 481; raw_beta + raw_beta_i4 = 2 → 1,252
- **v2i4 total ≈ 83,706 params** (estimate from architecture, NOT directly verified in log — train.log does not print param count).

Constraint: hypernet overhead ≤ 0.5 × 83,706 = **41,853 params**.

**Per-layer per-relation hypernet overhead (low-rank factorization rank r).**
Original per-layer rel embedding: 11 × 64 = 704 (one matrix per layer, shared across pairs).

R2 form (from ANALYSIS_REPORT lines 91-94): W_r^t(p) = A_r^t + U_r^t · diag(g(p)) · V_r^t. But the natural translation to EmerGNN's per-layer rel_kg embedding is:
- A^t (base, shared across pairs): 11 × 64 = 704 / layer
- U^t: 11 × r / layer (left low-rank factor over relations)
- V^t: r × 64 / layer (right low-rank factor over n_dim)
- g(p): MLP(pair_feat_dim=13 → r), say one hidden layer of width h=32: (13·32 + 32) + (32·r + r) = 448 + 33r
- Per layer hypernet ADD = 704·(no, the A^t already exists as the original rel_kg) — count only the EXTRA: 11r + 64r + (448 + 33r) = 108r + 448

Across L=3 layers (assume g(p) shared across layers, U/V per layer): extra params = 3·(11r + 64r) + 448 + 33r = 225r + 33r + 448 = 258r + 448.

**Budget solution.** 258r + 448 ≤ 41,853 → r ≤ 160. With per-layer g (not shared): 3·(75r + 33r) + 448 = 324r + 448 ≤ 41,853 → r ≤ 128.

**Minimum r recommendation.** r ≥ 4. Rationale: rank 1 makes W_r^t(p) − A_r^t a rank-1 outer product, gate g(p) acts as a single scalar — too restrictive to express different relation-specific modulations. r=4 lets g(p) modulate ≤ 4 independent relation directions (we have 11 relations; rank 4 covers ~36% of the relation slot space, comparable to relation_linear's 5-D bottleneck which is the paper's own design choice for n_base_rel=5).

**Maximum r recommendation.** r ≤ 32. Above r=32, two pathologies: (1) extra params 258·32 + 448 = 8,704 ≈ 10% of base — already substantial relative to backbone; (2) gradient norm scaling. The chained product diag(g)·V means ∂L/∂g_k = (V_k · ∂L/∂W) · U_k where the rank-r sum amplifies gradient magnitude as O(r). Empirically Xavier-init of U_r^t at 1/√(11) and V_r^t at 1/√r keeps initial output magnitude O(1), but the gradient through g has effective gain O(√r) — risk of vanishing for r > 64 (gate gets noisy signal), risk of exploding for r < 2 (gate must absorb all variance).

**Recommended sweep: r ∈ {4, 8, 16}.** First run r=8 single seed; if combined ≥ 0.785 try r=16; if regressing try r=4. 5-epoch gradient-norm smoke test BEFORE 100-epoch (CLAUDE.md training conventions support this).

**Gradient-norm risk identification.**
- Explosion: when g(p) outputs unbounded (no softmax/sigmoid). Mitigation: pass g through tanh or layer-norm, OR initialize MLP last-layer weights at zero so initial g=0 → W_r^t(p) = A_r^t (initial behavior identical to vanilla EmerGNN, then warmup).
- Vanishing: when g(p) collapses to constant (input pair_feat too low-rank). Mitigation: pair_feat must include continuous components — verified (jaccard_overall is continuous, lines 76-80 in v2i4_trainer.py).

**Theorist verdict T3.** r=8 default, sweep {4,8,16}. Overhead at r=8 = 2,512 params = **3.0% of v2i4 total** — well under 0.5× budget. Zero-init last layer of g(p) MLP for safe warmup. **Falsifiable claim.** If r=8 single-seed grad-norm exceeds 10× baseline at epoch 1, hypernet is unstable — reduce r or add layer-norm. If r=8 combined < 0.78, hypernet has not gained traction — try r=16 or warmup schedule on g.

---

### T4. R3 PMP identifiability

**R3 form (ANALYSIS_REPORT lines 108-111).** `combined = MLP_score([e, m_pool_ab, pair_feat])` with hidden dim h. Drops β_c·c and β_i·i.

**Claim 1: R3 ⊇ v2i4 as a function class.** Take MLP_score with one hidden ReLU layer of width h ≥ k+2 where k = dim(pair_feat). The first hidden unit can compute identity on e (one weight 1, others 0, bias 0). The remaining h−1 units can approximate `β_c·c_logit(feats22)` and `β_i·i_logit(pair_feat)` arbitrarily well by ReLU universal approximation (Cybenko 1989). The output linear layer sums them. So for any (β_c, β_i, count_mlp, i4_mlp) in v2i4, there exists a parameterization of R3 that reproduces v2i4's score function up to ε.

Caveat: m_pool_ab replaces feats22 input to count head. m_pool is computed from mediator embeddings (PMP) — these include information NOT in feats22 (typed-mediator embedding richness > 22-d count). So R3 trades feats22 input for m_pool input. **R3 is a function CLASS that under proper input augmentation contains v2i4 if both m_pool and the original feats22 are concatenated**. Without feats22 in the input, R3 is NOT a superset and could underperform on pairs where 22-d count carries information m_pool does not.

**Engineering recommendation.** Pass `[e, m_pool_ab, pair_feat, feats22]` to MLP_score (concatenate all four). This guarantees R3 ⊇ v2i4 strictly.

**Claim 2: when does R3 STRICTLY beat additive?** When the optimal score function s*(e, m_pool, pair_feat) has a non-linear cross-term in (e, m_pool) or (e, pair_feat). Concretely: if for some pair (a, b), `s*(e, m_pool, p) ≠ α(p)·e + β(p)·m_pool + γ(p)·p` for ANY scalar functions α, β, γ — i.e. the optimal mapping is intrinsically multiplicative or non-monotone in cross-features — then a linear-additive head cannot match it but an MLP can (Hornik 1991). Empirical signature: emergnn confidence (e magnitude) interacts with mediator richness (|m_pool|). High-e + high-|m_pool| pairs should get amplified confidence, low-e + low-|m_pool| pairs should get DAMPENED confidence (because both signals are weak). Additive cannot dampen — it just adds.

**Empty-mask risk (F3 from §Failure Analysis).** 70% of Path B pairs have empty mediator set → m_pool_ab = 0 vector. MLP_score must handle this. Two failure modes:
- (a) MLP_score learns `s ≈ MLP(e)` for the 70% empty subset, then fits to the m_pool-present 30% — risks two-mode optimization, slow convergence.
- (b) MLP_score introduces an implicit mask-aware bias: since m_pool=0 is a fixed point, the ReLU(W·[e, 0, p]) only sees e and p, so behavior on empty-mask is determined entirely by the (e, p) marginal. This is fine IF p (pair_feat) is non-trivial on empty-mask pairs (it is — pair_feat is computed from i4 typed-sets, not from mediator coverage).

**Mitigation.** Add explicit mask bit `m_present_ab ∈ {0,1}` as 14th pair_feat dim. Lets MLP_score learn a different sub-function for empty vs present pairs without entanglement. This is a 1-bit addition, ~0 param cost.

**Theorist verdict T4.** Formal claim:
> **R3 ⊇ v2i4 in function class if input includes feats22 + pair_feat alongside m_pool_ab.** R3 strictly beats v2i4 iff (a) m_pool carries information beyond feats22+pair_feat (verified: mediator embeddings include PubMedBERT text representations, paper:269-273), AND (b) optimal score has non-linear cross-term between emergnn confidence and mediator richness (plausible from paper §3.Z Finding 2). Add explicit mask bit for empty-mediator safety.

**Expected lift.** Realistic estimate +0.5 to +1.0 pp over 0.7804 = **0.785–0.790 single seed**, conditional on Path B mediator universe being non-trivially informative (currently 30% of pairs have nonzero coverage; if expanded via D2.6, could rise).

**Falsifiable claim T4.** If R3 with concatenated inputs `[e, m_pool, pair_feat, feats22, m_present]` lands below v2i4's 0.7804 single seed, then either (a) MLP_score is over-parameterized and overfits (try h=32 instead of 64), or (b) m_pool offers no additional signal beyond pair_feat (Path B is informationally redundant — falls into F3 territory).

---

### Cross-cutting questions

**Q-A. P(combined ≥ 0.79 single seed) ranking.**

Using point estimates above with 1σ uncertainty (assumed σ ≈ 0.3 pp from D2 K4 observation; user-stated σ in task brief):
- R1: mean 0.785, σ 0.3 pp → P(≥ 0.79) = P(Z ≥ (0.79−0.785)/0.003) = P(Z ≥ 1.67) ≈ **5%**
- R2: mean 0.787 (warmup risk-discounted from 0.79 ceiling), σ 0.4 pp (higher variance from hypernet) → P(≥ 0.79) = P(Z ≥ 0.75) ≈ **23%**
- R3: mean 0.788 (mass on PMP delivering), σ 0.35 pp → P(≥ 0.79) = P(Z ≥ 0.57) ≈ **28%**

**Q-A verdict.** R3 ≥ R2 > R1 by single-seed P(≥0.79). All three are below 50% chance — multi-seed runs essential for any meaningful claim at 0.79.

**Q-B. Seeds needed for 95% CI claim of "method M beats anchor 0.7804 by δ pp".**
Two-sided 95%: n ≥ (1.96·σ/δ)². With σ=0.3 pp:
- δ = 0.3 pp (i.e. claim 0.7834): n ≥ (1.96·0.3/0.3)² = 3.84 → **4 seeds**
- δ = 0.5 pp (claim 0.7854): n ≥ (1.96·0.3/0.5)² = 1.38 → **2 seeds** (but ≥3 for one-sided lower bound at 95% with stability)
- δ = 1.0 pp (claim 0.7904 i.e. cross 0.79): n ≥ (1.96·0.3/1.0)² = 0.35 → **1 seed sufficient** statistically, but **report 3 seeds for review robustness**

**Recommended seed plan.** R1 and R3: 3 seeds each (42, 7, 1234) — covers δ=0.5pp claim. R2: 5 seeds (R1 + 2026, 2027) — higher variance, plus the 5-epoch grad-norm smoke test before each long run.

---

### Final Theorist ranking of R1/R2/R3

| Rank | Direction | E[combined] | P(≥ 0.79) | Effort × Risk | Theorist confidence in feasibility |
|------|-----------|:-----------:|:---------:|:-------------:|:------------------------------------:|
| **1 (paper)** | **R3 PMP** | **0.788** | **28%** | medium / medium | **high — paper Section 4 commitment + R3 ⊇ v2i4 under input augmentation, formally safe** |
| **2 (novelty)** | **R2 Hypernet** | **0.787** | **23%** | high / high | medium — strongest architectural diff but grad-norm risk; do 5-epoch smoke first |
| **3 (enabler)** | **R1 Gated fusion** | **0.785** | **5%** | low / low | **high feasibility, low payoff alone**; treat as warmup / unblock for R3 |

**Reasoning for re-ordering vs Research Agent's R1→R3→R2.**
- Research Agent ranked R1 first on EFFORT (cheapest). I rank R3 first on EXPECTED PAYOFF given:
  (a) R3 is paper-promised (Section 4) — running it now de-risks paper claims.
  (b) R3 ⊇ v2i4 under augmented inputs is the most formally clean superset relationship of the three.
  (c) P(R3 ≥ 0.79 single seed) ≈ 28% vs R1's 5%.
- Recommend RUN R1 FIRST anyway as the **30-min smoke test** that validates the fusion-reform pipeline and gate-entropy diagnostics. Then promote to R3 as the main 0.79-hunting attempt. R2 only if both R1 and R3 fail.

**Top theorist risks.**
- R1: gate collapses to (1,0,0) — diagnose via gate entropy < 0.05 nats.
- R2: gradient explosion in U/V chain — mitigate with zero-init of g(p) last layer.
- R3: empty-mask 70% pairs collapse MLP into single-mode emergnn pass-through — mitigate with explicit m_present bit + ablation.

