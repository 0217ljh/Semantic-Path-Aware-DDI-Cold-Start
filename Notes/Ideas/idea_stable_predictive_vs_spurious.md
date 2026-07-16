# Idea: Stable-Predictive vs Stable-Spurious support decomposition (cold-start DDI)

2026-06-30. New line extending the OOD-support-degradation result (see
[[spmn_v2_autoresearch_plus1pt]]). Co-designed with codex; references FGCN-DKS (ICLR 2026,
Wang et al., dual knowledge separation) for the loss form + perturbation environments.

## Core novelty (one sentence)
Prior KG/DDI work splits structure into **general (stable) vs specific (drug-specific/variant)**.
We split the GENERAL/stable part ITSELF into two:
> OOD-stable support decomposes as S_stable = S_mech ⊔ S_hub, with I(Z_mech;Y)>0,
> I(Z_hub;Y)≈0, BOTH invariant across support-degradation environments.

Key theoretical point standard invariant learning misses: **stability across perturbation ⇏
usefulness for y**. HUB mediators (high-degree, connected to almost every drug) survive any
support-sparsification view → they are "invariant" by FGCN-DKS's criterion, yet
label-uninformative (present for + and − pairs alike). Pure cross-env invariance WRONGLY keeps
them. The novelty is the "stable informative vs stable nuisance" split, not "stable vs variant".

## What we borrow from FGCN-DKS (faithful, eq refs)
- Perturbation ENVIRONMENTS (§3.2): E perturbed views per graph = our OOD support-degradation views.
- MASK separation (Eq 1-4): mask M → Ā=M⊙A (keep) vs Ã=(1-M)⊙A (complement) + dual encoders.
- LOSS shape (Eq 5-8): cross-env invariance (L_env) + push-apart separation (L_div via inverse
  distance) + reconstruction. Decomposition opt (Eq 16): min ||G-(Ginv+Gvar)||²+λ||Ginv||².
We DIVERGE: their invariant=keep; ours adds a label-uninformativeness constraint on the
complement so the stable-but-spurious mass is broken off (their loss cannot do this).

## The new objective (codex, math-precise)
Mediator gate m_i = ε+(1-ε)σ(g_φ(x_i)), x_i = structural-only [type, log1p(deg),
deg_rank_within_type, rel_a, rel_b, d_a, d_b] (NO drug id → no leakage). Hub weight = 1-m_i.
Per type t, env e, masked attention pools (inject ±log gate into within-type softmax):
  z_M = Σ softmax(s_i+log m_i)·h_i ;  z_H = Σ softmax(s_i+log(1-m_i))·h_i
Heads f_M(z_M), f_H(z_H), f_J([z_M,z_H]). Loss:
  L = L_pred + λ_inv L_inv + λ_hub L_hubnull + λ_sep L_sep + λ_bud L_bud + λ_bal L_bal
- L_pred = mean_e BCE(f_M(z_M^e), y)  [+ small η·BCE(f_J,y)]   (mechanism is sufficient)
- L_inv  = Σ_{c∈{M,H}} mean_e ||z_c^e − z̄_c||²                (BOTH stable across views)
- L_hubnull = GRL adversary q_ψ on z_H: min_θ max_ψ −BCE(q_ψ(z_H),y) + ρ·KL(Bern(f_H(z_H))‖Bern(π))
  (π = train prevalence) ← THE NEW INGREDIENT: z_H stable but label-uninformative
- L_sep  = ||Cov(Z_M, Z_H)||_F²    (decorrelation, more stable than repulsion)
- L_bud  = (mean_i m_i − τ)²  ;  L_bal = pairwise hinge keeping mean m_p in [α,β]
codex picks GRL+prior-KL over MI-estimator (high-variance, fragile at 3 seeds) and over
pure prior-MSE (too weak). Optimize min_{θ,φ,heads} max_ψ L.

## Distinct from our corridor −log1p(deg)? YES (mathematically)
Corridor = fixed monotone prior on one scalar. New obj constrains I(z_H;y)≈0 while keeping z_H
stable — catches low-degree-but-ubiquitous nuisance (corridor misses) and keeps high-degree-
but-predictive mediators (corridor wrongly suppresses). Caveat: optimizer MAY still realize it
mostly by degree → must TEST that degree is/ isn't a sufficient statistic.

## Failure modes + guards (codex)
m→1 (hub empty, null vacuous) → L_bal + τ<1; m→0 (joint head rescues) → L_pred on z_M only,
η small; mechanism relearns count (our collapse) → count-residualize z_M = Σαh/(Σα+δ),
match view retained-count bins; hub leaks conditional on mech → L_sep + report ΔAUC of adding
hub; view-gen leakage (sparsify removes low-deg more) → per-type degree-matched retention;
adversarial instability vs epoch-1 collapse → warm-up λ_hub, shallow adversary, stop-grad z_H
into shared encoder first; tiny support → analyze only medium/high-support pairs.

## codex honest verdict
- Genuinely new, defensible decomposition + objective. Strong THEORY contribution.
- Metric: probably NOT a big win over a plain gate; adversarial min-max is fragile at epoch-1
  collapse + 3 seeds. Sell as principled correction to invariant-learning for support-based
  cold-start DDI, gains secondary.
- Plain structural gate (m_i + V-REx + budget + stability, NO hub-null) = the performance
  ANCHOR / control. Build it FIRST.

## Falsification experiment (smallest that can kill the hypothesis)
Train (1) plain gate, (2) split model. Freeze reps, fit probes on held-out train pairs:
A: y from z_M (should be HIGH) · B: y from z_H (should be ≈prior) · C: degree/count from z_M,z_H
(z_H should be MORE degree/count-loaded). Supports claim if: z_H near-prior for y, z_H predicts
degree≫z_M, adding z_H to z_M doesn't improve held-out AUC, and m_i NOT a monotone fn of degree
(R² of m_i from degree-only clearly < saturation). Falsifies if any of these fail.

## REFINEMENT (user, 2026-06-30): margin/overlap, NOT a hard node mask
A hard mediator-level partition (z_M in / z_H out) is WRONG: the discriminative unit is the
PATH, not the node. A node can be in BOTH a useful path and an over-general path, so forcing it
entirely to mech-or-hub damages the useful path. So: **both branches retain the FULL general
support; differentiate only at the MARGIN (distinctive, non-overlapping residual of each path),
softly.** "Over-general" signal = path REDUNDANCY / node-overlap with many other paths, NOT
degree (a path-set property → genuinely not the corridor).

### Revised formulation (codex) — split CONTRIBUTION, not membership
Per pair p, mediator embedding h_i. Pair prototype μ_p = Σα_i h_i over FULL support. Decompose:
h_i^∥ = Proj_{μ_p}(h_i) (common/redundant), h_i^⊥ = h_i−h_i^∥ (distinct/marginal). Soft gate
m_i = ε+(1-ε)σ(−γ1·r_sig −γ2·r_rep_loo +b) modulates ONLY the residual:
  z^C = Σ β_i^C h_i^∥           (common branch; full support, β^C∝exp(s_i))
  z^D = Σ β_i^D · m_i · h_i^⊥   (distinct/mechanism branch; full support, β^D∝exp(s_i+log m_i))
Redundancy descriptors (NOT degree):
- r_sig = within-pair signature dup = #{j≠i∈S_p:(t,rel_a,rel_b)_j=(...)_i}/(|S_p|−1)
- r_rep_loo = cos(h_i, μ_{p,−i})  (leave-one-out prototype cosine; high=generic, low=distinct)
- log1p(deg) only as NUISANCE control; claim must not depend on it.

### Revised loss (codex). GRL hub-null now on the COMMON channel z^C (not a removed node set)
min_{θ,φ,f_D,f_C,f_J} max_ψ:  L_pred(z^D→y BCE) + λ_inv(inv(z^D)+λ_C·inv(z^C)) cross-view
+ λ_null·[−BCE(q_ψ(z^C),y) + ρ·KL(Bern(f_C(z^C))‖Bern(π))]  (z^C stable but label-NULL)
+ λ_sep·cos²(z^D,z^C) orthogonality + λ_mass·(distinct residual-energy frac − τ)² + λ_bal hinge.
Falsifiable: I(z^D;y)>0, I(z^C;y)≈0, both stable across views, m_i NOT monotone in degree.

### codex minimal FIRST version (isolates hypothesis)
z^C=Σβ_i^C h_i^∥ ; z^D=Σβ_i^D·m_i·h_i^⊥ ; m_i from r_sig+r_rep_loo ONLY (no degree first).
L = L_pred(z^D)+λ_inv(inv^D+λ_C inv^C)+λ_null·GRL(z^C)+λ_sep+λ_bal. If z^C still carries label
→ decomposition wrong / projection too weak. Non-reducibility test: R²(m_i~deg) vs
R²(m_i~deg+r_sig+r_rep_loo); within-degree-bin association.

## Plan (anchor-first, per user)
1. **Anchor / control**: plain soft mediator gate reweighting the existing pool (no ∥/⊥ split,
   no GRL), gate-input ablation **degree-only vs degree+redundancy(r_sig,r_rep_loo)** + V-REx
   across views + budget. Tests "is there gate signal beyond degree" cheaply & stably.
2. **Margin/overlap method** (codex minimal first version) — if anchor + falsification probe
   show a separable stable-redundant mass.
## Loop log (ARIS auto-research, codex review each round)
- R-anchor1 (gate degree vs full; V-REx 0.5 + budget 0.1@τ0.7; locked det+cw0.1+bf; 3 seeds,
  bf_val selector): no-gate 0.7673/test0.7765 · gate-degree 0.7678/0.7754 · gate-full
  0.7685/0.7760. **full−degree bf_val +0.0008 (3/3 +.0007/.0005/.0011), test +0.0006**;
  full−no-gate val +0.0012 but test **−0.0005**. codex: weak directional evidence of signal
  beyond degree, near noise floor; gate too BLUNT (budget τ0.7 drops 30% mass unjustified).
- R-anchor2 ⏳ codex pick: **gate-full, budget REMOVED (weight 0), V-REx 0.5 kept**. Decision
  rule: must beat no-gate on bf_val by ≥~+0.001 with 3/3 → else STOP this line and pivot
  UPSTREAM to support-quality/path-selection (where >0.1pt still plausible), not downstream
  reweighting. Honest ceiling of this line: sub-0.1pt.

- R-anchor2 ✅ gate-full budget REMOVED, V-REx 0.5: bf_val 0.7688 (+0.0015 mean vs no-gate
  but only 2/3 per-seed), test 0.7763 (still < no-gate 0.7765). codex stop-rule (≥+0.001 3/3)
  NOT met → downstream-reweighting gate line CLOSED.
- R-upstream1 ⏳ codex pick: UPSTREAM hard prune of top-15% most signature-redundant (r_sig)
  pooled mediators per pair, deterministic ALL splits (no leakage), struct counts unchanged
  (`--prune-pool-frac 0.15`). Tests if redundant support hurts by ENTERING the pool. Locked
  pipeline det+cw0.1+bf, 3 seeds. Must beat no-gate test 0.7765 (or bf_val 0.7673 3/3) → else
  redundancy axis CLOSED → fallback = expand support with a NEW path family/relation schema
  (codex: the more plausible >0.1pt route; adds signal vs denoising an already-optimized pool).

- R-upstream1 ✅ hard prune top-15% redundant (r_sig), all splits: test 0.7724 vs no-gate
  0.7765 = **−0.0041 (HURTS)**, bf_val −0.0010. seed42 −0.0132. → removing "redundant"
  mediators destroys signal. **REDUNDANCY AXIS CLOSED** (gate no-op + prune hurts).

## VERDICT (codex, fresh review 2026-06-30): bipartition-for-gain is EXHAUSTED
On this 2-hop AND support the attention pool already handles redundancy; distinctiveness is
NOT a gain lever (confirms user's "don't hard-remove overlapping paths"). 0.7765 ≈ practical
standalone pure-KG CEILING. codex odds any new pure-KG lever >0.1pt ~30%, >0.3pt ~10-15%.
- Paper value of this line = the NEGATIVE ablations: "on AND-support, support SEMANTICS (not
  redundancy suppression) are the lever." Keep as limit-study evidence.
- One last standalone shot (~30% for >0.1pt): add **GO biological-process mediators** (new
  semantic axis; transfers when unseen drugs hit different proteins in same process). NOT
  generic 3-hop (noise) / anatomy (coarse) / disease (entangled w/ indication). Hard stop if
  <+0.002-0.003 test with bf_val win.
- codex TOP rec: LOCK standalone (0.7765/+1.08pt/negative-ablations) + invest in EmerGNN-FUSION
  extension (adapter's real purpose, more publishable) rather than chase more standalone AUROC.

## NEW LOOP (2026-06-30): ACTUAL invariance-bipartition IMPLEMENTED (the anchor was NOT this)
User correctly flagged: the anchor (scalar gate + prune) was NOT the invariance-training
bipartition. The real method is now built: `aware_bipartite.py` BipartiteAwareCore +
LabelAdversary, runner `--bipartite`.
- Per-mediator h_i, pair prototype μ_p (attention). Decompose h_i = h_i^∥ (proj on μ_p,
  COMMON) + h_i^⊥ (residual, DISTINCT). Redundancy gate m_i=ε+(1-ε)σ(mlp([r_sig,r_rep_loo])).
- Dual within-type pools (FULL support in both): z_C=pool(h^∥, β∝exp s), z_D=pool(m_i·h^⊥,
  β∝exp(s+log m)). Core returns cat[z_C,z_D,struct] → scorer = joint readout f_J (nothing
  removed from prediction, unlike prune). bf mech branch on top.
- Invariance-training criterion (FGCN-DKS Eq5-8 analog, on OUR OOD support-degradation views
  as environments): L = BCE(f_J) + view_bce·BCE(view) + cw·MSE + **λ_null·GRL-null(z_C→y)**
  (make common channel label-uninformative) + **λ_inv·‖z_C^o−z_C^v‖²+‖z_D^o−z_D^v‖²** (both
  stable across views) + **λ_sep·cos²(z_C,z_D)**. Deterministic. Smoke test OK (out_dim 884,
  2-ep bf test 0.7907).
- Round 1 ⏳: defaults null=inv=sep=0.1, grl_lambda=1.0, 3 seeds. Target beat no-gate 0.7765
  (goal +1pp → ~0.7865). 25-round loop: R1 baseline → tune λ_null/inv/sep, grl ramp, gate
  init, f_D-sufficiency term, per codex review + lit review each round.

### Lit-review (2026-06-30, snippet-level, verify before citing)
- Deep Spurious Infomax (arXiv 2407.11083): minimize I(spurious;y) = our GRL-null on z_C. ✓ design
- Invariant Distribution Criterion (arXiv 2510.20295): causal subgraph has SMALLER cross-env
  variation than non-causal → supports cross-view invariance on z_D; suggests ASYMMETRIC inv
  (weight z_D > z_C). tune knob.
- Learning Invariant Graph Reps Through Redundant Info (arXiv 2512.06154): redundancy in IGL.
- EERM: adversarial env editors = our OOD support-degradation views as environments.
- DDI: DSIL-DDI (domain-invariance for DDI shift); Benchmarking DDI distribution changes
  (arXiv 2410.18583) — validates our cold-start covariate-shift framing.

### Bipartite loop log
- R1 ✅ defaults (null=inv=sep=0.1): test 0.7685 vs no-gate 0.7765 = **−0.0080** (per-seed
  −.0098/+.0016/−.0159; bf_val −.0013). UNDERPERFORMS. codex: structural (b) — GRL-null on z_C
  is counterproductive because z_C carries real signal (consistent: prune hurt, gate no-op,
  now bipartite-null hurts most). The "common general = useless" premise looks FALSE on this
  2-hop AND substrate.
- R2 ⏳ GRL-null OFF (bp-null-weight 0, keep inv+sep). Disambiguates: recover→null was culprit;
  still worse→∥/⊥ split itself hurts. codex STOPPING: stop if R2 < no-gate −0.003; max 2 more
  rounds (R2,R3); continue only if a variant beats 0.7765 by ≥+0.003 mean, ≥2/3 seeds
  non-negative, bf_val-confirmed. Else conclude bipartition-SUPPRESSION premise FALSIFIED here.

- R2 ✅ GRL-null OFF: test 0.7702 (−0.0063 vs no-gate; per-seed −.0093/+.0022/−.0118). Removing
  null recovered only +0.0017; STILL −0.63pt below baseline → **the ∥/⊥ decomposition ITSELF
  underperforms the single undecomposed pool**, not just the null. codex pre-registered
  stop-criterion (R2 < no-gate −0.003) MET → STOP.

## FALSIFIED (2026-07-01): bipartition-suppression premise dead on this substrate
The ACTUAL invariance-training bipartition is now implemented & tested (R1 null-on −0.0080,
R2 null-off −0.0063), addressing the earlier gap (anchor ≠ real method). Full consistent
evidence — EVERY form of decomposing/suppressing the common/shared support hurts or is neutral:
prune −0.0041 · scalar gate no-op · bipartite+null −0.0080 · bipartite no-null −0.0063.
**Honest conclusion: in AND-INTERSECTION support the SHARED neighbors ARE the distilled
mechanism signal; the pool over the full intersection is already near-optimal, so
decomposing/suppressing the "common" component removes signal.** The invariance-style
common/variant separation (FGCN-DKS) does NOT transfer here — on intersection support, the
intersection IS the invariant. This is a legitimate NEGATIVE-result limit study for the paper.
+1pp over 0.7765 is NOT achievable via this mechanism family. Reallocate: lock standalone
0.7765/+1.08pt + negative-ablation limit study; optional GO-BP semantic-axis shot (~30%);
EmerGNN-fusion extension (adapter's real purpose).

relates: [[spmn_v2_autoresearch_plus1pt]] · [[project_semantic_path_ddi]]
