# i1-as-Routing-Principle v2 — LEARNED Asymmetric Mechanism (supersedes v1 hard-routing)

**Status**: DRAFT for codex review. Supersedes `README_i1_routing.md` (v1, 2026-05-23).
**User directive 2026-05-25**: a hard PK/PD modality split is acceptable to *try* but
too crude / not elegant. The headline mechanism must embody ML thinking — a *learned*
inductive bias, not a hand-coded router. v1's hard routing is demoted to an ablation/baseline.

## Why v1 hard-routing is both crude AND slightly wrong

v1 hard-assigned: molecular modality → PK channel only; PD channel → KG-effect only,
keyed off the PK/PD label classifier. Two problems:
1. **Not elegant / not ML**: it's a fixed if/else over mediator type, no learning, no
   discovery. The PK/PD assignment is imposed, not learned from data.
2. **Contradicts i1's own finding**. Per `Notes/Settings/Insights/i1.md` lines 48-51,
   108-110: i1 is **primary-layer asymmetric, NOT exclusive-layer**. "PK mainly via
   molecular, PD mainly via effect" — but PK-B (no KG mediator) borrows effect, PD-A
   borrows molecular. A hard router forbids exactly the cross-borrowing i1 documents.

## What i1.md actually prescribes (the elegant design is already implied)

From i1.md lines 51, 83-102 (verified on disk 2026-05-25):
- **Molecular-layer evidence = instance-level relational evidence**: shared enzyme/
  transporter/target. Fits standard GNN neighbor aggregation:
  `f ≈ combine(pool(N_mol(a)), pool(N_mol(b)))` — pair-decision LAST.
- **Effect-layer evidence = pair-conditioned semantic composition**: must SELECT the
  1-2 composable effect pairs out of the ~20×20 cross-product of the two drugs' effect
  neighbors: `f ≈ select_compose(N_eff(a) × N_eff(b))` — pair-aware selection FIRST.
- E8.6 (i1.md line 101) directly validated this: swapping bag-pool→reasoner for
  pair-conditional-selector→reasoner lifted PD-effect AUROC 0.652→0.739 (+8.7pt).
- Prescription (line 51): do pair-conditional perception on BOTH layers; PK mainly mol
  (PK-B fallback effect), PD mainly effect (PD-A fallback mol).

So the inductive bias is in the **operator asymmetry** (aggregate vs select-compose),
NOT in a fixed PK/PD routing table. The PK/PD distinction should EMERGE from which
operator carries signal for a given pair.

## v2 design — three learned components

### Component A — molecular-layer channel (symmetric aggregation + aligned modality)
- Standard order-invariant pooling over shared molecular-layer KG mediators (MNAH count
  features restricted to molecular-layer mediator types) — keep the MNAH backbone here.
- PLUS the v3 InfoNCE-aligned molecular modality embedding `proj_m(m_u)` for each drug
  (m_u aligned to KG-neighbor k_u; k_u cache DONE at
  `Code/data/_cache/kg_neighbor_target_pubmedbert.npz`). The molecular MODALITY enters
  the molecular CHANNEL because both are PK-mechanism evidence.

### Component B — effect-layer channel (pair-conditional cross-attention selection)
- The NEW mechanism i1/E8.6 prescribes. Over the cross-product N_eff(a) × N_eff(b) of the
  two drugs' effect-layer mediators, a learned cross-attention selects the few composable
  effect pairs (sign-aware for synergy/antagonism). This is where i3's pair-conditional
  selection naturally lives.
- No molecular modality here (i1: molecular not predictive of emergent effects), but the
  soft gate (Component C) can still admit a little for PD-A fallback.

### Component C — LEARNED soft gate (the elegant inductive bias, replaces hard routing)
- A light gate `g(pair) ∈ [0,1]` mixes the two channel logits:
  `logit = g · logit_mol + (1-g) · logit_eff`.
- g is a **learned** function of the pair's path-type composition (fractions of molecular
  vs effect mediators, counts), NOT a fixed PK/PD lookup. The inductive bias is structural
  (g conditioned on mediator-layer composition) — the model discovers the routing.
- Soft (continuous), so PK-B / PD-A cross-borrowing is representable — fixes v1's
  exclusivity bug while still biasing toward the i1 asymmetry.

### Optional, stronger variant — alignment-confidence self-gating (candidate, needs codex)
- Idea: the v3 InfoNCE alignment residual `||proj_m(m_u) - proj_k(k_u)||` is itself a
  learned proxy for PK-ness (molecular structure predicts KG neighbors for PK-mechanism
  drugs; not for PD). Use low residual → high molecular-channel weight. This makes the
  routing emerge from alignment quality with NO PK/PD labels at all — maximally elegant.
- Flag for codex: is this over-clever / circular? Keep as a v2.1 candidate, not the
  headline, until reviewed.

## What stays LEARNED vs what is the imposed prior
- Imposed (structural inductive bias): molecular channel = aggregate; effect channel =
  select-compose; gate conditioned on mediator-layer composition.
- Learned: the gate value g per pair, the cross-attention selection weights, the modality
  projections, and (candidate) the alignment-residual gating.
- NOT imposed: which pairs are PK vs PD. That emerges and is then VALIDATED against labels.

## Testable / falsifiable story (validation, not supervision)
Use `Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv` (verified on disk; per i1.md
line 13: 215 ddi_types → 184 PD / 30 PK / 1 Mixed; test positives PK≈966 / PD≈953):
- (a) learned gate g correlates with PK/PD labels (discovered, not fed in).
- (b) molecular channel lifts PK pairs, ~0 on PD; effect channel lifts PD pairs.
- (c) ablating Component B (replace select-compose with bag-pool) hurts PD >> PK
  (mirrors E8.6's +8.7pt on PD).
- (d) explains the observed PK(0.836) >> PD(0.702) difficulty asymmetry from Stage 3.
- (e) hard-routing v1 = ablation; expect v2 soft gate ≥ v1 because cross-borrow allowed.

## Expected GAIN mechanism
- Effect channel's select-compose recovers PD signal that bag-pool dilutes (E8.6 evidence).
- Molecular noise is down-weighted on PD pairs by the learned gate, not poured in uniformly
  (the TIGER/MKG-FENN collapse source) → clean additive PK gain.

## Codex review log
- **Round 1 (2026-05-25, gpt-5.2-codex): REVISE.** Core risk: the "emergent PK/PD inductive
  bias" can collapse into a post-hoc layer-composition classifier + two ordinary heads; the
  gate may just encode known layer composition while the HARD part (learnable effect-pair
  selection) stays under-validated. 7 required revisions integrated below (§Revisions r1).
- **Round 2 (2026-05-25, gpt-5.2-codex): PASS.** Attribution to operator asymmetry now
  defensible (decisive evidence = the asymmetric failure pattern, not aggregate AUROC).
  R5 sufficient to start; external auxiliary supervision (SIDER/UMLS) is OPTIONAL (uneven
  coverage could muddy the clean cold-start claim), but THREE negative controls are
  MANDATORY: (i) shuffle-effect-names (preserve degree) must break the effect channel;
  (ii) high-frequency generic effects must not dominate top attention; (iii) bag-pool
  replacement must materially hurt PD. Non-blocking tightening: **pre-register PASS/FAIL
  thresholds for the operator-asymmetry ablations BEFORE running variants** (else the story
  is post-hoc). No remaining blocker to implementation.

## Pre-registered ablation thresholds (set BEFORE running — codex hygiene)
- Operator asymmetry PASS: replacing effect cross-attention with bag-pool drops PD-subgroup
  AUROC by ≥ a pre-set margin (TBD, e.g. ≥1.5pt) while PK-subgroup drop is < half that.
- Negative controls PASS: shuffle-effect-names collapses effect-channel contribution to ≈0;
  top-k attention mass on generic high-frequency SEs stays below a pre-set ceiling.
- Gain PASS (per R8): mean S2 AUROC ~0.795 tight CI OR clean +2pt over MNAH with PD-subgroup
  win and no PK regression, across 5–10 seeds.
- (Fill exact numeric thresholds into Notes/Experiments ExpLog before the first full run.)

## Revisions r1 (addressing codex round 1) — these are now part of the method

**R1. Reframe gate↔PK/PD as VALIDATION, not discovery.** Since E1b already defines PK/PD
partly by mediator-layer composition, a gate conditioned on composition is EXPECTED to
correlate with PK/PD — that is a sanity check, not a finding. New framing: "the gate
operationalizes a topology-derived mechanistic prior and learns a soft, non-exclusive
allocation that correlates with independently curated PK/PD labels." The headline claim
shifts from correlation to **utility**: (i) does the learned gate beat using layer-composition
as plain scalar features; (ii) does it win specifically on the PK-B / PD-A cross-borrow
subclasses where hard routing must fail; (iii) does soft gate beat hard-PK/PD-routing,
hard-layer-routing, random gate, constant gate.

**R2. Prove the OPERATOR asymmetry (not just the gate) earns the gain.** The decisive
ablation: replace effect-channel cross-attention with bag-pool → must hurt PD >> PK
(mirrors E8.6). Molecular shared-mediator pooling must help PK; effect select-compose must
help PD. Subgroup-specific gains are the contribution, not aggregate AUROC.

**R3. Matched-capacity baselines (mandatory, else "just MoE with engineered features"):**
concat(mol-evidence, effect-evidence, aligned-structure, counts)→MLP/attention at matched
param count; two-head no-gate simple average; same-gate-but-both-heads-bag-pool (isolates
whether cross-attention matters); hard PK/PD router (= demoted v1); random/constant gate.

**R4. Strict information boundaries + low-capacity gate.** Molecular head: molecular
mediators + molecular embeddings ONLY. Effect head: effect/effect-pair composition ONLY.
Gate: low-capacity, **layer-composition features ONLY — no drug IDs, no learned drug
embeddings, no DDI-type labels** (prevents the gate from memorizing pair/type artifacts and
keeps R1's validation honest). Inference uses no train/test edge info.

**R5. Effect cross-attention regularization (the #1 technical risk — Stage 3 collapse).**
Required: sparse attention (entropy penalty or top-k entmax/sparsemax / hard top-k);
frequency-degree debiasing (penalize attention to globally common SEs like
nausea/headache/dizziness unless pair-specific evidence supports); effect-pair dropout of
high-frequency generic effects during training; weak supervision from SIDER/UMLS/MedDRA
hierarchy or known synergistic/antagonistic effect categories if available; auxiliary
contrastive task (true DDI pairs have more compatible selected effect-pair reps than matched
non-DDI). Diagnostics: top-k attention mass concentration; whether top-attended pairs are
rare/composable vs generic; shuffle-effect-names control (preserve degree) must break it.

**R6. Alignment-residual self-gating = EXPLORATORY variant only, not headline.** Risks:
residual may measure alignment-training-quality / drug degree / KG-neighbor density / encoder
confidence rather than PK-ness; circular if the same InfoNCE objective both trains alignment
and is read as mechanistic evidence. Required before promoting: stop-gradient on the residual
gate input; correlate residual with degree, KG-neighbor count, molecular/effect mediator
counts, drug popularity, split status; show it predicts PK/PD AFTER controlling for layer
composition; must beat dumb proxies (molecular-mediator fraction, molecular degree). Keep out
of main method unless it survives all controls.

**R7. Novelty vs arXiv 2511.06662 — verify from the actual PDF before drafting.** Build a
contrast table: data source (KG-internal effect mediators vs EHR co-occurrence); routing
basis (mechanistic layer composition vs data-source fusion); operator (effect-effect
select-compose vs pathway-level fusion); validation (independent PK/PD labels + hard-router
failure cases); setting (both-drugs-unseen S2). The differentiator is narrow and sharp:
KG-internal mechanistic split + pair-conditional effect-effect cross-attention + soft
non-exclusive PK/PD-asymmetry gate — NOT generic dual-pathway fusion.

**R8. Gain-claim evidence bar (codex):** 5–10 seeds, CIs, paired tests vs MNAH; report S2
AUROC overall + PK + PD + Mixed; report AUPRC too (imbalanced); gain on PD WITHOUT
sacrificing PK; gain survives controlling for drug degree / mediator count / DDI-type
frequency; no leakage via mediator construction. Bar to claim "toward 0.80": mean AUROC
~0.795 with tight CIs, OR clean significant +2pt over MNAH with mechanistic subgroup wins.
A single seed >0.80 does NOT count.

## Protocol next steps
1. Lit support (→ `Notes/Settings/Paper-Review/`): pair-conditional / cross-attention
   mediator selection for DDI; learned modality gating / MoE for molecular+KG fusion;
   sign-aware synergy/antagonism GNN; evidence molecular fails for PD; gap check for any
   PK/PD-aware or mechanism-typed DDI architecture (novelty).
2. codex review of THIS v2 design until PASS (cwd = project root; codex can't read D:\ —
   paste this file's content into the prompt).
3. Code: integrate into v3 trainer as a new versioned model folder (new files; do NOT
   modify MNAH backbone signatures). Keep v1 hard-routing as a registered ablation variant.
4. Run seed42, controls (uniform-fusion = collapse baseline; v1 hard-routing; swapped-gate;
   shuffled-align), per-class PK/PD analysis. Target: net gain toward 0.80 + the i1 story.
