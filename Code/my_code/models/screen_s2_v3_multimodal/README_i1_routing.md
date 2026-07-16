# i1-as-Routing-Principle — Mechanism-Typed Inductive Bias (REQUIRED, co-equal with multimodal)

**Status**: DRAFT for codex review. User directive 2026-05-23: i1 must be integrated as a
required contribution, elevated to the same status as the multimodal-alignment part, via a
CLEVER inductive bias — and is expected to also bring a performance GAIN (not just story).

## Central thesis (unifies i1 + multimodal + explains the cold-start collapse)
- PK interactions act through the MOLECULAR layer (enzyme/transporter/target/protein/pathway)
  → chemistry-predictable → MOLECULAR modality is informative.
- PD interactions act through the EFFECT-SYSTEM layer (side-effect/phenotype/anatomy/disease)
  → physiologically emergent → molecular structure is NOT predictive (it's noise for PD).
- THEREFORE: the molecular modality should be routed ONLY to the PK channel. Injecting it
  uniformly (TIGER/MKG-FENN) pours noise into PD pairs → the cold-start collapse.
- **i1 is the routing principle that makes multimodal alignment WORK under cold-start.**
  Not an add-on module — it's the cure for the multimodal collapse. This unifies the two
  user requirements into ONE mechanism.

## Why Stage 3 (naive PK/PD count split) failed, and how this fixes it
Stage 3 = soft count split, two SYMMETRIC heads, no modality routing, no asymmetry → weak
routing (spec_PK negative). Fixes (3 hard inductive biases):
1. **Modality routing (hard prior)**: molecular embedding + its alignment feed ONLY the
   PK/molecular-path channel. PD channel = KG-effect-layer ONLY.
2. **Asymmetric aggregation**: PK = SYMMETRIC pooling over shared molecular mediators
   (order-invariant "shared mechanism", e.g. both CYP3A4 substrates). PD = pair-conditional,
   sign-aware composition (synergy/antagonism) — i3 selection lives in the PD channel
   (i1 says PD-effect is the paradigm that NEEDS pair-conditional perception).
3. **Paradigm gate**: light gate estimates P(PK vs PD) from path-type composition, mixes
   the two channel logits; REGULARIZED toward the i1 prior (molecular-mediator-rich → PK),
   not free-learned.

## Expected GAIN mechanism (user's intuition, agreed)
- Removes molecular noise from PD pairs (the collapse source) → clean additive PK gain.
- The aligned molecular signal (v3 InfoNCE) enters only where it's predictive (PK) → net +.

## Testable story (use existing ddi_pk_pd_labels.csv: PK=966 / PD=953 test positives)
- (a) molecular modality improves PK pairs, ~0 on PD pairs.
- (b) ablating molecular-routing hurts PK >> PD.
- (c) paradigm-gate output correlates with PK/PD labels.
- (d) explains the observed PK(0.836) >> PD(0.702) difficulty asymmetry: PD has no
  molecular shortcut.

## Integration with v3 (same architecture, two faces)
- v3 InfoNCE aligns molecular m_u ↔ KG-neighbor k_u (edge-independent; k_u cache DONE).
- i1 routing: the aligned molecular signal flows ONLY into the PK channel of the meeting-node
  head; PD channel stays KG-effect-layer. MNAH backbone + meeting-node head unchanged otherwise.

## Lit support (DONE — 4 papers saved to Paper-Review)
- Takeda 2017 (J Cheminf): PK→enzymes/transporters (structure-predictable); enzyme channel
  strongest predictor. Grounds PK=molecular.
- Huang 2013 (PLOS CB): PD DDIs predicted from PPI/target network proximity (S-score),
  structure UNNECESSARY. Grounds PD=effect-layer/emergent. → best cite for "PD is network-level".
- Im 2025 (in-KB): protein/pathway modality actively HURT; model confused PK(enzyme) with
  PD(QTc). → empirical anchor for "uniform molecular modality degrades PD".
- SynerGNet 2024: sign-aware synergy/antagonism GNN over PPI = template for PD channel.
- MedMoE 2025: report-type-conditioned modality routing works in biomed ML, but NEVER for
  DDI/PK-PD → our mechanism-typed routing is a CLEAN NOVELTY GAP.
- "molecular noise collapses PD under cold-start" = no prior isolates it → OUR experiment to own.

## CODEX r23 = FIX → APPROVED MINIMAL BUILD (code this first)
MINIMAL model (falsifies the core claim cheaply; NOT the full 3-bias):
  base   = MNAH/EmerGNN logit
  pk_logit = MLP([ PK/molecular-mediator counts ; aligned-molecular-emb z_m ])
  pd_logit = MLP([ PD/effect-layer mediator counts ])      # NO sign-aware, NO molecular
  g_pk = sigmoid(MLP(path-type composition));  g_pd = 1-g_pk
  final = base + g_pk*pk_logit + g_pd*pd_logit
  loss = ddi_BCE + lambda_gate*BCE(g_pk, weak_pk_label) [lambda~0.05-0.1] + (align loss if joint)
DECISIONS: hard molecular→PK (with PD-injected control); WEAK gate supervision from
ddi_pk_pd_labels (noisy, small weight, ablate); NO sign-aware PD in v1; NO full 3-bias yet.
DECISIVE EXPERIMENT (S2): A=MNAH | B=uniform molecular | C=molecular→PK-only | D=molecular→PD-only.
  Report overall + PK-subset + PD-subset AUC + gate g_pk separation. Desired: C↑(esp PK),
  B weak/hurts, D hurts, PD preserved in C.
ABLATIONS: no-gate-loss (identifiability), uniform, PD-only, both-channel, no-molecular.
PREREQUISITES to build: (1) molecular m_u precompute (Morgan), (2) InfoNCE alignment trainer
  (m_u↔k_u, k_u cache DONE) → aligned z_m, (3) PK/PD count subsets from the 22-dim (molecular
  kinds vs effect kinds), (4) per-pair weak PK/PD label via pair→ddi_type map (built in Stage 3),
  (5) path-type composition feature for the gate (= the 22-dim kind histogram, normalized).

## Protocol next steps
1. Lit support (→ Paper-Review): mechanism-typed/relation-typed path reasoning for DDI;
   sign-aware synergy/antagonism GNN; evidence that molecular fails for PD / PK has molecular
   basis; any PK/PD-aware DDI architecture (gap check).
2. codex review of THIS inductive-bias design until PASS.
3. Code: integrate PK-routed molecular alignment + PD KG-effect channel into the v3 trainer.
4. Run seed42, controls (no-routing uniform-fusion = collapse baseline; swapped-routing;
   shuffled-align), per-class PK/PD analysis. Target: net gain toward 0.80 + the i1 story.
