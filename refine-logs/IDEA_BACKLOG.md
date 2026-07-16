# IDEA_BACKLOG — substrate-change program (>=30 ideas, evolutionary, ~1 week)

Directive 2026-05-26: original KG+Morgan substrate is exhausted at ~0.77 (all v2 variants tie
MNAH; i1 per-neighbor effect-embedding composition falsified). Bring in TWO new substrates:
(A) richer MOLECULAR info (SMILES/fragments/pretrained encoders); (B) LLM-DISTILLED semantics
beyond the KG (Claude). Keep the 4 hard constraints (learned PK/PD routing, InfoNCE multimodal
alignment, i1 PK=molecular/PD=effect, i2 MNAH backbone). Target >0.80; honest ceiling ~0.77-0.79
so a strong mechanism/story + any gain is the win. Story must be written well (user).

Reference baselines (seed42, legacy split, test_s2 AUROC): EmerGNN 0.746 | MNAH 0.772 |
v2(xattn/aux/count) 0.768-0.772 | mol-channel-alone PK 0.71 PD 0.65 | KG-effect channel = chance.

LEAKAGE RULE (all ideas): LLM/molecular features must be PER-DRUG (or per-drug-structure),
NEVER per-(drug_a,drug_b) interaction labels — a pairwise LLM "do A,B interact?" is oracle
leakage on cold-start. Features computed on train+unseen drugs alike, no DDI-edge info.

## Wave 1 — substrate feasibility probes (cheap, decide which substrate carries signal)
- I1 (B): per-drug LLM pharmacology/MoA description (Claude) -> PubMedBERT embed -> drug feature
  cache. Probe: does it beat KG k_u as the alignment target / as a PD feature? [HEADLINE start]
- I2 (A): ChemBERTa (SMILES transformer) per-drug embedding -> replace/augment Morgan m_u.
- I3 (A): BRICS fragment multiset + Morgan; fragment vocabulary per drug.
- I4 (B): per-drug LLM STRUCTURED attributes (CYP substrate/inhibitor/inducer, transporter,
  pharmacologic class, MoA) -> typed features (PK-side from enzymes, PD-side from class/MoA).
- I5 (A): pretrained molecular GNN embedding (MolCLR/GIN-pretrain) per drug, if loadable.
- I6 (probe): LLM-feature ALONE as a logistic head on test_s2 (no GNN) -> upper-bound the
  signal the LLM substrate carries for PK vs PD subgroups.

## Wave 2 — integrate the winning substrate into the routing architecture
- I7: molecular channel = best-of(ChemBERTa/Morgan) aligned via InfoNCE; PD channel = LLM-PD
  features (I1/I4) instead of dead KG-effect neighbors. Learned gate. [core new method]
- I8: dual InfoNCE alignment — align BOTH molecular and LLM features to KG k_u (or to each
  other) in a shared space; the gate routes among aligned views.
- I9: fragment-level select-compose for PK (the i1 select-compose mechanism applied to
  MOLECULAR fragments A x B, where it may actually work, unlike effect neighbors).
- I10: LLM-PD select-compose — compose drug A's LLM-derived adverse-effect set with drug B's
  (the i1 PD-composition, but with LLM semantics that may carry the missing signal).
- I11: typed routing — PK channel fed enzyme/transporter LLM features (I4), PD channel fed
  class/MoA/AE LLM features; gate learned. Cleanest i1 realization on the new substrate.
- I12: distillation student — train a small structure->LLM-feature student encoder so unseen
  drugs get LLM-quality features from structure alone (true distillation; cold-start clean).

## Wave 3 — alignment / objective refinements (evolve from Wave-2 winner)
- I13: contrastive multi-view (molecular, LLM, KG) alignment with a shared projector.
- I14: gate aux-supervision with PK/PD labels (interpretability + routing sharpness).
- I15: residual/late-fusion vs gated-mix ablation for the new channels.
- I16: temperature/sparsity on any composition operator that survives Wave 2.
- I17: per-drug LLM CONFIDENCE/uncertainty as a feature reliability gate.
- I18: align molecular->LLM (structure predicts pharmacology) as the cold-start bridge
  (analog of the m_u->k_u idea but with LLM target, likely richer than KG k_u).
- I19: ensemble of substrate channels (side-study only per locked constraint).
- I20: distill LLM pairwise *rationales* into per-drug vectors (decompose pairwise->per-drug).

## Wave 4 — scale, robustness, story (evolve from best)
- I21: multi-seed (3 inits) + release seed43/44 generalization of the best model.
- I22: ablate each substrate to attribute the gain (molecular-only, LLM-only, KG-only, all).
- I23: PK/PD subgroup attribution of the gain (where does LLM/molecular help?).
- I24: richer LLM prompt designs (CoT, structured JSON, role-play pharmacologist) A/B.
- I25: bigger LLM (Sonnet) for a subset to test feature-quality ceiling vs Haiku.
- I26: embedding backbone swap for LLM text (PubMedBERT vs Claude embeddings vs e5).
- I27: negative controls — shuffle LLM features (preserve dist), random-drug LLM text.
- I28: leakage audit — confirm no test-drug pharmacology memorized via DDI in LLM output.
- I29: cost/latency-efficient feature caching + incremental update for new drugs.
- I30: calibration + AUPRC + per-class reporting for the final model.
- I31: failure analysis of remaining PD-hard pairs after LLM features.
- I32: write the mechanism/story draft (KG-insufficient -> LLM-distilled semantics rescue PD).

## REFINED PLAN (codex 019e6747, 2026-05-26) — shared/residual alignment
Root cause of molecular no-lift: aligning molecular to k_u (KG-neighbor mean) makes it
reconstruct what KG already has. FIX (keeps required molecular<->KG alignment as core):
- molecular -> m_shared (InfoNCE-aligned to a RELATION-TYPED KG target, not plain mean)
            + m_resid (KG-orthogonal residual: orthogonality + decorrelation loss).
- fuse [KG backbone, m_shared, m_resid] on top of MNAH; judge by LIFT over 0.772
  (>0.775 real, >0.778 strong; rerun noise ~+-0.3pt).
Evolutionary experiment order (each = "X on top of KG"):
- E-frag (NEXT): BRICS/Murcko fragment molecular source + shared/residual align + typed KG target.
- E-chemberta: ChemBERTa source, same residual-align design.
- E-llm: sanitized LLM-pharmacology as an additive residual channel (leakage-controlled;
  ~26% of distill outputs tripped sanitizer -> redacted+flagged, run flag-stratified ablation).
- E-combined: ChemBERTa+fragments+LLM residual fusion (highest ceiling, last).
Prereqs to build: (1) fragment features per drug; (2) relation-typed KG target k_typed
(rebuild k_u with typed buckets: targets/enzymes/transporters/pathways/indications/phenotypes);
(3) shared/residual alignment trainer; (4) fused channel in v2.

## Status
- ACTIVE: Wave 2 / E-frag. LLM distill running (~26% leak-flagged, sanitized) for E-llm.
- Each idea: codex plan-review -> build -> run -> analyze (LIFT over KG) -> codex patch -> log.
- Evolutionary: Wave N+1 pruned/added from Wave N results (living file).
