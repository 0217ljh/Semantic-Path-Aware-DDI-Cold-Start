# EXPERIMENT_PLAN — i1-routing v2 (learned) on MNAH backbone

Source idea: `Code/my_code/models/screen_s2_v3_multimodal/README_i1_routing_v2_learned.md` (codex PASS r2).
Target: **test_s2 AUROC > 0.80** on ColdDDI DrugBank seed42, building on MNAH ~0.77.
NO extra comparison baselines (user directive). MNAH 0.77 is the reference, not re-run.

## Hard requirements (must all hold in the winning run)
1. Learned PK/PD soft routing (NOT hard split).
2. Multimodal alignment InfoNCE m_u <-> k_u (edge-independent).
3. Aligns i1: molecular channel = aggregate; effect channel = pair-conditional select-compose.
4. Aligns i2: MNAH meeting-node head retained, complementary to EmerGNN path-flow.

## Prerequisites (deterministic precomputes)
- [DONE] P1 molecular m_u: `molecular_mu_morgan.npz` (1994 drugs, 2048-d).
- [DONE] P0 alignment target k_u: `kg_neighbor_target_pubmedbert.npz` (7754 drugs, 768-d).
- [TODO] P2 effect-neighbor sets: per-drug effect-layer KG neighbor IDs + PubMedBERT emb
  (CSR), kinds in {side_effect, disease, anatomy, effect/phenotype, symptom}. Input to
  the effect-channel cross-attention. Adapt precompute_kg_neighbor_target.py.
- [TODO] P3 InfoNCE alignment trainer: proj_m(m_u) <-> proj_k(k_u), train-graph drugs only;
  outputs aligned molecular embedding per drug (frozen for downstream, or joint-trained).

## Experiments (must-run)
- E1 (main): v2 full model = MNAH + molecular channel (molecular-layer counts + aligned m_u)
  + effect channel (pair-conditional cross-attention select-compose, sign-aware, R5 reg)
  + learned soft gate (R4 low-capacity, layer-composition-only). seed42, 3 repeats,
  within-run Δ vs MNAH. Metric: test_s2 AUROC overall + PK/PD subgroup + AUPRC.
- S1 (sanity): shuffle effect-neighbor names (preserve degree) → effect channel must
  collapse to ~0 contribution. Confirms effect signal is real, not degree artifact.
- S2 (sanity/ablation): replace effect cross-attention with bag-pool → PD-subgroup AUROC
  must drop materially while PK drop < half. Confirms operator asymmetry (the i1 claim).
- A1 (analysis): per-class PK/PD breakdown vs ddi_pk_pd_labels.csv; learned gate g vs PK/PD
  correlation (validation, not discovery); top-k attention mass on generic vs rare SEs.

## Pre-registered PASS/FAIL (codex hygiene, set before runs)
- Operator asymmetry PASS: S2 drops PD AUROC ≥ 1.5pt, PK drop < 0.75pt.
- Negative control PASS: S1 effect-channel contribution → |Δ| < 0.3pt vs no-effect-channel.
- Gain PASS: E1 mean test_s2 AUROC ≥ 0.79 with PD-subgroup gain and no PK regression
  (toward 0.80; clean +2pt over MNAH 0.77 with subgroup win also qualifies).

## Loop (analyze-iterate)
After E1: if gain PASS not met → codex (Theorist+Reviewer) does failure analysis + targeted
patch → ML Engineer implements → re-run. Repeat until all 4 hard requirements hold AND gain
PASS, or theoretical ceiling. Every round logged to AGENT_LOG.jsonl + EXPERIMENT_TRACKER.md.
