# Predicting Drug–Drug Interactions through Drug Structural Similarities and Interaction Networks Incorporating Pharmacokinetics and Pharmacodynamics Knowledge

- **Authors**: Takeda T, Hao M, Cheng T, Bryant SH, Wang Y
- **Year / Venue**: 2017 / Journal of Cheminformatics, 9:16
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5340788/
- **Read depth**: full-text summary
- **Cluster**: c5

## TL;DR
Logistic-regression DDI predictor whose feature design is explicitly split along the pharmacological PK vs PD axis. PK is operationalised through enzyme and transporter overlap/similarity; PD through target-receptor and protein-protein-interaction proximity. The paper is one of the cleanest published statements that PK and PD are mechanistically distinct interaction routes that need different feature sources, and that the enzyme (PK) channel is the single strongest predictor.

## Problem & Setting
DrugBank-derived DDI set. Twenty-one feature-set combinations built from structural similarity plus PK descriptors (enzymes, transporters) and PD descriptors (targets, PPI network). Transductive logistic regression, AUC evaluation.

## Method (core)
- **PK features**: shared/similar metabolising enzymes (Se), shared/similar transporters. Grounded in the idea that structurally similar drugs share enzymes/transporters and therefore compete on the same ADME route.
- **PD features**: target overlap and target-receptor proximity over an interaction network.
- **Decoder**: single unified logistic regression over the concatenated PK+PD feature vector (NOT two separate models / NOT routed by interaction type).

## Cold-start handling
Not evaluated. Transductive AUC over edge splits; no inductive S1/S2 protocol.

## Key contributions
- Explicit pharmacological framing: PK = enzyme/transporter (metabolism, ADME) level; PD = receptor/target level.
- Quantitative channel decomposition: enzyme (PK) score is the strongest single signal (Se mean 0.800); swapping transporter for enzyme features raised AUC 0.741 → 0.827 (+11%).
- PK-alone AUC 0.657, PD-alone AUC 0.627, PK+PD combined 0.834 — both channels carry complementary signal, supporting a two-paradigm decomposition.

## Limitations / gaps (as relevant to our insights)
- Fusion is a flat feature concatenation into one head — the model never **routes** a modality conditioned on whether the target interaction is PK or PD. The PK/PD distinction lives only in feature engineering, not in the architecture.
- No molecular-structure deep encoder; "structural similarity" is Tanimoto, not a learned modality that could be routed.
- No cold-start evaluation.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: PRIMARY pharmacology citation. Directly grounds the claim that PK interactions act through the molecular layer (enzymes/transporters, structure-predictable) while PD interactions act through the target/effect layer. The enzyme channel being the strongest PK predictor supports "molecular structure is informative for PK".
- **i3 (pooling noise)**: The PK+PD-combined model gains over either alone, but because it concatenates rather than routes, it cannot suppress molecular-layer features when scoring a PD-type interaction — exactly the uniform-fusion failure mode our routing proposal targets.
- **Routing-novelty gap**: This is the closest prior work that *separates PK and PD at the feature level*, yet it still uses one unified head. It is therefore strong evidence that mechanism-typed **architectural** modality routing is unoccupied territory.

## Notes
- Use as the anchor citation for "PK = molecular/enzyme layer, PD = effect/target layer" in the i1 motivation.
- Pair with Huang 2013 (PD via PPI S-score) to cover both halves of the split with primary computational-pharmacology evidence.
