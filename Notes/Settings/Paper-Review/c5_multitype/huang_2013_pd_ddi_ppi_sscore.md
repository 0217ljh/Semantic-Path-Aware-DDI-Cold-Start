# Systematic Prediction of Pharmacodynamic Drug-Drug Interactions through Protein-Protein-Interaction Network

- **Authors**: Huang J, Niu C, Green CD, Yang L, Mei H, Han JD
- **Year / Venue**: 2013 / PLOS Computational Biology, 9(3):e1002998
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC3605053/
- **Read depth**: full-text summary
- **Cluster**: c5

## TL;DR
Predicts pharmacodynamic (PD) DDIs purely from the closeness of two drugs' targets in a protein-protein-interaction network via an "S-score" (network proximity + cross-tissue co-expression of target systems). The premise is that PD interactions are NOT predictable from molecular structure but instead emerge from the topological/functional proximity of the two drugs' effect systems. This is the canonical "PD = effect-layer / network-emergent" computational-pharmacology paper.

## Problem & Setting
Genome-scale prediction of PD DDIs. Gold-standard PD-DDI positives from clinical sources; human PPI network; target annotations; gene expression across 79 tissues.

## Method (core)
- Map each drug to its targets plus first-step PPI neighbours.
- **S-score** = tightness of connection between the two target-centred subsystems, combining (1) shortest-path proximity in the PPI network, (2) Pearson correlation of target-system expression across 79 tissues.
- Integrate S-score with clinical side-effect similarity in a Bayesian model to raise specificity.

## Cold-start handling
Not framed as ML cold-start, but mechanistically inductive: a new drug only needs its target set and PPI placement, no co-occurrence history. Relevant analog for PD-channel cold-start reasoning.

## Key contributions
- Establishes that PD DDIs are detectable at the PPI-network level — "PD DDIs can be discerned at the PPI network level."
- Target-based features (target overlap, target distance, S-score) beat indication / gene-expression-signature / side-effect-similarity features — topological target proximity is the dominant PD signal.
- S-score alone AUC 0.731, rising to 0.812 with side-effect integration; 9,626 predicted PD DDIs at 82% accuracy / 62% recall.

## Limitations / gaps (as relevant to our insights)
- Pure PD method; no PK channel and no molecular-structure channel at all (by design — structure is treated as uninformative for PD).
- PPI-network-only; does not combine with a molecular branch, so cannot itself demonstrate the "molecular noise hurts PD" interference effect (only that structure is unnecessary for PD).
- 2013-era statistical method, not a GNN.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: PRIMARY pharmacology citation for the PD half. Grounds "PD interactions act through the effect/target layer (PPI, phenotype) and are physiologically emergent, not chemistry-predictable." Complements Takeda 2017 (PK = enzyme/transporter half).
- **PD-channel design (sign/synergy aggregation)**: The S-score is a pair-conditional, network-proximity quantity over the two drugs' target systems — direct precedent for keeping the PD channel KG-effect-layer-only and pair-conditional rather than molecular.
- **Routing-novelty gap**: Shows PD prediction was historically solved with a target/PPI feature, deliberately discarding molecular structure. Reinforces that pouring molecular features into a PD prediction is unmotivated — but no modern multimodal DDI model encodes this as an architectural routing prior.

## Notes
- Cite alongside Takeda 2017 as the two-sided pharmacological foundation: Takeda = "structure/enzyme predicts PK", Huang = "target/PPI proximity predicts PD, structure not needed".
- Supports the asymmetric-aggregation design: PD aggregation is pair-conditional and network-mediated, distinct from a symmetric shared-mechanism PK readout.
