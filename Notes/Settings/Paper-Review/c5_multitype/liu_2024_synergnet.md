# SynerGNet: A Graph Neural Network Model to Predict Anticancer Drug Synergy

- **Authors**: Liu M, Srivastava G, Ramanujam J, Brylinski M
- **Year / Venue**: 2024 / Biomolecules, 14(3):253
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10967862/
- **Read depth**: full-text summary
- **Cluster**: c5

## TL;DR
GNN that classifies a drug pair as synergistic vs antagonistic by mapping drug-target and omics features onto the human PPI network and reading out a pair-conditional graph embedding. It is a clean example of a directional (synergy/antagonism), pair-conditional, effect-layer aggregation that operates entirely on the target/network layer rather than on molecular structure — the design template our PD channel needs.

## Problem & Setting
Binary synergy/antagonism classification for anticancer drug pairs across cancer cell lines. Cancer-specific featured graphs built per (drug pair, cell line).

## Method (core)
- Build a pair-and-context-specific reduced PPI graph; protein nodes carry 218-d features (expression, CNV, mutation, drug-protein affinity, GO); drug-target nodes marked.
- Two GENConv graph-conv blocks → Jumping Knowledge aggregation → global max+avg pooling → MLP head.
- Aggregation is implicitly pair-conditional via the marked target nodes of the two drugs.

## Cold-start handling
Not an explicit DDI S1/S2 protocol, but pair graphs are built from targets/omics, so a drug needs only its target set — structurally inductive on the effect layer.

## Key contributions
- Demonstrates synergy/antagonism (a signed/directional PD outcome) can be predicted from a PPI-network graph with pair-conditional readout.
- Shows the effect-layer signal is carried by target placement + omics, not molecular fingerprints.

## Limitations / gaps (as relevant to our insights)
- Cancer-synergy specific, not general DrugBank DDI typing.
- Sign is a label, not an aggregation primitive — the model does not use signed/antisymmetric message passing; it classifies after symmetric pooling.
- Does not contrast against or combine with a molecular-structure branch, so it cannot show molecular interference.

## Relevance to our insights
- **PD-channel design (sign/synergy-aware, pair-conditional)**: Best in-KB template for the PD branch. Confirms a viable recipe: pair-conditional aggregation over the effect/target network, predicting a directional synergy/antagonism outcome, with no molecular modality.
- **i1 (PK/PD split)**: Reinforces that PD-type outcomes (synergy/antagonism) live on the target/PPI layer and are learnable there without molecular structure.
- **Routing-novelty gap**: Even this synergy GNN is single-paradigm (PD-only). It does not co-model PK or route a molecular modality by interaction type — consistent with the finding that mechanism-typed routing is unoccupied.

## Notes
- For a stronger antisymmetric-aggregation primitive (true sign-aware message passing) see the signed-GNN line (decoupled signed link prediction, KBS 2025); SynerGNet is the biomedical-grounded synergy/antagonism instance to cite for the PD-channel motivation.
- Cite as the architectural precedent for "PD = pair-conditional, sign/synergy-aware, effect-layer-only".
