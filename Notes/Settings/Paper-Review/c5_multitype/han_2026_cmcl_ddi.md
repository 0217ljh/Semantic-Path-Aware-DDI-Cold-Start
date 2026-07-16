# CMCL-DDI: Pharmacophore-Aware Cross-View Contrastive Learning for DDI Prediction

- **Authors**: Yehong Han, Lin Du (Qilu Normal Univ. / Karlstad Univ.)
- **Year / Venue**: 2026 / (PMC PMC12928573) ; code https://github.com/95LY/CMCL-DDI
- **Read depth**: abstract
- **Cluster**: c5 (multi-view DDI) — relevant to cross-modal alignment

## TL;DR
Two intrinsic-molecular views: (1) pharmacophore-based subgraph (functional features), (2) SMILES sequence. A cross-view contrastive loss aligns both in a shared latent space; a cross-attention module fuses them for the DDI head.

## Method (core)
- Pharmacophore subgraph encoder → graph-level embedding.
- SMILES sequence encoder → sequential embedding.
- Cross-view contrastive learning aligns the two views (mutual representation enhancement).
- Cross-attention fusion → DDI classifier.

## Cold-start handling
No explicit S1/S2 cold-start evaluation reported in the abstract. Both views are INTRINSIC to the molecule (pharmacophore + SMILES), so the alignment is naturally label-independent and would transfer to unseen drugs — but it does NOT use a KG modality, so it is a within-molecule alignment, not molecule↔KG.

## Relevance to our project
- Demonstrates that cross-view contrastive alignment of two molecular views works for DDI, and the alignment is label-free (good precedent).
- Gap we exploit: CMCL-DDI aligns molecule-to-molecule views; we need molecule-to-KG-neighborhood. The contrastive + cross-attention fusion template is reusable.
- Useful as a contrastive-alignment baseline that does NOT collapse the way label-fused TIGER/MKG-FENN do, supporting our hypothesis that alignment objective (not fusion alone) is the key.
