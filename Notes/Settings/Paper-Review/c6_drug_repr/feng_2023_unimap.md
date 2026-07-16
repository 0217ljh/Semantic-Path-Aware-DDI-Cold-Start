# UniMAP: Universal SMILES-Graph Representation Learning

- **Authors**: (UniMAP authors)
- **Year / Venue**: 2023 / arXiv:2310.14216
- **Link**: https://arxiv.org/abs/2310.14216
- **Read depth**: abstract
- **Cluster**: c6 (molecular representation, cross-view alignment)

## TL;DR
Joint SMILES + molecular-graph pretraining via a multi-layer Transformer with cross-modality fusion. Four pretraining tasks including Multi-Level Cross-Modality Masking and Fragment-Level Alignment to fuse 1D (SMILES) and 2D (graph) views.

## Method (core)
- Transformer fuses SMILES tokens and graph nodes.
- Fragment-Level Alignment explicitly aligns sub-structures across the two modalities — a structured (not just global) contrastive/masking alignment, all self-supervised (no task label).

## Relevance to our project
- Fragment-level alignment is a finer-grained alternative to global contrastive: align molecular substructures to KG sub-neighborhoods rather than a single pooled vector. Could reduce the saturation failure seen in MKG-FENN.
- Self-supervised, so transfers to unseen drugs. Both modalities here are intrinsic-molecular (SMILES+graph); for us one modality becomes the KG neighborhood.
