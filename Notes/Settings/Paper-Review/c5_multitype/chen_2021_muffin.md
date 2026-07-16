# MUFFIN: Multi-Scale Feature Fusion for Drug-Drug Interaction Prediction

- **Authors**: Yujie Chen, Tengfei Ma, Xixi Yang, Jianmin Wang, Bosheng Song, Xiangxiang Zeng
- **Year / Venue**: 2021, Bioinformatics, 37(17):2651-2658
- **Link**: https://academic.oup.com/bioinformatics/article/37/17/2651/6171181
- **Read depth**: full-text
- **Cluster**: c5

## TL;DR
Combines molecular-graph features (MPNN over SMILES) with knowledge-graph embeddings (TransE on DRKG) via a bi-level (cross-product + scalar-product) fusion module. Evaluated on binary, multi-class (81 DrugBank events), and multi-label (200 TWOSIDES side effects) DDI tasks.

## Problem & Setting
- **Classes**: 
  - Binary: 1.18M DDIs (DRKG).
  - Multi-class (81 types): 172,426 DDIs from DrugBank/DeepDDI split.
  - Multi-label (200 types): 99,002 DDIs from TWOSIDES.
- One of the few c5 papers that simultaneously benchmarks across all three DDI task formulations.

## Method (core)
- Drug structural encoder: MPNN over SMILES -> molecular embedding.
- Drug semantic encoder: TransE over DRKG (drugs, proteins, diseases, pathways, side-effects).
- Bi-level fusion:
  - Cross-level: CNN over outer-product of structure + KG vectors (captures all pairwise feature interactions).
  - Scalar-level: element-wise product (fine-grained).
- Concat fused vectors -> MLP -> task-specific output head.

## Cold-start handling
- Not explicitly evaluated. Random pair split for all three tasks.
- Argues that fusing KG semantics "alleviates the restriction of limited labeled data" - an indirect cold-start argument but not measured at the drug level.
- For drugs missing from DRKG the KG branch falls back to zero embedding - undocumented cold-start failure mode.

## Key contributions
- First c5 method to span all three DDI task formulations (binary / multi-class / multi-label) under one architecture.
- Bi-level cross+scalar fusion is a richer alternative to concat or attention.
- Showed KG semantics complement molecular structure for the 200-label TWOSIDES task.

## Limitations / gaps
- No drug-disjoint cold-start evaluation - the headline numbers do NOT correspond to S2.
- Drugs absent from DRKG suffer silently.
- 81 and 200 labels treated flat; no PK/PD or anatomical hierarchy.
- KG features come from biomedical KG only - drug names enter as entity IDs, not as text.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: The dual-branch (structure + KG) architecture is the closest a c5 paper gets to our i1: structure branch ~ PD-side (chemistry of action), KG branch ~ PK-side (metabolism partners, transporters via KG). However, both are fused early and used uniformly across all 81/200 classes - the paper does NOT route classes to branches by mechanism. Strong evidence that the i1 split is *latent* in MUFFIN's design but unexploited.
- **i2 (meeting node + over-smoothing)**: MPNN is per-molecule (no over-smoothing at drug-drug scale). KG embedding via TransE is shallow (no propagation). The model avoids over-smoothing by avoiding deep drug-drug GNNs entirely - aligned with i2's prescription.
- **i3 (pooling noise + attention limit)**: Replaces attention with bi-level cross+scalar product fusion. No external prior injection. The cross-product is dense and uniform across feature dims - a different but still uninformed pooling.
- **i4 (node-name semantic prior)**: NOT used. KG entity IDs are dense vectors learned from TransE; the textual name of a drug (or its target) is discarded once it's an entity in the KG.

## Notes
- The 200-label TWOSIDES setup is a useful counterpart to our 65/100 DrugBank setup.
- MUFFIN is a strong baseline for any method that wants to argue "structure + KG is good but not enough without text/mechanism".
