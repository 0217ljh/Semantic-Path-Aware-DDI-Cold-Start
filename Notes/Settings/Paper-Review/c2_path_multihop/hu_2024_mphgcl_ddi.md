# MPHGCL-DDI: Meta-Path-Based Heterogeneous Graph Contrastive Learning for Drug-Drug Interaction Prediction

- **Authors**: Baofang Hu, Zhenmei Yu, Mingke Li
- **Year / Venue**: 2024, Molecules (Vol. 29, Issue 11, 2483)
- **Link**: https://www.mdpi.com/1420-3049/29/11/2483 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11173658/
- **Read depth**: full-text (abstract + PMC body)
- **Cluster**: c2

## TL;DR
MPHGCL-DDI defines five hand-crafted meta-paths over a drug-centric heterogeneous graph (D-C-D, D-P-D, D-E-D, D-T-D, D-P-P-D) and uses meta-path-induced subgraphs to build two contrastive views (average view, augmented view). A multi-task loss combines DDI prediction with unsupervised and supervised contrastive objectives.

## Problem & Setting
Multi-typed DDI event prediction with three tasks; emphasises rare DDI events. Heterogeneous graph nodes = drugs (D), chemical substructures (C), proteins (P), enzymes (E), pathways (T).

## Method (core)
1. **Meta-paths (hand-crafted)**:
   - D-C-D (drugs sharing chemical substructure)
   - D-P-D (drugs targeting the same protein)
   - D-E-D (drugs sharing an enzyme)
   - D-T-D (drugs in the same pathway)
   - D-P-P-D (proteins linked by PPI)
2. **Two views**:
   - **Average view** — average of meta-path-specific drug embeddings.
   - **Augmented view** — three-level augmentation (feature masking, edge masking, sub-graph masking) then attention-fused.
3. **Losses**: multi-relation DDI prediction + unsupervised contrastive (view-to-view) + supervised contrastive (label-aware).
4. PPI is highlighted as the most useful indirect signal.

## Cold-start handling
The paper **explicitly identifies** poor performance on Task 3 (DDIs between two new drugs) as an unsolved problem and calls it the cold-start challenge for future work. PPI is the only signal they find that helps new drugs at all.

## Key contributions
- Hand-crafts meta-paths covering both PK-like (E, P) and PD-like (T, C) layers.
- Empirically shows PPI (the D-P-P-D 3-hop meta-path) is the strongest signal for unseen drugs.
- Three-level augmentation scheme tailored to meta-path subgraphs.

## Limitations / gaps (as relevant to our insights)
- Meta-paths are hand-crafted, not learned — limits transfer to new schemas.
- Class-imbalance bias toward frequent DDI events.
- Explicitly admits failure on the cold-start (new-new) split — exactly our target setting.
- No node-name text used.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: implicit. D-E-D / D-P-D look PK-leaning (enzyme, target), D-T-D / D-C-D look more PD-leaning (pathway, phenotype-proxy via substructure). But the model averages them rather than separating reasoning paradigms — a missed opportunity that motivates our i1 split.
- **i2 (meeting node + over-smoothing)**: the meta-path intermediate node *is* a meeting node by construction (P, E, T), but the readout is still drug-anchored after averaging.
- **i3 (pooling noise + attention limit)**: averaging across meta-paths is exactly the "uniform aggregation over low-SNR neighborhoods" weakness we critique.
- **i4 (node-name semantic prior)**: not used.
- **Cold-start failure admission** is a clean citation to motivate our work.

## Notes
The clearest published statement that meta-path DDI models fail on new-new drugs. Cite as motivation.
