# HLN-DDI: Hierarchical Molecular Representation Learning with Co-Attention Mechanism for Drug-Drug Interaction Prediction

- **Authors**: Yue Luo, Lei Deng, Zhijian Huang (Central South University)
- **Year / Venue**: 2025 / BMC Bioinformatics, 26:152 (June 2025)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12135231/ ; https://pubmed.ncbi.nlm.nih.gov/40468206/
- **Read depth**: full-text (PMC open access)
- **Cluster**: c6

## TL;DR
HLN-DDI learns molecular representations at **three explicit hierarchy levels — atom, motif, molecule** — where motifs come from an **enhanced BRICS** decomposition (extra rule: split large rings, keep smallest constituent ring as the motif). A **co-attention** mechanism scores cross-drug interactions *across hierarchy levels*, with a relation-type-specific bilinear form for the final probability. Pure molecular, no KG.

## Problem & Setting
Multi-type DDI on DrugBank. Warm-start plus two inductive partitions, and a temporal generalization test on 25 FDA-approved post-2017 drugs (3,467 interactions).

## Method (core)
- Atom-level GNN, motif-level graph over BRICS fragments, molecule-level pooled vector.
- Co-attention: r_xy = b^T tanh(W_x V_i^x + W_y V_j^y) across the hierarchy-level embeddings of the two drugs.
- Prediction: P = sigma( sum r_xy V_i^x M_r V_j^y ), M_r relation-type-specific bilinear.

## Cold-start handling
**Explicit and cleanly separated.** P1 = both drugs unseen (= our S2): 75.44% accuracy. P2 = one drug unseen (= our S1): 83.38% accuracy. Also a forward-temporal new-drug test. This is one of the few hierarchical-molecular papers that reports a clean **double-drug-cold** number.

## Key contributions
- Three-level (atom/motif/molecule) hierarchy with cross-level co-attention, BRICS-grounded motifs.
- Reports an explicit both-unseen (S2-equivalent) accuracy, useful as a molecular-only S2 reference point (~75% acc).

## Relevance to our insights
- **Adding hierarchical molecular substructure to MNAH**: HLN-DDI confirms that a BRICS atom/motif/molecule hierarchy with cross-level co-attention still functions under **both-drugs-unseen** (their P1 = our S2), giving direct empirical support that hierarchical molecular substructure is double-cold-stable in principle. The cross-hierarchy co-attention is a candidate fusion pattern: the meeting-node/shared-mediator embedding in MNAH could act as an additional "level" the co-attention attends to, mixing KG-mediator semantics with atom/motif/molecule levels.
- Provides a concrete molecular-only S2 accuracy (~0.75) as a sanity reference for how much the KG path-flow + meeting-node head should add on top.
