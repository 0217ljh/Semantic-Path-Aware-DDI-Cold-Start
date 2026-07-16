# BioPathNet: Path-Based Reasoning for Biomedical Knowledge Graphs (NBFNet-based)

- **Authors**: Yue Hu, Svitlana Oleshko, Samuele Firmani, Zhaocheng Zhu, Hui Cheng, Maria Ulmer, Matthias Arnold, Maria Colomé-Tatché, Jian Tang, Sophie Xhonneux, Annalisa Marsico
- **Year / Venue**: 2024 (bioRxiv); peer-reviewed version in *Nature Biomedical Engineering* (2025)
- **Link**: https://www.biorxiv.org/content/10.1101/2024.06.17.599219v2.full ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11326122/ ; https://www.nature.com/articles/s41551-025-01598-z
- **Read depth**: full-text (PMC version)
- **Cluster**: c2

## TL;DR
BioPathNet adapts the Neural Bellman-Ford Network (NBFNet) to biomedical KGs, learning **path representations** between an anchor query node and all candidate tails via generalized dynamic programming. Adds two domain extensions: a background regulatory graph (BRG) for message passing only, and entity-type-aware negative sampling. Evaluated on gene function, drug repurposing (zero-shot disease splits), synthetic lethality, and lncRNA-mRNA — the drug repurposing task is conceptually closest to DDI.

## Problem & Setting
General path-based link prediction on biomedical KGs. The drug-disease (repurposing) task is a strict zero-shot setting: all triplets for the held-out disease area are removed, plus 95% of biomedical connections.

## Method (core)
1. **NBFNet backbone** — for query (u, r, ?), compute pair representation h(u, v) by running a generalized Bellman-Ford relational message passing from u to all v.
2. **Path representations**, not node embeddings — h(u, v) summarises all paths from u to v.
3. **BRG message-passing edges** — extra edges (PPI etc.) used only during propagation, not for scoring.
4. **Type-aware negative sampling** — only sample negatives of the matching node type.

## Cold-start handling
Strong zero-shot/inductive evaluation by construction: the model can score any (u, v) pair as long as both are in the KG, since it learns relations and path patterns, not entity embeddings. Beat TxGNN by +20.2% AUPRC on zero-shot disease splits.

## Key contributions
- Pair-anchored (not node-anchored) representation — directly aligned with our i2.
- Path-representation framework is task-agnostic and inductive.
- Negative-sampling and BRG tricks are clean domain adaptations.

## Limitations / gaps (as relevant to our insights)
- DDI specifically isn't the headline task — drug repurposing (drug-disease) is. Adapting to drug-drug requires choosing how to anchor the pair.
- Still no node-text semantics — entity initial features are ID/embedding.
- High variance on contraindication task.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not addressed.
- **i2 (meeting node + over-smoothing)**: this is the **methodological template** for "pair-anchored, path-derived" representation we want. NBFNet's h(u, v) is mathematically a path-aggregated representation, not a drug-aggregated one — closest existing match to our meeting-node intuition.
- **i3 (pooling noise + attention limit)**: BioPathNet inherits NBFNet's relational message passing; still aggregation-based, no external semantic prior, similar to SumGNN's weakness but with better inductive structure.
- **i4 (node-name semantic prior)**: not used. Plugging biomedical text into NBFNet's initial node features is a natural extension and a direct contribution we can claim.

## Notes
Methodologically the strongest c2 template. The NBFNet pair-representation is what we should likely build on, and adding node-text semantics is the natural i4 contribution.
