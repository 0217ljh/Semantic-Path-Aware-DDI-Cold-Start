# MIRACLE: Multi-view Graph Contrastive Representation Learning for Drug-Drug Interaction Prediction

- **Authors**: Yingheng Wang, Yaosen Min, Xin Chen, Ji Wu
- **Year / Venue**: 2021 / The Web Conference (WWW)
- **Link**: https://arxiv.org/abs/2010.11711 ; https://dl.acm.org/doi/10.1145/3442381.3449786 ; code https://github.com/isjakewong/MIRACLE
- **Read depth**: abstract-only
- **Cluster**: c1

## TL;DR
MIRACLE treats the DDI graph as a multi-view structure where each node in the macro-level interaction graph is itself a molecular graph; it learns molecular and interaction representations jointly with a contrastive objective enforcing consistency between the two views.

## Problem & Setting
Binary DDI prediction on standard DDI benchmarks (DrugBank-derived). Evaluation is transductive; cold-start / S2 not explicitly evaluated, though the dual-view design (molecular + interaction) makes the molecular branch in principle usable for unseen drugs.

## Method (core)
- Inter-view: GCN on the macro DDI interaction graph treating drugs as nodes.
- Intra-view: Bond-aware attentive MPNN on each molecular graph.
- Contrastive loss aligning the two views' drug representations.
- Pair scorer over fused multi-view embeddings.

## Cold-start handling
Partial. The molecular view does not require KG membership and could in principle handle an unseen drug, but the paper does not run a formal inductive split.

## Key contributions
- First clean "molecular structure + interaction graph" dual encoder with contrastive alignment.
- Demonstrates that contrastive regularization between views beats simple concatenation.
- Provides ablation isolating the contribution of each view.

## Limitations / gaps (as relevant to our insights)
- No biomedical KG context beyond direct DDI edges.
- No mediator-anchored reasoning.
- No text features at the node level.
- Cold-start unseen-drug performance not reported.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. Binary DDI only, no PK/PD split.
- **i2 (meeting node + over-smoothing)**: Not addressed. Macro-graph view is still drug-as-node aggregation.
- **i3 (pooling noise + attention limit)**: Macro graph is direct DDI edges only, so noise concern is lower, but no external prior is used either.
- **i4 (node-name semantic prior)**: Not used. Drugs are SMILES + interaction edges.

## Notes
Useful baseline for "dual encoder" framing. Confirms that combining molecular + interaction views beats either alone; orthogonal to our KG-text axis.
