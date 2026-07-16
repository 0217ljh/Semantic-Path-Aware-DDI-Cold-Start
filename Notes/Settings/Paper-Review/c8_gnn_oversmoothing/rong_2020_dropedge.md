# DropEdge: Towards Deep Graph Convolutional Networks on Node Classification

- **Authors**: Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang (Tencent AI Lab)
- **Year / Venue**: 2020 / ICLR 2020
- **Link**: https://arxiv.org/abs/1907.10903
- **Read depth**: abstract + key-claim level
- **Cluster**: c8

## TL;DR
DropEdge randomly removes a fraction of edges from the message-passing graph at each training epoch. The paper provides theoretical evidence that this either slows down the convergence rate of over-smoothing or reduces the information loss caused by it, and empirically lets GCN/GraphSAGE/JKNet train deeper (up to 64 layers) without collapse.

## Problem & Setting
Two coupled obstacles for deep GCNs: (1) over-fitting on small labeled sets and (2) over-smoothing that washes out features as depth grows. Both individually addressed by prior work, but no method tackles them jointly.

## Method (core)
- Sample a binary mask over edges with rate $p$ per epoch; train GCN on the masked adjacency.
- Theoretical claim: dropping edges enlarges the principal eigenvalue gap of the residual graph, slowing the contraction in the Oono-Suzuki sense.
- Works as a plug-in for any message-passing backbone.

## Cold-start handling
Authors do not test on cold-start, but the random-edge perturbation is implicitly related to data augmentation that has been used in cold-start GNN work.

## Key contributions
1. A simple, model-agnostic regularizer that demonstrably enables much deeper GCNs.
2. Theoretical bridge between edge sparsification and slower over-smoothing.
3. Strong empirical depth-vs-accuracy curves on Cora / Citeseer / Pubmed.

## Limitations / gaps
- Edge dropping is uniform; biomedical KGs have heavily skewed edge importance (drug-protein vs drug-side-effect-of-side-effect), so uniform dropping is suboptimal.
- No DDI evaluation.
- Improvement saturates at moderate depth; the underlying collapse is delayed, not removed.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: N/A.
- **i2 (meeting node + over-smoothing)**: SUPPORTING. DropEdge belongs to the family of "patch the deep GNN" methods that we explicitly argue against in favor of a structural fix (anchor at the meeting node). It is the right baseline reference for "even with mitigation, the exponential collapse still bites."
- **i3 (pooling noise + attention limit)**: N/A.
- **i4 (node-name semantic prior)**: N/A.

## Notes
Important to cite alongside PairNorm and GCNII as the canonical depth-mitigation toolkit our method bypasses rather than competes with.
