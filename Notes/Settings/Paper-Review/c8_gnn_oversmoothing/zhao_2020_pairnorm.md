# PairNorm: Tackling Oversmoothing in GNNs

- **Authors**: Lingxiao Zhao, Leman Akoglu
- **Year / Venue**: 2020 / ICLR 2020
- **Link**: https://arxiv.org/abs/1909.12223
- **Read depth**: abstract + key-claim level
- **Cluster**: c8

## TL;DR
PairNorm is a parameter-free normalization layer applied after each GCN propagation step that constrains the total pairwise squared distance between node embeddings to remain constant across layers. This prevents the global collapse onto a single point and lets deep GCNs at least retain inter-node separation, although task accuracy improvements are mostly on synthetic deep-only setups.

## Problem & Setting
Standard GCN node embeddings converge in $L_2$ distance to the over-smoothed subspace within ~6 layers. Existing fixes (residual, JK) preserve information but do not constrain the geometry. PairNorm targets the geometric symptom directly.

## Method (core)
- After each layer, center embeddings to zero mean, then rescale so the total pairwise distance equals a target constant.
- No learnable parameters, no per-layer cost beyond a global statistic.
- Two interpretations: (a) preserves Dirichlet energy proxy, (b) prevents the rank-1 subspace collapse.

## Cold-start handling
N/A.

## Key contributions
1. Geometric framing of over-smoothing as Dirichlet-energy collapse.
2. Practical normalization that delays collapse without architectural changes.
3. Demonstration that deeper PairNorm GCNs remain useful in scenarios with high label sparsity.

## Limitations / gaps
- Normalization preserves pairwise distance but does not preserve task-relevant signal; accuracy gain is often marginal at the natural depth.
- No biomedical/DDI evaluation.
- The constraint is uniform across pairs; for KG-DDI where two drugs share a single meeting node, uniform pairwise normalization is not selective.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: N/A.
- **i2 (meeting node + over-smoothing)**: SUPPORTING. Establishes that the geometry of node embeddings collapses, not just discriminability. Useful citation when arguing that the symptom of over-smoothing is empirically visible (Dirichlet energy decay) and a real engineering concern.
- **i3 (pooling noise + attention limit)**: Tangential.
- **i4 (node-name semantic prior)**: N/A.

## Notes
A standard "over-smoothing mitigation" reference. We cite it as part of the cluster of normalization-based remedies that we contrast against the structural meeting-node design.
