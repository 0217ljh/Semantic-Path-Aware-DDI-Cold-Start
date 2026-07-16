# Understanding Over-Squashing and Bottlenecks on Graphs via Curvature

- **Authors**: Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, Michael M. Bronstein
- **Year / Venue**: 2022 / ICLR 2022 (Outstanding Paper Honorable Mention)
- **Link**: https://arxiv.org/abs/2111.14522
- **Read depth**: abstract + key-claim level
- **Cluster**: c8

## TL;DR
Formalizes over-squashing through discrete graph curvature. Defines a Balanced Forman curvature on edges and proves that negatively-curved edges (those lying on the unique short path between two communities) are exactly where information is squashed. Proposes Stochastic Discrete Ricci Flow (SDRF), a curvature-guided rewiring that adds edges around the most negative-curvature edges to relieve the bottleneck.

## Problem & Setting
Alon and Yahav identified over-squashing empirically but did not characterize which edges suffer it. This paper closes that gap with a geometric / topological criterion.

## Method (core)
- Define Balanced Forman curvature $\text{Ric}(u,v)$ on an edge as a combinatorial quantity involving the number of triangles and 4-cycles supported on the edge.
- Theorem: information flow through edge $(u,v)$ is bounded above by a quantity that decays with $|\text{Ric}(u,v)|$ when curvature is negative.
- Algorithm SDRF: iteratively add edges adjacent to the most negative-curvature edge until the bottleneck is removed; optionally remove the most positive edges to keep cost constant.

## Cold-start handling
N/A.

## Key contributions
1. Rigorous theoretical link between graph curvature and over-squashing.
2. Practical, principled rewiring algorithm with consistent improvements on heterophilous and long-range benchmarks.
3. Geometric language that unifies over-smoothing and over-squashing as two sides of mass-displacement on graphs.

## Limitations / gaps
- Rewiring modifies the input graph, which is unattractive when the graph has biological meaning (e.g., a curated drug-protein-disease KG).
- Computing curvature on a multi-million-edge biomedical KG is expensive.
- No DDI evaluation.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: N/A.
- **i2 (meeting node + over-smoothing)**: SUPPORTING. The curvature lens gives us another principled reason to avoid forcing drug-to-drug propagation through narrow biological bottlenecks. The "meeting node" can be interpreted as a positively-curved hub that already shortcuts what SDRF would have to rewire toward.
- **i3 (pooling noise + attention limit)**: N/A.
- **i4 (node-name semantic prior)**: N/A.

## Notes
Strongest theoretical foundation for "structure-induced bottleneck" arguments. Cite alongside Alon and Yahav 2021.
