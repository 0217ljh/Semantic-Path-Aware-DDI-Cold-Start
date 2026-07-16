# Representation Learning on Graphs with Jumping Knowledge Networks

- **Authors**: Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka
- **Year / Venue**: 2018 / ICML 2018
- **Link**: https://arxiv.org/abs/1806.03536
- **Read depth**: abstract + key-claim level
- **Cluster**: c8

## TL;DR
JKNet observes that different nodes need different effective neighborhood radii because of variations in local graph structure, and that stacking deeper to enlarge the receptive field globally is wasteful and harmful. It adds skip-connection-like "jumps" from every layer to the final readout so each node can pick the right depth.

## Problem & Setting
In a single GCN of fixed depth k, every node integrates information from a k-hop neighborhood. The authors point out that the rate at which a random walk expands depends on local topology, so the same k is too small for some nodes and already in the over-smoothed regime for others.

## Method (core)
- For each node, expose intermediate representations $h_v^{(1)}, ..., h_v^{(L)}$ from every layer to a final aggregator.
- Aggregators considered: concatenation, max-pooling, LSTM-attention.
- The model implicitly learns a per-node effective depth.

## Cold-start handling
N/A directly, but the per-node adaptive depth is mechanically related: if the cold drug is at the periphery of the KG, JKNet would let it terminate at a shallow layer while the well-connected partner can integrate deeper.

## Key contributions
1. First framework to allow adaptive neighborhood depth per node, motivated by the over-smoothing limitation of fixed-depth GCNs.
2. Theoretical analysis using random-walk influence distributions.
3. Improvements over GCN/GraphSAGE/GAT on bioinformatics (PPI), social, and citation benchmarks.

## Limitations / gaps
- Still uses message passing between drug and drug, not between drug and meeting node.
- The depth tradeoff is handled but not eliminated.
- No DDI / polypharmacy evaluation.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: N/A.
- **i2 (meeting node + over-smoothing)**: SUPPORTING. Strong evidence that the GNN community has long recognized that uniform deep stacking is harmful, and that adaptive depth is the second-best workaround. Our meeting-node argument can be framed as "rather than adaptively tune depth as JKNet does, we sidestep depth entirely by anchoring on the meeting biological entity."
- **i3 (pooling noise + attention limit)**: Related — JKNet's LSTM-attention readout is one of the early demonstrations that uniform pooling across layers is insufficient.
- **i4 (node-name semantic prior)**: N/A.

## Notes
Useful for the related-work paragraph to position our method as "orthogonal to depth-adaptive approaches like JKNet (Xu et al., 2018)."
