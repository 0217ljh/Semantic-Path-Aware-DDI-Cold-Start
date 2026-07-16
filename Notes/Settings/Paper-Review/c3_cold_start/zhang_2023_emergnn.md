# Emerging Drug Interaction Prediction Enabled by Flow-based Graph Neural Network with Biomedical Network (EmerGNN)

- **Authors**: Yongqi Zhang, Quanming Yao, et al. (LARS-research)
- **Year / Venue**: 2023, Nature Computational Science
- **Link**: https://www.nature.com/articles/s43588-023-00558-4 (preprint arXiv:2311.09261)
- **Read depth**: full-text (abstract+method through GitHub README)
- **Cluster**: c3

## TL;DR
Flow-based GNN that propagates drug A's features along all multi-hop biomedical-KG paths to drug B, attending over path edges. Anchors the entire prediction at the *path between two drugs* rather than at drug node embeddings — directly relevant to our i2.

## Problem & Setting
Targets **emerging-drug DDI**: drugs that have just been approved and have few or no known interactions. Train/test split is chronological — emerging drugs are held out and treated as cold-start. Equivalent to S2 when both drugs in a test pair are emerging; mostly evaluated in the "one emerging + one existing" regime (S1 in some papers' notation). Uses DrugBank-derived KG plus HetioNet-style biomedical entities (gene, protein, anatomy, disease).

## Method (core)
1. For each target drug pair (u,v), extract the path-induced subgraph G_uv = all KG paths of length ≤ K from u to v.
2. Flow-based GNN: messages start at u, are propagated layer-by-layer along edges of G_uv toward v, with an attention/relevance weight per edge.
3. Final pair representation is the receiver-side aggregation at v after K steps.

## Cold-start handling
- **Drug representation**: drug node features are KG identifiers + initial GNN embeddings learned from KG topology. **No SMILES**, **no text descriptions**.
- An emerging drug only needs to be connected to the biomedical KG by at least one edge (e.g., "drug-targets-gene") to be embeddable.
- Cold-start power comes entirely from biomedical KG anchoring and flow paths — emerging drug's prediction reuses the path semantics learned from existing drugs through the same KG structure.

## Key contributions
- First flow-based directional GNN over biomedical paths for DDI.
- Demonstrates strong performance on emerging-drug splits (significantly above SumGNN / Decagon / KG-DDI baselines).
- Edge-attention enables interpretable path-level evidence.

## Limitations / gaps (as relevant to our insights)
- Path attention is computed over KG topology only; **no external semantic priors** injected. Attention is structural/learned-from-scratch.
- KG quality / sparsity is a hard ceiling for emerging drugs with few neighbors.
- No PK/PD-aware modelling of path types; all relation types treated symmetrically in the flow.
- Treats drug-pair flow as the unit; relies on deep paths, which is exactly the over-smoothing regime our i2 critiques.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. The KG mixes targets, enzymes, pathways, side-effects — no architectural distinction between PK paths (drug→enzyme→drug) and PD paths (drug→phenotype→drug).
- **i2 (meeting node + over-smoothing)**: **Highly relevant** — EmerGNN *does* enumerate paths, but it forces flow drug-to-drug rather than anchoring on a meeting node. With K=3-4 hops on dense biomedical KG, this is the classic over-smoothing regime.
- **i3 (pooling noise + attention limit)**: Uses attention but attention weights are learned purely from end-task signal — no external semantic prior. Likely under-performs on noisy emerging-drug neighborhoods.
- **i4 (node-name semantic prior)**: **Missing** — drug nodes are IDs+topology embeddings only. No biomedical text features. This is a clean gap for our work.

## Notes
- This is *the* canonical c3 baseline. Public code at github.com/LARS-research/EmerGNN.
- Most subsequent emerging-drug DDI papers compare against EmerGNN.
- Reports substantial gap between transductive and emerging-drug settings — but does not break down by interaction mechanism.
