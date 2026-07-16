# SkipGNN: Predicting Molecular Interactions with Skip-Graph Networks

- **Authors**: Kexin Huang, Cao Xiao, Lucas M. Glass, Marinka Zitnik, Jimeng Sun
- **Year / Venue**: 2020 / Scientific Reports
- **Link**: https://www.nature.com/articles/s41598-020-77766-9 ; https://arxiv.org/abs/2004.14949 ; code https://github.com/kexinhuang12345/SkipGNN
- **Read depth**: abstract-only (Nature page returned 303 redirect; abstract from search and arXiv)
- **Cluster**: c1

## TL;DR
SkipGNN constructs an auxiliary "skip graph" linking nodes that share a common second-order neighbor and iteratively fuses messages from both the original and the skip graph, so that biologically meaningful two-hop signals (drug-protein-drug) get a direct channel.

## Problem & Setting
Four molecular-interaction tasks evaluated under the same architecture: DDI, drug-target, PPI, gene-disease. Training is transductive on each interaction network; the paper claims robustness on noisy/incomplete networks but does not run a formal cold-start (S2) protocol.

## Method (core)
- Build skip graph: connect two nodes if they share a neighbor in the original graph (captures "skip similarity").
- Two parallel GNNs run on the original graph and the skip graph.
- Iterative fusion: at each layer, embeddings from one graph condition messages on the other.
- Pair scorer over fused node embeddings.

## Cold-start handling
N/A. Drugs need to exist in the training graph to be embedded. The "robustness to incomplete networks" result is on edge-removal, not on node-removal.

## Key contributions
- Identifies that biological "two-hop" patterns (drug-protein-drug) deserve a first-class architectural channel.
- The skip-graph construction is generic and applies to DDI, DTI, PPI, GDA.
- Empirically robust under sparse / noisy input graphs.

## Limitations / gaps (as relevant to our insights)
- The "shared neighbor" is implicit in the skip edge but is collapsed; the mediator's identity (which protein) is not foregrounded in the representation.
- Still anchored at the drug node, not the mediator.
- Two GNN towers double parameters without addressing what to do with very noisy KGs.
- No text/name features.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. Skip edges are typeless.
- **i2 (meeting node + over-smoothing)**: Important partial precedent. Skip-graph is essentially "let drug-drug talk through their shared neighbor without deep GNN" — same motivation as our meeting-node thesis, but the mediator is hidden, not exposed.
- **i3 (pooling noise + attention limit)**: Skip-graph construction is uniform (any shared neighbor counts the same). No mechanism to inject external semantic priors about which mediator matters.
- **i4 (node-name semantic prior)**: Not used.

## Notes
SkipGNN's two-hop motivation is the closest existing argument for our i2. We should cite it as the inspirational precedent while pointing out it (a) hides the mediator identity and (b) does not address noise via semantics.
