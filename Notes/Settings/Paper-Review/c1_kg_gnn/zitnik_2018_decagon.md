# Decagon: Modeling Polypharmacy Side Effects with Graph Convolutional Networks

- **Authors**: Marinka Zitnik, Monica Agrawal, Jure Leskovec
- **Year / Venue**: 2018 / Bioinformatics (ISMB 2018)
- **Link**: https://arxiv.org/abs/1802.00543 ; https://academic.oup.com/bioinformatics/article/34/13/i457/5045770
- **Read depth**: abstract-only (binary contents on direct PDF; abstract + project page only)
- **Cluster**: c1

## TL;DR
Decagon is a multimodal GCN on a heterogeneous graph of protein-protein, drug-target, and drug-drug edges (one edge type per polypharmacy side effect). An R-GCN-style encoder produces node embeddings and a tensor-factorization decoder scores side-effect-specific drug pair links.

## Problem & Setting
Task is multi-relational link prediction over 964 polypharmacy side-effect relation types between drug pairs. Graph integrates STITCH/SIDER/BioGRID/TWOSIDES-style sources. Evaluation is transductive over edge splits; cold-start over unseen drugs is not the focus (S0/S1/S2 not discussed).

## Method (core)
- Heterogeneous graph with two node types (proteins, drugs) and many edge types (one per side effect plus PPI and drug-target).
- Relational GCN encoder, per-relation weight matrices, neighbor aggregation across edge types.
- Tensor-factorization decoder DEDICOM-style: each side effect has its own bilinear matrix sharing a global drug embedding.
- Negative sampling for link prediction; AUROC/AP/AP@50 per side effect.

## Cold-start handling
N/A. Drugs must be present in the training graph; no inductive evaluation is reported.

## Key contributions
- First scalable multi-relational GCN for polypharmacy side-effect prediction.
- Demonstrates parameter sharing across hundreds of side-effect relations improves rare-class prediction.
- Establishes the dominant graph schema later borrowed by many KG-DDI works.

## Limitations / gaps (as relevant to our insights)
- Drug embeddings are anchored at the drug node, not at any shared mediator.
- Uses purely structural neighborhood; no text or name semantics.
- Cold-start unseen drug case is unaddressed; new drugs have no embedding.
- Decoder treats every side effect symmetrically with no PK/PD distinction.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not differentiated. All 964 side effects are scored by the same encoder; PK (enzyme/transporter) and PD (system phenotype) collapsed into one tensor.
- **i2 (meeting node + over-smoothing)**: Counter-example: representation lives at the drug node and aggregates two-hop protein context. Over-smoothing risk at deeper layers is implicitly avoided by using only two GCN layers.
- **i3 (pooling noise + attention limit)**: Uses uniform mean aggregation per relation. No external semantic prior is injected, exactly the failure mode i3 anticipates.
- **i4 (node-name semantic prior)**: Not used. Nodes are atom-free IDs.

## Notes
Decagon is the canonical baseline for any KG-DDI work and the structural ancestor of KGNN, SumGNN, KnowDDI. Its absence of any cold-start protocol is why many subsequent papers do not even pose S2.
