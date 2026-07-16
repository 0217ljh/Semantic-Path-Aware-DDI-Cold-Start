# Decagon: Modeling polypharmacy side effects with graph convolutional networks

- **Authors**: Marinka Zitnik, Monica Agrawal, Jure Leskovec
- **Year / Venue**: 2018, Bioinformatics (ISMB), 34(13):i457-i466
- **Link**: https://arxiv.org/abs/1802.00543
- **Read depth**: full-text (arXiv preprint + Stanford project page)
- **Cluster**: c5

## TL;DR
Seminal multi-relational GCN for polypharmacy. Builds a multimodal heterogeneous graph with PPI + drug-target + drug-drug edges where each of 964 polypharmacy side effects is its own edge type, and decodes side-effect-specific link scores from learned drug embeddings. Establishes the TWOSIDES/STITCH 964-edge-type benchmark.

## Problem & Setting
- **Classes**: 964 polypharmacy side-effect types (= 964 edge relations).
- **Datasets**: TWOSIDES (4.6M drug-drug-side-effect tuples filtered to 964 frequent side effects) + STITCH drug-protein + BioGRID/HumanPPI protein-protein.
- Multi-label: a drug pair can produce many side effects (multiple edges).

## Method (core)
- Heterogeneous graph: ~645 drug nodes, ~19k protein nodes; edges typed by relation.
- Per-layer R-GCN-style propagation: each relation has its own weight matrix; messages aggregated per relation then summed.
- Decoder: per-side-effect bilinear scoring on the two drug embeddings; the per-relation diagonal vector R_r is the side-effect's signature.
- Trained end-to-end as a multi-label link prediction problem.

## Cold-start handling
- Edges are masked for evaluation, but drugs themselves remain in the graph during training (transductive). Not designed for true S2 (both drugs new) cold-start.
- Authors note Decagon "models particularly well side effects with a strong molecular basis" because protein-target sharing transfers across drugs - this is the closest thing to a cold-start mechanism (via target overlap).
- For drugs with no known targets the method collapses, which is exactly the cold-start gap our work targets.

## Key contributions
- First multi-relational GCN for DDI side effects at 964-relation scale.
- Demonstrated +69% gain over Tensor Factorization / RESCAL baselines.
- Showed side-effect-specific learned R_r vectors cluster by anatomical system - first hint of latent mechanism structure.

## Limitations / gaps
- Transductive: cannot score a brand-new drug with no graph connections.
- "Particularly well on side effects with strong molecular basis" implies a known head/tail performance asymmetry tied to mechanism - directly relevant to our i1.
- 964 side effects flat-classified; no PK/PD hierarchy.
- Each relation has its own weight matrix - parameter cost scales linearly with number of side effects, and tail effects are starved of supervision.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Implicit support. The paper itself reports that side effects with a "strong molecular basis" (loosely, PD-mechanism observable through target/PPI sharing) are predicted better than non-molecular ones. This is empirical evidence that the model's homogeneous treatment underperforms for the non-molecular (effect-layer) regime - aligns with i1's claim that PK-layer reasoning and PD-layer reasoning have different evidence sources.
- **i2 (meeting node + over-smoothing)**: Highly relevant. Decagon's R-GCN explicitly passes messages drug -> protein -> drug, so the protein layer is a natural "meeting node". But it only uses 2-hop propagation; deeper stacks would over-smooth the heterogeneous graph. The paper's success at 2 hops is partial evidence that drug -> meeting-node -> drug is the right depth.
- **i3 (pooling noise + attention limit)**: Uses unweighted mean aggregation per relation. No attention - exactly the uniform pooling baseline our i3 calls out. A natural place to inject attention + a mechanism prior.
- **i4 (node-name semantic prior)**: Drug names not used; only structural graph IDs.

## Notes
- Per-side-effect AUROC reported in supplementary: head side effects 0.95+, tail side effects ~0.7. This head/tail gap is one of the most-cited motivations for class imbalance + contrastive approaches in c5.
- Drug-pair-as-edge formulation means there is no notion of "drug pair representation" - decoding is direct from individual drug embeddings, a key architectural difference from DDIMDL family.
- Code: https://github.com/mims-harvard/decagon
