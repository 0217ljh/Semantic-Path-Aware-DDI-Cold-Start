# Graph Optimal Transport for Cross-Domain Alignment (GOT)

- **Authors**: Liqun Chen, et al.
- **Year / Venue**: 2020 / ICML
- **Link**: https://arxiv.org/abs/2006.14744
- **Read depth**: abstract + method
- **Cluster**: c7 (cross-modal alignment methodology) — general method, not DDI

## TL;DR
General OT framework for aligning two structured (graph) representations across domains/modalities. Casts cross-domain alignment as graph matching with two OT distances: Wasserstein (WD) for node/entity matching and Gromov-Wasserstein (GWD) for edge/structure matching. No paired supervision required.

## Method (core)
- Represent each modality as a (dynamically built) graph of feature embeddings.
- WD aligns nodes across the two graphs; GWD aligns relational structure (distance-preserving, so it works even without a shared coordinate frame).
- Fused as a transport-cost objective added to the task loss.

## Relevance to our project (DDI-edge-independent alignment)
- GWD is the key tool: it aligns the STRUCTURE of the molecular graph to the STRUCTURE of the KG neighborhood without needing the two embedding spaces to be pre-registered and WITHOUT any DDI label — exactly the "intrinsic structure only" property we want.
- For an unseen drug we can OT-align its molecular-graph node set to its KG-neighborhood node set, producing a shared latent that both modalities agree on before any classifier.
- Heavier than contrastive (transport plan computation), but principled for the cold-start regime where contrastive negatives are hard to define for unseen drugs.
- Cf. "Optimal Transport-Based Multi-Grained Alignments for Text-Molecule Retrieval" (arXiv 2411.11875) — OT alignment already shown for molecule↔text.
