# Multimodal Contrastive Representation Learning in Augmented Biomedical Knowledge Graphs (BioMedKG / PrimeKG++)

- **Authors**: Tien Dang, Viet Thanh Duy Nguyen, Minh Tuan Le, Truong-Son Hy
- **Year / Venue**: 2025 / Frontiers in Systems Biology (also arXiv:2501.01644)
- **Link**: https://arxiv.org/abs/2501.01644
- **Read depth**: abstract + method overview
- **Cluster**: c7

## TL;DR
Augments PrimeKG with biological sequences and entity description text (PrimeKG++), then runs a multimodal contrastive framework that combines specialized language models, graph contrastive learning, and a KGE backbone. Claims generalisation to **unseen nodes** for link prediction on drug-disease and DrugBank DTI tasks.

## Problem & Setting
Biomedical KG link prediction on PrimeKG++ and DrugBank. Targets drug-disease relations and drug-target interactions. Not a DDI paper, but the cold-start-on-unseen-nodes claim makes it directly relevant.

## Method (core)
- Build PrimeKG++ by adding biological sequence features and textual entity descriptions to PrimeKG.
- Three subsystems:
  1. Specialized LM encodes textual entity descriptions into semantic vectors.
  2. Graph Contrastive Learning over the KG to refine intra-entity representation.
  3. A KGE backbone for inter-entity relational signal.
- Multimodal contrastive loss aligns the three views.

How node text is incorporated: pre-trained biomedical LM produces an entity-level semantic vector that is contrastively aligned with the graph-derived vector.

## Cold-start handling
Authors explicitly claim "strong generalizability, enabling accurate link predictions even for unseen nodes." This is one of the few c7 papers to advertise inductive performance, made possible because the LM gives an embedding for any entity with a description string.

## Key contributions
- PrimeKG++: text + sequence augmented version of PrimeKG.
- Multimodal contrastive recipe combining LM, GCL, KGE.
- Reports unseen-node generalisation on drug-disease and DTI link prediction.

## Limitations / gaps
- Specific LM identity not made explicit in the abstract; need to check the paper for which biomedical encoder.
- No DDI evaluation in the abstract.
- Contrastive alignment is symmetric across modalities; no separation of mechanism types (PK vs PD).

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not separated.
- **i2 (meeting node)**: not addressed; standard GCL.
- **i3 (pooling noise + attention limit)**: contrastive alignment is the implicit fusion; external prior is injected via contrastive loss rather than attention. This is a more flexible alternative to scalar fusion but still does not gate per relation type.
- **i4 (node-name semantic prior)**: **primary prior art, current SOTA-class.** Encoder: specialized biomedical LM (e.g., PubMedBERT class, exact identity to verify). Pretraining: biomedical corpora. Fusion with graph: multimodal contrastive learning rather than direct concat or scalar weighted sum. Crucially **demonstrates cold-start (unseen node) inductive link prediction**, which is the property we want to claim for our i4 method.

## Notes
- This is probably the single most up-to-date c7 baseline if we make an i4 claim in 2026.
- Cross-link with c3 cold-start cluster: papers that test unseen-entity splits are rare in biomedical KG; BioMedKG is one and should appear in both clusters' related work paragraphs.
