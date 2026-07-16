# FuseLinker: Leveraging LLM's Pre-trained Text Embeddings and Domain Knowledge to Enhance GNN-Based Link Prediction on Biomedical Knowledge Graphs

- **Authors**: Yongkang Xiao, Sinian Zhang, Huixue Zhou, Mingchen Li, Han Yang, Rui Zhang
- **Year / Venue**: 2024 / Journal of Biomedical Informatics (Sep 2024)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12079804/
- **Read depth**: abstract + method + results
- **Cluster**: c7

## TL;DR
Initializes biomedical-KG node features with LLM-derived text embeddings of entity names/descriptions, projects them to the same dimension as a domain-knowledge embedding, and **fuses by weighted average** before feeding to a GNN for link prediction. Benchmarks five text encoders (BERT, PubMedBERT, Flan-T5, Llama2, PMC-LLaMA) on four biomedical KGs.

## Problem & Setting
Biomedical-KG link prediction: KEGG50k, Hetionet, SuppKG, ADInt. Includes drug repurposing case studies on Hetionet but not a dedicated DDI / S2 split.

## Method (core)
- For each entity, build a text string from its name + description.
- Encode with an "embedding-visible" LLM: [CLS] for encoder-only, mean pool for encoder-decoder, last non-pad token for decoder-only.
- Run a domain KGE model (e.g., TransE / RotatE class) to get a "knowledge" embedding.
- Align dimensions with an autoencoder + FC layer.
- **Fusion**: `e_final = w * e_text + (1 - w) * e_knowledge`, w tuned per dataset.
- Pass fused features through a GNN-based link predictor.

How node-name text is incorporated: full pre-trained LLM encoding of entity-name+description string, projected and linearly combined with the structural embedding.

## Cold-start handling
Not explicitly tested as an inductive split. However, the design (text features available for any named entity) implies the architecture is portable to unseen nodes if w is large enough; this is unexplored experimentally in the paper.

## Key contributions
- Systematic comparison of five biomedical text encoders for KG node initialization on four KGs.
- Reports best results with balanced fusion (w around the middle), not pure text or pure knowledge.
- Best on KEGG50k: MRR 0.969, AUROC 0.987.

## Limitations / gaps
- No inductive / cold-start evaluation despite the architecture supporting it.
- Linear weighted fusion is simple; no attention or path-aware fusion.
- No analysis per relation type or per drug-pair regime.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not addressed.
- **i2 (meeting node + over-smoothing)**: uses a GNN over fused features; meeting-node behaviour not analysed.
- **i3 (pooling noise + attention limit)**: scalar w controls text/structure mixture, which is exactly the "can't inject external priors" concern in a simplified form. Provides empirical evidence that text-as-prior helps even when added linearly.
- **i4 (node-name semantic prior)**: **primary prior art.** Encoder choices include PubMedBERT and PMC-LLaMA (biomedical specialised). Pretraining is general LLM pretraining. Fusion is a per-dataset-tuned weighted average between LLM text embedding and TransE-family knowledge embedding. This is the closest published architecture to "text gives cold-start-stable node feature, structure refines it" — but they stop short of cold-start splits. Our angle: do their cold-start evaluation, plus replace scalar w with a structure-aware gate.

## Notes
- This is the **single closest baseline** to our i4 instantiation. Must compare on at least one of their benchmarks if we make an i4 claim.
- Dataset choice: Hetionet is shared with much DDI/repurposing literature.
