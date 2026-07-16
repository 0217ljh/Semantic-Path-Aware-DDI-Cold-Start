# LLM-DDI: Leveraging Large Language Models for Drug-Drug Interaction Prediction on Biomedical Knowledge Graph

- **Authors**: Dongxu Li, Yue Yang, Ziwen Cui, Hengchuang Yin, Pengwei Hu, Lun Hu
- **Year / Venue**: 2026 / IEEE Journal of Biomedical and Health Informatics, 30(1):773-781
- **Link**: https://pubmed.ncbi.nlm.nih.gov/40601466/
- **Read depth**: abstract-only (paywalled; PubMed metadata)
- **Cluster**: c4

## TL;DR
GPT-based embeddings of each drug/molecule are injected as node features into a message-passing GNN over a biomedical knowledge graph; the GNN learns relational representations that are then used for DDI prediction. LLM is a feature extractor, GNN is the predictor.

## Problem & Setting
DDI prediction on a biomedical KG combining drug-drug, drug-target, drug-disease relations. Specific datasets and split protocol not visible in abstract.

## Method (core)
Three stages:
1. **LLM embedding** - GPT generates rich text-based embeddings for each molecule (likely drug name + description + properties), capturing semantic and pharmacological context.
2. **Message-passing GNN** - propagates and fuses these LLM embeddings across KG edges to learn relationally-aware drug representations.
3. **DDI prediction head** - score interactions from the learned representations.

LLM role: text feature extractor (frozen, not fine-tuned). GNN does the relational reasoning. Hybrid LLM+KG+GNN.

## Cold-start handling
Not detailed in available metadata. Title and abstract do not mention S1/S2 or inductive splits. The architecture is *compatible* with cold-start (since LLM can embed any new drug from text alone, GNN can extend if KG includes the new drug), but evaluation must be checked in full text.

## Key contributions
- Uses GPT embeddings rather than chemical fingerprints as the LLM-provided drug feature.
- Couples LLM text knowledge with KG topology via message passing - direct integration target for c4.

## Limitations / gaps (as relevant to our insights)
- Abstract does not specify cold-start protocols, leakage controls, or comparison baselines.
- Treating GPT as a black-box text encoder reuses any DDI memorization in the embeddings.
- Closeness of approach to standard KG-GNN + text-feature pipelines means novelty may sit mainly in the choice of LLM.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed in abstract.
- **i2 (meeting node + over-smoothing)**: Uses message-passing GNN with no apparent meeting-node concept. If layers are deep, over-smoothing risk applies; abstract doesn't say.
- **i3 (pooling noise + attention limit)**: The LLM embedding *is* the external prior injected into the GNN - structurally similar to our i3 prescription, but pooling at the GNN side is likely still uniform.
- **i4 (node-name semantic prior)**: Strong overlap - GPT embeddings of drugs are exactly the node-name + text semantics signal we hypothesize is the only cold-start-stable channel. This paper is a direct reference for our i4 design.

## Notes
- Full-text access required to verify whether they evaluate S2 inductive splits and whether GPT embedding alone (no GNN) is ablated - that ablation is critical for the i4 question.
- IEEE JBHI 2026, peer-reviewed (unlike most c4 candidates).
- Need to find arXiv preprint or institutional access.
