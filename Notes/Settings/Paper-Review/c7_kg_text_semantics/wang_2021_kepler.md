# KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation

- **Authors**: Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, Jian Tang
- **Year / Venue**: 2021 / TACL (preprint arXiv:1911.06136, Nov 2019)
- **Link**: https://arxiv.org/abs/1911.06136
- **Read depth**: abstract + method overview
- **Cluster**: c7

## TL;DR
Jointly trains a transformer (RoBERTa-style) on (a) masked language modeling and (b) a knowledge-embedding objective (TransE-style) where every entity embedding is the **encoded textual description** of that entity. The result is a single model that produces text-derived entity embeddings that work for both NLP tasks and **inductive** KG link prediction. Introduces Wikidata5M with aligned entity descriptions.

## Problem & Setting
General KG completion + LM pretraining on Wikidata5M. Not biomedical per se, but the inductive-KE evaluation makes it a methodological template for cold-start node embedding in any KG, including biomedical ones.

## Method (core)
- For each entity e with description text d_e: embedding(e) = PLM(d_e)[CLS] (or pooled).
- Relation embeddings are still learned id vectors.
- KE objective: TransE margin loss on (encoded h, learned r, encoded t).
- Jointly trained with MLM on the same encoder. Single shared encoder.

How node text is incorporated: the entity embedding is **literally** the PLM encoding of its description. There is no parallel id-embedding table that would be needed for new entities.

## Cold-start handling
**Explicit inductive evaluation.** At test time, KEPLER can score triples involving entities that were never seen during KE training, as long as a description text exists. Reports inductive link-prediction numbers on a held-out entity subset of Wikidata5M. This is the cleanest cold-start-by-text demonstration in the c7 line.

## Key contributions
- First model to jointly learn LM and KE such that text-derived entity vectors are competitive on KG completion.
- Establishes inductive KE evaluation as a standard protocol for text-aware KG embedding.
- Wikidata5M dataset with description alignment.

## Limitations / gaps
- Pretraining cost is large (RoBERTa scale).
- Relations are still id-embedded, so brand-new relations are not handled.
- General-domain, not biomedical; transferring to UMLS/Hetionet/PrimeKG requires re-pretraining or domain adaptation.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not modeled.
- **i2 (meeting node)**: not modeled; KEPLER is structure-blind beyond triple supervision.
- **i3 (pooling noise + attention limit)**: only LM self-attention; no graph aggregator, so no over-smoothing or pooling story.
- **i4 (node-name semantic prior)**: **strongest direct prior art** in c7. Encoder: RoBERTa initialised from RoBERTa-base, jointly fine-tuned with TransE-style KE loss. Fusion: there is no fusion, the text encoding *is* the entity embedding. The fact that this works inductively on Wikidata5M is empirical evidence that for our S2 setting an LM encoding of node name/description is a viable cold-start-stable signal. Our differentiation: we should **combine** this signal with structure (path, meeting node), not replace structure with it.

## Notes
- When we cite c7 in related work, KEPLER is the canonical example that "text-encoder = entity embedding" works inductively.
- For biomedical drugs, the analog is BioBERT/PubMedBERT-encoded drug descriptions, which Alshahrani 2022 and FuseLinker 2024 explore.
