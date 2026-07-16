# KGNN: Knowledge Graph Neural Network for Drug-Drug Interaction Prediction

- **Authors**: Xuan Lin, Zhe Quan, Zhi-Jie Wang, Tengfei Ma, Xiangxiang Zeng
- **Year / Venue**: 2020 / IJCAI
- **Link**: https://www.ijcai.org/proceedings/2020/380
- **Read depth**: abstract-only (full PDF fetch returned binary)
- **Cluster**: c1

## TL;DR
KGNN samples a fixed-size neighborhood per entity in a biomedical KG and aggregates multi-hop neighbor messages biased by the current entity's representation, then scores the drug pair as a function of two drug embeddings. Designed to expand each drug's receptive field across the KG beyond direct DDI edges.

## Problem & Setting
Binary DDI prediction on DrugBank (v5.1.4) using a preprocessed KEGG-drug knowledge graph as external context. Evaluation is transductive (random edge split); cold-start / S2 is not the focus.

## Method (core)
- For every entity, sample a fixed-size neighbor set per hop (similar to GraphSAGE).
- Per-layer aggregation mixes neighbor messages with the entity's current vector ("bias from representation of the current entity").
- Stacked H hops to give receptive field over the KG.
- Pair scorer over two drug embeddings (concat + MLP / inner product variant).

## Cold-start handling
N/A. Drug embeddings are learned per ID; an unseen drug has no embedding at test time.

## Key contributions
- One of the first end-to-end GNNs that learns drug representations from a biomedical KG specifically for DDI.
- Shows KG context (KEGG entities and relations) raises DDI AUC over DDI-only baselines.
- Provides a clean ablation over hop depth and sampling size.

## Limitations / gaps (as relevant to our insights)
- Pure structural aggregation, neighbor identity is an ID embedding only.
- Aggregation is uniform within the sampled neighborhood (sum / mean).
- No mechanism to separate PK-style from PD-style neighbors.
- Strictly transductive; cold-start drugs cannot be embedded.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. KG edges (target, pathway, disease, etc.) are typed but the model does not route PK vs PD reasoning through separate channels.
- **i2 (meeting node + over-smoothing)**: Reinforces the "drug node as anchor" paradigm. Increasing hop depth past 2 hurts performance in the paper's own ablation, consistent with over-smoothing.
- **i3 (pooling noise + attention limit)**: Uniform sampling makes the low-SNR-neighborhood problem worse, the paper has no attention or prior to gate noise.
- **i4 (node-name semantic prior)**: Not used. KEGG entity names are reduced to IDs.

## Notes
Frequently cited as the prototypical "KG + GNN for DDI" baseline. Useful as a published demonstration that hop depth has a hard ceiling on this KG schema (motivates i2).
