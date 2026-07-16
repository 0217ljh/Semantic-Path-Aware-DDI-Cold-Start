# KnowDDI: Accurate and Interpretable Drug-Drug Interaction Prediction Enabled by Knowledge Subgraph Learning

- **Authors**: Yaqing Wang, Zaifei Yang, Quanming Yao
- **Year / Venue**: 2024 / Communications Medicine (Nature)
- **Link**: https://www.nature.com/articles/s43856-024-00486-y ; https://arxiv.org/abs/2311.15056 ; code https://github.com/LARS-research/KnowDDI
- **Read depth**: full-text (PMC extraction)
- **Cluster**: c1

## TL;DR
KnowDDI learns a pair-specific knowledge subgraph from a large biomedical KG (Hetionet) for each drug pair, assigning a "connection strength" to every edge, pruning irrelevant ones and adding "resemble" edges between similar drugs. The subgraph is then encoded by a GNN whose attention weights and surviving edges form a human-readable explanation.

## Problem & Setting
DrugBank (86 relation types) and TWOSIDES (200 relation types) for multi-class / multi-label DDI prediction, with Hetionet as the external KG. Triplet split 7:1:2; transductive. No formal S2 cold-start protocol.

## Method (core)
- Extract enclosing subgraph around (drug_i, drug_j) from Hetionet.
- Estimate per-edge connection strength as a soft mask; weakly connected edges are dropped.
- Add resemble edges between drugs whose latent features are similar — propagates DDI signal to drugs with sparse KG context.
- GNN over the refined subgraph; concatenate pair embedding for relation classification.

## Cold-start handling
Partial / implicit. The resemble-edge mechanism is designed to help drugs with sparse KG neighborhoods, which is exactly the S2 failure mode, but the paper does not report an S2 split. Authors note molecular features are not used — a direct extension axis.

## Key contributions
- "Connection strength" learning is a principled way to denoise the KG subgraph at the edge level (more than attention reweighting — it can drop edges).
- Resemble edges propagate signal across structurally similar drugs.
- Genuinely interpretable: the explanation is a small surviving subgraph.

## Limitations / gaps (as relevant to our insights)
- Edge-strength learning still operates only on neighbors present in Hetionet — cannot inject external semantic priors.
- KG entities are IDs; node names not encoded as text.
- No PK/PD typing of edges beyond Hetionet's relation labels.
- Transductive evaluation only.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. All Hetionet relations are scored by the same connection-strength head.
- **i2 (meeting node + over-smoothing)**: Strong precedent. By learning connection strength on a pair-conditioned subgraph, KnowDDI implicitly emphasizes the mediator chain between the two drugs. Closer to our framing than KGNN/SkipGNN.
- **i3 (pooling noise + attention limit)**: Direct evidence in favor of i3. The paper's whole motivation is that KG is noisy; their answer (connection strength + resemble edges) is more powerful than attention but still cannot inject knowledge absent from the KG.
- **i4 (node-name semantic prior)**: Not used. Node text not encoded.

## Notes
The strongest single c1 paper for "denoise the KG subgraph" framing. Direct competitor; our i4 (text-name prior) and explicit S2 protocol differentiate us.
