# GL-Fusion: Rethinking the Combination of Graph Neural Network and Large Language Model

- **Authors**: Haotong Yang, Xiyuan Wang, Qian Tao, Shuxian Hu, Zhouchen Lin, Muhan Zhang
- **Year / Venue**: 2024 (arXiv:2412.06849, late 2024 / active 2025)
- **Link**: https://arxiv.org/html/2412.06849v1
- **Read depth**: abstract + method
- **Cluster**: c7

## TL;DR
A graph-language architecture that keeps raw node/edge text available to a GNN through graph-text cross-attention and structure-aware transformer layers, with a twin GNN+LLM predictor. For knowledge-graph completion it uses **GNN predictions only** at inference, with text embeddings preserved through the layers and feeding the GNN readout. Evaluated inductively on FB15k-237-ind (entities unseen at train time).

## Problem & Setting
General text-attributed graph learning and KG completion (FB15k-237, citation graphs). **Not biomedical and not DDI** — included here as the closest published example of the *mechanism* we want: text/LLM signal feeding the GNN readout for inductive link prediction.

## Method (core)
- Structure-aware transformer layers inject message passing into attention so text and structure are processed jointly.
- Graph-text cross-attention reads from raw node text instead of compressing it to a single vector.
- Twin predictor: LLM head (autoregressive text) + GNN head (parallel classification). KG completion uses the GNN head.
- Inductive eval: test entities unseen during training.

How node-name text is incorporated: node and edge text embeddings are carried through every transformer layer via cross-attention and feed the **GNN readout components**, rather than being used only as a frozen node-init feature.

## Cold-start handling
Genuinely inductive on FB15k-237-ind: predicts links for entities unseen at training, which is the structural analog of S2 cold-start on a general KG.

## Key contributions
- Demonstrates that keeping text live and fusing it at the GNN readout (not just at init) improves inductive KG completion.
- Provides an architectural template for "text-at-readout" link prediction.

## Limitations / gaps
- No biomedical / DDI evaluation; encoders are general LLMs, not PubMedBERT.
- Heavy joint transformer architecture; no notion of a single mediator/meeting node.
- No relation-type or PK/PD-aware gating.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not addressed.
- **i2 (meeting node + over-smoothing)**: no explicit meeting-node anchor; pair representation is implicit in the joint model.
- **i3 (pooling noise + attention limit)**: directly relevant — it argues the same thing we do, that compressing text into a single node-init vector loses information, so it keeps text live into the readout. Constructive support for "inject external text priors at pooling."
- **i4 (node-name semantic prior)**: **the closest published instance of the readout-stage text-injection mechanism**, but on general (Freebase-style) KGs, not biomedical, not DDI, and with no mediator-node focus. It establishes that "text at readout for inductive link prediction" is a known idea in general graph ML, so our novelty must be the **biomedical-mediator + DDI cold-start + PubMedBERT** specialization, not the bare readout-injection mechanism.

## Notes
- Important for honest novelty positioning: cite as prior art for the *generic* text-at-readout idea, then carve our contribution as the DDI-mediator-S2 instantiation.
- Not a DDI baseline (different domain), but a methodological reference.
