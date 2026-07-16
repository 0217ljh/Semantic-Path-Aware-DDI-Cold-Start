# MSEDDI: Multi-Scale Embedding for Predicting Drug-Drug Interaction Events

- **Authors**: Liyi Yu, Wangren Qiu, Weizhong Lin, Xiang Cheng, Xuan Xiao, Jiexia Dai
- **Year / Venue**: 2023, International Journal of Molecular Sciences, 24(5):4500
- **Link**: https://www.mdpi.com/1422-0067/24/5/4500
- **Read depth**: abstract-only (MDPI returned 403 on full-text fetch; details from abstract + survey citations)
- **Cluster**: c5

## TL;DR
Three-channel multi-scale drug encoder: (1) knowledge-graph embeddings from a biomedical KG, (2) SMILES notation embeddings (character-level), (3) molecular graph embeddings (GNN). Fuses the three drug-pair channels via self-attention for multi-type DDI event prediction.

## Problem & Setting
- **Classes**: 65 DDI events (Deng-65 / DS1).
- **Dataset**: DS1 (572 drugs, 74,528 DDIs); a small large-scale extension reported.
- Multi-class single-label.

## Method (core)
- Channel A: TransE-style embedding over biomedical KG (drugs, targets, etc.).
- Channel B: SMILES character-level CNN/Transformer encoder.
- Channel C: GNN over molecular graph.
- Per-channel drug-pair representation -> self-attention fusion across channels -> classifier.

## Cold-start handling
- Standard Tasks 1/2/3 protocol carried over from DDIMDL family.
- Multi-scale embeddings improve Task 2/3 (new-drug) over DDIMDL because SMILES + molecular graph are inductive features (no training-drug similarity basis).
- KG channel is partially transductive (new drugs absent from KG fall back).

## Key contributions
- One of the first c5 methods to explicitly use *three* drug representation scales (KG + SMILES + molecular graph) under one model.
- Self-attention across channels is a cleaner version of the MDF-SA-DDI multi-fusion idea.
- Targets the 65-event Deng benchmark with stronger inductive features.

## Limitations / gaps
- Limited evaluation on the larger 100-event DS2 in the abstract.
- No reported class-imbalance handling.
- KG channel suffers for cold drugs (no KG node).
- Multi-scale fusion is uniform across event classes; no mechanism-aware routing.

## Relevance to our insights
- **i1 (PK/PD)**: NOT differentiated. The three channels (KG / SMILES / graph) are roughly aligned with our i1 split (KG ~ PK/system-level via target overlap; SMILES & graph ~ PD/chemistry), but the self-attention fusion is symmetric across event classes - the model never says "for class k use KG more". Strong opportunity for our PK/PD-routed extension.
- **i2 (meeting node + over-smoothing)**: GNN is per-molecule only; no drug-drug propagation -> no over-smoothing concern. KG channel could in principle anchor a meeting node (drug -> target -> drug) but TransE-style decoding doesn't exploit it explicitly.
- **i3 (pooling noise + attention limit)**: Self-attention over 3 channel tokens. Compact, but again no external prior - the attention has to learn from data what to weight.
- **i4 (node-name semantic prior)**: SMILES is used as a text-like sequence (character-level), but the *drug name* and *target name* are still treated as entity IDs in the KG, not as biomedical text.

## Notes
- MSEDDI is a useful "kitchen-sink multi-source" baseline to outperform.
- The fact that SMILES character encoding helps Task 3 (both new) supports the broader thesis that inductive features matter for S2 cold-start.
