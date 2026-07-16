# K-Paths: Reasoning over Graph Paths for Drug Repurposing and Drug Interaction Prediction

- **Authors**: Tassallah Abdullahi, Ioanna Gemou, Nihal V. Nayak, Ghulam Murtaza, Stephen H. Bach, Carsten Eickhoff, Ritambhara Singh
- **Year / Venue**: 2025, KDD '25 (31st ACM SIGKDD); arXiv 2502.13344
- **Link**: https://arxiv.org/abs/2502.13344 ; code: https://github.com/rsinghlab/K-Paths
- **Read depth**: full-text (abstract + arXiv overview)
- **Cluster**: c2

## TL;DR
K-Paths is a **training-free retrieval** framework: for a query drug pair, it extracts the K shortest *loopless and diversity-aware* paths between them in a biomedical KG (a diversity-extended Yen's algorithm). The retrieved paths are then fed either to an LLM or a GNN as concise interpretable evidence. Achieves a 90% KG-size reduction while preserving GNN performance and gives large gains for LLM-based prediction (e.g., Tx-Gemma 27B +19.8 F1).

## Problem & Setting
Drug repurposing + DDI prediction, including the inductive setting where one or both entities are unseen in training. Model-agnostic — paths are inputs to any downstream predictor.

## Method (core)
1. **Yen's K-shortest loopless paths** between query entities, modified with a diversity penalty that down-weights paths sharing intermediate edges with already-selected paths.
2. **Path serialization** — each path becomes a relational chain (e.g., drug → enzyme → drug) that can be tokenized for an LLM or vectorized for a GNN.
3. **No training** for the path-retrieval stage; the downstream predictor is whatever the user picks.

## Cold-start handling
Explicit support for inductive reasoning: paths can be retrieved as long as the two query entities are connected in the KG, regardless of whether the DDI was seen at training. This is the cleanest cold-start story among c2 papers.

## Key contributions
- Decouples reasoning (paths) from prediction (model) — a clean retrieval interface.
- Diversity-aware Yen's algorithm explicitly addresses the noise / redundancy problem in path enumeration.
- Demonstrates that very small path subsets (≤K paths) carry most of the predictive signal.

## Limitations / gaps (as relevant to our insights)
- Path **selection** is graph-topological (shortest + diverse); it does not use biomedical text semantics to pick paths.
- Path scoring is heuristic, not learned end-to-end.
- For complex DDI mechanisms, K shortest paths may miss longer mechanistically correct paths.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not addressed at the retrieval stage. Downstream LLM could potentially separate PK/PD, but paths themselves are unlabeled.
- **i2 (meeting node + over-smoothing)**: paths bypass GNN message passing entirely on the retrieval side — this is a different answer to the over-smoothing problem (don't aggregate at all, just retrieve). Endpoints are still drugs.
- **i3 (pooling noise + attention limit)**: directly addresses i3 — the diversity-aware Yen's procedure is an explicit external prior over which paths to keep (vs attention which only reweights existing edges).
- **i4 (node-name semantic prior)**: only used downstream by LLM, not in retrieval. A natural extension is text-aware path scoring — close to what we propose.

## Notes
Most recent and closest in philosophy to our angle. The "retrieve diverse paths, then predict" decomposition is a strong baseline structure to compare against. Their LLM variant is essentially a c2-c4 hybrid.
