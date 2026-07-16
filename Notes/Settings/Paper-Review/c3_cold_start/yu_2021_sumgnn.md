# SumGNN: Multi-typed Drug Interaction Prediction via Efficient Knowledge Graph Summarization

- **Authors**: Yue Yu, Kexin Huang, Chao Zhang, Lucas M. Glass, Jimeng Sun, Cao Xiao
- **Year / Venue**: 2021, Bioinformatics
- **Link**: https://arxiv.org/abs/2010.01450
- **Read depth**: abstract + method (via Communications Medicine + GitHub)
- **Cluster**: c3 (inductive low-resource DDI on KG subgraphs)

## TL;DR
Subgraph-based KG-DDI predictor with self-attention pruning. Inductive by construction — different subgraph per pair — and explicitly designed to help **low-resource (rare) DDI types**. Foundational baseline for KG-based cold-start DDI.

## Problem & Setting
Multi-typed (multi-relation) DDI prediction over DrugBank + biomedical KG (HetioNet-derived). Inductive setting: per drug-pair, only the local subgraph is used, so unseen drugs only need KG attachment. Evaluation focuses on rare DDI types (relations occurring <50 times).

Cold-start framing in this paper is **relation-frequency cold-start** rather than full drug-cold-start. Still considered a c3 anchor because the inductive subgraph mechanism is the architectural template most later cold-start DDI papers extend.

## Method (core)
1. **Subgraph extraction**: for drug pair (u,v), extract the k-hop enclosing subgraph in the KG.
2. **R-GCN over subgraph**: per-relation message passing.
3. **Layer-independent self-attention summarization**: assigns importance scores to edges and prunes low-importance edges, yielding a "reasoning path."
4. **Integration**: subgraph representation + node embeddings → multi-channel decoder.

## Cold-start handling
- Subgraph anchoring → inductive: a new drug pair just needs a connected subgraph in the KG.
- Drug node features come from KG structure / initial embeddings; no text.
- Performance under truly new drugs depends on whether the new drug has enough KG attachment edges.

## Key contributions
- Subgraph reformulation = scalable + inductive bias for DDI prediction on biomedical KG.
- Up to 5.54% F1 over best baseline; **38.19% improvement on F1** for rare relation types (<50 examples).
- Edge-level attention scores provide interpretable reasoning paths.

## Limitations / gaps (as relevant to our insights)
- Attention scores are learned end-to-end from end-task supervision — no external semantic prior.
- No PK/PD-aware path-type distinction.
- For truly emerging drugs with sparse KG attachment, the subgraph becomes degenerate (single-edge or empty).
- No text features on KG nodes.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. All KG relations (protein-target, gene-expression, side-effect, etc.) flow through the same attention head.
- **i2 (meeting node + over-smoothing)**: Subgraph anchoring is closer to meeting-node spirit than pure drug-to-drug GNN, but doesn't explicitly identify which node is the mediator. With k=3-4 hops, classical over-smoothing risk applies.
- **i3 (pooling noise + attention limit)**: Pure end-task-learned attention — exactly the case our i3 argues is insufficient. Their own results (degradation on low-resource relations even after attention) implicitly support this.
- **i4 (node-name semantic prior)**: Not used. KG nodes are entities without text features.

## Notes
- Standard baseline for any new cold-start DDI method (used by KnowDDI, EmerGNN, KAGE, TextDDI as baseline).
- F1 on DrugBank: ~88.88% (transductive); drops considerably under S2 (re-benchmarked by DDI-Ben).
- Code: github.com/yueyu1030/SumGNN.
