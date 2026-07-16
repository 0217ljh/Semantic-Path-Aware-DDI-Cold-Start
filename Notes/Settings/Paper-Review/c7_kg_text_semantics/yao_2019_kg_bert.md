# KG-BERT: BERT for Knowledge Graph Completion

- **Authors**: Liang Yao, Chengsheng Mao, Yuan Luo
- **Year / Venue**: 2019 / arXiv:1909.03193 (widely cited preprint, code released)
- **Link**: https://arxiv.org/abs/1909.03193
- **Read depth**: abstract + key sections
- **Cluster**: c7

## TL;DR
Treats each KG triple (h, r, t) as a natural-language sentence built from entity/relation **text descriptions**, feeds it to BERT, and predicts triple plausibility from the [CLS] vector. First widely used demonstration that pre-trained LMs can encode KG entities directly from their natural-language descriptions, removing the strict dependence on a learned entity-id embedding table.

## Problem & Setting
General-purpose KG completion (triple classification, link prediction, relation prediction) on WN18RR, FB15k-237, UMLS. Not DDI-specific, but the UMLS subset is biomedical and the architecture has been widely adapted to biomedical KGs.

## Method (core)
- Each entity and each relation has a textual "description" (entity name + gloss).
- Input sequence: `[CLS] desc(h) [SEP] desc(r) [SEP] desc(t) [SEP]`.
- Fine-tunes BERT-Base; the [CLS] embedding is fed to a binary classifier (triple positive vs. corrupted negative).
- For link prediction, scores all candidate tails by running BERT once per candidate.

How node names / textual node descriptions are incorporated: **the entity is represented exclusively through its text** during inference. There is no separate learned id-embedding per entity. The graph topology is only seen indirectly via the supervision signal (positive triples).

## Cold-start handling
Not its stated goal, but architecturally KG-BERT is naturally inductive: any new entity that has a text description can be scored without retraining the entity table. This is the property later c7 work (KEPLER, FuseLinker, BioMedKG) builds on for unseen entities. The original paper does not run an explicit inductive split, however.

## Key contributions
- First clean recipe for "BERT-encodes-the-triple-as-text" KGC.
- Established that textual descriptions of entities carry enough signal for competitive link prediction on standard benchmarks, including UMLS.
- Triggered the "text-aware KG embedding" line that c7 is anchored on.

## Limitations / gaps
- Inference is O(|E|) BERT passes per query; expensive at biomedical-KG scale.
- Ignores graph structure beyond positive-triple supervision (no message passing, no path).
- No multi-hop / relational reasoning module.
- Not benchmarked on inductive / cold-start splits in the original paper.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not addressed; KG-BERT treats all relations uniformly.
- **i2 (meeting node + over-smoothing)**: not addressed; no GNN involved.
- **i3 (pooling noise + attention limit)**: BERT self-attention is the only aggregator; no explicit prior injection.
- **i4 (node-name semantic prior)**: **direct ancestor**. KG-BERT is the textbook example that entity-name and entity-description text alone, encoded by a pre-trained LM, can drive KG completion. For our cold-start S2 setting, this validates that the node-name text is a usable cold-start-stable signal independent of the graph table. Encoder used: BERT-Base (general English). For biomedical work, the natural replacements are BioBERT/PubMedBERT/SciBERT, which downstream c7 papers do.

## Notes
- Anchor citation for the "text-as-feature" line we want to differentiate from.
- Differentiation angle for us: KG-BERT uses text in place of structure; our i4 uses text **as a prior alongside** path/meeting-node structure.
