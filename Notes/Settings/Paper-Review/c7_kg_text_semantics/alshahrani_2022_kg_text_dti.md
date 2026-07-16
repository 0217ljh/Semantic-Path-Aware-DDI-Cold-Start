# Combining Biomedical Knowledge Graphs and Text to Improve Predictions for Drug-Target Interactions and Drug-Indications

- **Authors**: Mona Alshahrani et al.
- **Year / Venue**: 2022 / PeerJ
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8988936/
- **Read depth**: abstract + method
- **Cluster**: c7

## TL;DR
Compares graph-only, text-only, and graph+text embedding strategies for drug-target interaction and drug-indication prediction. Text comes from PubMed abstracts (PubTator-annotated, entities normalised to KG IRIs), encoded with Word2Vec skip-gram. KG embedding is TransE on random-walk corpora. Two fusion strategies: concatenated vs jointly-learned.

## Problem & Setting
Drug-target interaction (STITCH, Yamanishi) and drug-indication (SIDER). Not DDI directly, but the methodology (text + KG fusion for drug entities) is the c7 template applied to drug repurposing.

## Method (core)
- Build a heterogeneous biomedical KG from public sources.
- Text corpus: PubTator-annotated PubMed abstracts (~27.6M), entity mentions replaced by KG IRIs so the same token is reused in graph and text.
- Encoders: TransE (graph) and Word2Vec skip-gram (text).
- Fusion strategies:
  1. **Concatenate** Word2Vec(text) and Word2Vec(graph-from-random-walks).
  2. **Jointly learn** one Word2Vec by merging text and walk corpora before training.

How node-name text is incorporated: at the token level. Because PubTator-normalised abstracts substitute KG IRIs for entity mentions, the same vocabulary item gets a Word2Vec embedding from both literature context and random-walk context.

## Cold-start handling
**Acknowledged but not solved.** The authors explicitly note both modes "fail in the zero-shot scenario when an entity is absent from either the knowledge graph or the text corpus." They expand the evaluation set to the union of KG and literature, so an entity present in only one source still gets an embedding — a soft form of cold-start coverage but not unseen-by-both.

## Key contributions
- Clean controlled comparison of graph-only vs text-only vs fused.
- Shows joint training of one Word2Vec over text + walks generally beats concatenation.
- Empirically validates that text adds non-redundant signal over a biomedical KG for drug-target and drug-indication tasks.

## Limitations / gaps
- Word2Vec, not BERT-family; no contextual encoding.
- Zero-shot for fully novel entities is not solved.
- DTI/DTA setting, not DDI; transfer to S2-DDI is not shown.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not separated.
- **i2 (meeting node)**: not addressed.
- **i3 (pooling noise + attention limit)**: shallow embedding only.
- **i4 (node-name semantic prior)**: **moderate.** Shows that even shallow text signals over normalised biomedical literature meaningfully complement KG embeddings for drug tasks. Encoder: Word2Vec skip-gram (predecessor to BERT-style encoders). Fusion: either concat or shared-vocabulary joint training. Provides evidence that the cold-start benefit from text is real but ceiling-limited by the encoder; modern BERT-family work (FuseLinker, BioMedKG) extends this.

## Notes
- Pair with FuseLinker for "shallow vs deep" text-encoder comparison in our related-work paragraph.
