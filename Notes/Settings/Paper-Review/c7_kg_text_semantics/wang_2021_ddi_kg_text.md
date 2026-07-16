# Drug-Drug Interaction Predictions via Knowledge Graph and Text Embedding: Instrument Validation Study

- **Authors**: Meng Wang, Haofen Wang, Xing Liu, Xinyu Ma, Beilun Wang
- **Year / Venue**: 2021 / JMIR Medical Informatics
- **Link**: https://medinform.jmir.org/2021/6/e28277 (also PMC8277366)
- **Read depth**: abstract + method
- **Cluster**: c7

## TL;DR
Joint TransR-based KG embedding plus an autoencoder over biomedical-text-derived DDI label vectors, optimised in a shared low-dimensional space. One of the earliest end-to-end frameworks that explicitly fuses biomedical-KG structure and biomedical-text signal for DDI prediction.

## Problem & Setting
DDI prediction. Builds a drug KG from Bio2RDF and extracts DDI label sets from the 2013 DDI corpus. Predicts multi-label DDI relations between drug pairs.

## Method (core)
- KG branch: TransR translation embedding over Bio2RDF triples (71k triples, 305k entities).
- Text branch: each DDI sentence is mapped to a binary label vector (1,053 distinct DDI label types); a deep autoencoder with tanh nonlinearity compresses this into the **relation** embedding.
- Joint loss: TransR margin loss + autoencoder reconstruction loss, sharing the relation space.

How node-name/text is incorporated: text here is used to define **relation** semantics (DDI label vocabulary), not node features. The drugs themselves are still id-embedded from the KG.

## Cold-start handling
Not addressed. Drugs are assumed present in the KG during training. Evaluation removes 30% of DDI edges at random, so it is a transductive S0-like setting. Not appropriate as cold-start prior art, but useful as a "KG+text for DDI" reference.

## Key contributions
- First joint translation-based DDI model that learns from KG and biomedical text in a shared space.
- Shows that text-derived relation representations improve DDI multi-label prediction over pure KG embedding.

## Limitations / gaps
- Drug nodes are id-embedded, not text-encoded. So new drugs cannot be embedded.
- Uses binary label-set autoencoder rather than a contextual LM (no BERT family).
- No inductive / S2 evaluation.
- Bio2RDF as the only KG.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not separated; all DDI labels treated as one relation type space.
- **i2 (meeting node)**: not addressed; pairwise translation only.
- **i3 (pooling noise + attention limit)**: autoencoder reconstruction, no attention.
- **i4 (node-name semantic prior)**: **partial.** Text is leveraged but at the relation-label level, not at the drug-node level. For our purposes this paper shows DDI benefits from text signals, but it does **not** demonstrate the cold-start-stable property because drugs are still id-embedded. We should cite as "text-aware DDI exists, but not at node-name level for cold drugs."

## Notes
- Useful contrast: c7's mainstream way of fusing text (at node level, via BERT) versus this paper's way (at relation level, via autoencoder).
