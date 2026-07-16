# MUFFIN: Multi-Scale Feature Fusion for Drug-Drug Interaction Prediction

- **Authors**: Yujie Chen, Tengfei Ma, Xixi Yang, Jianmin Wang, Bosheng Song, Xiangxiang Zeng
- **Year / Venue**: 2021 / Bioinformatics, 37(17):2651-2658
- **Link**: https://academic.oup.com/bioinformatics/article/37/17/2651/6171181 ; code https://github.com/xzenglab/MUFFIN
- **Read depth**: full-text
- **Cluster**: c6 (borderline c1; included here because the MPNN molecular branch is central and tested independently)

## TL;DR
MUFFIN fuses two complementary signals: an **MPNN over the molecular graph** (chemistry side) and **TransE embeddings over a biomedical knowledge graph** (semantic side). Fusion is bi-level: a cross-product matrix processed by a CNN (local + global) plus element-wise scalar interaction.

## Problem & Setting
Three DDI tasks on three datasets: binary (DRKG, 1.18M pairs), multi-class (DeepDDI's split, 81 types), multi-label (TWOSIDES, 200 types). All transductive.

## Method (core)
- **Molecular branch**: SMILES → 2D molecular graph → MPNN → mean readout over atom embeddings.
- **KG branch**: TransE on biomedical KG (DRKG-derived) yielding entity embeddings for each drug.
- **Fusion**:
  - Cross-level: outer product of the two drug vectors → d×d matrix → CNN → local + global features.
  - Scalar-level: element-wise product captures co-active dimensions.
- Concatenated features → MLP classifier.

## Cold-start handling
Not addressed. Evaluation is fully transductive — both drugs are in the training pool. KG embeddings (TransE) are non-inductive by construction: a drug missing from the KG has no embedding.

## Key contributions
- Clean architectural pattern for **molecular + KG** fusion at the drug-pair level.
- Bi-level (cross-product CNN + element-wise) fusion that outperforms naive concatenation.
- Strong transductive numbers on three diverse DDI tasks.

## Limitations / gaps (as relevant to our insights)
- No cold-start evaluation; KG branch is non-inductive.
- Mean pooling on the MPNN side is the canonical "uniform pooling" we critique.
- No mechanism for injecting external pharmacological priors into the fusion.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Partial. The molecular MPNN is PK-flavoured; the KG side carries some PD-ish signal (gene/target/disease relations) but the paper does not separate the two paradigms. We can cite MUFFIN as a precedent for "two complementary signal sources" while noting it does not articulate the PK/PD split.
- **i2 (meeting node + over-smoothing)**: KG branch uses TransE, no path / no meeting-node reasoning. The molecular branch uses mean pool, which is exactly the over-smoothed readout.
- **i3 (pooling noise + attention limit)**: Direct example of the i3 critique. The MPNN readout is **mean over atoms**, no attention, no prior injection. The bi-level fusion is on the **drug-level** vectors, not at substructure resolution, so any substructure noise is already baked in.
- **i4 (node-name semantic prior)**: MUFFIN uses **structured KG triples** as the non-molecular signal, not node-name text. This is the closest cluster-c6 precedent for "non-molecular signal helps", but it still requires KG membership (no cold-drug solution). Our node-name text approach is strictly more inductive.

## Notes
Useful as the "molecular + KG combined" baseline in the related-work table. Cite when arguing that fusion alone is not enough — the underlying readout and KG embedding are both non-inductive.
