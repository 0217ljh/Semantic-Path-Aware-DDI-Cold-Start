# MUFFIN: Multi-Scale Feature Fusion for Drug-Drug Interaction Prediction

- **Authors**: Yujie Chen, Tengfei Ma, Xixi Yang, Jianmin Wang, Bosheng Song, Xiangxiang Zeng
- **Year / Venue**: 2021 / Bioinformatics (vol 37, issue 17)
- **Link**: https://academic.oup.com/bioinformatics/article/37/17/2651/6171181
- **Read depth**: full-text (Oxford academic page extracted)
- **Cluster**: c1

## TL;DR
MUFFIN fuses two complementary drug encoders — an MPNN over SMILES molecular graphs and a TransE embedding over an external biomedical KG (DRKG) — through a bi-level cross-and-scalar fusion module for binary, multi-class, and multi-label DDI prediction.

## Problem & Setting
Three regimes:
- Binary DDI on DRKG (1,178,210 pairs).
- Multi-class on DeepDDI/DrugBank (172,426 pairs / 81 types).
- Multi-label on TWOSIDES (99,002 pairs / 200 types).
5-fold CV; transductive. Cold-start / S2 not evaluated.

## Method (core)
- Molecular branch: MPNN on SMILES-derived molecular graph → structure embedding.
- KG branch: TransE on DRKG entities/relations → semantic embedding.
- Bi-level fusion: cross-level (CNN extracting local + global pair features) and scalar-level (element-wise product).
- Task-specific classification head.

## Cold-start handling
N/A. Both branches need the drug present at training time (TransE embedding lookup, SMILES MPNN trained jointly).

## Key contributions
- Clean dual-branch (molecular + KG-embedding) fusion with substantial macro-F1 gain (up to +12%).
- Shows KG semantic features are complementary to molecular features.
- Strong multi-task results across binary / multi-class / multi-label.

## Limitations / gaps (as relevant to our insights)
- Authors explicitly note DRKG redundancy remains a problem → KG noise.
- TransE embeddings are global ID embeddings; no neighborhood-conditioning, no mediator anchoring.
- No text / name semantics, only ID-based TransE.
- No inductive protocol.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. The KG branch is a single TransE embedding per drug.
- **i2 (meeting node + over-smoothing)**: Not addressed. The fusion is at drug-pair scoring stage, with no shared mediator in the architecture.
- **i3 (pooling noise + attention limit)**: TransE is global and not noise-aware; the paper notes KG noise but does not solve it.
- **i4 (node-name semantic prior)**: Not used. KG nodes are IDs, not text-embedded names.

## Notes
A useful dual-encoder baseline. MUFFIN exemplifies the standard "KG-embedding + molecular" recipe that ignores both i2 and i4.
