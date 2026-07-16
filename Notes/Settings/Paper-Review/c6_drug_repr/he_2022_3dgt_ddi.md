# 3DGT-DDI: 3D Graph and Text Based Neural Network for Drug-Drug Interaction Prediction

- **Authors**: Haohuai He, Guanxing Chen, Calvin Yu-Chian Chen
- **Year / Venue**: 2022 / Briefings in Bioinformatics, 23(3):bbac134
- **Link**: https://academic.oup.com/bib/article/23/3/bbac134/6576451
- **Read depth**: full-text
- **Cluster**: c6

## TL;DR
3DGT-DDI argues that **2D molecular graphs lose conformational information** relevant to DDI. It generates 3D conformations via MMFF force-field optimisation, runs SchNet (a 3D-aware GNN) over them, and fuses the resulting drug embedding with SciBERT-encoded textual context from biomedical sentences (DDIExtraction-2013 task setting).

## Problem & Setting
DDIExtraction 2013 shared task: classify drug-pair sentences from DrugBank / MEDLINE into five interaction categories (Advice / Effect / Mechanism / Int / Negative). Secondary evaluation on DrugBank multi-class DDI.

## Method (core)
- **3D conformation**: SMILES → RDKit + MMFF → 3D coordinates.
- **3D GNN**: SchNet over the 3D molecular graph (atoms + interatomic distances) gives a conformation-aware drug embedding.
- **Text branch**: SciBERT encodes the sentence containing the drug pair; position embeddings mark where each drug appears.
- **Fusion**: hidden-layer feature fusion via CNN over concatenated branches; softmax over five DDI categories.

## Cold-start handling
Not explicitly evaluated for cold drugs. However, both branches are inductive in principle — SchNet works from coordinates, SciBERT works from text — so the architecture is cold-compatible by construction.

## Key contributions
- First DDI paper to use **3D conformations** rather than 2D graphs.
- Demonstrates value of fusing molecular structure with biomedical literature semantics.
- 84.48% macro-F1 on DDIExtraction 2013.

## Limitations / gaps (as relevant to our insights)
- 3D conformation generation is force-field-based, not experimental; quality varies.
- Text branch uses **sentence-level** context from literature, not entity-level node-name text — so it requires a relevant sentence to exist (not pure cold-start).
- DDIExtraction 2013 task is relation extraction from text, slightly different from DDI link prediction.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: 3D structure is even more strongly PK-side than 2D graphs (encodes binding-pocket-relevant geometry). The text branch adds some PD signal (clinical effect language). Mixed paradigm, but PK-dominated.
- **i2 (meeting node + over-smoothing)**: 3D atomic distances are the "meeting geometry" between atoms. Not a biomedical meeting node, but conceptually the closest thing in cluster c6 to "anchor on the interaction site." Useful contrast.
- **i3 (pooling noise + attention limit)**: SchNet readout still pools over atoms; CNN fusion does not inject external pharmacological priors. Same i3 limitation.
- **i4 (node-name semantic prior)**: Critically relevant. 3DGT-DDI uses **biomedical text from sentences**, which is the closest c6 precedent for using text as a complementary signal. Our work argues for **node-name** text rather than sentence-level text — node names are always available (cold-stable), whereas sentences may not exist for new drugs. Strong contrast point.

## Notes
Most directly aligned c6 paper with our text-prior intuition. Use as the citation showing that "text helps DDI" while differentiating on (a) what text (node-name vs sentence) and (b) cold-start guarantees.
