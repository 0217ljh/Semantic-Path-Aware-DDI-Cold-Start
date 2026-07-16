# DDIMDL: A multimodal deep learning framework for predicting drug-drug interaction events

- **Authors**: Yifan Deng, Xinran Xu, Yang Qiu, Jingbo Xia, Wen Zhang, Shichao Liu
- **Year / Venue**: 2020, Bioinformatics (Oxford), 36(15):4316-4322
- **Link**: https://academic.oup.com/bioinformatics/article/36/15/4316/5837109
- **Read depth**: full-text
- **Cluster**: c5

## TL;DR
Multimodal DNN that fuses four drug-level similarity features (chemical substructures, targets, enzymes, pathways) via per-modality sub-DNNs and an average ensemble to classify DDIs into 65 event types. Establishes the de facto Deng-65 benchmark widely reused by later multi-type DDI work.

## Problem & Setting
- **Classes**: 65 DDI event types extracted from DrugBank text descriptions by dependency parsing + filter (events with > 10 DDIs kept).
- **Dataset**: 572 drugs, 74,528 pairwise DDIs (DrugBank), called the "Deng-65" or DS1 benchmark.
- Multi-class single-label (one event per pair).

## Method (core)
- For each of 4 modalities (substructures 881-D PubChem fp, targets 1162-D, enzymes, pathways KEGG), drugs are encoded as Jaccard similarity vectors against all training drugs.
- Each modality fed through its own 3-layer DNN (512->256->65).
- Final prediction = average ensemble of sub-DNN softmaxes.

## Cold-start handling
- Three explicit tasks:
  - **Task 1** (warm): 5-fold CV on DDI pairs; both drugs seen.
  - **Task 2** (S1): one drug new, other known.
  - **Task 3** (S2): both drugs new.
- Drug-disjoint split for Tasks 2-3. Acc drops 0.8852 -> 0.4075 from Task 1 -> Task 3.
- No special cold-start machinery; relies on feature generalization.

## Key contributions
- First widely adopted multi-event-type DDI benchmark (65 classes, Deng-65 / DS1).
- Demonstrated complementarity of substructure + target + enzyme modalities.
- Three-task protocol (warm / one-new / both-new) reused by MDF-SA-DDI, MDDI-SCL, DPSP, MSEDDI.

## Limitations / gaps (as relevant to our insights)
- Severe class imbalance unhandled (event #1 has 19,620 DDIs; tail events ~10).
- Drug features are pre-computed Jaccard similarity vectors against training drugs - truly novel drugs at inference are projected onto the training-drug basis (a soft cold-start workaround, not a true inductive solution).
- Event types treated as flat 65-way classes; no mechanism-aware grouping.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: NOT differentiated. All 65 events flat-classified; events mixing metabolism (PK) and "increased therapeutic efficacy" (PD) share one softmax. Their per-event accuracy table likely hides systematic PK-vs-PD asymmetry that future work could exploit.
- **i2 (meeting node + over-smoothing)**: No GNN; pure DNN over similarity vectors. Doesn't engage the over-smoothing question but also doesn't anchor on any meeting node - drug-drug pair is the unit.
- **i3 (pooling noise + attention limit)**: Uses unweighted average ensemble across modalities. No attention, no prior injection - directly demonstrates the "uniform pooling is noisy" weakness; later works (MDF-SA-DDI, MSEDDI) replace it with transformers.
- **i4 (node-name semantic prior)**: Not used. Drug name is only an identifier, not a feature. Features are all structural/biological.

## Notes
- Class imbalance acknowledged in limitations: "we will consider novel techniques of dealing with the imbalanced dataset." No focal loss, no resampling.
- Per-class performance not reported in detail; aggregate accuracy 0.8852, AUPR 0.9208.
- Task 3 (both new) collapse to 0.41 acc is the canonical evidence that similarity-vector inputs fail S2 cold-start.
