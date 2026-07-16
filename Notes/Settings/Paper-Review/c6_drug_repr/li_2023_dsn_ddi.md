# DSN-DDI: An Accurate and Generalized Framework for Drug-Drug Interaction Prediction by Dual-View Representation Learning

- **Authors**: Zimeng Li, Shichao Zhu, Bin Shao, Xiangxiang Zeng, Tong Wang, Tie-Yan Liu
- **Year / Venue**: 2023 / Briefings in Bioinformatics, 24(1):bbac597
- **Link**: https://academic.oup.com/bib/article/24/1/bbac597/6966537 ; https://pubmed.ncbi.nlm.nih.gov/36592061/
- **Read depth**: full-text (publisher HTML)
- **Cluster**: c6

## TL;DR
DSN-DDI ("Dual-view Substructure Network") learns drug substructures from two complementary views simultaneously: the **intra-view** (a drug attends within its own molecular graph) and the **inter-view** (a drug's atoms attend to *all* atoms of its partner via a bipartite cross-drug graph). Local (atom-level) and global (drug-level, via SAGPooling) modules are stacked iteratively, and **all hierarchical layer representations are retained** (not just the final layer) for the decoder. Pure molecular, no KG, no domain knowledge.

## Problem & Setting
Multi-class / multi-label DDI on DrugBank and TWOSIDES. Evaluates transductive (warm) plus two inductive partitions. Note their label convention is **swapped vs ours**: their "S1" = both drugs unseen (= our S2); their "S2" = one drug unseen (= our S1).

## Method (core)
- SMILES to molecular graph, 55-dim atom features.
- DSN encoder block fuses intra-view and inter-view messages at each layer; receptive fields at different scales across stacked blocks yield substructures of different sizes (the "hierarchical" aspect).
- SAGPooling produces importance-weighted drug-level substructure embeddings.
- Decoder integrates all hierarchical global representations for the final relation-typed score.

## Cold-start handling
**Explicit and strong.** Both-drugs-unseen (their "S1"): ~81.79% AUC (their reported +7.07% relative over second-best). One-drug-unseen (their "S2"): ~91.01% AUC. The inter-view (partner-conditioned) cross-attention is the key reason it generalizes to unseen drugs better than SSI-DDI / GMPNN-CS, while staying purely structural.

## Key contributions
- Joint intra-view + inter-view substructure learning in one encoder (vs SSI-DDI's separate-then-co-attend).
- Retaining all hierarchical layer representations rather than only the last layer.
- Best-in-class purely-molecular cold-start numbers at publication; de facto strong molecular baseline.

## Relevance to our insights
- **Adding hierarchical molecular substructure to MNAH (KG + meeting-node)**: DSN-DDI is the strongest evidence that a *partner-conditioned, multi-scale* substructure encoder is cold-start-stable on its own (works from chemistry alone, no drug-ID). Its inter-view bipartite cross-attention is the mechanism most worth borrowing — it conditions one drug's substructure salience on the partner, which composes naturally with a shared mediator/meeting-node head. The "keep all layer representations" trick gives a ready hierarchy to fuse at multiple depths.
- **i3 (pooling/attention limit)**: improves on flat co-attention but still injects no external (KG/biomedical) prior — exactly the gap MNAH's KG side fills.
- Convention warning: when citing its cold-start numbers, remap their S1/S2 to ours.
