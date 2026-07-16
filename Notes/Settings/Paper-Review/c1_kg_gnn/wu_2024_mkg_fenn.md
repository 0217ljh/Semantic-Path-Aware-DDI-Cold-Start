# MKG-FENN: A Multimodal Knowledge Graph Fused End-to-End Neural Network for Accurate DDI Prediction

- **Authors**: Di Wu, Wu Sun, Yi He, Zhong Chen, Xin Luo
- **Year / Venue**: 2024 / AAAI (vol. 38, 10216-10224)
- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/28887 ; code https://github.com/wudi1989/MKG-FENN
- **Read depth**: abstract + architecture + ColdDDI re-eval numbers
- **Cluster**: c1 (multimodal molecular+KG fusion)

## TL;DR
Builds a multimodal KG (MKG) from four relation views (drug-chemical-entity, drug-substructure, drug-drug, molecular-structure), runs a four-channel GNN, then fuses all channel features through an MLP trained end-to-end against DDI events. Near-perfect on transductive DrugBank; saturates/collapses on double-drug cold-start.

## Method (core)
- Four-view MKG: (1) drugs-chemical-entities, (2) drug-substructures, (3) drugs-drugs, (4) molecular structures.
- Four-channel GNN extracts high-order + semantic features per view.
- FENN: an MLP fuses the four channel embeddings, learned end-to-end with the DDI-event classifier.
- The drug-drug view bakes the supervised relation INTO the representation; molecular/substructure features are aligned to KG features only through the joint classifier loss.

## Cold-start handling
None. Reported as transductive (drugs appear in both train and test). The drug-drug view is a transductive shortcut — at cold-start both drugs are unseen so that channel is empty, and the fusion has never learned to operate without it.

## ColdDDI re-evaluation numbers (Liang et al. ColdDDI v7, 1,900-drug set, 3-seed)
- AUC-ROC: S0 0.999, S1 0.703, **S2 0.563** (largest absolute drop, ~0.44 from S0; LOWEST S2 AUC of every baseline in the table)
- Recall: S0 0.987, S1 0.566, **S2 1.000** — but flagged ‡: "S2 Recall = 1.000 arises because it predicts nearly all S2 pairs as positive" (degenerate, not real recall).
- Stratified S2 Recall (800-drug subset): PK-A/PK-B/PD-A/PD-B all ~0.99, A–B gap +0.006, but flagged † as a SATURATION ARTIFACT (predicts everything positive), explicitly EXCLUDED from bolding.

## Why it may fail under cold-start (modality misalignment)
This is the canonical "collapse" example. With the drug-drug KG view dead (both drugs unseen) and the molecular/chemical channels never aligned to the KG except through the now-broken classifier, the fused model degenerates to a near-constant positive predictor — AUC 0.563 (near chance) while Recall = 1.000 because it labels almost everything positive. The extra modalities yield NEGATIVE value at cold-start.

## Relevance to our project
- Strongest empirical motivation for our problem statement: a 4-view multimodal KG model collapses to near-random AUC and degenerate recall on S2.
- Confirms the diagnosis: end-to-end label-supervised fusion gives no usable cross-modal correspondence for unseen pairs. Need an alignment objective that does NOT route through the DDI edge.
