# TIGER: Dual-Channel Learning Framework for DDI Prediction via Relation-Aware Heterogeneous Graph Transformer

- **Authors**: Xiaorui Su, Pengwei Hu, Zhu-Hong You, et al.
- **Year / Venue**: 2024 / AAAI (vol. 38)
- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/27777 ; code https://github.com/Blair1213/TIGER
- **Read depth**: abstract + architecture + ColdDDI re-eval numbers
- **Cluster**: c1 (also relevant to our multimodal molecular+KG line)

## TL;DR
Dual-channel DDI model. One channel runs a relation-aware heterogeneous-graph Transformer over a biomedical KG; the other encodes the drug molecular graph. Graph-level and node-level embeddings from both channels are concatenated/fused for the DDI head. SOTA on transductive (warm) DrugBank splits; collapses on double-drug cold-start.

## Method (core)
- KG channel: relation-aware self-attention over the heterogeneous KG, capturing long-range / high-order semantic relations between entity pairs (Transformer instead of stacked GNN layers).
- Molecular channel: GNN over the 2D molecular graph for structural drug features.
- Fusion: combines node-level (KG) and graph-level (molecular) embeddings; the two channels are trained jointly against the DDI label.
- The fusion is supervised end-to-end by the DDI edge — there is no modality-alignment objective independent of the label.

## Cold-start handling
None explicit. Original paper evaluates three real-world datasets in a transductive regime. The KG channel relies on the unseen drug already having a populated KG neighborhood; the molecular channel always works, but the two channels are only made compatible THROUGH the supervised DDI signal, so for an unseen pair (both drugs absent at train time) the fused space is uncalibrated.

## ColdDDI re-evaluation numbers (Liang et al. ColdDDI v7, 1,900-drug set, 3-seed)
- AUC-ROC: S0 0.976, S1 0.771, **S2 0.630** (drop ~0.35 from S0)
- Recall: S0 0.918, S1 0.595, **S2 0.426** (lowest-but-one Recall among all baselines on S2; below KG-only EmerGNN's 0.619)
- Stratified S2 Recall (800-drug subset): PK-A 0.457, PK-B 0.443, PD-A 0.617, PD-B 0.489; A–B gap +0.071.

## Why it may fail under cold-start (modality misalignment)
The two channels are aligned only implicitly via the supervised DDI edge. When both drugs are unseen, neither the KG embedding (sparse/absent neighborhood) nor the joint fusion has a calibrated correspondence to the molecular embedding, so the extra KG modality cannot rescue the molecular channel and may add noise. TIGER's S2 AUC (0.630) is BELOW the pure molecular DSN-DDI (0.711) and KG-only EmerGNN (0.707) — direct evidence the fusion collapses rather than helps.

## Relevance to our project
- This is one of the two named "collapse" baselines (KG-integrated group). Our MNAH multimodal extension must beat TIGER's S2 0.630 AUC to claim a real gain.
- Motivates DDI-edge-INDEPENDENT alignment: TIGER's only alignment signal is the held-out label, which is exactly unavailable at cold-start.
