# MDDI-SCL: predicting multi-type drug-drug interactions via supervised contrastive learning

- **Authors**: Shenggeng Lin, Weizhi Chen, Gengwang Chen, Songchi Zhou, Dong-Qing Wei, Yi Xiong
- **Year / Venue**: 2022, Journal of Cheminformatics, 14:81
- **Link**: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00659-8
- **Read depth**: full-text
- **Cluster**: c5

## TL;DR
Adds supervised contrastive loss on top of an autoencoder + self-attention encoder for multi-type DDI. The contrastive objective pulls same-event pairs together and pushes different-event pairs apart, directly attacking the head/tail imbalance problem in Deng-65 / DS2-100.

## Problem & Setting
- **Classes**: 65 (DS1) and 100 (DS2) events.
- **Datasets**: DS1 = 572 drugs, 74,528 DDIs; DS2 = 1,258 drugs, 323,539 DDIs.
- Same Task 1 / 2 / 3 cold-start protocol as DDIMDL.

## Method (core)
- Drug encoder: self-attention over concatenated 4-feature input + autoencoder reconstruction (MSE loss).
- Drug-pair latent: concatenate two drug latents -> fusion MLP.
- Three losses simultaneously:
  - MSE reconstruction (drug encoder),
  - Supervised contrastive loss (pair latent),
  - Cross-entropy + focal + label smoothing (classifier).

## Cold-start handling
- Tasks 1/2/3 follow DDIMDL drug-disjoint split.
- Contrastive loss is the key cold-start lever: it shapes the latent space using only label information, so unseen drugs encoded from the same modalities project into the right cluster.
- Reports gains on Tasks 2 (S1) and 3 (S2) over MDF-SA-DDI and DDIMDL.

## Key contributions
- First to apply supervised contrastive learning to multi-type DDI.
- Three-level loss (MSE + SCL + classification) ablation isolates SCL's contribution.
- AUPR 0.9782, accuracy 0.9378 for unseen interaction types between known drugs (Task 1 variant).

## Limitations / gaps
- Still drug-level similarity features; no graph, no KG, no text.
- Performance on Tasks 2-3 varies between DS1 and DS2 - hyperparameter sensitivity admitted.
- PK / PD events not distinguished; the supervised contrastive objective lumps every event into its own cluster regardless of mechanistic family, which could fight a coarser PK-vs-PD structure rather than exploit it.

## Relevance to our insights
- **i1 (PK/PD)**: NOT differentiated. The supervised contrastive loss enforces ONE cluster per event type. This is the strongest form of "treat all types uniformly" - directly counter to our i1 hypothesis that PK and PD form different reasoning regimes. Could be repurposed by adding a hierarchical contrastive objective (event -> PK/PD super-class).
- **i2 (meeting node + over-smoothing)**: No GNN, no over-smoothing concern. No meeting node either.
- **i3 (pooling noise + attention limit)**: Self-attention over feature modalities, not over neighborhoods. Contrastive learning is exactly the kind of "external prior" that attention alone can't inject - so this paper indirectly supports our i3 claim that attention-only is insufficient.
- **i4 (node-name semantic prior)**: Drug name not used.

## Notes
- Class imbalance handling is the cleanest in c5: focal + label smoothing + SCL stacked.
- Tail-class per-event accuracy still not reported individually.
- The SCL framework would be a natural baseline if we want to argue that PK-vs-PD hierarchical contrastive > flat contrastive.
- Code: https://github.com/ShenggengLin/MDDI-SCL
