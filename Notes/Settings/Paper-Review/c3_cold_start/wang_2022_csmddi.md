# Predict Multi-Type Drug-Drug Interactions in Cold-Start Scenario (CSMDDI)

- **Authors**: Zun Liu, Xing-Nan Wang, Hui Yu, Jian-Yu Shi, Wen-Min Lu (et al.)
- **Year / Venue**: 2022, BMC Bioinformatics
- **Link**: https://link.springer.com/article/10.1186/s12859-022-04610-4 (PubMed: 35172712)
- **Read depth**: full-text (abstract+method)
- **Cluster**: c3

## TL;DR
Multi-type DDI predictor explicitly designed for two cold-start scenarios (one-new-drug = S1, two-new-drugs = S2). Learns a *mapping function* from drug attributes to drug network embeddings so that new drugs (with attributes but no edges) can be embedded into the DDI graph.

## Problem & Setting
Defines exactly the two cold-start scenarios our paper uses:
- **S1**: one new drug + one known drug.
- **S2**: two new drugs (both unseen).

Multi-type DDI: predicts both *whether* an interaction occurs and *what type*. Built on DrugBank multi-type DDI labels.

## Method (core)
1. Train on known drugs: learn DDI-graph embedding via SVD / GAE / TransE / RESCAL (four variants tested).
2. Simultaneously learn a **mapping function**: drug attributes (chemical descriptors, target profiles, side-effects, etc.) → embedding space.
3. At test time, for a new drug with no DDI edges: feed its attributes through the mapping → use predicted embedding for downstream DDI scoring.

## Cold-start handling
- Cold-start handled via **attribute → embedding regression**, not message passing.
- Requires drug attributes (chemical descriptors, targets, side-effects); new drug needs at least some attribute coverage.
- No KG, no text descriptions. Attributes are structured tabular features.

## Key contributions
- Explicit S1 and S2 splits with proper cross-validation.
- Demonstrates that multi-type DDI is harder in S2 than binary DDI.
- Tests four embedding backbones (SVD, GAE, TransE, RESCAL).

## Limitations / gaps (as relevant to our insights)
- Attribute-based mapping is shallow; struggles when new drugs have unusual attribute profiles.
- No biomedical KG, no semantic text — purely tabular attribute regression.
- Performance drop from transductive to S2 is substantial (reported in their tables but not summarized in one line).
- Baselines (DeepDDI, DDIMDL) are pre-GNN.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Partial — uses multiple attribute groups (targets ≈ PK; side-effects ≈ PD), but treats them as a flat feature vector, no architectural distinction.
- **i2 (meeting node + over-smoothing)**: Not applicable (no GNN).
- **i3 (pooling noise + attention limit)**: Not applicable.
- **i4 (node-name semantic prior)**: Not addressed — attributes are numerical fingerprints, not text.

## Notes
- Provides a clean reference for the S1 vs S2 distinction in the *drug-cold-start* sense (complementary to Pahikkala 2021).
- Useful low baseline to bracket KG/GNN methods against.
