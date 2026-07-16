# Cold-Start Problems in Data-Driven Prediction of Drug-Drug Interaction Effects

- **Authors**: Pahikkala, Airola, Stock, Waegeman, et al. (re-published taxonomy work in J. Chem. Inf. Model.)
- **Year / Venue**: 2021, JCIM / PMC8147651
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8147651/
- **Read depth**: full-text
- **Cluster**: c3

## TL;DR
Foundational taxonomy paper that formally defines four DDI-prediction tasks of increasing difficulty based on which entities are unseen at test time. Establishes the canonical S0/S1/S2 cold-start vocabulary the field now uses and demonstrates the AUC drop from transductive to "two-new-drugs" setting.

## Problem & Setting
DDI effect prediction is cast as triplet link prediction over (drug, drug, effect). Four tasks:
- `dde^` — known pair, unknown effect (tensor completion, NOT cold-start).
- `dd^e` — unknown pair (both drugs seen separately). First cold-start task = S1 in modern literature.
- `d^de` — one new drug, one seen drug = S2-equivalent (some papers call this S1).
- `d^d^e` — two new drugs = hardest cold-start = canonical S2 in our setting.

The naming differs across the literature, but this paper's distinction is the universally-cited reference. Negative sampling discussion: stresses that proper CV scheme must avoid leakage of the new entity across folds.

## Method (core)
Three-step kernel ridge regression with separable regularization on drugs and effects, enabling efficient cross-validation. The method itself matters less than the taxonomy.

## Cold-start handling
Method is similarity-kernel based — new drugs are represented through their chemical/biological similarity to training drugs. No explicit representation learning; relies entirely on hand-crafted drug feature kernels.

## Key contributions
- Formal taxonomy of the four prediction settings (still the citation backbone for the entire field).
- Demonstrates that AUC-ROC drops from 0.957 (tensor completion) to 0.843 (two new drugs).
- Provides per-task CV schemes that prevent leakage.

## Limitations / gaps (as relevant to our insights)
- Pre-GNN era — no representation learning, no KG integration.
- Drug similarity kernels are hand-crafted features (Tanimoto, etc.), not semantic.
- No discussion of mechanism (PK vs PD).
- No path-level / mediator-node reasoning.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. Treats DDI as a flat link-prediction task.
- **i2 (meeting node + over-smoothing)**: Not applicable (no GNN).
- **i3 (pooling noise + attention limit)**: Not applicable.
- **i4 (node-name semantic prior)**: Not addressed — uses structural kernels only.

## Notes
Most-cited foundational reference for the S0/S1/S2 split convention used throughout c3. Best to cite this when introducing the problem setting; the gap is that nothing in this paper addresses *how* to represent unseen drugs semantically.
