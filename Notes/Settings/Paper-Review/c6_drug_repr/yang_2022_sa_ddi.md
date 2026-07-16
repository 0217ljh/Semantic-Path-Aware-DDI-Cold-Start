# Learning Size-Adaptive Molecular Substructures for Explainable DDI by Substructure-Aware GNN (SA-DDI)

- **Authors**: Ziduo Yang, Weihe Zhong, Qiujie Lv, Calvin Yu-Chian Chen
- **Year / Venue**: 2022 / Chemical Science, 13(29):8693-8703
- **Link**: https://pubs.rsc.org/en/content/articlehtml/2022/sc/d2sc02023h ; code https://github.com/guaguabujianle/SA-DDI
- **Read depth**: full-text
- **Cluster**: c6

## TL;DR
SA-DDI builds on D-MPNN with a **substructure attention** mechanism that scores substructures of varying radii, plus a **Substructure-Substructure Interaction Module (SSIM)** that uses one drug to identify which substructures matter in its partner (asymmetric, conditional attention). Evaluates both warm-start and **cold-start** (new-drug-pair and new-pair) on DrugBank and TWOSIDES. Provides pharmacologist-consistent visual interpretations.

## Problem & Setting
Multi-class DDI on DrugBank (86 types, 1706 drugs, ~192K tuples) and multi-label DDI on TWOSIDES (645 drugs, 963 types). Three splits: warm-start, S1 (one drug new), and S2 (both drugs new).

## Method (core)
- Atom + bond features via RDKit from SMILES.
- D-MPNN encoder with substructure-attention scoring at each iteration, producing variable-radius substructure embeddings.
- **SSIM**: rather than pool drug A independently, SSIM uses drug B's representation as a query to weight A's substructures (and vice versa). This is partner-conditioned attention, an upgrade over symmetric co-attention.
- Final score = aggregated SSIM-weighted interactions across substructure pairs.

## Cold-start handling
**Explicit**. Reports both S1 (new drug paired with seen drug) and S2 (both-drug-new) performance. Notes significant performance degradation in cold-start, reflecting the structural-generalisation ceiling of purely molecular methods.

## Key contributions
- Partner-conditioned substructure attention (SSIM) — moves beyond symmetric co-attention.
- Strong interpretability: visualised clusters recover known pharmacological functional groups (e.g. barbituric acid, sulfonamide for dicoumarol).
- Explicit cold-start benchmark numbers that the cluster can cite.

## Limitations / gaps (as relevant to our insights)
- **Cold-start drop is large** — the paper itself acknowledges performance "degraded significantly in cold-start." Pure molecular features hit a ceiling.
- No external prior injection: attention is fully data-driven.
- No biomedical / KG / phenotype signal; PK-only.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Pure PK. Reinforces that molecular substructure methods cannot reach the PD ceiling because they have no access to downstream effects.
- **i2 (meeting node + over-smoothing)**: SSIM's partner-conditioned attention is a small step toward "anchor on the interaction not on the drug" — the partner's representation is used as a query. Useful precedent for our meeting-node argument, but the anchor is still chemical.
- **i3 (pooling noise + attention limit)**: SSIM tries to **fix the symmetric-pooling problem** by making attention asymmetric (partner-conditioned). This is a partial fix consistent with our i3 narrative. But it still cannot inject *external* priors (e.g. known DDI pathway annotations) — the upper bound on data-driven attention.
- **i4 (node-name semantic prior)**: SA-DDI's cold-start degradation is the empirical case for adding an external cold-stable signal. Our argument: node-name biomedical text fills exactly this gap.

## Notes
Best citation for "even partner-conditioned substructure attention degrades in cold-start." Strong empirical anchor for the i3 + cold-start composite argument.
