# DPSP: a multimodal deep learning framework for polypharmacy side effects prediction

- **Authors**: Raziyeh Masumshah, Changiz Eslahchi
- **Year / Venue**: 2023, Bioinformatics Advances, 3(1):vbad110
- **Link**: https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad110/7243195
- **Read depth**: full-text
- **Cluster**: c5

## TL;DR
Multimodal DDI prediction that builds Jaccard-similarity features from five modalities (mono side effects, targets, enzymes, substructures, pathways) and feeds them to a DNN. Benchmarks on DS1/DS2 (Deng) + DS3 (185 polypharmacy adverse effects).

## Problem & Setting
- **Classes**: 65 (DS1), 100 (DS2), 185 (DS3 - new larger polypharmacy benchmark).
- **Datasets**:
  - DS1: 572 drugs, 37,264 DDIs, 65 events.
  - DS2: 1,258 drugs, 161,770 DDIs, 100 events.
  - DS3: 645 drugs, 63,473 DDIs, 185 polypharmacy adverse effects (from TWOSIDES/SIDER/OFFSIDES).

## Method (core)
- Per-modality feature: build mono-feature vector for each drug (e.g., target set, mono side effect set), then compute Jaccard similarity to all training drugs -> per-drug similarity vector.
- Concat / average the per-modality similarity vectors -> deep MLP -> per-event sigmoid (multi-label).
- BCE loss; no class-imbalance-specific objective.

## Cold-start handling
- Not addressed. The paper assumes drugs are in the training similarity basis.
- DS3 includes mono side effects from OFFSIDES, so for a brand-new drug without prior side-effect annotations, the most informative modality is unavailable.

## Key contributions
- Introduces DS3 (185 polypharmacy adverse effects), a useful intermediate between Deng-65/100 and Decagon-964.
- Direct comparison against GNN-DDI, MSTE, MDF-SA-DDI, NNPS, DDIMDL on the same protocol.
- Shows mono-side-effect features (when available) dominate for polypharmacy prediction.

## Limitations / gaps
- Pure similarity-feature DNN - no graph, no text, no attention.
- No cold-start protocol; reported numbers are random pair split.
- Class imbalance acknowledged in related work but unmitigated.
- Mono side effects (drug -> single-drug side effect catalog) is a strong feature precisely because it leaks information from clinical knowledge - making the model dependent on a feature that brand-new drugs lack.

## Relevance to our insights
- **i1 (PK/PD)**: NOT differentiated; 65/100/185 events flat-classified. The fact that mono-side-effect features dominate is indirect evidence for our i1: known clinical phenotype (PD-layer evidence) carries more signal than chemistry (PK-layer features) - exactly the asymmetry i1 predicts.
- **i2 (meeting node + over-smoothing)**: No GNN, no propagation. Drug-drug interaction is computed directly from individual feature concat - no meeting node, no over-smoothing risk.
- **i3 (pooling noise + attention limit)**: Uniform concat/average fusion across modalities. No attention. Directly demonstrates the i3 weakness; later approaches outperform DPSP partly because they add attention or contrastive shaping.
- **i4 (node-name semantic prior)**: NOT used.

## Notes
- DPSP is the cleanest baseline to argue "5-modality similarity features + plain MLP is a strong floor, but transductive only".
- DS3 (185 effects) is a useful evaluation target if we want to bridge to Decagon-style multi-label polypharmacy.
- Code: https://github.com/raziyehmasumshah/DPSP
