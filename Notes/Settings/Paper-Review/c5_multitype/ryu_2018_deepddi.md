# DeepDDI: Deep learning improves prediction of drug-drug and drug-food interactions

- **Authors**: Jae Yong Ryu, Hyun Uk Kim, Sang Yup Lee
- **Year / Venue**: 2018, PNAS, 115(18):E4304-E4311
- **Link**: https://www.pnas.org/doi/10.1073/pnas.1803294115
- **Read depth**: abstract-only (PNAS paywall returned 403; details from official summary + downstream papers)
- **Cluster**: c5

## TL;DR
Original 86-DDI-type benchmark. Uses Structural Similarity Profile (SSP) - each drug encoded as a vector of Tanimoto similarities to a fixed reference drug set - fed to a multi-label DNN producing human-readable sentence templates for 86 DDI types.

## Problem & Setting
- **Classes**: 86 DDI sentence-template types curated from DrugBank.
- **Dataset**: 192,284 DDIs from 191,878 drug pairs (DrugBank gold standard).
- Multi-label (a pair can fire multiple sentence templates).
- Mean per-type accuracy 92.4%.

## Method (core)
- Each drug -> Tanimoto similarity vector against a fixed structural reference set (the SSP).
- Drug-pair input = concat of two SSPs.
- DNN classifier with 86 sigmoid outputs (multi-label).
- Output piped through a sentence template generator using drug names for human-readable text.

## Cold-start handling
- Inductive in the sense that SSP only requires SMILES, so unseen drugs can be encoded.
- But evaluation is random pair-split, NOT drug-disjoint - so true S2 cold-start performance is not measured.
- Drug-food extension shows the SSP generalizes to ligands not in training data, partial evidence for inductive capability.

## Key contributions
- Established the 86-type DDI prediction task and SSP feature.
- First to output human-readable DDI sentences (useful for clinician trust).
- Drug-food extension applied to 3,288 food compounds.

## Limitations / gaps
- Random pair split inflates accuracy; the 92.4% does not transfer to S1/S2.
- SSP requires a fixed reference set; new drugs can be encoded but the reference set itself is a transductive dependency.
- 86 templates conflate PK-mechanism events (metabolism, serum concentration) and PD-mechanism events (additive efficacy, antagonism).
- Class imbalance not addressed.

## Relevance to our insights
- **i1 (PK/PD)**: NOT differentiated. The 86 templates are a flat label space; many templates describe explicit PK mechanism (e.g., "the metabolism of X can be decreased") while others describe PD outcome (e.g., "the risk of QT prolongation"), but the model has no way to use this structure.
- **i2 (meeting node + over-smoothing)**: No graph at all - pure DNN over similarity vectors.
- **i3 (pooling noise + attention limit)**: No attention; no pooling over neighborhoods. The SSP is itself a uniform-similarity aggregation - the very baseline our i3 critiques.
- **i4 (node-name semantic prior)**: Drug names used ONLY for output sentence templating, not as a feature. The name-as-feature opportunity is left untouched.

## Notes
- Despite its age, the 86-template space is still cited in 2024-2026 surveys.
- Random pair split is a key reason later work (DDIMDL, MDF-SA-DDI, MDDI-SCL) introduced the Task 1 / 2 / 3 protocol.
