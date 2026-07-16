# Deep Learning Improves Prediction of Drug-Drug and Drug-Food Interactions (DeepDDI)

- **Authors**: Jae Yong Ryu, Hyun Uk Kim, Sang Yup Lee
- **Year / Venue**: 2018 / PNAS (Proceedings of the National Academy of Sciences USA)
- **Link**: https://www.pnas.org/doi/10.1073/pnas.1803294115
- **Read depth**: full-text (abstract + key sections)
- **Cluster**: c6

## TL;DR
DeepDDI is the canonical early deep-learning DDI baseline. It takes only drug names + SMILES as input, converts each drug to a Structural Similarity Profile (SSP) by computing Tanimoto similarity to a reference set of drugs, and feeds the concatenated SSP through a feed-forward DNN to predict one of 86 DDI types with 92.4% mean accuracy.

## Problem & Setting
Predict DDI type (one of 86 semantic types in DrugBank) for arbitrary drug pairs given only molecular structure. Setting is mostly transductive (pairs of drugs already in DrugBank). The paper also applies the trained model to drug-food constituent pairs and adverse-event mechanism prediction.

## Method (core)
- Drug representation = **Structural Similarity Profile (SSP)**: for each drug d, compute Tanimoto similarity of its ECFP4 fingerprint to every reference drug, yielding a length-|R| vector.
- SSP is dimensionality-reduced (PCA) and concatenated for the two drugs in the pair.
- Concatenated vector fed to a deep feed-forward neural network (8 hidden layers) that does softmax over 86 DDI types.
- Human-readable interaction sentences are generated from the predicted type.

## Cold-start handling
Implicit only. SSP is computable for any drug with a SMILES, so cold drugs can in principle be embedded without retraining. However the paper does not evaluate a true S2 (both-drugs-unseen) cold split. Performance on held-out cold drugs is not separately reported.

## Key contributions
- First large-scale, multi-class (86-type) DDI prediction with deep learning.
- Establishes the SSP feature as a strong purely-structural baseline.
- Demonstrated transfer to drug-food constituent pairs.

## Limitations / gaps (as relevant to our insights)
- Ordering-sensitive: different SSP concatenation order can yield different predictions for the same unordered pair.
- Purely PK-flavoured: features encode molecular similarity only; no phenotype/effect signal.
- No explicit S2 cold-start evaluation; cold performance unknown.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Pure PK-side method. Captures only molecular-similarity signal; cannot reason about downstream effect overlap. A motivating "PK-only" baseline showing why PD signal is missing.
- **i2 (meeting node + over-smoothing)**: N/A. No graph reasoning; flat similarity vector.
- **i3 (pooling noise + attention limit)**: No pooling over substructures. SSP is a global similarity descriptor, so substructure noise is averaged into similarity. Illustrates the "uniform global descriptor" failure mode our work argues against.
- **i4 (node-name semantic prior)**: Contrast point. DeepDDI uses **molecular** features (ECFP4) which are cold-start-stable in the chemistry sense (computable from SMILES). This contrasts with our use of **node-name biomedical text** as the cold-stable signal. Both share the "no training-pair exposure required" property but live in different feature spaces.

## Notes
Single best baseline to cite for "purely molecular structural similarity, ignoring KG and effect-layer information." Useful in our related-work as the canonical PK-only deep model.
