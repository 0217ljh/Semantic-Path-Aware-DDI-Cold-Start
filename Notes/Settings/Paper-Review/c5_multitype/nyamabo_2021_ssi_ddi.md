# SSI-DDI: substructure-substructure interactions for drug-drug interaction prediction

- **Authors**: Arnold K. Nyamabo, Hui Yu, Jian-Yu Shi
- **Year / Venue**: 2021, Briefings in Bioinformatics, 22(6):bbab133
- **Link**: https://academic.oup.com/bib/article/22/6/bbab133/6265181
- **Read depth**: full-text
- **Cluster**: c5

## TL;DR
Decomposes DDI prediction into pairwise interactions between drug substructures (atom-radius receptive fields from a GAT) and aggregates them with a co-attention mechanism. Multi-type setting (86 DrugBank sentence types), interpretable at the substructure-pair level.

## Problem & Setting
- **Classes**: 86 DrugBank sentence templates (multi-type).
- **Dataset**: DrugBank v5.0.3, 1,704 drugs, 191,400 DDIs.
- Random 60/20/20 split (drug-pair split); also reports inductive (new-drug) results.

## Method (core)
- Each drug = molecular graph; GAT layers produce substructure embeddings at increasing radii (1 to L hops).
- Substructure-substructure interaction score: bilinear between drug-A substructure i and drug-B substructure j, summed over L radii.
- Co-attention: importance weights gamma_{i,j} for each substructure pair.
- Final score: weighted sum -> sigmoid per DDI type.

## Cold-start handling
- Inductive evaluation reported: model trained on a drug subset and tested on unseen drugs.
- Substructure features come from raw molecular graph -> no transductive dependency on training drug IDs.
- Performance degrades inductively vs transductively but remains usable.
- The paper explicitly contrasts SSI-DDI's structure-from-SMILES inductive ability against DDIMDL-style similarity-vector inputs.

## Key contributions
- First to model DDI as a sum of substructure-substructure interactions (chemistry-motivated).
- Co-attention over substructure pairs -> interpretability (highlights which atoms interact).
- Strong inductive baseline thanks to graph-based featurization.

## Limitations / gaps
- Performance is sensitive to drug-pair ordering - hints at noise leakage during substructure extraction (authors flag this).
- Substructures are chemistry-only; no targets, enzymes, or KG semantics.
- 86 DDI types treated as flat sigmoid outputs; no PK/PD separation.
- The mechanism is appropriate for PD-mechanism interactions (where physical chemistry of substructures matters) but underpowered for PK-mechanism interactions (which are mediated by transporters / enzymes - external knowledge).

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Strongly relevant. Substructure-substructure attention is essentially a PD-style reasoning channel - the model assumes interaction is determined by physical chemistry of co-occurring substructures. This is exactly half of what i1 says we need; SSI-DDI is the canonical example of "molecular-layer-only" reasoning that should be paired with a PK channel.
- **i2 (meeting node + over-smoothing)**: Uses L=2 GAT hops on each molecule independently, then bilinear cross-product between drugs. There is no drug-drug graph and no over-smoothing at the inter-drug level. The "meeting node" is implicit in the substructure-pair bilinear, not learned.
- **i3 (pooling noise + attention limit)**: Co-attention over O(L^2) substructure pairs is exactly the kind of attention we critique. The authors themselves report sensitivity to noise during substructure extraction - this directly supports i3's claim that attention cannot fully de-noise without an external prior.
- **i4 (node-name semantic prior)**: Drug names not used as features (only as IDs).

## Notes
- One of the few c5 papers with native inductive evaluation; the closest direct competitor for our S2 setting.
- Order-sensitivity (DDI(A,B) different from DDI(B,A)) is a documented artifact - relevant if we benchmark.
- Code: https://github.com/kanz76/SSI-DDI
