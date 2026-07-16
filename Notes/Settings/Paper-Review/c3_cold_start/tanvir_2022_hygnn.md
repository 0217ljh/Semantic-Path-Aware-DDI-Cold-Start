# HyGNN: Drug-Drug Interaction Prediction via Hypergraph Neural Network

- **Authors**: Khaled Mohammed Saifuddin, Briana Bumgardner, Farhan Tanvir, Esra Akbas
- **Year / Venue**: 2022 (arXiv) / 2023 IEEE ICWS conference; arXiv:2206.12747
- **Link**: https://arxiv.org/abs/2206.12747
- **Read depth**: abstract + method overview
- **Cluster**: c3

## TL;DR
Hypergraph neural network over **SMILES-extracted chemical substructures**, treating each drug as a hyperedge over substructure nodes. Targets new drugs by depending only on SMILES — universally available — but uses zero text/KG semantic information.

## Problem & Setting
DDI prediction with focus on new drugs where comprehensive databases (protein targets, side-effect labels, KG) are missing. Cold-start setting is "new drugs with only SMILES available." Evaluated on DrugBank-derived DDI dataset. Negative sampling: standard random non-edges.

## Method (core)
1. Parse SMILES → frequent chemical substructures → vocabulary of substructure nodes.
2. Build hypergraph: each drug is a hyperedge connecting its constituent substructure nodes.
3. **Attention-based hyperedge encoder**: aggregates substructure node embeddings into a drug-level representation.
4. **Decoder** scores DDI between drug-pair representations.

## Cold-start handling
- Drug representation = function of its SMILES substructures only. Any new SMILES → immediately embeddable.
- No KG nodes, no biomedical text, no targets/enzymes.
- Cold-start strength comes from substructure vocabulary being shared across drugs — new drugs reuse seen substructure embeddings.

## Key contributions
- Hypergraph (rather than pairwise graph) over chemical substructures.
- Attention over substructure → drug aggregation.
- Reported ROC-AUC 97.9%, PR-AUC 98.1% — but this appears to be transductive evaluation; the paper does not present a rigorous S2 split.

## Limitations / gaps (as relevant to our insights)
- Despite framing as cold-start, evaluation protocol does not clearly separate train and test drugs (no S2 numbers reported).
- Substructure vocab can miss novel chemotypes (rare substructures with few seen drugs → poor embedding).
- Pure structural — no PK/PD context, no semantic priors.
- Attention is learned end-to-end with no external supervision.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed.
- **i2 (meeting node + over-smoothing)**: Hyperedge view is a soft analog of "meeting node" (substructures act as shared mediators between drugs), but mediators are *chemical*, not *biological*. Doesn't address PK/PD-level mediators.
- **i3 (pooling noise + attention limit)**: Uses attention; no external prior — classic case of insufficient signal under cold-start. Confirms our concern.
- **i4 (node-name semantic prior)**: Substructures are SMILES fragments without biomedical text — opposite of our position. The high reported numbers are likely transductive-leakage rather than true cold-start.

## Notes
- Cited often but the cold-start evaluation rigor is weak. Should be used as evidence that "SMILES-only methods report high numbers but don't pass rigorous S2 evaluation."
- Cross-check against DDI-Ben benchmark numbers for the same model class.
