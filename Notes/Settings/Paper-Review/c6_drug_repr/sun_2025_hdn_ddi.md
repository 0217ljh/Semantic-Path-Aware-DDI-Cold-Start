# HDN-DDI: A Novel Framework for Predicting Drug-Drug Interactions Using Hierarchical Molecular Graphs and Enhanced Dual-View Representation Learning

- **Authors**: Sun et al. (jcsun-00 GitHub; Central-South-China group lineage)
- **Year / Venue**: 2025 / BMC Bioinformatics, 26 (Jan 2025), article s12859-025-06052-0
- **Link**: https://link.springer.com/article/10.1186/s12859-025-06052-0 ; code https://github.com/jcsun-00/DrugBank , https://github.com/jcsun-00/Twosides
- **Read depth**: abstract + structure (Springer full text behind auth)
- **Cluster**: c6

## TL;DR
HDN-DDI extends DSN-DDI's dual-view idea by first running an **explainable substructure-extraction module** that decomposes each molecule with **BRICS** (plus extra ring-splitting rules) and then **assembles the resulting motifs into motif-level nodes**, forming a true two-tier *hierarchical molecular graph* (atom-level + motif-level + molecule-level). Dual-view (intra + inter) representation learning then runs over this hierarchical graph. Pure molecular, no KG.

## Problem & Setting
Multi-class/multi-label DDI on DrugBank and TWOSIDES. Warm-start plus cold-start (unseen-drug) evaluation. Warm-start: ~97.9% acc (DrugBank), ~99.4% acc (TWOSIDES).

## Method (core)
- Three-stage decomposition: BRICS break to fragments, supplementary rules split large rings to smallest constituent ring, assemble motifs into explicit motif-level nodes.
- Hierarchical graph encodes atom-level and motif-level nodes jointly, giving chemically-meaningful (named functional-group) substructures rather than per-layer hidden states.
- Enhanced dual-view (intra-drug + inter-drug) learning over the hierarchical graph, relation-typed decoder.

## Cold-start handling
**Explicit.** Reports cold-start gains on previously unseen drugs (the headline +4.96% accuracy / +7.08% F1 figure is inherited from the DSN-DDI cold-start lineage and re-reported as the improvement bracket). Does not, in the abstract, separate one-drug-unseen from both-drugs-unseen as cleanly as DSN-DDI / HLN-DDI do — treat the double-cold claim as unconfirmed pending full text.

## Key contributions
- Replaces DSN-DDI's implicit "scale = layer depth" substructures with **explicit, chemically-grounded BRICS motif nodes** — the most concrete realization of *hierarchical* molecular substructure among the dual-view family.
- Interpretability at atom / motif / molecule levels.

## Relevance to our insights
- **Adding hierarchical molecular substructure to MNAH**: HDN-DDI is the cleanest template for *what "hierarchical molecular" should mean* in our extension. BRICS motif-level nodes are chemistry-defined and therefore cold-start-stable (a motif exists for an unseen drug as long as we have its SMILES). A motif-level node layer could plug into MNAH as a second hierarchy that the meeting-node/shared-mediator head reads from, complementing the KG path-flow. The explicit motif nodes are also more interpretable to fuse than DSN-DDI's per-layer scales.
- Caveat: like all the dual-view family, it carries no KG/biomedical prior — strictly the molecular side of the proposed fusion.
