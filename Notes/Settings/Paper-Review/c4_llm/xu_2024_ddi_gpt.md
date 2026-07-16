# DDI-GPT: Explainable Prediction of Drug-Drug Interactions using Large Language Models Enhanced with Knowledge Graphs

- **Authors**: Chengqi Xu, Krishna C Bulusu, Heng Pan, Olivier Elemento (Weill Cornell)
- **Year / Venue**: 2024 (Dec) / bioRxiv preprint (also indexed on PubMed PMC11661079)
- **Link**: https://www.biorxiv.org/content/10.1101/2024.12.06.627266v1
- **Read depth**: abstract + PMC summary (full PDF blocked)
- **Cluster**: c4

## TL;DR
Hybrid LLM+KG framework for DDI: combines pretrained LM representations with knowledge-graph context over biomedical entities to predict DDIs and produce attribution-based explanations (pathways, interactome). 0.964 AUROC on TWOSIDES; zero-shot 0.84 AUROC on 9,480 FDA Adverse Event Reporting System DDIs (442 distinct drugs).

## Problem & Setting
Two tasks:
- Supervised binary DDI on TWOSIDES.
- Zero-shot generalization to FDA-FAERS real-world adverse-event-derived DDIs.

## Method (core)
- LLM provides contextual embeddings for biomedical entities (drug, gene, pathway).
- KG provides relational context (interactome).
- A predictor on top fuses LM + KG signals.
- Feature attribution gives per-pathway explanations.

LLM role: contextual encoder over biomedical entity text; the KG is the relational scaffold. Hybrid model in spirit close to LLM-DDI (Li 2026) but with explicit explainability stack.

## Cold-start handling
The 9,480 FAERS DDI evaluation is labelled "zero-shot" - presumably means the model is not trained on FAERS pairs, but the constituent drugs may overlap with the TWOSIDES training set. Without a drug-disjoint S2 split it is hard to call this strict cold-start. Still, +14% over published baselines on an out-of-distribution real-world set is meaningful.

## Key contributions
- Combined LLM+KG approach with explicit feature attribution for clinical credibility.
- Two-task evaluation including a real-world adverse-event source (FAERS).
- BTK-inhibitor case study identifies CYP3A-enriched signals - actually a PK-mechanism story (i1-shaped).

## Limitations / gaps (as relevant to our insights)
- Drug-disjoint inductive split not confirmed; "zero-shot" here is dataset-shift, not necessarily new-drug.
- LLM identity not disclosed in abstract; pretraining leakage on TWOSIDES likely.
- Explanation modality is post-hoc attribution, not architected.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Indirect support - the CYP3A/BTK case study is a PK-mediated finding emerging from the explanation layer. Suggests the architecture is biased toward PK paths through interactome.
- **i2 (meeting node + over-smoothing)**: KG-based reasoning is present but architecture details limited; cannot confirm meeting-node design.
- **i3 (pooling noise + attention limit)**: Explanation layer effectively *prunes* the noisy interactome to attributable subpaths - aligned with i3's prescription of injecting priors instead of uniform pooling.
- **i4 (node-name semantic prior)**: Strong overlap - LM provides text-grounded contextual embeddings for biomedical entities, which is the i4 mechanism. The +14% zero-shot improvement is suggestive evidence that LM-derived text features generalize beyond the training distribution.

## Notes
- Best published numbers in c4 cluster (0.964 AUROC).
- Released as web platform + software package (per abstract).
- A high-priority paper to obtain full text for: need to verify whether the inductive split is drug-disjoint or merely dataset-disjoint.
- The explainability angle is unusually well-developed for c4.
