# A Systematic Review of Molecular Structures, Knowledge Graphs, and Cold-Start Scenario in Drug-Drug Interaction Prediction

- **Authors**: Mir Mansoor Ahmad, Zuraini Binti Ali Shah, Hui Wen Nies
- **Year / Venue**: 2025 / Computers in Biology and Medicine, Vol. 190 (May 2025)
- **Link**: https://pubmed.ncbi.nlm.nih.gov/40187181/ ; https://www.sciencedirect.com/science/article/abs/pii/S0010482525004731
- **Read depth**: abstract + structure (publisher full text behind paywall)
- **Cluster**: c3

## TL;DR
A 2025 systematic review that organizes DDI-prediction research along three axes: (1) molecular representations, (2) knowledge-graph approaches, and (3) the cold-start scenario. It frames cold-start (drugs with limited interaction data or unknown structure) as a central, still-open generalization challenge and argues that exhaustive clinical-trial DDI discovery is infeasible, so computational cold-start prediction is essential.

## Problem & Setting
Survey, not a method. Covers molecular-structure methods, KG/GNN methods, and how each handles cold-start. Confirms cold-start as the field's hard frontier as of 2025.

## Method (core)
- Taxonomy of molecular representations used for DDI.
- Taxonomy of KG-based integration of heterogeneous biomedical data (drugs, proteins, ontologies).
- Discussion of cold-start failure modes and candidate mitigations.

How node-name text is incorporated: as a survey it catalogs text/semantic augmentation strategies but, per the abstract, does not single out node-name text at a mediator readout as an established technique.

## Cold-start handling
This is the review's organizing theme. It treats cold-start as the primary generalization bottleneck and surveys structure-based and KG-based attempts to address it.

## Key contributions
- An up-to-date (2025) consolidation of the cold-start DDI landscape, useful as a single citable source for "cold-start is the open problem."
- Joins molecular, KG, and cold-start literatures that are usually surveyed separately.

## Limitations / gaps
- Survey only; no new method or benchmark.
- Abstract does not detail per-method cold-start numbers or a unified split protocol (DDI-Ben remains the benchmark reference for that).

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not a focus; supports our claim that mechanism-typed reasoning is unexplored.
- **i2 (meeting node + over-smoothing)**: not discussed.
- **i3 (pooling noise + attention limit)**: indirectly supports — current KG methods generalize poorly under cold-start.
- **i4 (node-name semantic prior)**: useful framing citation. The review confirms cold-start is unsolved and that text/semantic augmentation is an active but unconsolidated direction, supporting our position that "node-name text at the mediator readout for S2 DDI" is not an established recipe.

## Notes
- Cite in the intro/related-work to anchor the cold-start problem statement with a 2025 survey.
- Pair with DDI-Ben (Zhang 2024) for the concrete benchmark/split side of the same argument.
