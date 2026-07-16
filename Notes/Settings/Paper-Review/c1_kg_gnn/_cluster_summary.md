# Cluster c1 Summary — KG-based DDI Prediction with GNN

## Recurring patterns
Across 2018–2024, c1 work converges on a small set of architectural recipes for DDI prediction over biomedical knowledge graphs. The earliest works (Decagon, KGNN) learn one embedding per drug node by aggregating multi-relational neighborhoods from a KG (proteins, targets, pathways), then score a drug pair by a bilinear or MLP decoder. This paradigm is structurally elegant but brittle: it ties prediction to a single global drug embedding, suffers when the KG neighborhood is noisy or sparse, and collapses entirely when the drug is unseen at training time. The second wave (SkipGNN, SumGNN, KnowDDI) reacts to this by shifting the unit of computation from "drug embedding" to "pair-conditioned subgraph or 2-hop pattern", with attention-based or learnable-strength pooling to denoise. A parallel molecular-only thread (MIRACLE, GMPNN-CS, DGNN-DDI) abandons the KG entirely and scores DDIs from substructure pairs of two SMILES graphs, with substructure-attention or co-attention; GMPNN-CS is the only paper in the cluster that runs a clean inductive S1/S2 cold-start protocol. MUFFIN bridges molecular and KG branches via late fusion of TransE+MPNN. No cluster paper uses node-name text semantics, and PK/PD reasoning is never separated.

## Common limitations
- Cold-start (S2) is almost universally not evaluated; only GMPNN-CS reports an explicit inductive protocol.
- KG is treated as a typed but text-less structure — entity names are IDs, never embedded.
- "Denoising" reduces to attention or learned edge gates over existing neighbors; no mechanism injects external semantic priors absent from the KG.
- PK and PD reasoning paradigms are collapsed: the same model scores enzyme-mediated and phenotype-level interactions through one head.
- Drug embedding is the dominant representation anchor; the shared mediator between drug pairs is at best implicit (SkipGNN, SumGNN) and never has its own foregrounded representation.
- Evaluation focuses on transductive AUROC over edge splits, which inflates apparent generalization.

## Insight coverage (i1–i4)
- **i1 (PK/PD two paradigms)**: NOT addressed by any paper in c1. Every work scores all interaction types with one unified head.
- **i2 (meeting node + over-smoothing)**: Partial. SkipGNN gives drug-drug 2-hop a dedicated channel but hides the mediator; SumGNN's pair-subgraph and KnowDDI's connection-strength subgraph implicitly emphasize the meeting region but do not foreground the mediator node identity. No paper makes the mediator the architectural anchor.
- **i3 (pooling noise + attention limit)**: Partially addressed. SumGNN uses self-attention pooling; KnowDDI advances to learnable edge-strength with edge dropping. Both confirm i3's diagnosis but neither injects priors external to the KG, which is exactly what i3 says is needed.
- **i4 (node-name semantic prior)**: NOT addressed. Zero papers in c1 encode entity names as text embeddings. This is the cleanest gap our proposal addresses.

## Files
- zitnik_2018_decagon.md
- lin_2020_kgnn.md
- huang_2020_skipgnn.md
- wang_2021_miracle.md
- chen_2021_muffin.md
- yu_2021_sumgnn.md
- nyamabo_2022_gmpnn-cs.md
- ma_2023_dgnn-ddi.md
- wang_2024_knowddi.md
