# Paper Review — Semantic-Path-Aware-DDI-Cold-Start

Year scope: **2018–2026**. Each paper has its own `.md` file inside the appropriate cluster folder.

## Clusters

| # | Folder | Topic |
|---|---|---|
| 1 | `c1_kg_gnn/` | KG-based DDI prediction with GNN (Decagon / KGNN / SkipGNN family) |
| 2 | `c2_path_multihop/` | Path-based / multi-hop reasoning for DDI |
| 3 | `c3_cold_start/` | Cold-start / inductive DDI prediction (S1/S2 settings) |
| 4 | `c4_llm/` | LLM-based DDI prediction (2023–2026) |
| 5 | `c5_multitype/` | Multi-type / event-type / multi-label DDI |
| 6 | `c6_drug_repr/` | Drug representation (SMILES / structure / fingerprint) for DDI |
| 7 | `c7_kg_text_semantics/` | Biomedical KG embedding + textual/semantic features |
| 8 | `c8_gnn_oversmoothing/` | GNN over-smoothing / expressivity in biomedical context |

## Per-paper file template

Each paper gets a markdown file named `<lead-author>_<year>_<short-slug>.md` with the following sections:

```markdown
# [Paper Title]

- **Authors**: ...
- **Year / Venue**: ...
- **Link**: ...
- **Read depth**: full-text | abstract-only
- **Cluster**: cN

## TL;DR
1–2 sentences.

## Problem & Setting
Task framing, dataset, split convention (S0/S1/S2 if applicable).

## Method (core)
Architecture / key idea in 3–6 bullets.

## Cold-start handling
How (if at all) the paper addresses unseen drugs. "N/A" if not discussed.

## Key contributions
- ...

## Limitations / gaps (as relevant to our insights)
- ...

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: ...
- **i2 (meeting node + over-smoothing)**: ...
- **i3 (pooling noise + attention limit)**: ...
- **i4 (node-name semantic prior)**: ...

## Notes
Anything else worth keeping.
```

## Cluster index files
Each cluster also has a `_cluster_summary.md` written after all its papers are reviewed, capturing the recurring patterns, gaps, and aggregated relevance to i1–i4.

## Master index
`_INDEX.md` at this level aggregates every paper across all clusters with one-line summaries, sortable by cluster and year.
