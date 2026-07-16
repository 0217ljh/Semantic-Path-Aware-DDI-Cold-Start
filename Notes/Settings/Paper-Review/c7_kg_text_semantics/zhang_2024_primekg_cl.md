# PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs

- **Authors**: PrimeKG-CL authors (Harvard / Zitnik lab and collaborators)
- **Year / Venue**: 2025 / arXiv:2605.10529 (preprint)
- **Link**: https://arxiv.org/html/2605.10529
- **Read depth**: abstract-only (benchmark paper, used here as a reference for the canonical node-feature recipe)
- **Cluster**: c7

## TL;DR
A continual learning benchmark over evolving versions of PrimeKG. The reason this matters for c7 is its **standardized node-feature recipe**: it ships three node-feature modalities for the same KG — textual (768-d BiomedBERT [CLS] of entity descriptions), molecular (Morgan fingerprints + MLP for drug SMILES), and structural (R-GCN embeddings). It locks in the convention of "BiomedBERT-encoded entity description" as the canonical textual node feature for PrimeKG.

## Problem & Setting
Continual link prediction on snapshots of PrimeKG. Not a DDI cold-start paper, but functions as a feature-extraction blueprint that downstream DDI / repurposing methods (including ours) can re-use.

## Method (core)
- Three pre-computed node-feature views per PrimeKG node:
  1. **Textual**: 768-d [CLS] embedding from **BiomedBERT** applied to the entity's clinical description text.
  2. **Molecular**: 1024-bit radius-2 Morgan fingerprints for drugs (SMILES), projected through a learned MLP.
  3. **Structural**: R-GCN embeddings over multi-relational neighbourhood.
- Benchmark protocol over PrimeKG snapshots for continual link prediction.

How node-name text is incorporated: at preprocessing time, every entity's clinical description string is encoded by BiomedBERT and the [CLS] vector is cached as the textual feature.

## Cold-start handling
The continual-learning split tests adaptation to **new entities arriving in later snapshots**, which is structurally similar to cold-start S2 for DDI: at test time, some drug nodes are unseen at train time. The standardized text feature is what lets baselines handle these new entities without retraining the entity table.

## Key contributions
- Canonicalises the (text / molecular / structural) three-view node feature for PrimeKG.
- Provides reproducible BiomedBERT-encoded textual features for the entire PrimeKG entity set.
- Continual / inductive evaluation protocol that overlaps conceptually with cold-start S2.

## Limitations / gaps
- It is a benchmark, not a method paper; no novel modeling contribution.
- Continual-learning split is not identical to our DDI S2 split.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not modeled.
- **i2 (meeting node)**: not addressed; R-GCN baseline only.
- **i3 (pooling noise + attention limit)**: standard R-GCN; the benchmark exposes how brittle structure-only embeddings are on new entities, motivating text features.
- **i4 (node-name semantic prior)**: **infrastructure-level prior art.** Encoder: **BiomedBERT** (a.k.a. PubMedBERT, MSR biomedical pretrained). Fusion: not prescribed by the benchmark itself; left to each method. The benchmark essentially says "here are the cold-start-stable text features, see what you can do with them." For our paper we should adopt this exact textual feature definition if we run on PrimeKG, so reviewers can compare apples-to-apples.

## Notes
- Use as the source for our PrimeKG text-feature definition.
- Pair with BioMedKG (Dang 2025) as the "benchmark + method" duo for PrimeKG-with-text.
