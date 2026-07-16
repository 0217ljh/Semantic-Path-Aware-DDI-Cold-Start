# Benchmarking Drug-Drug Interaction Prediction Methods: A Perspective of Distribution Changes (DDI-Ben)

- **Authors**: Yongqi Zhang et al.
- **Year / Venue**: 2024-2025, Bioinformatics (arXiv:2410.18583)
- **Link**: https://academic.oup.com/bioinformatics/article/41/11/btaf569/8285831 (arXiv:2410.18583)
- **Read depth**: full-text (abstract+benchmark overview)
- **Cluster**: c3 (benchmark/meta paper)

## TL;DR
Benchmark of 10 representative DDI methods under simulated distribution-change splits (the practical cold-start regime). Headline finding: **most methods collapse under distribution shift; LLM-based and text-augmented methods are systematically more robust.** This is the single best citation for justifying our i4.

## Problem & Setting
Introduces DDI-Ben framework that explicitly simulates distribution changes between drug sets as a proxy for real-world emerging-drug deployment. Compatible with multiple drug-split strategies (chronological, scaffold, attribute-based). Critique target: standard i.i.d. splits are "unrealistic" for cold-start evaluation.

## Method (core)
- Not a single method — a benchmarking framework.
- Re-evaluates 10 representative methods (Decagon, SumGNN, KnowDDI, EmerGNN, MSTE, Pure-DDI, KG-DDI baselines, and LLM-based / text-based methods like TextDDI) under controlled distribution shifts.
- Provides splits, metrics, and code release.

## Cold-start handling
- The benchmark itself doesn't propose a method; it stress-tests existing ones.
- Splits are designed to maximize distribution gap between train and test drugs (mimics emerging-drug regime).

## Key contributions
- Standardized cold-start evaluation suite for the DDI field.
- Demonstrates substantial performance degradation under distribution shift for graph/KG methods that excel transductively.
- Identifies that **text-augmented and LLM-based methods are notably more robust** under distribution change.

## Limitations / gaps (as relevant to our insights)
- Benchmark, not a method — provides evidence but no new architecture.
- Doesn't disaggregate by interaction mechanism (PK vs PD), which we may need to extend.
- LLM baselines may be vulnerable to leakage from pretraining corpus; not always controlled.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not directly addressed. Splits are drug-based, not mechanism-based. Suggests a gap our paper can extend.
- **i2 (meeting node + over-smoothing)**: Implicit support — KG/GNN methods that rely on dense propagation are exactly the ones that collapse under distribution shift in the benchmark.
- **i3 (pooling noise + attention limit)**: Implicit support — attention-based GNN methods are not robust.
- **i4 (node-name semantic prior)**: **Strongest external evidence** — quantitatively shows text-based methods are more robust. Most useful citation in this entire cluster for motivating our approach.

## Notes
- Will be the headline citation in our "why cold-start needs semantics" paragraph.
- Provides ready-made baselines and splits we can adopt directly.
- Code/data on the paper's GitHub repo.
