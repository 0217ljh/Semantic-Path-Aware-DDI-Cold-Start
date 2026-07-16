# LLMs for Drug-Drug Interaction Prediction: A Comprehensive Comparison

- **Authors**: Gabriele De Vito, Filomena Ferrucci, Athanasios Angelakis
- **Year / Venue**: 2025 / arXiv:2502.06890 (cs.LG)
- **Link**: https://arxiv.org/abs/2502.06890
- **Read depth**: full-text (abstract + paper summary)
- **Cluster**: c4

## TL;DR
Benchmarks 18 LLMs (proprietary GPT-4/Claude/Gemini and open-source 1.5B-72B) as direct DDI predictors. Zero-shot is weak (avg sensitivity 0.55); fine-tuning is decisive, and a small Phi-3.5 (2.7B) matches/beats GPT-4 after tuning (sensitivity 0.978, accuracy 0.919).

## Problem & Setting
Binary DDI classification from text inputs (SMILES + target organisms + gene interactions formatted as raw text). Evaluated on DrugBank with cross-validation across 13 external DDI datasets.

## Method (core)
LLM is the predictor (not a feature extractor). Inputs are pure-text serializations of drug pair information; output is a classification token. Two regimes are compared:
1. Zero-shot prompting (baseline of memorized knowledge).
2. Supervised fine-tuning on DDI labels.

No KG, no GNN. The contribution is the systematic scaling/architecture comparison.

## Cold-start handling
The paper does NOT report S2/inductive splits. Splits appear to be random over drug pairs, which is transductive at the drug-pair level and likely contains drug-overlap leakage between train and test. This is a critical gap for cold-start interpretation: fine-tuned LLMs may largely memorize seen-drug patterns.

## Key contributions
- First broad LLM benchmark covering 18 models for DDI.
- Empirical claim that small fine-tuned LMs (Phi-3.5 2.7B) rival GPT-4 after tuning, undermining the "bigger is better" narrative.
- Cross-dataset validation set (13 external DDI sources).

## Limitations / gaps (as relevant to our insights)
- No cold-start (S1/S2) split → headline numbers may overstate generalization to novel drugs.
- No mechanism analysis (PK vs PD), no path/KG grounding; binary label only.
- Leakage of DDI facts from pretraining corpora into zero-shot eval not controlled (DrugBank is in Common Crawl).
- Closed evaluation on a single label type per pair (binary, not 86-type).

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed. Treats DDI as monolithic classification; loses the PK-vs-PD distinction.
- **i2 (meeting node + over-smoothing)**: Not applicable - no GNN architecture; the "meeting" is implicit in transformer attention over a serialized drug pair.
- **i3 (pooling noise + attention limit)**: Indirect evidence that scale alone does not solve the task (zero-shot is poor); fine-tuning, which essentially injects task-specific priors, is what works. Consistent with our claim that uniform pooling/attention over raw signal is insufficient without external priors.
- **i4 (node-name semantic prior)**: Partially supports - LLMs encode drug-name semantics natively, but the paper does not isolate the contribution of name vs SMILES vs gene-list inputs.

## Notes
- Models tested: GPT-4, Claude, Gemini, Llama, Qwen, Phi, Gemma, DeepSeek (full list of 18).
- Fine-tuned set: GPT-4, Phi-3.5 2.7B, Qwen-2.5 3B, Gemma-2 9B, DeepSeek R1 distilled Qwen 1.5B.
- No discussion of memorization tests or held-out-drug evaluation - a major concern given DrugBank's public availability.
- Useful as a "supervised LLM" baseline ceiling for c4 comparisons.
