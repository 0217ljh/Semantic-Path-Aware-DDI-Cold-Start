# Improving Drug-Drug Interaction Prediction via In-Context Learning and Judging with Large Language Models (DDI-JUDGE)

- **Authors**: He Qi, Xiaoqiang Li, Chengcheng Zhang, Tianyi Zhao
- **Year / Venue**: 2025 / Frontiers in Pharmacology, Vol 16
- **Link**: https://doi.org/10.3389/fphar.2025.1589788
- **Read depth**: full-text
- **Cluster**: c4

## TL;DR
Prompt-engineering framework with two pieces: (i) ICL sample selection via drug similarity (Tanimoto, cosine, graph-based), and (ii) a GPT-4 "judge" that ensembles candidate predictions from multiple LLMs along axes of accuracy/clarity/evidence/relevance. Reports modest absolute AUC (zero-shot 0.642, 8-shot 0.788) but consistent gains over each individual LLM.

## Problem & Setting
Binary DDI prediction on Luo's heterogeneous dataset (708 drugs, 10,036 known interactions from DrugBank 3.0 + protein/disease/side-effect side info). 10-fold CV.

## Method (core)
Inference-only pipeline (no fine-tuning):
1. **ICL sample selection** - for each test pair, retrieve k similar positive and k similar negative examples from training data using one of 5 similarity metrics.
2. **Prompt template** - structures four blocks: input requirements, prediction task, consideration factors, in-context examples.
3. **Judging stage** - GPT-4 evaluates outputs of multiple base LLMs against 4 rubric criteria, then weighted fusion yields the final prediction.

LLM role: pure predictor + meta-judge. No KG paths, no GNN.

## Cold-start handling
Authors frame the work around zero/few-shot scenarios but evaluation is 10-fold CV with no explicit drug-disjoint (S2) split. Cold-start claim is weak: zero-shot here means no fine-tuning, not novel drugs. Pretraining leakage of DrugBank 3.0 (released ~2014) into all GPT-class models is almost certain.

## Key contributions
- Ensemble-of-LLMs framework with explicit judging step.
- Systematic comparison of 5 ICL similarity metrics for example selection.
- Identifies 3 case-study DDIs absent from DrugBank as discovery validation.

## Limitations / gaps (as relevant to our insights)
- No drug-disjoint evaluation → cold-start claim unsupported.
- Acknowledges GPT-4-as-judge may bias toward GPT-style answers ("LLM-judge bias").
- Absolute performance (AUC 0.78 in best few-shot) is below GNN baselines on the same dataset - the value is interpretability/robustness, not raw accuracy.
- No mechanistic axis (PK vs PD); single binary label.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not addressed.
- **i2 (meeting node + over-smoothing)**: Not applicable.
- **i3 (pooling noise + attention limit)**: Tangential - the judge can be seen as an attention mechanism over candidate predictions, but it does not inject external priors beyond the prompt template.
- **i4 (node-name semantic prior)**: Indirect - similarity metrics include both molecular (Tanimoto) and embedding-based, but does not isolate node-name/text semantics from molecular structure.

## Notes
- LLMs tested: GPT-4, GPT-3.5, GPT-4o, Claude 3.5, Llama 2/3, Davinci-003, DeepSeek V3.
- Best baseline: GPT-4o 0.585 AUC zero-shot, 0.681 8-shot.
- DDI-JUDGE: 0.642 zero-shot, 0.788 8-shot.
- No memorization audit; given dataset age, results should be read cautiously.
- Useful as a "prompting-only LLM ceiling" reference.
