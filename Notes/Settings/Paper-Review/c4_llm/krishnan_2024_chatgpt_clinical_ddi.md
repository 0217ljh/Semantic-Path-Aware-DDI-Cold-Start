# Evaluating the Capability of ChatGPT in Predicting Drug-Drug Interactions: Real-World Evidence Using Hospitalized Patient Data

- **Authors**: Radha Krishnan et al. (Univ of Sydney, 11 authors)
- **Year / Venue**: 2024 / British Journal of Clinical Pharmacology, 90(12)
- **Link**: https://bpspubs.onlinelibrary.wiley.com/doi/10.1111/bcp.16275
- **Read depth**: full-text summary
- **Cluster**: c4

## TL;DR
Audit study, not a method paper. Tests ChatGPT-3.5 zero-shot on 11,698 real drug pairs from 120 hospitalized polypharmacy patients (median age 68). Result: sensitivity 0.06-0.16 per prompt, 0.24 combined, specificity >95%, kappa vs pharmacists 0.077-0.143. Misses ~75% of clinically relevant DDIs.

## Problem & Setting
Real-world clinical DDI detection: each patient's actual medication list scored pairwise. Three prompt variants:
- P1: demographic + clinical context + emotional framing,
- P2: drugs + doses + explicit DDI question,
- P3: drugs only + DDI question.

## Method (core)
LLM is the predictor; pure zero-shot prompting of ChatGPT-3.5 web interface. No KG, no fine-tuning, no retrieval. Output binarized (yes/no DDI). Pharmacist gold standard from same clinical workflow.

## Cold-start handling
Not framed as cold-start, but in practice every drug in the cohort is well-known and almost certainly in ChatGPT's pretraining. The poor sensitivity therefore is not a cold-start limitation - it's a memorization+retrieval failure even on familiar drugs. This is a crucial datapoint: LLMs do NOT reliably retrieve their own DDI knowledge under realistic prompts.

## Key contributions
- Real clinical-pair benchmark (11,698 pairs), not a curated test set - rare for c4.
- Prompt sensitivity quantified across three styles.
- Honest external comparator (pharmacists at the same hospital).

## Limitations / gaps (as relevant to our insights)
- Only ChatGPT-3.5 tested; GPT-4 / Claude / open-source not assessed.
- Binary outcome flattens severity.
- Patient data given to LLM is partial vs pharmacist's full chart - confound for direct comparison.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Not explicitly. But the binary-outcome design hides whether ChatGPT does better on one paradigm than the other.
- **i2 (meeting node + over-smoothing)**: Not applicable.
- **i3 (pooling noise + attention limit)**: Strong support - attention over a flat drug-list prompt cannot inject the clinical priors a pharmacist uses (severity, mechanism, dose adjustment).
- **i4 (node-name semantic prior)**: Negative evidence for naive use - even with full drug names, zero-shot LLM fails. Implies: node-name semantics is necessary but not sufficient; needs structured grounding (KG/RAG).

## Notes
- Sensitivity 0.06-0.16 per prompt, specificity ~0.95.
- Kappa vs pharmacists 0.077-0.143 (poor agreement).
- AUC ~0.5.
- Useful as the strongest "naive LLM is dangerous in DDI" citation.
- Clinical realism makes this a high-impact reference for the limits-of-LLMs paragraph.
