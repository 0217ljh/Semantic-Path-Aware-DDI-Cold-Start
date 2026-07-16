# c4 Cluster Summary: LLM-Based DDI Prediction (2023-2026)

## Scope
LLM-anchored DDI prediction papers from 2023-2026. Eight papers covered, spanning four LLM-usage modes:
- LLM as direct predictor (zero-shot or fine-tuned) - De Vito 2025, Qi 2025, Krishnan 2024.
- LLM as feature extractor for downstream classifier/GNN - Li 2026, Im 2025, Hakim 2025.
- LLM + KG hybrid with retrieval/explanation - Liu 2025 (CBR-DDI), Xu 2024 (DDI-GPT).

## Paper list
| File | Lead | Year | LLM Role | Cold-start eval |
|---|---|---|---|---|
| devito_2025_llm_comprehensive_comparison.md | De Vito | 2025 | Predictor (ft/zs) | No |
| liu_2025_cbr_ddi.md | Liu (Yao group) | 2025 | RAG + reasoning | Yes (S1/S2) |
| qi_2025_ddi_judge.md | Qi | 2025 | Predictor + judge | No (10-fold CV) |
| li_2026_llm_ddi.md | Li | 2026 | Feature extractor | Unclear (paywalled) |
| im_2025_llm_multimodal_ddi.md | Im | 2025 | BioBERT extractor | No |
| hakim_2025_polyllm.md | Hakim | 2025 | SMILES encoder | No |
| krishnan_2024_chatgpt_clinical_ddi.md | Krishnan | 2024 | Predictor (clinical audit) | No (real-world pairs, known drugs) |
| xu_2024_ddi_gpt.md | Xu | 2024 | LLM + KG hybrid | Partial (dataset-shift ZS) |

8 papers total; ~7 full-text-summary level, 1 abstract-only (Li 2026 IEEE JBHI, paywalled).

## Key cross-paper findings

### 1. Cold-start rigor is the cluster's biggest weakness
Of 8 papers, only **CBR-DDI (Liu 2025)** uses an explicit drug-disjoint S2 split matching our setting. DDI-GPT does dataset-shift zero-shot (FAERS) which may still share drugs with TWOSIDES training. The remaining 6 use transductive splits with drug-overlap leakage; their headline numbers must be discounted when read as "novel drug" performance.

### 2. Pretraining-corpus memorization is universally under-audited
DrugBank, TWOSIDES, and the DDI Extraction 2013 corpus are all in Common Crawl / PubMed. No paper in this cluster runs a memorization audit (e.g., shuffled drug names, novel-suffix probes). Krishnan 2024's negative clinical result is consistent with the hypothesis that even memorized knowledge is poorly retrievable under prompts.

### 3. LLM-as-extractor often beats LLM-as-predictor at fixed compute
- Im 2025: BioBERT name/description embedding alone gives 0.958 acc on 79-type DDI typing.
- Hakim 2025: ChemBERTa SMILES embedding + GNN beats Decagon.
- Liu 2025: small open LLMs with retrieval beat raw GPT-4.
The pure-predictor route (De Vito) requires fine-tuning to compete, and even then leakage clouds the claim.

### 4. Hybrid LLM+KG is the convergent direction
Liu 2025 (CBR-DDI), Li 2026 (LLM-DDI), Xu 2024 (DDI-GPT) all converge on the same template: LLM-derived text features + KG relational structure + downstream predictor. This is the c4 modal architecture.

## Coverage of our 4 insights

### i1 - PK vs PD as two reasoning paradigms
- **Direct support**: Im 2025 explicitly reports the model confusing upstream PK classes (enzyme inhibition → concentration change) with downstream PD classes (QTc) within the DeepDDI 79-type taxonomy. Best single citation.
- **Indirect support**: Xu 2024's CYP3A/BTK case study is a PK-mechanism finding; Hakim 2025's failure on immune-mediated effects implies a PD-side gap.
- No paper explicitly architects a PK/PD split. **Gap**: our work would be the first to make this an explicit design axis.

### i2 - Meeting node + over-smoothing
- Largely orthogonal to c4. Most c4 papers use shallow architectures or transformer attention rather than deep GNNs.
- Liu 2025 sidesteps over-smoothing by handing reasoning back to LLM; Li 2026 uses message-passing GNN but depth unstated.
- **Gap**: meeting-node-style design is absent in c4; our contribution here remains differentiated.

### i3 - Pooling noise + attention can't inject priors
- **Direct support**: Hakim 2025's sum-pooling-of-drug-embeddings is the canonical failure mode; their best AUC is bounded by it.
- **Direct support**: Im 2025's PSP modality (random-walk protein features) actively *hurts* when combined - the noisy-pooling result we predict.
- **Constructive support**: Liu 2025 (CBR-DDI) bypasses uniform pooling by injecting curated cases as external prior. This is the in-cluster proof-of-concept that i3's prescription works.
- **Strongly covered** - we have both negative and positive citations.

### i4 - Node-name biomedical text semantics as cold-start-stable signal
- **Strongest in-cluster evidence**: Im 2025 shows BioBERT-on-drug-name alone gives 0.958 multiclass acc, with full 768-d > PCA-reduced (semantic density matters).
- **Architectural overlap**: Li 2026 and Xu 2024 inject LLM text embeddings as drug-node features into KGs.
- **Counter-example**: Krishnan 2024 - naive LLM use on drug names is dangerous; node-name semantics needs structured grounding (RAG/KG) to deliver.
- **Comparative angle**: Hakim 2025 uses SMILES not name/description - useful contrast that helps us argue *which text* the LLM should see.
- **Best citation chain for i4**: Im 2025 (name semantics work) → Liu 2025 (LLM description + retrieval works in cold-start) → Krishnan 2024 (naive use fails) → our framing as structured node-name retrieval.

## Takeaway for our paper
1. **Cold-start gap is real** - we are competing primarily against Liu 2025 (CBR-DDI) on rigorous S2 evaluation. Most other c4 numbers are not comparable.
2. **Architectural niche**: meeting-node-anchored design with LLM-derived node-name text features is unoccupied. Hybrid LLM+KG papers (Li, Xu, Liu) all do drug-anchored or case-anchored reasoning.
3. **i1 narrative** is supported by Im 2025's empirical failure modes - cite as evidence that the PK/PD axis matters in practice.
4. **Memorization audit** for any LLM component is mandatory and absent from the cluster - low-hanging differentiator if we run it.
5. **Direct baselines to implement**: CBR-DDI (Liu 2025, S1/S2), BioBERT-only (Im 2025, feature-extractor), LLM-DDI (Li 2026, LLM+GNN). All three are reproducible from descriptions; CBR-DDI likely has public code.
