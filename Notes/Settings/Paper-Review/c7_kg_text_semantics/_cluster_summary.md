# c7 Cluster Summary — Biomedical KG Embedding + Textual / Semantic Features

**Cluster anchor insight**: **i4** — node-name biomedical text semantics is the cold-start-stable signal. This is the **most relevant** cluster of the literature review for our paper's central claim.

## Papers Reviewed

| # | Key | Year | Venue | Domain | Text encoder | Inductive / cold-start? |
|---|-----|------|-------|--------|--------------|--------------------------|
| 1 | Yao 2019 — KG-BERT | 2019 | arXiv (cited heavily) | General KG + UMLS | BERT-Base | Architecturally yes, not evaluated |
| 2 | Wang 2021 — KEPLER | 2021 | TACL | General KG (Wikidata5M) | RoBERTa, jointly trained with TransE | **Yes — explicit inductive eval** |
| 3 | Wang 2021 — DDI KG+Text | 2021 | JMIR Med Inform | DDI (Bio2RDF + DDI corpus) | Autoencoder over DDI label vectors (no contextual LM) | No |
| 4 | Alshahrani 2022 — KG+Text DTI | 2022 | PeerJ | DTI / drug-indication (STITCH, SIDER, Yamanishi) | Word2Vec on PubTator-normalised PubMed | Partial — union of KG and text coverage |
| 5 | Xiao 2024 — FuseLinker | 2024 | J. Biomed. Informatics | Biomedical KG link prediction (KEGG50k, Hetionet, SuppKG, ADInt) | BERT / **PubMedBERT** / Flan-T5 / Llama2 / **PMC-LLaMA** | Architecturally supports it, not evaluated |
| 6 | Dang 2025 — BioMedKG / PrimeKG++ | 2025 | Frontiers Sys. Biol. | Drug-disease + DTI (PrimeKG++, DrugBank) | Specialized biomedical LM + Graph Contrastive Learning | **Yes — claims unseen-node generalisation** |
| 7 | Zhang 2025 — PrimeKG-CL | 2025 | arXiv (benchmark) | Continual link prediction (PrimeKG snapshots) | **BiomedBERT** [CLS] of entity description (canonical recipe) | Yes — new entities in later snapshots |

7 papers total. Full-text or near-full coverage: KG-BERT, KEPLER, Wang-2021-DDI, Alshahrani 2022, FuseLinker, BioMedKG (6). Abstract-only with strong design detail: PrimeKG-CL (1).

## Cross-cutting findings

1. **The c7 mechanism is convergent.** Across years and venues, the recipe is:
   - take an entity's name + description text,
   - run it through a pre-trained (often biomedical) LM,
   - use the resulting vector as (or fused into) the entity's node feature in the KG.

   The cold-start-stable property is a structural consequence of this design: any new entity with a description gets an embedding without retraining the entity table.

2. **Evolution of the text encoder.** Word2Vec (Alshahrani 2022) → BERT-Base (KG-BERT 2019) → RoBERTa joint-trained with KE (KEPLER 2021) → PubMedBERT / BiomedBERT (FuseLinker 2024, PrimeKG-CL 2025) → biomedical LLMs (PMC-LLaMA in FuseLinker; specialized LM in BioMedKG 2025). PubMedBERT/BiomedBERT has emerged as the de facto biomedical default for entity-description encoding.

3. **Fusion strategy is the underexplored axis.**
   - KG-BERT and KEPLER: **replace** structure with text (no fusion).
   - Alshahrani: concat or joint-vocabulary Word2Vec (shallow).
   - FuseLinker: scalar weighted average `w * e_text + (1-w) * e_knowledge` per-dataset tuned.
   - BioMedKG: multimodal contrastive alignment.
   No paper conditions the text–structure fusion on **relation type** or **drug-pair regime**, which is an open gap our i4 method can claim.

4. **Cold-start evaluation is rare.** Only KEPLER, BioMedKG, and PrimeKG-CL run a genuine inductive split. KG-BERT, FuseLinker, and Wang-2021-DDI architecturally allow it but do not run it. This is a citable gap for our paper.

5. **DDI-specific c7 work is thin.** Wang-2021-DDI is the only true DDI paper in this cluster, and it uses text only at the relation-label level, not at the drug-node level. Cold-start S2-DDI with PubMedBERT-encoded drug-name features is **largely unstaked methodological territory.**

## Key takeaway for i4 (the primary insight this cluster supports)

The c7 literature gives us **two things to lean on and one gap to claim**:

- **Lean on**: text-encoder-of-entity-description as a cold-start-stable node feature is empirically validated (KEPLER, BioMedKG) and is the standard practice on biomedical benchmarks like PrimeKG (PrimeKG-CL). We do not need to defend this design choice from scratch.
- **Lean on**: PubMedBERT / BiomedBERT is the consensus encoder. We should use it (or PMC-LLaMA) to avoid encoder-choice nitpicks.
- **Claim**: no published method runs a **proper cold-start S2-DDI evaluation** that uses node-name text as the cold-start-stable signal **in combination with** path/meeting-node structural reasoning. FuseLinker comes closest but uses a structure-blind scalar fusion and does not run inductive splits. BioMedKG runs inductive but on drug-disease / DTI, not S2 DDI.

## Anti-duplication checks

- KG-BERT and KEPLER are general-domain. Kept here because they are the methodological backbone of the c7 line; not duplicated in c1 (those are GNN-only DDI methods).
- BioMedKG is multimodal but text is the primary cold-start mechanism — belongs in c7, not c5 (multi-event) or c6 (drug repr).
- Wang-2021-DDI is at the boundary with c1 (DDI + KG) but kept here because text is an explicit part of the design.

## Open gaps relevant to our paper

1. No published c7 paper does **S2-DDI** with cold-start evaluation.
2. No fusion mechanism is gated by **relation/mechanism type** (PK vs PD).
3. No method combines **meeting-node structural reasoning (i2)** with **node-name text prior (i4)**; FuseLinker and BioMedKG do graph + text but the graph side is generic GNN/contrastive, not meeting-node aware.
