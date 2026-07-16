# Paper-Review Master Index

**Total files**: 65 per-paper markdown files + 8 cluster summaries
**Unique papers**: ~54 (some appear in multiple clusters; cross-references noted below)
**Year range**: 2018–2026
**Date built**: 2026-05-13

## Cluster-by-cluster paper list

### c1 — KG-based DDI prediction with GNN (9 papers)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2018 | zitnik | Decagon — multi-relational GNN for polypharmacy SE | abstract |
| 2020 | lin | KGNN — KG-aware embedding for DDI | abstract |
| 2020 | huang | SkipGNN — skip-graph for biomedical edge prediction | abstract |
| 2021 | wang | MIRACLE — multi-view contrastive DDI | abstract |
| 2021 | chen | MUFFIN — molecular + KG fusion DDI | full-text |
| 2021 | yu | SumGNN — KG-subgraph + relation summarization | abstract |
| 2022 | nyamabo | GMPNN-CS — gated message passing with co-attention | full-text |
| 2023 | ma | DGNN-DDI — directed GNN | abstract |
| 2024 | wang | KnowDDI — knowledge-injected DDI | full-text |

### c2 — Path-based / multi-hop reasoning for DDI (9 papers)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2021 | yu | SumGNN — *also in c1, c3* | full-text |
| 2021 | yu | RANE-DDI — random walk + attention | abstract |
| 2023 | zhang | EmerGNN — flow-based for emerging drugs *(also c3)* | full-text |
| 2024 | gao | MedKGQA — KG question answering for medication | full-text |
| 2024 | wang | KnowDDI — *also in c1, c3* | full-text |
| 2024 | hu | MPHGCL-DDI — multi-meta-path heterogeneous CL | full-text |
| 2024 | hu | BioPathNet — NBFNet for biological paths | full-text |
| 2025 | abdullahi | K-Paths — RL path retrieval | full-text |
| 2026 | xie | RISE-DDI — RL subgraph extraction | abstract |

### c3 — Cold-start / inductive DDI prediction (10 papers)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2021 | zitnik | Cold-start problems in DDI (taxonomy) | full-text |
| 2021 | yu | SumGNN — *also in c1, c2* | full-text |
| 2022 | wang | CSMDDI — first explicit S1+S2 method | full-text |
| 2022 | tanvir | HyGNN — hypergraph cold-start | full-text |
| 2022 | deng | META-DDIE — few-shot event types | full-text |
| 2023 | zhang | EmerGNN — *also in c2* | full-text |
| 2023 | zhu | TextDDI — text-only zero-shot | full-text |
| 2024 | wang | KnowDDI — *also in c1, c2* | full-text |
| 2024 | geng | ZeroDDI — zero-shot event class | full-text |
| 2024 | zhang | DDI-Ben — benchmark, distribution-shift splits | full-text |

### c4 — LLM-based DDI prediction (8 papers, all 2024-2026)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2024 | krishnan | ChatGPT clinical DDI audit | full-text |
| 2024 | xu | DDI-GPT — LLM + KG with attribution | full-text |
| 2025 | devito | 18-LLM comprehensive comparison | full-text |
| 2025 | liu | CBR-DDI — case-based + KG (**only S2-eval paper in c4**) | full-text |
| 2025 | qi | DDI-Judge — GPT-4-as-judge ensemble | full-text |
| 2025 | im | LLM multimodal DDI (BioBERT + ECFP + PPI) | full-text |
| 2025 | hakim | PolyLLM — ChemBERTa + GNN | full-text |
| 2026 | li | LLM-DDI — GPT embeddings → GNN | abstract |

### c5 — Multi-type / event-type / multi-label DDI (9 papers)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2018 | ryu | DeepDDI — *also in c6* | abstract |
| 2018 | zitnik | Decagon — *also in c1* | full-text |
| 2020 | deng | DDIMDL — multi-source DDI | full-text |
| 2021 | nyamabo | SSI-DDI — *also in c6* | full-text |
| 2021 | chen | MUFFIN — *also in c1, c6* | full-text |
| 2022 | lin | MDF-SA-DDI — multi-source + focal + mixup | full-text |
| 2022 | lin | MDDI-SCL — supervised contrastive learning | full-text |
| 2023 | masumshah | DPSP — multi-relational SE prediction | full-text |
| 2023 | yang | MSEDDI — multi-source SE-DDI | abstract |

### c6 — Drug representation (SMILES / structure) for DDI (7 papers)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2018 | ryu | DeepDDI — *also in c5* | abstract |
| 2020 | chithrananda | ChemBERTa (foundation for SMILES LM) | full-text |
| 2021 | nyamabo | SSI-DDI — *also in c5* | full-text |
| 2021 | chen | MUFFIN — *also in c1, c5* | full-text |
| 2022 | nyamabo | GMPNN-CS — *also in c1* (explicit S1/S2) | full-text |
| 2022 | he | 3DGT-DDI — 3D + SciBERT | full-text |
| 2022 | yang | SA-DDI — partner-conditioned SSIM (S1+S2 explicit) | full-text |

### c7 — Biomedical KG embedding + textual / semantic features (7 papers)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2019 | yao | KG-BERT — first text+KG | full-text |
| 2021 | wang | KEPLER — RoBERTa + TransE joint, inductive eval | full-text |
| 2021 | wang | DDI KG+Text — JMIR Med Inform | full-text |
| 2022 | alshahrani | KG+Text DTI — PubMed Word2Vec | full-text |
| 2024 | xiao | FuseLinker — PubMedBERT / PMC-LLaMA scalar fusion | full-text |
| 2025 | dang | BioMedKG / PrimeKG++ — contrastive multimodal | full-text |
| 2025 | zhang | PrimeKG-CL — BiomedBERT continual link prediction | abstract |

### c8 — GNN over-smoothing / expressivity (8 papers)

| Year | Key | Paper | Read depth |
|---|---|---|---|
| 2018 | li | Deeper Insights into GCN (Laplacian smoothing) | abstract+claim |
| 2018 | xu | JKNet — jumping knowledge | abstract+claim |
| 2020 | oono | Exponential expressive loss theorem | abstract+claim |
| 2020 | rong | DropEdge | abstract+claim |
| 2020 | zhao | PairNorm | abstract+claim |
| 2020 | chen | GCNII — deep GCN via initial residual | abstract+claim |
| 2021 | alon | Over-squashing bottleneck | abstract+claim |
| 2022 | topping | Curvature & over-squashing | abstract+claim |

## Cross-cluster duplicates (paper appears in N folders)

| Paper | Clusters | Reason |
|---|---|---|
| SumGNN (Yu 2021) | c1, c2, c3 | KG+GNN + path-based + inductive eval |
| KnowDDI (Wang 2024) | c1, c2, c3 | KG+GNN + path-based + cold-start |
| EmerGNN (Zhang 2023) | c2, c3 | Path-based + emerging-drug cold-start |
| MUFFIN (Chen 2021) | c1, c5, c6 | KG+GNN + multi-type + molecular fusion |
| Decagon (Zitnik 2018) | c1, c5 | KG+GNN + multi-SE |
| GMPNN-CS (Nyamabo 2022) | c1, c6 | Molecular + explicit S1/S2 |
| SSI-DDI (Nyamabo 2021) | c5, c6 | Multi-type + molecular |
| DeepDDI (Ryu 2018) | c5, c6 | Multi-type + SSP molecular |

## File naming convention
`<leadauthor>_<year>_<short-slug>.md` (lowercase, hyphens or underscores)

## Where to look first
- For overall novelty argument → `_SYNTHESIS.md` (this folder)
- For PK/PD framing — i1 gap → c1, c3, c5 cluster summaries (all confirm wide-open lane)
- For meeting-node + over-smoothing — i2 → c8 cluster summary (theory) + c2 cluster summary (closest path-anchored prior art)
- For pooling/attention noise — i3 → c1 cluster summary (attention insufficient) + c3 (DDI-Ben empirical)
- For node-name text semantics — i4 → c7 cluster summary (recipe established outside DDI) + c3 cluster summary (validated indirectly via TextDDI/ZeroDDI/DDI-Ben)
