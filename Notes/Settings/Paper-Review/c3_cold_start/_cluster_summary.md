# Cluster c3 — Cold-Start / Inductive DDI Prediction: Summary

## How cold-start is typically framed in c3
Cold-start DDI work fragments along two orthogonal axes. The first axis is **what is unseen at test time**: most papers target *drug cold-start* (a new drug never appears in the training DDI graph), while a smaller branch targets *event/relation cold-start* (a new interaction type with no labeled examples but with seen drugs). The drug-cold-start branch further splits into **S1** (one new drug + one known drug) and **S2** (two new drugs — both unseen, the strictest setting), following the Pahikkala 2021 taxonomy that the field has near-universally adopted. The second axis is **what information substitutes for missing DDIs**: structural (SMILES substructures), topological (biomedical KG paths), or semantic (textual descriptions / class names). Recent benchmark evidence (DDI-Ben 2024) shows that purely structural and topological methods collapse under realistic distribution shift, while text/LLM-augmented methods retain substantially more performance — strongly motivating semantic priors. However, no published method anchors prediction on a **named mediator (meeting node)** with **PK/PD-aware** path typing while also using **node-name semantics** — this is the joint gap our paper addresses.

## Split convention table

| Paper                | Cold-start axis           | Splits used                     | New-drug feature source                   |
|----------------------|---------------------------|---------------------------------|--------------------------------------------|
| Pahikkala 2021       | Drug (taxonomy)           | dde / dd / d^de / d^d^e         | Similarity kernels                         |
| CSMDDI (2022)        | Drug                      | S1 + S2 (canonical names)       | Tabular attributes (targets, SEs)          |
| SumGNN (2021)        | Relation (low-resource)   | Per-relation inductive          | KG topology                                |
| HyGNN (2022)         | Drug (claimed)            | Standard CV (rigor weak)        | SMILES substructures                       |
| META-DDIE (2022)     | Event type (few-shot)     | C-way K-shot meta               | SMILES SPM                                 |
| EmerGNN (2023)       | Drug (emerging)           | Chronological emerging-drug     | KG topology only                           |
| TextDDI (2023)       | Drug (zero-shot)          | Chronological, drug-disjoint    | DrugBank/PubChem text                      |
| KnowDDI (2024)       | Mixed (KG-augmented)      | Transductive + inductive        | KG topology + GraphSAGE                    |
| ZeroDDI (2024)       | Event class (zero-shot)   | 175 seen / 68 unseen classes    | GIN + BioBERT class text                   |
| DDI-Ben (2024)       | Benchmark (all)           | Distribution-shift splits       | Method-dependent                           |

## Which of our insights are addressed by prior c3 work

| Insight                                       | Addressed by | Gap |
|-----------------------------------------------|------|-----|
| **i1** PK and PD as two paradigms            | None — all papers treat DDI relations as a flat label set. ZeroDDI's (Effect, Sign, Pattern) decomposition is the closest analog but is not mechanism-typed. | **Full gap.** Our paper can be the first to architecturally separate PK and PD reasoning. |
| **i2** Meeting node + over-smoothing         | Partially — SumGNN/KnowDDI/EmerGNN use subgraph or path anchoring (closer to mediator spirit), but all still force drug-to-drug flow with k≥3 hops on dense biomedical KGs, putting them squarely in over-smoothing regime. None explicitly identify a *single mediator node* as the prediction anchor. | **Strong gap** with adjacent prior art to build on. |
| **i3** Pooling noise + attention insufficient | Partial validation — DDI-Ben confirms attention-only KG methods collapse under shift. ZeroDDI injects BioBERT prior on event side. TextDDI's RL selector acts as a semantic external supervisor. No method injects external semantic prior on the *drug-node side* of a KG GNN. | **Targeted gap** — combine path GNN with external semantic priors on drug-node attention. |
| **i4** Node-name biomedical text semantics    | Strongly validated — TextDDI (zero-shot DDI), ZeroDDI (zero-shot event), DDI-Ben (benchmark) all show text/semantic features are the most robust signal under cold-start. **But** the leading KG/GNN cold-start methods (EmerGNN, KnowDDI, SumGNN) explicitly do NOT use text features on KG nodes. | **Open lane** — bring node-name semantics into KG-path-based DDI prediction. |

## Bottom line for our paper

Prior c3 work has separately demonstrated each ingredient of our thesis:
- Path/subgraph anchoring helps under cold-start (SumGNN, EmerGNN, KnowDDI).
- Semantic text features are more robust than structure under distribution shift (TextDDI, ZeroDDI, DDI-Ben).
- Event decomposition into reusable attributes enables zero-shot (ZeroDDI).

**No published method combines: (a) explicit meeting-node anchoring, (b) PK/PD-aware path typing, and (c) node-name semantic priors over a biomedical KG.** This is a clean, defensible position for the paper.

## Files in this cluster
- `zitnik_2021_cold_start_problems_ddi.md` — taxonomy
- `wang_2022_csmddi.md` — first explicit S1/S2 method
- `yu_2021_sumgnn.md` — KG-subgraph inductive
- `tanvir_2022_hygnn.md` — SMILES hypergraph
- `deng_2022_meta_ddie.md` — few-shot event (contrast paper)
- `zhang_2023_emergnn.md` — flow-based KG for emerging drugs
- `zhu_2023_textddi.md` — text-only zero-shot
- `wang_2024_knowddi.md` — KG subgraph + resemble edges
- `geng_2024_zeroddi.md` — semantic zero-shot event
- `zhang_2024_ddi_ben.md` — benchmark / robustness evidence
