# Cluster c2 — Path-Based / Multi-Hop Reasoning for DDI

## Papers in this cluster (8)

| # | Year | Paper | Venue | Read |
|---|------|-------|-------|------|
| 1 | 2021 | SumGNN (Yu et al.) | Bioinformatics | full |
| 2 | 2021 | RANEDDI (Yu et al.) | Information Sciences | abstract |
| 3 | 2022 | MedKGQA (Gao et al.) | arXiv (rev. 2024) | full |
| 4 | 2023 | EmerGNN (Zhang et al.) | Nat. Computational Science | full |
| 5 | 2024 | KnowDDI (Wang et al.) | Nat. Communications Medicine | full |
| 6 | 2024 | MPHGCL-DDI (Hu et al.) | Molecules | full |
| 7 | 2024 | BioPathNet (Hu et al.) | bioRxiv / Nat. Biomed. Eng. (2025) | full |
| 8 | 2025 | K-Paths (Abdullahi et al.) | KDD '25 | full |
| 9 | 2026 | RISE-DDI (Xie et al.) | AAAI-26 | abstract |

Full text: 7, Abstract only: 2.

## How methods score / select paths

| Method | Selection mechanism | Endpoints | Learned? |
|--------|--------------------|-----------|----------|
| SumGNN | Self-attention edge pruning over h-hop enclosing subgraph | drug-pair | yes |
| RANEDDI | Relation-aware aggregation (implicit paths) | drug-pair | yes |
| MedKGQA | Directed metabolic pathway + MRC | drug → protein → ... → drug | partly |
| EmerGNN | Flow-based GNN over union of paths ≤ L | drug-pair (pair-anchored flow) | yes |
| KnowDDI | Subgraph pruning + injected drug-similarity edges | drug-pair | yes |
| MPHGCL-DDI | Five hand-crafted meta-paths + contrastive views | drug-pair | no (paths are fixed) |
| BioPathNet | NBFNet Bellman-Ford path representation | pair-anchored (h(u, v)) | yes |
| K-Paths | Diversity-aware Yen's K-shortest loopless paths | drug-pair | no (retrieval) |
| RISE-DDI | RL-policy edge-by-edge subgraph extraction | drug-pair | yes |

## Cold-start coverage

- **Explicit S2 (both-drugs-unseen) benchmarks**: EmerGNN (gold standard), RISE-DDI (inductive but not split-typed).
- **Zero-shot / inductive but not strict S2**: K-Paths (training-free), BioPathNet (zero-shot disease split, not drug split), KnowDDI (sparse-KG robustness).
- **Admits failure on S2**: MPHGCL-DDI — explicit acknowledgement that meta-path models fail on new-new drugs.
- **Not evaluated**: SumGNN, RANEDDI, MedKGQA.

## Cross-cutting takeaways (vs i1–i4)

- **i1 (PK/PD two paradigms)**: essentially **uncovered in the c2 literature**. MPHGCL-DDI mixes PK-leaning and PD-leaning meta-paths but averages them; MedKGQA covers only PK (drug-protein-protein-drug). No paper separates reasoning for the two regimes. Strong open-gap signal.
- **i2 (meeting node + over-smoothing)**: best-supported insight in the cluster. BioPathNet (NBFNet pair representation) and EmerGNN (flow-based pair-anchored) give the methodological template; MedKGQA confirms the protein-as-mediator intuition for PK paths.
- **i3 (pooling noise + attention limit)**: every paper acknowledges this implicitly — SumGNN/RANEDDI rely on attention reweighting; KnowDDI is the only one that *injects* extra edges (drug similarity); K-Paths is the only one that does explicit external retrieval. No paper yet injects **text-derived** semantic priors.
- **i4 (node-name semantic prior)**: completely absent from c2. Every method uses ID-initialised or KG-embedding-initialised intermediate nodes. This is the most uncontested gap.

## Two-sentence takeaway for i1–i4 coverage

Path-based DDI methods strongly support **i2** (BioPathNet and EmerGNN both move from drug-anchored to pair/path-anchored representations) and partially address **i3** (KnowDDI injects similarity edges, K-Paths retrieves diverse paths, RISE-DDI uses RL pruning) but no c2 method separates PK vs PD reasoning regimes (**i1** is wide open) and none uses biomedical text on intermediate node names (**i4** is wholly uncontested). The most relevant baselines to compare against are EmerGNN (cold-start S2 splits), KnowDDI (similarity-edge injection), BioPathNet (NBFNet path-representation template), and K-Paths (training-free path retrieval).

## Recommended baseline lineup for our experiments

1. **EmerGNN** — same S1/S2 split, direct head-to-head.
2. **KnowDDI** — strongest "subgraph + injected edges" baseline.
3. **BioPathNet (adapted to DDI)** — strongest pair-anchored NBFNet baseline; needs a drug-drug adaptation.
4. **K-Paths + GNN** — training-free retrieval baseline.
5. **SumGNN** — classical attention-pruning baseline for the i3 ablation.
6. **MPHGCL-DDI** — meta-path baseline; useful to show i1 split (PK vs PD meta-paths) improves over averaging.

## Open directions left untouched by c2

- Typing paths into PK vs PD reasoning streams (i1).
- Using node-name biomedical text to score / initialise paths (i4).
- Combining (i) external retrieval (K-Paths style) with (ii) text-prior injection (i4) on top of (iii) pair-anchored readout (BioPathNet/EmerGNN style) — this composition is the natural opening for our work.
