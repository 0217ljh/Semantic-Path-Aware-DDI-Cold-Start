# Cold-Start Audit: 8 Papers (original paper + original GitHub, verified 2026-06-22)

**Purpose**: determine, for each method, (a) whether it addresses cold-start (unseen drugs) and (b) HOW it represents a cold drug. Decisive for Section 2 narrative. Verified against ORIGINAL upstream papers + GitHub (NOT local modified copies — user warned local TIGER/MKG-FENN were modified).

## Representation classes
- (a) **inductive-encode**: new drug's own SMILES/graph + own KG neighborhood through trained encoders
- (b) **kNN / similar-drug substitution**: replace new drug with most-similar known drug(s)
- (c) **embedding-table / transductive**: per-drug id embedding; unseen drug has no entry
- (d) **not addressed**: warm-start / transductive link split only

## MASTER TABLE

| Paper | Venue | Multi-modal tier | Paper addresses cold-start? | Cold-drug representation | Evidence |
|---|---|---|---|---|---|
| **MKG-FENN** | AAAI24 | A (clean dual-modal) | YES (Task2=S1, Task3=S2) | **(b) kNN similar-drug substitution** | `modeltask2.py:55-60` cold drug rep = avg of nearest training drugs; `MKG-FENN-task2.py:341-392` test_adj via drug_sim1-4 |
| **TIGER** | AAAI24 | A | **NO** (zero mention; 5-fold link split) | (d) not addressed | `main.py:97-105` only `scenario_type=='random'` StratifiedKFold over links |
| **MUFFIN** | Bioinf21 | A | **NO** (5-fold random edge) | (c) embedding-table/transductive | `main.py:224` random KFold; `:309-320` precomputed TransE + GIN `.npy` tables by id |
| **MIRACLE** | WWW21 | B (DDI-graph) | **NO** (4:1 random edge, link prediction) | (c) embedding-table/transductive | paper §4 4:1 edge split; intra-view GCN over fixed DDI adjacency |
| **MolecBioNet** | KDD25 | A | YES (paper §4.2 "novel drugs") | (c) in released code (paper claim not reproducible) | paper Table 2 novel-drug; but repo only `StratifiedKFold(5)` random + `nn.Embedding(num_nodes)` id table + per-id CenterLoss |
| **DDKG** | BIB22 | C (init-only) | YES (DW-CV≈S1, PW-CV≈S2) | (a) inductive (SMILES→LSTM enc-dec + KG aggregation) | paper Table 3; `models/ddkg.py:147-185`. Caveat: DW/PW-CV driver not in released run.py (only random CV wired) |
| **KnowDDI** | NatCommMed24 | (KG-only) | **NO** (edge holdout, DDI scarcity not unseen drugs) | (c) embedding-table/transductive | `GraphSAGE.py` `pre_embed=nn.Parameter(num_nodes,dim)` by global id |
| **K-Paths** | KDD25 | (KG-only) | YES (inductive, core) | (a) LLM-path inductive / (c) GNN variant | uses EmerGNN inductive split; LLM mode injects KG paths as text (no per-drug param); GNN mode `nn.Embedding(num_nodes)` |
| **EmerGNN** | NatComputSci23 | (KG-based, mol only inits) | YES (CORE: S0/S1/S2) | **(a) inductive KG-structure + frozen Morgan FP** | `load_data.py:136-175` shuffle_train holds out DRUGS; `models.py:19-29` frozen Morgan FP as flow source, no learnable id embedding in cold mode |

## KEY FINDINGS

### Finding 1 — Genuine multi-modal methods do NOT properly do cold-start
Of the 4 clean dual-modal methods (Tier A: MKG-FENN, TIGER, MUFFIN, MolecBioNet):
- **TIGER**: no cold-start at all (zero mention, transductive link split)
- **MUFFIN**: no cold-start at all (random edge split)
- **MKG-FENN**: has cold-start splits BUT uses naive kNN similar-drug substitution (NOT inductive encoding) — the cold drug is literally replaced by the average of its nearest training drugs
- **MolecBioNet**: claims novel-drug results in the paper, but the released code ships only a random edge split + transductive id-embedding table; the novel-drug split is not reproducible from the repo

→ **No genuinely multi-modal method demonstrably implements proper inductive cold-drug encoding.** They are warm-start methods; where cold-start appears, it is naive substitution (MKG-FENN) or unreproducible (MolecBioNet).

### Finding 2 — Cold-start is properly done by KG-based methods (EmerGNN)
- **EmerGNN** is the only method that both claims AND demonstrably implements end-to-end inductive unseen-drug inference: holds out DRUGS (not edges), represents cold drug via KG-structure flow propagation seeded by frozen Morgan fingerprint, no learnable per-drug id embedding in cold mode. This is its central contribution; it is the source of the S0/S1/S2 naming.
- **K-Paths** (2025) also genuinely inductive (in LLM training-free mode), uses EmerGNN's inductive split.
- But EmerGNN is NOT multi-modal in our sense — molecular info only seeds KG node init (Tier C-equivalent).

### Finding 3 — EmerGNN's path learning is a NOISY SUBGRAPH, not interpretable mechanism paths (verified, supports our critique)
- EmerGNN propagates over the FULL augmented KG sparse adjacency `[all_ent, all_ent, 2*all_rel+1]` (`load_data.py:133`), NOT a pre-extracted mechanism subgraph.
- "Path" structure emerges only from a soft per-relation sigmoid attention applied L=3 times (`models.py:64-67`).
- Interpretable paths are reconstructed POST-HOC by beam search in `visualize()` (`models.py:131-217`), SEPARATE from prediction.
- The paper itself notes the original subgraph has "thousands of edges / too much irrelevant information" → it is attention-pruned noisy propagation, NOT explicit interpretable mechanism paths.

## RESULTING NARRATIVE (locked direction)

The audit supports the user's fallback narrative:

> "Multi-modal DDI prediction is predominantly warm-start. Genuine multi-modal methods (TIGER, MUFFIN) do not even evaluate cold-start; those that touch it either use naive kNN similar-drug substitution (MKG-FENN) or do not release a reproducible cold-start pipeline (MolecBioNet). The methods that DIRECTLY construct cold-start representations are KG-based (e.g., EmerGNN), but EmerGNN's path learning is deficient: it does not learn concrete, interpretable mechanism paths — it propagates over a noisy full-KG subgraph with soft attention, so its 'paths' are attention-pruned noise rather than the interpretable mechanism corridors through which DDIs actually arise."

### Two-gap positioning for our paper
```
Gap 1 (multi-modal side):  multi-modal DDI methods are warm-start;
                           they have no transferable cold-start design
Gap 2 (KG-based side):     KG-based methods (EmerGNN) DO cold-start,
                           but learn noisy subgraphs, not interpretable mechanism paths

Our work fills BOTH: multi-modal (fragment + KG) alignment that is cold-start
                     transferable, built on interpretable mechanism-level hyper-edges
                     (not noisy subgraphs)
```

## Caveats to state honestly in paper
1. MKG-FENN's cold-start = kNN substitution is the ORIGINAL author design (`modeltask2.py`); when we benchmark on our cold-start split we should either (a) use their kNN as-is, or (b) note any adaptation.
2. TIGER/MUFFIN have no native cold-start; any cold-start number we report for them is from a cold-start adaptation (e.g., ColdDDI protocol), NOT the original method. Must state this.
3. MolecBioNet's novel-drug claim is in the paper but not reproducible from released code (random split + id-embedding table only).
4. EmerGNN's "noisy subgraph" critique is verified from code (full-KG propagation + post-hoc beam-search visualization).

## Cross-references
- PDFs: `Paper/Reference/`
- Prior axis table: `Notes/Log/section2_multimodal_axes.md`
- Section 2 logic: `Notes/Log/paper_locked_logic.md`
- EmerGNN code: github.com/LARS-research/EmerGNN
