# PD higher-order structure probe (S2 cold-start)

TL;DR. Pair-level higher-order graph structure (degree-corrected common-neighbors and subgraph/motif density of the shared neighborhood) does NOT separate PD positives from negatives meaningfully better than a plain walk-count baseline. Best AUROC moves from 0.6570 (F_count) to 0.6617 (F_degcorr) to 0.6605 (F_motif), so the motif gain over walk-counts is only +0.0035 and the degree-corrected gain is only +0.0047. Pure topology is tapped. The remaining PD signal must come from molecular or text sources.

## Research question

For pharmacodynamic (PD) DDIs in cold-start (S2), can pair-level higher-order graph structure (path multiplicity, degree-corrected common-neighbors, and especially subgraph/motif density of the shared neighborhood) separate positives from negatives better than a simple walk-count baseline. This tests whether structure ABOVE NBFNet's relational-1-WL ceiling carries PD signal. No semantics, no molecular, no node identity. Pure topology only.

## Setup

This is a no-model probe. We hand-craft pair-level topology features and fit two cheap classifiers, then report TEST AUROC and AUPRC. The graph is the merged KG (DrugBank plus Hetionet plus PrimeKG, 178029 nodes, 7099528 edges). The adjacency `A` is symmetric, binary, self-loop-free, and relation-agnostic over all merged-KG nodes (adjacency nnz 9675024). Drug rows are the 1900 `kind == "Drug"` nodes. We build the dense 1900x1900 pair matrices once and index per pair.

The split, PD bucket, cap, and seed are copied exactly from the PD effect-identity probe so the numbers are comparable. Train pairs are PD-positive (cap 8000) plus train negatives (8000). Test pairs are test_s2 PD-positive (cap 8000) plus test_s2 negatives (cap 8000). Seed 42 throughout. Splits live under `Code/data/KG/drugbank/splits/seed42/`.

### Feature families (nested, pure topology)

- **F_count** (the walk-count baseline NBFNet implicitly aggregates). len-2 walks (common neighbors), len-3 walks, len-4 walks.
- **F_degcorr** = F_count plus Adamic-Adar and Resource-Allocation degree-corrected common-neighbor heuristics.
- **F_motif** = F_degcorr plus subgraph/motif density of the shared neighborhood. For each pair we take `S` = common neighbors of the two drugs, then compute `n_common`, edges within `S`, density of `S`, and a 4-cycle proxy (equal to edges within `S`).

For each family we fit LogisticRegression (L2, class_weight balanced, max_iter 2000) on StandardScaler'd features, and HistGradientBoostingClassifier (random_state 42) on raw features.

## Results (TEST, all verified from this run)

| Feature family | LR AUROC | LR AUPRC | GBDT AUROC | GBDT AUPRC | best AUROC |
|---|---|---|---|---|---|
| F_count   | 0.6410 | 0.6459 | 0.6570 | 0.6519 | 0.6570 (GBDT) |
| F_degcorr | 0.6413 | 0.6466 | 0.6617 | 0.6607 | 0.6617 (GBDT) |
| F_motif   | 0.6442 | 0.6487 | 0.6605 | 0.6605 | 0.6605 (GBDT) |

Headline gains over the walk-count baseline (best-classifier AUROC per family).

- F_degcorr minus F_count = **+0.0047**
- F_motif minus F_count = **+0.0035**

GBDT is the best classifier for all three families. Note that F_motif's best AUROC (0.6605) is actually slightly below F_degcorr's (0.6617), so adding the motif features brought no improvement over the degree-corrected family and the small headline gain comes from the degree-correction not from the motif/subgraph signal.

## Which feature drives any gain

Permutation importance on the F_motif GBDT (10 repeats, scoring roc_auc, computed on TEST) ranks the features as follows.

| feature | perm importance mean | perm importance std |
|---|---|---|
| resource_allocation | 0.1168 | 0.0048 |
| walks_len3 | 0.0561 | 0.0036 |
| walks_len4 | 0.0181 | 0.0021 |
| adamic_adar | 0.0093 | 0.0019 |
| walks_len2_common_neighbors | 0.0023 | 0.0011 |
| motif_n_edges_within_S | 0.0022 | 0.0006 |
| motif_density_S | 0.0020 | 0.0008 |
| motif_n_common | 0.0000 | 0.0000 |
| motif_n_4cycle_proxy | 0.0000 | 0.0000 |

The dominant feature is Resource-Allocation (a degree-corrected common-neighbor count), followed by len-3 walks. The genuinely beyond-1-WL motif/subgraph features (edges within `S`, density of `S`, the 4-cycle proxy) sit at the very bottom with near-zero permutation importance. In other words the small bit of separation that exists is carried by degree-discounted walk counts, not by subgraph motif structure.

## Verdict

**Tapped.** Even motif/subgraph structure is exhausted. With F_motif minus F_count at only +0.0035 (well under the ~0.01 borderline and far under the +0.03 win threshold) and F_motif sitting at 0.6605, higher-order topology does NOT separate PD positives from negatives beyond walk counts. Pure structure is tapped. The remaining PD signal must come from molecular or text sources rather than from a subgraph-GNN, SEAL, or k-WL model over this KG topology.

## Context (reference numbers, cited not recomputed)

- Real NBFNet v1.7 cold-start S2 PD AUROC = 0.727 (learned model, the 1-WL-level structural ceiling on PD).
- Prior hand-crafted shared-effect probes on PD pos-vs-neg topped out around 0.60 to 0.64 AUROC.

All three hand-crafted topology families here land around 0.64 to 0.66, in line with the prior hand-crafted ceiling and well below the learned NBFNet 0.727. So the gap between hand-crafted topology and the learned model is NOT closed by adding higher-order or motif structure.

## Caveats (honest)

- Single seed (42). No multi-seed mean or std. Treat the small deltas as noise-level.
- Each class capped at 8000 train and 8000 test pairs. Larger pools could shift absolute numbers slightly.
- The |S| > 1000 motif subsample cap was hit on **0 pairs** (train 0, test 0), so the motif density features are exact for every sampled pair in this run. No silent truncation occurred.
- The adjacency `A` is undirected, binary, and relation-agnostic. It collapses all relation types and edge directions, so any signal that lives in typed or directed structure is invisible to this probe by construction. This matches the spec (pure topology only).
- F_motif's best AUROC was slightly below F_degcorr's, so the motif features added nothing on top of the degree-corrected family.

## Reproduce

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/analyze_pd_higher_order_structure_probe.py"
```

Outputs.

- `Code/runs/2026-06-08__pd_higher_order_structure_probe/metrics.json`
- `Code/runs/2026-06-08__pd_higher_order_structure_probe/feature_importance.csv`
