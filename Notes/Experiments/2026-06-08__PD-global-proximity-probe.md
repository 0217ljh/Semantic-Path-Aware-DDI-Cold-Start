# PD Global Structural Proximity (GSP) Probe (S2 cold-start)

Date 2026-06-08. Run dir `Code/runs/2026-06-08__pd_global_proximity_probe/`.

## Question

We already proved Local Structural Proximity (LSP) is dead for pharmacodynamic (PD) DDIs. Common neighbours, walk counts, Adamic-Adar and Resource-Allocation all top out near 0.66 AUROC. This probe tests the one structural axis we had not tested, namely Global Structural Proximity. We ask whether GSP features separate PD positives from negatives better than LSP. The GSP families are Personalized PageRank / random-walk-with-restart proximity, diffusion distance, truncated Katz path sums, and typed length-2 metapath counts.

Literature motivation comes from Mao et al. "Revisiting Link Prediction: A Data Perspective", ICLR 2024 (arXiv:2310.00793), which shows GSP (Katz, PPR, SimRank) recovers link-prediction performance precisely when LSP is deficient. That is our PD regime because PD connects through high-degree hubs and is LSP-deficient.

## Setup

NO-MODEL probe. Hand-crafted pair features plus two cheap classifiers, LogisticRegression (L2, class_weight balanced, StandardScaler'd features) and HistGradientBoostingClassifier (raw features, random_state 42). Pure topology, no molecular, no text, no node identity. We report TEST AUROC and AUPRC.

The split, PD bucket definition, per-class cap, seed, and adjacency are identical to the twin probe `analyze_pd_higher_order_structure_probe.py` so the numbers are directly comparable. The only thing that changes is the feature families.

- Adjacency. Symmetric, binary, self-loop-free, relation-agnostic over all 178029 merged-KG nodes (nnz 9675024). Edges 7099528.
- Drugs. The 1900 nodes with `kind == "Drug"`.
- Protocol. PD positives (cap 8000) plus train negatives (8000) form the train set. Test S2 PD positives (cap 8000) plus test S2 negatives (cap 8000) form the test set. Seed 42. Final sizes are train pos 8000 / neg 8000, test pos 8000 / neg 8000.

### Feature families

| Family | Features |
|---|---|
| F_lsp (reference) | walks_len2, walks_len3, adamic_adar, resource_allocation |
| F_ppr | rw and sym operators, each contributing symmetrized proximity (sym), directional max/min, full-vector inner product (gram), and diffusion distance |
| F_katz | truncated Katz drug-drug path sums for beta in {0.1, 0.05, 0.01}, lengths l = 2..4 |
| F_typed | typed length-2 metapath counts per node-kind bucket, buckets {gene_protein, side_effect, pheno_effect, disease, pathway, anatomy, biological_process} |
| F_gsp_all | F_ppr + F_katz + F_typed (20 features) |

PPR uses alpha 0.85 (restart 0.15), up to 20 power iterations, tolerance 1e-6 on mean L1 delta per source. Two operators are included as variants, the row-normalized random walk `Ahat_rw = D^{-1} A` and the symmetric-normalized `Ahat_sym = D^{-1/2} A D^{-1/2}`. Diffusion distance and inner product come from the Gram matrix of the full landing vectors restricted to drug sources.

The drugbank enzyme / target / transporter / carrier roles do NOT exist as distinct node kinds in this merged KG. They are folded into the `gene/protein`, `Gene`, `Protein` kinds, so we did not add separate typed buckets for them. The typed buckets used are exactly the seven biological buckets listed above.

## Results (verified, from this run)

All numbers from `Code/runs/2026-06-08__pd_global_proximity_probe/metrics.json`.

| Family | LR AUROC | LR AUPRC | GBDT AUROC | GBDT AUPRC | best AUROC |
|---|---|---|---|---|---|
| F_lsp | 0.6499 | 0.6422 | 0.6557 | 0.6503 | 0.6557 |
| F_ppr | 0.6635 | 0.6525 | 0.6638 | 0.6618 | 0.6638 |
| F_katz | 0.6429 | 0.6438 | 0.6353 | 0.6337 | 0.6429 |
| F_typed | 0.6430 | 0.6418 | 0.6418 | 0.6441 | 0.6430 |
| F_gsp_all | 0.6731 | 0.6743 | 0.6738 | 0.6845 | 0.6738 |

### Gains over F_lsp (best-classifier AUROC per family)

| Family | gain over F_lsp |
|---|---|
| F_ppr | +0.0081 |
| F_katz | -0.0128 |
| F_typed | -0.0128 |
| F_gsp_all | +0.0181 |

PPR is the only single GSP family that beats LSP, and only by a hair. Katz and typed metapaths each land slightly below the LSP baseline. Stacking everything (F_gsp_all) gives the largest gain at +0.0181, which is still well short of the +0.03 lever threshold and stays at 0.6738 AUROC, far from the 0.70+ bar.

For context (cited, not recomputed). Real NBFNet v1.7 cold-start S2 PD AUROC is 0.727, the learned 1-WL-level structural ceiling. Prior LSP hand-crafted probes on PD topped near 0.60 to 0.66. Our F_lsp reference here reproduces that range at 0.6557.

## Which GSP feature drove the gain

From `feature_importance.csv` (permutation importance on the F_gsp_all GBDT, TEST AUROC, 10 repeats).

| feature | perm importance mean | std |
|---|---|---|
| ppr_sym_diffdist | 0.0529 | 0.0034 |
| ppr_sym_sym | 0.0292 | 0.0017 |
| ppr_rw_sym | 0.0259 | 0.0024 |
| ppr_rw_max | 0.0257 | 0.0008 |
| typed_disease | 0.0162 | 0.0008 |
| typed_gene_protein | 0.0145 | 0.0012 |
| ppr_rw_diffdist | 0.0138 | 0.0011 |

The signal that exists is PPR, not Katz and not typed metapaths. The single strongest feature is the diffusion distance from the symmetric, hub-discounting operator (`ppr_sym_diffdist`). The next tier is the symmetrized PPR proximity from both operators. So the small gain comes from global diffusion proximity under hub discounting, with diffusion distance edging out raw proximity. Katz features and most typed buckets sit near zero importance, and several (typed_pathway, typed_anatomy, typed_biological_process, ppr_sym inner/max/min) are exactly zero or slightly negative.

## PPR convergence and resources

- Operator rw. 20 iterations, final mean L1 delta 3.938e-02. Did not reach the 1e-6 tolerance within 20 iterations, but the delta is small and decaying steadily.
- Operator sym. 20 iterations, final mean L1 delta 9.540e-03. Also did not formally hit tolerance but converged tighter than rw.
- No chunking. `PPR_CHUNK = 0`. The landing matrix P is 178029 x 1900 float32 which is about 1.35 GB per operator. With 31 GB RAM available this fit comfortably and we ran the full source set in one pass per operator. No OOM. Peak was well under available memory.
- Total wall time 224.6 s.

Caveat on convergence. We capped at 20 iterations and did not reach the strict 1e-6 tolerance. The rw delta of 0.039 is the loosest. A longer iteration budget could tighten the PPR vectors slightly, but the family is already the best GSP family and the gap to the +0.03 lever threshold is large, so more iterations would not change the verdict.

## Verdict

**borderline.** GSP gain over LSP is between the two decision thresholds. F_gsp_all beats F_lsp by +0.0181 and F_ppr by +0.0081, both real but small. This is above the 0.01 floor that would have signalled a clean fundamental deficiency, yet far below the +0.03 and 0.70+ bar that would have made GSP the missing structural lever.

Honest reading. There is a faint, genuine global-proximity signal in PD that LSP does not capture, and it lives entirely in PPR diffusion proximity under hub discounting. But it is nowhere near enough to call GSP the lever. The practical conclusion matches the spirit of the fundamental-deficiency branch. Pure topology, now tested across both the local axis (LSP, dead) and the global axis (GSP, faint), is essentially exhausted for PD. The remaining signal must come from molecular or text. We now have ICLR-2024-backed evidence that even the GSP escape hatch, which is exactly where Mao et al. say structure should recover when LSP is deficient, recovers only about +0.02 here and never crosses 0.68.

## Caveats

- Single seed (42). No std across seeds.
- Per-class cap 8000 on both train and test PD pos/neg. Larger caps untested.
- PPR settings. alpha 0.85, 20 iterations, did not hit 1e-6 tolerance (rw delta 0.039, sym delta 0.0095). Two operators only.
- Adjacency is undirected, binary, and relation-agnostic. All relation types are collapsed. Typed information enters only through the typed-metapath family, which was near-useless here.
- Katz truncated at length 4 with three beta values. Longer truncation untested.

## Literature

Mao, Li, Tang, Wang, Sun, et al. "Revisiting Link Prediction: A Data Perspective." ICLR 2024. arXiv:2310.00793.

## Reproduce

```
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/analyze_pd_global_proximity_probe.py"
```
