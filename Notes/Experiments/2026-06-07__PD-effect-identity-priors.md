# 2026-06-07 — PD effect-identity STRUCTURAL PRIORS probe (L=3)

> TL;DR: At BFS depth L=3, no purely-structural prior (IDF / degree-norm /
> top-k-specific / PMI) recovers pos-vs-neg signal beyond the count-only baseline
> F0. Best gain = **+0.0124 AUROC** (IDF-LR), far below the +0.03 bar. The
> "most specific" effects the priors surface are themselves near-universal hubs
> (reached by ~1740/1900 drugs). **Verdict: structural priors CANNOT denoise L=3
> -> confirms go-molecular.**

## Question

At L=3 the discriminative shared-effect signal is buried under high-degree HUB
nodes that almost every drug pair reaches. Can a label-free structural PRIOR /
inductive weighting on the shared-effect identity multi-hot RECOVER signal that
raw 0/1 identity cannot? No model training — hand-crafted features + LR/GBDT.

## Setup

- Script: `Code/scripts/analyze_pd_effect_identity_priors.py`
- BFS/KG/assembly machinery copied verbatim from the working L=1 probe
  `Code/scripts/analyze_pd_effect_identity_probe.py`. Only `L=3` and the
  prior-variant feature construction are new.
- 1900-drug DrugBank split seed42, S2 (train=G1, test=G2). PD-bucket positives,
  sampled negatives. Cap **8000/class** for both train and test → 16k train, 16k test.
- Vocab V = effect nodes shared by ≥20 distinct TRAIN pairs, capped to top-3000
  by train-pair frequency. **|V| = 3000** (hit the cap).
- Undirected merged KG: 178,029 nodes, 9,675,024 dir-edges; 1900 drugs, 38,700
  effect-type nodes. `max drugreach = 1828/1900` (some effect nodes are reached
  by 96% of all drugs at L=3 — the hub problem made concrete).
- Effect kinds: Side Effect / effect/phenotype / Symptom / disease / Disease.
- Artifacts: `Code/runs/2026-06-07__pd_effect_identity_priors/metrics.json`,
  `.../effect_node_weights__IDF.csv`. Single seed (42). Runtime 204s.

### Feature sets

- **F0** (count, no identity): `[total #shared, #shared min-depth≤1, #shared min-depth≤2, total]`.
- Identity variants = F0 ++ weighted multi-hot over V:
  - **raw**: 1 if pair shares X.
  - **IDF**: `log(n_drugs / drugreach[X])` — down-weight nodes many drugs reach.
  - **degnorm**: `1 / sqrt(max(deg[X],1))`.
  - **topk_specific**: keep only the K=30 shared nodes per pair with smallest `drugreach`; value 1.
  - **PMI**: `max(0, log( (cooc[X]/N_train) / (p[X]^2 + 1e-9) ))`, `p[X]=drugreach[X]/n_drugs`.

`drugreach[X]` built from a full BFS over **all 1900 drug-pool drugs** at L=3
(genuine corpus-level hub statistic, not just assembled-pair drugs).

## Results (TEST AUROC / AUPRC, gain = AUROC − F0)

| feature set | LR AUROC | LR AUPRC | LR gain | GBDT AUROC | GBDT AUPRC | GBDT gain | **best gain** |
|---|---|---|---|---|---|---|---|
| **F0** (count) | 0.6358 | 0.6307 | — | 0.6345 | 0.6265 | — | — |
| raw | 0.6126 | 0.5699 | −0.0232 | 0.6328 | 0.6187 | −0.0017 | −0.0017 |
| IDF | **0.6481** | 0.6307 | **+0.0124** | 0.6328 | 0.6187 | −0.0017 | **+0.0124** |
| degnorm | 0.6438 | 0.6201 | +0.0080 | 0.6328 | 0.6187 | −0.0017 | +0.0080 |
| topk_specific | 0.6363 | 0.6126 | +0.0005 | 0.6364 | 0.6286 | +0.0019 | +0.0019 |
| PMI | 0.6439 | 0.6339 | +0.0081 | 0.6337 | 0.6215 | −0.0008 | +0.0081 |

All numbers from `metrics.json` (run 2026-06-07__pd_effect_identity_priors).

### Reading the table

- **raw identity at L=3 HURTS** (LR −0.0232): the multi-hot is dominated by hub
  columns shared by nearly every pair, so it adds noise/overfitting rather than signal.
- IDF / degnorm / PMI each pull LR back up a little (+0.008 to +0.012) by
  down-weighting hubs — but the best of any prior is **+0.0124**, well under the
  +0.03 recovery bar.
- **GBDT raw/IDF/degnorm are byte-identical (0.6328)**: HistGradientBoosting bins
  each feature, so it is invariant to monotonic per-column rescaling — multiplying
  a column by a constant prior weight produces identical splits. The prior only
  moves a *linear* model (LR). topk_specific/PMI differ for GBDT because they
  change the sparsity pattern / zero-floor, not just scale. So the priors'
  whole effect is the small LR bump above.

## Interpretability — best variant (IDF-LR) top positive-weight effects

The failure mode is explicit. The "most-up-weighted specific" effects are
themselves near-universal hubs (`drugreach` ≈ 1725–1767 out of 1900):

| name | kind | degree | drugreach | train_pair_freq | lr_weight |
|---|---|---|---|---|---|
| postpartum depression | disease | 11 | 1740 | 13562 | 1.885 |
| Creutzfeldt Jacob disease | disease | 171 | 1732 | 13514 | 1.434 |
| narcolepsy without cataplexy | disease | 25 | 1745 | 13680 | 1.110 |
| C-reactive protein increased | Side Effect | 6 | 1738 | 13487 | 1.102 |
| neurocirculatory asthenia | disease | 24 | 1752 | 13738 | 1.070 |
| Aneurysm | Side Effect | 12 | 1733 | 13529 | 1.022 |
| Elevated mood | Side Effect | 5 | 1725 | 13498 | 0.997 |
| idiopathic bronchiectasis | disease | 50 | 1755 | 13626 | 0.939 |
| Impetigo | Side Effect | 9 | 1733 | 13464 | 0.937 |
| Focal sensory seizure | effect/phenotype | 101 | 1765 | 13934 | 0.893 |
| acquired polycythemia vera | disease | 63 | 1767 | 13871 | 0.870 |
| Asthenopia | Side Effect | 17 | 1734 | 13588 | 0.766 |
| polycystic kidney disease | disease | 111 | 1751 | 13622 | 0.752 |
| Illusion | Side Effect | 14 | 1737 | 13474 | 0.734 |
| Syncope vasovagal | Side Effect | 17 | 1752 | 13736 | 0.734 |

None of these are mechanistically real PD effects — they are NOT Torsade de
pointes / serotonin syndrome / myelosuppression. They are low-*degree* nodes that
nonetheless get reached by ~91–93% of all drugs through 3-hop paths (note `polycystic
kidney disease`, the canonical L=3 hub failure example, appears here with
drugreach 1751 ≈ 92%). IDF down-weights them because every drug reaches them,
but they still co-occur in ~13,500/16,000 train pairs, so even after re-weighting
they carry a positive linear weight. The structural prior cannot tell "specific PD
mechanism" from "low-degree-but-globally-reachable hub" — at L=3 those are the same.
(Contrast: the L=1 probe surfaced REAL effects — Torsade de pointes, Myelosuppression,
long QT syndrome, neuroleptic malignant syndrome — because at L=1 only true shared
direct effects qualify.)

## Decision

> **Structural priors CANNOT denoise L=3.** Best prior gain over F0 = **+0.0124**
> (IDF-LR), below the +0.03 bar; AND the up-weighted effects are near-universal
> hubs, not real PD mechanisms. The relevant core PD effects are themselves hubs
> at 3 hops, so degree/IDF/PMI suppress them together with the noise.
>
> → **Confirms go-molecular.** A purely-structural KG re-weighting will not
> rescue the L=3 regime; the discriminative PD signal has to come from molecular
> substrate, not from deeper KG hops.

## Caveats (honest)

- Single seed (42); caps 8000/class (16k train, 16k test) and vocab capped at 3000.
- L=3, **undirected** symmetric adjacency on the merged KG.
- Hand-crafted features only — absolute AUROC ~0.63 is below NBFNet's ~0.727; we
  read the **increment** vs F0, not the absolute level.
- GBDT identity variants are scale-invariant, so the LR bump is the only channel a
  prior can act through here; a prior that helped GBDT would have to change feature
  *presence*, not weight (only topk_specific does, and it gives +0.0019).
- Known failure mode (observed directly): core PD effects are themselves hubs at
  L=3, so no monotone per-node prior separates them from globally-reachable noise.
