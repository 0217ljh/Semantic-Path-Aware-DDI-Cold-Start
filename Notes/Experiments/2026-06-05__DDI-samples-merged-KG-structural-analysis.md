# DDI Samples: Structural Analysis in the Merged KG

**Date** 2026-06-05
**Script** `Code/scripts/analyze_ddi_merged_kg_structure.py`
**Artifacts** `Code/runs/analyze_ddi_merged_kg_structure/` (`summary.json`, `pos_features.parquet`, `neg_features.parquet`, `wl_collision.json`)
**Motivation** The C-MPNN / NBFNet / ULTRA discussion. DDI is a pairwise, query-conditioned link-prediction task whose only signal in the merged KG flows through the non-DDI neighbourhood. This note measures (1) how much path evidence DDI pairs actually have, (2) where the relational-WL ceiling of C-MPNN bites, and (3) whether structure alone separates positives from negatives.

All numbers below are read directly from the run output and `summary.json`. Nothing is from memory.

---

## 0. Setup (verified)

| Quantity | Value |
|---|---|
| Merged KG nodes | 178,029 |
| Merged KG edges | 7,099,528 (59 base relations, drugbank + hetionet + primekg) |
| Symmetric binary adjacency nnz | 9,675,024 |
| DDI drugs (`kind == Drug`) | 1,900 (all present as KG nodes) |
| Positive DDI pairs | 565,731 (0 dropped) |
| Negative pool (seed42 val+test, dedup, pos-filtered) | 348,256 |

The merged KG contains **no drug–drug DDI edges**. It does contain ~5.5k `het:CrC` (compound-*resembles*-compound) drug–drug edges, which are structural similarity, not interaction labels. So DDI is genuine link prediction over the surrounding target / enzyme / gene / pathway / side-effect / disease structure. This is exactly the C-MPNN setting.

---

## 1. Drug KG-degree (per-drug signal budget)

mean **166.2**, median **85**, min **0**, max **1476**.

**65 drugs (3.4%) have zero KG degree.** For these, C-MPNN has literally no neighbourhood to propagate over. Combined with cold-start (S2), where the drug is also unseen in training, these are the structurally-invisible core where path-based propagation provides nothing.

---

## 2. Pair-level path evidence (positives)

Shortest-distance proxy from walk matrices (1 = direct edge, 2 = shared neighbour, 3 = length-3, 4 = ≥4 / weakly connected within radius 3):

| Distance | Fraction of positive pairs |
|---|---|
| 1 (direct resemblance edge) | 0.57% |
| 2 (share ≥1 neighbour) | 76.50% |
| 3 | 14.89% |
| ≥4 / disconnected | 8.04% |
| **connected within 3 hops** | **91.96%** |

Per-feature (positives):

| Feature | median | %>0 |
|---|---|---|
| common neighbours (len-2 walks) | 6 | 77.1% |
| length-3 walks | 96 | 88.0% |
| shared gene/protein (≈ shared targets) | 2 | 57.8% |
| shared side effects | 0 | 35.3% |
| shared disease | 0 | 30.8% |
| shared pathway | 0 | **0.0%** |

**Reads:**
- ~76% of positive pairs are at distance 2, so most positives are reachable by the short 2-hop metapaths C-MPNN propagates over. Path evidence exists for the bulk.
- But **8% of positives are essentially disconnected within 3 hops**, and shared-target overlap is thin (median 2 genes, 42% share none). The dominant short-range signal is mostly the gene/protein layer.
- **Pathway overlap is effectively zero** (drugbank `db:pathway` is too sparse, ~1k edges). Pathway is not a usable structural channel in the current merged KG.

---

## 3. Pos-vs-neg contrast and trivial-predictor AUC

AUC of each single structural feature as a standalone interact/no-interact predictor (565,731 pos vs 348,256 neg):

| Feature | POS mean | NEG mean | AUC |
|---|---|---|---|
| shared gene/protein | 3.33 | 0.83 | **0.6881** |
| common neighbours | 27.28 | 10.63 | 0.6812 |
| shortest distance | 2.30 | 2.75 | 0.6550 |
| length-3 walks | 430.95 | 204.95 | 0.6343 |
| shared side effects | 14.12 | 5.66 | 0.5827 |
| shared disease | 1.75 | 0.72 | 0.5795 |
| direct edge | 0.01 | 0.00 | 0.5022 |
| shared pathway | 0.00 | 0.00 | 0.5000 |

**Reads:**
- There **is** a structural signal but it is modest. The single strongest structural cue (shared gene/protein neighbours) gives AUC **0.69**, and common-neighbour count gives **0.68**. This is the floor a pure C-MPNN exploits, and it is far from the project target (S2 AUROC > 0.80).
- The gap between ~0.69 (best raw structural cue) and >0.80 (target) is precisely the room a method must earn from something beyond raw KG structure. Direct-edge and pathway channels carry no signal.

---

## 4. Relational-WL collision diagnostic (the C-MPNN ceiling)

Relational 1-WL is the expressivity ceiling of C-MPNN. Two drug nodes that converge to the same WL colour are provably indistinguishable to any C-MPNN, so a pair built from such drugs cannot be separated from another pair with the same colour signature, regardless of label. Color-refinement run over the whole merged KG (117 relations incl. inverses, 14.2M message edges):

| Round | distinct drug colours / 1900 | twin drugs | pos pairs mechanism-ambiguous | pos signature also in negatives |
|---|---|---|---|---|
| 0 (init = node kind) | 1 | 100% | 100% | 100% |
| 1 | 1763 | 9.2% | 7.6% | 6.8% |
| 2 | 1826 | 4.3% | 4.0% | 4.0% |
| 3 (converged) | 1828 | **4.2%** | **4.0%** | **4.0%** |

Definitions:
- **twin drugs** drugs that share their final colour with ≥1 other drug (WL-indistinguishable).
- **mechanism-ambiguous** fraction of positive pairs whose (unordered) colour signature maps to >1 distinct `ddi_type`. Structure cannot decide *which* mechanism.
- **pos signature also in negatives** fraction of positive pairs whose colour signature also occurs among negative pairs. Structure cannot even decide *interact vs not*.

**Reads (the centrepiece):**
- After convergence, **~4.2% of drugs are WL-twins** and **~4.0% of positive DDI pairs sit on a colour signature that is irreducibly ambiguous** — either it carries multiple mechanisms, or it is shared with a negative pair. For this ~4% core, *no* C-MPNN built on this KG can be correct on all of them. This is the relational-WL ceiling landing on the hardest samples, made concrete and quantified.
- This ~4% is the **optimistic** ceiling. It assumes full-power WL (infinite rounds, exact colours, all 117 relations). A real finite-layer, finite-width C-MPNN will collapse *more* pairs, so 4% is a lower bound on the structurally-irreducible error.
- 6.27% of *negative* signatures also appear among positives (`frac_neg_signature_also_in_pos`, round 3), i.e. the collision is two-sided.

---

## 5. Implication for the method line

1. **Path evidence is real but the structural ceiling is ~0.69 AUC** (best single cue) and a hard ~4% WL-collision core. The leap to the >0.80 target cannot come from richer KG propagation alone.
2. **The differentiation point is exactly the WL-collision core.** Injecting non-structural signal that breaks the relational-WL symmetry — molecular fingerprints / SMILES, LLM-distilled pharmacology — on the ~4% structurally-equivalent-but-behaviorally-different pairs is where a C-MPNN backbone can be beaten. This matches the ColdDDI Tanimoto-vs-Recall diagnostic line.
3. **Two structural dead ends to drop:** pathway overlap (0% signal, sparse `db:pathway`) and direct-edge / resemblance (AUC 0.50). The live structural channels are gene/protein and common-neighbour count.
4. **Cold-start floor:** 65 zero-degree drugs (3.4%) + 8% of positives weakly connected (≥4 hops). In S2 these get neither training exposure nor usable path evidence — the hardest slice for any propagation model.

---

## Reproduce

```bash
# from project root, via WSL conda env project_1
python Code/scripts/analyze_ddi_merged_kg_structure.py --wl-rounds 3 --seed 42
```

Per-pair feature tables (`pos_features.parquet`, `neg_features.parquet`) carry every feature per pair for downstream slicing (e.g. cross with Tanimoto, or with S0/S1/S2 membership).
