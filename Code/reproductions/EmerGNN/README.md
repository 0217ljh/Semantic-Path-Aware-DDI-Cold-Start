# EmerGNN Reproduction

Reproduce the cold-start S1 and S2 results in Zhang et al. (Nat Comp Sci
2023), Table 1, using:
- **Data**: official `data-DB.zip` from the Zenodo mirror
  ([10.5281/zenodo.10016715](https://zenodo.org/records/10016715)),
  unpacked into `_Original-Dataset/DrugBank/data/`. MD5 of the outer zip
  matches Zenodo metadata `51f92e5a1cfbf745d82f32087f830e76`. Shipped
  vocab + Morgan-FP artefacts live under
  `_Original-Dataset/DrugBank/data/necessary/` with `__official` suffix
  per CLAUDE.md §"复现代码 (Reproduction) 规范" §2.
- **Model**: local copy at `model_mc.py` (independent of `baseline/`
  per CLAUDE.md §"Baseline 规范" line 447-448 "严禁 import 互调"),
  `EmerGNN_MC(n_classes=86, n_dim=64, length=3, feat='M')`. Algorithm
  ported from upstream `LARS-research/EmerGNN/DrugBank/models.py`
  (snapshot at `_paper-and-GitHub/LARS-research__EmerGNN/`); 2026-05-18
  audit confirmed structural correspondence on dimensions + module
  layout, but **bit-for-bit equivalence under identical inputs has not
  been programmatically verified yet** (see `_reviews/2026-05-18__paper_faithful.md`).
- **Task**: per-pair multi-class over 86 DDI relation types
  (paper §Methods, eq. 5: softmax CE). Drug pairs are positives only;
  the implicit "negatives" are the 85 other relations in the softmax
  denominator.

## Paper's reported numbers (Table 1, DrugBank, 5-seed mean ± std)

| Setting | F1-Score (macro) | Accuracy | Cohen's κ |
|---|---|---|---|
| **S1** (emerging ↔ existing) | 62.0 ± 2.0 | 68.6 ± 3.7 | 62.4 ± 4.3 |
| **S2** (emerging ↔ emerging) | 25.0 ± 2.8 | 46.3 ± 3.6 | 31.9 ± 3.8 |

Primary metric: **macro F1-score**. Paper additionally reports S0 (warm
start) in Supp Table 3 but we focus on S1/S2 which are the headline
results.

## Data layout (after `unzip`)

```
Original-Dataset/
└── DrugBank/data/
    ├── DB_molecular_feats.pkl       # pandas DataFrame, 1710 × {DrugBank ID, Node ID, SMILES, Morgan_Features, RDKit2D_Features}
    ├── node2id.json                 # 1710 drugs → int ID  (drugs occupy IDs 0..1709)
    ├── entity_drug.json             # 32414 non-drug entities → int ID
    ├── relation2id.json             # 109 relations (86 DDI + 23 HetioNet KG)
    ├── S0/                          # warm-start (single split)
    ├── S1_{1,12,123,1234,12345}/    # cold-start S1, 5 seeds
    └── S2_{1,12,123,1234,12345}/    # cold-start S2, 5 seeds

Each S1_<seed>/ or S2_<seed>/ contains:
    train_ddi.txt valid_ddi.txt test_ddi.txt   # "h t r" per line, r ∈ {0..85}
    train_KG.txt  valid_KG.txt  test_KG.txt    # "h t r" per line, r ∈ {0..108}
```

Important: `valid_KG.txt` and `test_KG.txt` hold **deltas** that are
*added* to the train KG at validation / test time. Paper convention
(load_data.py lines 117-119):

```python
vKG = train_kg + valid_kg_delta              # KG visible at validation
tKG = train_kg + valid_kg_delta + test_kg_delta  # KG visible at test
```

The training KG is rebuilt every epoch by `shuffle_train` — see below.

## `shuffle_train` per-epoch protocol (key inductive design)

EmerGNN's `load_data.py:shuffle_train` resamples the train DDI set every
epoch (paper §Methods "EmerGNN flow-based GNN"). For S1 / S2 with
`ratio=0.8`:

1. `train_ent` = entities appearing in any train_ddi triplet
2. `ddi_in_kg` ⊆ `train_ent` (entities that also appear in the KG)
3. Sample 20% of `ddi_in_kg` to mark as **removed_ent** (treated as
   "emerging" this epoch)
4. `kept_ent` = `train_ent − removed_ent`
5. Partition train_ddi into two pieces:
   - **fact_triplet**: both endpoints in `kept_ent` → added as KG edges
     for this epoch's `KG`
   - **train_data**: training targets the model must predict
     - **S1**: exactly one endpoint in `kept_ent` (one "new" + one "old")
     - **S2**: neither endpoint in `kept_ent` (both "new")
   - Other triplets are silently discarded this epoch
6. `epoch_KG = build_graph(fact_triplet ⊕ train_kg)`
7. Model is trained to predict `train_data` using `epoch_KG`

This per-epoch resampling forces the model to learn from *partial*
DDI knowledge, simulating the inductive cold-start scenario at train
time. Without `shuffle_train`, training collapses to a transductive
setup and S1/S2 numbers degrade significantly.

## Reproduction protocol

For each setting in {S1_1, S1_12, S1_123, S1_1234, S1_12345,
                    S2_1, S2_12, S2_123, S2_1234, S2_12345}:

1. Load vocab + paper splits + Morgan FP
2. Build `vKG` and `tKG` once (static through training)
3. For each epoch (1..100):
   - Run `shuffle_train` to get `epoch_KG` + `train_data`
   - Mini-batch (B=32) over `train_data` in input order, softmax CE
     (paper does NOT extra-shuffle targets after `shuffle_train` builds them;
     we match this behavior)
4. Every `epoch_per_test` (=5) epochs, evaluate on valid_ddi using vKG;
   keep best checkpoint by **macro F1**
5. After training, load best checkpoint, evaluate on test_ddi using tKG
6. Record macro F1 / accuracy / Cohen κ

Aggregate 5 seeds per setting; report `mean ± std`.

## Hyperparameters — paper's S1/S2 HARDCODED overrides, NOT argparse defaults

⚠️ **Critical**: `evaluate.py`'s argparse defaults (`lr=0.03, n_batch=512, n_dim=128, lamb=7e-4`)
are **misleading**. Inside `run_model()` lines 56-62 the code overrides them
for S1/S2 settings. Our `--default` flags below match the overrides.

| Setting | Value | Source |
|---|---|---|
| `n_dim` | **64** | `evaluate.py:run_model` S1/S2 override |
| `length` (L) | 3 | paper |
| `lr` | **0.001** | `evaluate.py:run_model` S1/S2 override |
| `weight_decay` (lamb) | **1e-8** | `evaluate.py:run_model` S1/S2 override |
| `n_epoch` | 100 | `evaluate.py` argparse default |
| `n_batch` (train) | **32** | `evaluate.py:run_model` S1/S2 override |
| `test_batch_size` | 16 | `evaluate.py` argparse default |
| `epoch_per_test` | 5 | `evaluate.py` argparse default |
| `feat` | 'M' (Morgan FP) | `evaluate.py:run_model` S1/S2 override |
| `shuffle_ratio` | 0.8 | `load_data.py` default |
| `eval_rel` | 86 | hardcoded in `load_data.py:67` |
| `optimizer` | Adam | `base_model.py` |
| `scheduler` | `ReduceLROnPlateau(mode='max')` with PyTorch defaults (factor=0.1, patience=10) | `base_model.py` (uses `mode='max'` only) |
| **RNG seed** | **0** | `evaluate.py:run_model(0)` — paper's 5 "seeds" (1, 12, ..., 12345) are *split folder names*, NOT this RNG seed |
| Loss reduction | **sum** | `base_model.py`: `loss = loss.sum()` then `backward()` |

> Paper also has `tune_hyperms.py` running hyperopt over `(lr, lamb, n_batch, n_dim, length)`
> with ~100 evals. The overrides above are the post-tuning values that
> ship in `evaluate.py`. Paper §Methods supplementary notes 360 configs;
> we match the released code rather than re-tuning.

## How to run

```bash
# Smoke (one setting, fewer epochs)
python reproductions/EmerGNN/run_reproduction.py --setting S1_1 --n-epoch 10 --tag s1_1_smoke

# Full single setting / seed
python reproductions/EmerGNN/run_reproduction.py --setting S1_1 --n-epoch 100 --tag s1_1_full

# Full 5 seeds × 2 settings (long)
for setting in S1_1 S1_12 S1_123 S1_1234 S1_12345 S2_1 S2_12 S2_123 S2_1234 S2_12345; do
  python reproductions/EmerGNN/run_reproduction.py --setting $setting --n-epoch 100 --tag ${setting}_full
done
```

Results saved to `Code/runs/<timestamp>__emergnn_reproduce__<tag>__seed42/results.json`.
The `seed42` in the run_id is a meta-seed for the Python RNG; the
paper's S1/S2 seed (1/12/.../12345) is in the **setting** name (e.g.
`S1_1` → paper seed = 1).

## Known caveats

* **Model code path** uses our pure-PyTorch `_compute_messages` instead
  of paper's `torchdrug.layers.functional.generalized_rspmm` CUDA kernel.
  Mathematically equivalent (same `out[t] += rel_w[r] * hid[h]` per edge);
  only differs in speed and memory profile.
* **Loss** uses `F.cross_entropy(logits, y, reduction='sum')` to match
  paper's `loss = loss.sum()` (sum vs mean changes effective lr).
* **Morgan FP** loaded directly from `DB_molecular_feats.pkl` (paper's
  pre-computed 1024-bit fingerprints), NOT recomputed from SMILES.
  Bit-exact via `Node ID == arange(1710)` assert in `data_loader.py`.
* **eval_ent / all_ent**: paper computes these dynamically from the data;
  we use `vocab["n_entity"]` (= 1710 drugs + 32414 entities = 34124).
* **`ddi_in_kg` now matches paper**: union of (train + valid + test KG)
  entities is passed via `extra_kg_ent` kwarg to `shuffle_train` —
  matches paper's `process_files_kg` 92-109 which iterates all 3 splits.
* **`coalesce()` only affects unused path**: `kg_builder.py:build_sparse_adj`
  coalesces COO and `edges_as_dense_lists` returns indices (without
  values), which would silently dedup multi-edges. Our `run_reproduction.py`
  bypasses this path via `build_kg_edges` direct construction, so the
  reproduction itself is not affected. Other baselines that use
  `coldddi.baselines.emergnn.kg_builder` should be audited separately.

## What Codex review caught (and we fixed)

**Round 1 — CRITICAL issues** (all fixed):
1. ❌ argparse defaults used (lr=0.03, n_batch=512, n_dim=128, lamb=7e-4)
   → ✅ replaced with `run_model()` S1/S2 overrides (lr=0.001, n_batch=32, n_dim=64, lamb=1e-8)
2. ❌ `F.cross_entropy` default `reduction='mean'`
   → ✅ changed to `reduction='sum'` matching paper
3. ❌ `ReduceLROnPlateau(factor=0.5, patience=5)`
   → ✅ changed to PyTorch defaults (`factor=0.1, patience=10`) matching paper
4. ❌ default RNG seed=42
   → ✅ changed to 0 matching `run_model(0)` call
5. ❌ Morgan FP loading didn't verify Node ID ordering
   → ✅ added `assert Node ID == arange(1710)` to data_loader

**OOM debugging (separate from fidelity)** — model's `_compute_messages`
materialized a full (E, B, n_dim) tensor (~28 GB at E=3.6M, B=32, D=64)
via `torch.cat(outs)`, exceeding 32GB VRAM. Fixed by rewriting `_propagate`
to chunk + immediate `scatter_add` + gradient checkpoint during training.
Codex round-2 verified the rewrite preserves model semantics (still
computes `out[t] += hiddens[h] * rel[r]` per edge); only difference is
floating-point reduction order from chunking (sub-bit numerical drift).

**Round 2 — MEDIUM issues** (all fixed):
6. ❌ `shuffle_train`'s `ddi_in_kg` only used train_kg (paper uses union of
   train+valid+test KG entities via `process_files_kg`)
   → ✅ added `extra_kg_ent` kwarg; runner passes union of test_kg entities
7. ❌ runner extra-shuffled `train_targets` after `shuffle_train`
   (paper does sequential batching via `utils.batch_by_size`)
   → ✅ removed permutation; per-epoch randomness comes solely from `shuffle_train`
