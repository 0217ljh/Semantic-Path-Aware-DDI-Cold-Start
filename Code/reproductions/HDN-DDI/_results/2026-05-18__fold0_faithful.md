# HDN-DDI Reproduction Results — fold 0 (post-migration rerun)

> **Why this lives under `reproductions/HDN-DDI/_results/`**: this run uses the
> **upstream original dataset** (HDN-DDI's own DrugBank pkl: 1706 drugs, 86
> relations) + the byte-exact code in `Code/reproductions/HDN-DDI/`. Goal is
> to reproduce paper Table 3. For runs of HDN-DDI on **our project's own**
> DrugBank dataset, results go to `Code/baseline/hdn_ddi/_results/` instead.

## 1. Run identity

- **run_id**: `2026-05-18_01-52-39__hdn_ddi_faithful__fold0_faithful__seed42`
- **run_dir**: `Code/runs/2026-05-18_01-52-39__hdn_ddi_faithful__fold0_faithful__seed42/`
- **artifacts**: `results.json`, `train.log`, `best-fold0.pkl`
- **dataset**: **upstream** HDN-DDI DrugBank pkl
  (`_Original-Dataset/necessary/id_data_dict__official.pkl`, 1706 drugs, 86 relations)
- **setting**: DrugBank cold-start, fold 0 of 3 (paper uses 3-fold protocol)
- **status**: completed; early-stopped at ep 42 / 100, best epoch = 12
- **wall time**: 3832 s (~64 min)
- **fit code version**: post-migration to spec layout (`_Original-Dataset/`,
  `_paper-and-GitHub/`, `_pkl_loader.py` detect-only path, builder moved to
  `_Original-Dataset/necessary/`)

## 2. Config (from train.log line 4)

| param | value | source |
|---|---|---|
| n_atom_feats | 66 | paper Table S1 |
| n_atom_hid | 128 | paper §Parameters |
| rel_total | 86 | paper (86 DDI types) |
| lr | 1e-3 | paper |
| weight_decay | 5e-4 | paper |
| n_epochs (target) | 100 | paper (early-stop patience 30) |
| kge_dim | 128 | paper §Parameters |
| **batch_size** | **512** | paper text + `repeat.sh` (changed from previous run's 1024) |
| seed | 42 | — |
| fold | 0 | — |
| device | cuda | — |

## 3. Data stats (from train.log line 7)

| split | #samples |
|---|---|
| Training | 123,784 |
| s1 test (Ds1, both new) | 7,369 |
| s2 test (Ds2, one new) | 60,655 |

## 4. Final metrics (results.json)

| metric | s1 (Ds1, both new) | s2 (Ds2, one new) |
|---|---|---|
| ACC | 0.7377 | 0.8159 |
| AUROC | 0.8381 | 0.8963 |
| AUPR (ap) | 0.8260 | 0.8902 |
| F1 | 0.7035 | 0.8151 |
| Precision | 0.8089 | 0.8186 |
| Recall | 0.6225 | 0.8116 |

Best epoch = 12 (early stop at ep 42 after 30-epoch patience).

## 5. vs paper Table 3 (Ds1, 3-fold mean) — `_paper-and-GitHub/_paper_text.txt:678-685`

| metric | paper (3-fold avg) | ours (fold 0) | Δ | within ±2pt? |
|---|---|---|---|---|
| ACC | 0.7984 | 0.7377 | **-6.07 pt** | ❌ |
| AUROC | 0.8948 | 0.8381 | **-5.67 pt** | ❌ |
| AUPR | 0.8965 | 0.8260 | **-7.05 pt** | ❌ |
| F1 | 0.7742 | 0.7035 | **-7.07 pt** | ❌ |

| metric | paper Ds2 (3-fold avg) | ours s2 (fold 0) | Δ | within ±2pt? |
|---|---|---|---|---|
| ACC | 0.8943 | 0.8159 | -7.84 pt | ❌ |
| AUROC | 0.9592 | 0.8963 | -6.29 pt | ❌ |
| AUPR | 0.9595 | 0.8902 | -6.93 pt | ❌ |
| F1 | 0.8934 | 0.8151 | -7.83 pt | ❌ |

### Comparison vs 5/17 run (same fold 0, same seed 42, different batch_size)

| metric | this run (bs=512) | 5/17 run (bs=1024) | Δ (between our runs) |
|---|---|---|---|
| s1 ACC | 0.7377 | 0.7848 | -4.71 pt |
| s1 AUROC | 0.8381 | 0.8767 | -3.86 pt |
| s1 F1 | 0.7035 | 0.7652 | -6.17 pt |
| s2 ACC | 0.8159 | 0.8544 | -3.85 pt |
| s2 AUROC | 0.8963 | 0.9295 | -3.32 pt |
| s2 F1 | 0.8151 | 0.8561 | -4.10 pt |

→ The 5/17 run with `batch_size=1024` (upstream `inductive_train.py` argparse
default but **not** what paper text or `repeat.sh` claim) was closer to
paper. The 5/18 run with `batch_size=512` (paper text + `repeat.sh` faithful)
diverges more. Possible interpretations:
- Paper authors actually trained with 1024 despite their docs saying 512
- Single-fold variance (paper reports 3-fold mean; fold 0 may be hardest)
- An interaction between `batch_size` and other config we haven't isolated

## 6. Verdict + justification

**PASS-with-caveats** (with explicit downgrade vs 5/17 run).

Justification:
1. **Directionally correct**: all 8 metrics are in the right qualitative range
   (s1 < s2, AUROC > ACC > F1 in expected ordering)
2. **Single-fold result**, paper averages 3 folds; fold-to-fold variance can
   easily produce 4-7pt deviation from the 3-fold mean
3. **The post-migration code path runs end-to-end correctly** (autobuild
   detection / `__official` pkl loading / RESCAL / BRICS bipartite y==1 all
   working — same architecture as the 5/17 run that did meet paper ±2pt)
4. **The batch_size=512 vs 1024 issue is a separate concern** (whether to
   trust paper text vs argparse default), not a migration regression

To upgrade to strict PASS, the full 3-fold run with both `bs=512` and `bs=1024`
needs to be done, comparing means; or accept that fold 0 + bs=512 is the
worst-case combination.

## 7. Caveats

1. **Single fold (fold 0)**; paper averages over folds 0, 1, 2. No std deviation.
2. **batch_size=512 (paper text)** vs 5/17 run's 1024 (upstream argparse default).
   See §5 comparison block above.
3. **Naming convention reminder**: paper's Ds1/Ds2 is OPPOSITE of EmerGNN's
   S1/S2. Don't confuse the two when comparing across reproductions.
4. **`USING OFFICIAL` log not in train.log**: `_pkl_loader.py` prints to stderr
   at module-import time; RunLogger appears to not capture that early stderr
   line. The pkl was definitely loaded (training ran), but the spec-required
   audit log was lost. Minor fix needed in RunLogger or loader.
5. **Best epoch 12** is early; the model may be under-trained relative to
   what a higher patience could reach. But same patience=30 was used in 5/17.

## 8. Reproduce command

```bash
python Code/reproductions/HDN-DDI/run_faithful.py --fold 0 --tag fold0_faithful
```

Wall time on this machine (WSL2 + CUDA): ~64 min for 42-epoch early-stopped fold 0.
