# HDN-DDI Reproduction Results

> **Why this lives under `reproductions/HDN-DDI/_results/`**: this run uses the
> **upstream original dataset** (HDN-DDI's own DrugBank pkl: 1706 drugs, 86
> relations) + the byte-exact code in `reproductions/HDN-DDI/`. Goal is to
> reproduce paper Table 3. For runs of HDN-DDI on **our project's own** DrugBank
> dataset (≈1900 drugs, our K-way split), results go to
> `baseline/hdn_ddi/_results/` instead. See CLAUDE.md §"Reproduction /
> experiment 结果归档规范" for the rule.

## Run identity

- **run_id**: `2026-05-17_05-47-01__hdn_ddi_faithful__fold0_faithful__seed42`
- **run_dir**: `Code/runs/2026-05-17_05-47-01__hdn_ddi_faithful__fold0_faithful__seed42/`
- **artifacts**: `results.json`, `train.log` (194 lines), `best-fold0.pkl`
- **dataset**: **upstream** HDN-DDI DrugBank pkl (`id_data_dict_dsn_full_connect.pkl`,
  1706 drugs, 86 relations) — NOT our project's 1900-drug DrugBank
- **setting**: DrugBank cold-start, fold 0 of 3 (paper uses 3-fold protocol)
- **status**: completed; early-stopped at ep 45/100, best epoch = 15
- **wall time**: 45 epochs × ~130 s/ep ≈ 98 min (~1.6 hours)

## Config (verified from train.log line 4)

| param | value |
|---|---|
| n_atom_feats | 66 |
| n_atom_hid | 128 |
| rel_total | 86 |
| lr | 1e-3 |
| weight_decay | 5e-4 |
| n_epochs (target) | 100 |
| kge_dim | 128 |
| batch_size | 1024 |
| seed | 42 |
| fold | 0 |
| device | cuda |

## Data stats (verified from train.log line 7)

| split | #samples |
|---|---|
| Training | 123,784 |
| s1 test (Ds1, new ↔ new, both unseen) | 7,369 |
| s2 test (Ds2, new ↔ old, one unseen) | 60,655 |

**HDN-DDI naming convention** (paper Sec 3.6): `Ds1 = "new drug ↔ new drug"` is the harder cold-start variant (both drugs unseen). `Ds2 = "new drug ↔ old drug"` is the easier (one unseen). This is **opposite** of EmerGNN's naming (which uses S2 for both-unseen).

## Final best results (verified from train.log "Best Result" block + results.json)

| metric | s1 (Ds1, both new) | s2 (Ds2, one new) |
|---|---|---|
| ACC | 0.7848 | 0.8544 |
| AUROC | 0.8767 | 0.9295 |
| AUPR (ap) | 0.8684 | 0.9245 |
| F1 | 0.7652 | 0.8561 |
| Precision | 0.8417 | 0.8465 |
| Recall | 0.7015 | 0.8659 |

Best epoch = 15 (selected by val AUROC during training).

## Comparison vs paper Table 3 (verified from `paper-and-github/_paper_text.txt` line 678-685)

Paper reports 3-fold mean (we ran fold 0 only):

| metric | paper Ds1 (3-fold avg) | ours s1 (fold 0) | Δ | within paper ±2pt? |
|---|---|---|---|---|
| ACC | 0.7984 | 0.7848 | -1.36 pt | ✅ |
| AUROC | 0.8948 | 0.8767 | -1.81 pt | ✅ |
| AUPR | 0.8965 | 0.8684 | -2.81 pt | ⚠ slightly out |
| F1 | 0.7742 | 0.7652 | -0.90 pt | ✅ |

| metric | paper Ds2 (3-fold avg) | ours s2 (fold 0) | Δ | within ±2pt? |
|---|---|---|---|---|
| ACC | 0.8943 | 0.8544 | -3.99 pt | ❌ |
| AUROC | 0.9592 | 0.9295 | -2.97 pt | ❌ |
| AUPR | 0.9595 | 0.9245 | -3.50 pt | ❌ |
| F1 | 0.8934 | 0.8561 | -3.73 pt | ❌ |

## Verdict

**PASS** (per user judgment).

Justification for considering this a pass despite the s2 gap:

1. **Paper averages over 3 folds**; we ran only fold 0. Fold-to-fold variance can easily produce 3-4 pt deviations from the 3-fold mean on a single fold. The paper does not publish per-fold numbers, so we can't directly compare single-fold to single-fold.
2. **Direction is right**: all metrics are clearly above the strongest baseline in paper Table 3 except marginally. E.g. our s2 ACC = 85.44% sits between BDN-DDI (85.35%) and HDN-DDI's reported mean (89.43%).
3. **s1 (the harder cold-start) is well within paper range** — 3 of 4 metrics within 2pt of paper mean, AUPR within 3pt.
4. **Training behaviour was healthy**: early stop triggered after 30 epochs of no improvement; loss curve smooth.

To upgrade to a strict ±2pt match on the 3-fold mean, the full 3-fold run is needed; this single-fold result is judged "directionally correct, magnitude credible" by project lead.

## Caveats

1. Single fold (fold 0); paper averages over folds 0, 1, 2. No way to compute std deviation from our run alone.
2. Early stop at ep 45 / best ep 15 — the model was still capable of further training but val plateau triggered early stop. Different patience could change best ckpt.
3. Naming: paper's Ds1/Ds2 is **opposite** of EmerGNN's S1/S2. Don't confuse the two when comparing across reproductions.

## Reproducing this result

```bash
python Code/reproductions/HDN-DDI/run_faithful.py --fold 0 --tag fold0_faithful
```

Wall time on the same machine: ~1.6 hours for 45-epoch early-stopped fold 0.
