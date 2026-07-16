# HDN-DDI Baseline multi_cls — seed 42, 100 epochs (K-way DDI type prediction)

> **Why this lives under `baseline/hdn_ddi/_results/`**: this run uses
> **our project's own** 800-drug DrugBank cold-start dataset
> (`Code/data/coldddi_legacy/800drug/seed42.pkl`) + the baseline-side code at
> `Code/baseline/hdn_ddi/multi_cls/`. Companion file to the binary run at
> `2026-05-18__binary_cls__seed42.md` (same data, same encoder, different head).

## 1. Run identity

- **run_id**: `2026-05-18_04-00-14__run_baseline__hdn_ddi_mc_seed42_full__seed42`
- **run_dir**: `Code/runs/2026-05-18_04-00-14__run_baseline__hdn_ddi_mc_seed42_full__seed42/`
- **artifacts**: `results.json`, `train.log`, mirror at `Code/runs/_logs/<run_id>.log`
- **dataset**: our project's 800-drug DrugBank cold-start split (seed42 PKL bundle)
- **mol-pkl**: cache HIT on `Code/baseline/hdn_ddi/_data/necessary/hdn_ddi_mol_graphs__mine.pkl`
  (built fresh during binary run earlier today, reused here)
- **status**: completed; ran full 100 epochs
- **wall time**: fit=1727.5 s + eval ~5 s = **~29 min total** (faster than binary
  because no per-epoch negative sampling pass; pos-only K-way training)
- **fit code version**: post-migration to spec layout + post-codex-multi_cls-fix
  (best_ckpt selection moved from `val_top1` → `val_macro_f1`, train-side
  ddi_type NaN fail-fast added)

## 2. Config (from results.json)

| param | value | source |
|---|---|---|
| baseline | `hdn_ddi_mc` | CLI `--baseline` |
| kg_source | `drugbank` (ignored by hdn_ddi_mc) | CLI `--kg-source` |
| epochs | 100 | CLI `--epochs` |
| seed | 42 | CLI `--seed` |
| batch_size | 512 (default in `binary_cls/baseline.py:113`, inherited) | spec |
| optimizer | Adam(lr=1e-3, wd=5e-4) | spec §4 case A |
| scheduler | LambdaLR(0.96^epoch) | paper §Parameters |
| loss | `F.cross_entropy(logits, labels)` (default reduction='mean') | K-way task |
| best ckpt metric | **`val_macro_f1`** (post codex round-2 fix) | spec §4 case A "paper 主指标" |
| **n_classes (K)** | **152** (observed unique ddi_type in train; project taxonomy is finer than paper's 86) | observed |
| K-way head | `HDN_DDI_MC` einsum `bif,rfg,bjg->brij` → sum(-2,-1) → [B,K] | `multi_cls/model.py:72-75` |
| eval_strategy | epoch | spec |
| n_atom_feats | 66 (Table S1), kge_dim 128, n_blocks 6 | shared w/ binary |

## 3. Data stats

| split | #positives | n_oov_excluded (ddi_type unseen in train) | eval n |
|---|---|---|---|
| train | 53,743 | 0 (fail-fast guard added) | — |
| test_s0 (warm) | 2,986 | 1 | 2,985 |
| test_s1 (1 cold) | 15,258 | 119 | 15,139 |
| test_s2 (2 cold) | 1,919 | 32 | 1,887 |

**OOV note**: cold-start splits contain `ddi_type` values not seen in training
(152 train types vs slightly larger taxonomy in test). These rows are
silently excluded from the K-way eval since the model has no logit for them.

**Negative sampling**: NOT used (K-way trains on positives only with their true
ddi_type as label; each pos has exactly one true class).

## 4. Final metrics (results.json `eval` block, best-val-macro_f1 ckpt)

| split | n | top1_acc | top3_acc | top5_acc | macro_f1 | macro_auc |
|---|---|---|---|---|---|---|
| test_s0 (warm) | 2,985 | **0.8044** | 0.9595 | 0.9806 | 0.6722 | **0.9717** |
| test_s1 (1 cold) | 15,139 | **0.4330** | 0.6804 | 0.7790 | 0.2648 | **0.8616** |
| test_s2 (2 cold) | 1,887 | **0.2109** | 0.4293 | 0.5427 | 0.0619 | **0.6775** |

**Trend**: monotonic degradation s0 > s1 > s2 on every metric (consistent
with cold-start ranking).

## 5. Comparison vs binary baseline (same seed, same data)

| split | binary AUC | MC macro_auc | gap (MC − binary) | interpretation |
|---|---|---|---|---|
| s0 (warm) | 0.6595 | 0.9717 | **+31.2 pt** | warm-pair memorization dominates |
| s1 (1 cold) | 0.6412 | 0.8616 | +22.0 pt | partial memorization (1 unseen drug) |
| s2 (2 cold) | 0.6201 | 0.6775 | **+5.7 pt** | true cold-start: MC advantage shrinks dramatically |

| split | binary AUC s0→s2 drop | MC macro_auc s0→s2 drop |
|---|---|---|
| | **-3.9 pt** | **-29.4 pt** |

→ **The MC task's headline numbers (0.97 / 0.86 / 0.68) are dominated by
warm-start memorization**. In true cold-start (s2), the MC advantage over
binary shrinks to ~6pt, and both are near-equivalent in difficulty.

## 6. Verdict + justification

**PASS-as-baseline** (no implementation bug; numbers reflect honest model
generalization on our cold-start setting).

Justification:
1. **Pipeline correctness**: K-way einsum head produces correct shape `[B, 152]`
   logits; cross_entropy + Adam + LambdaLR matches binary (spec-compliant);
   codex multi_cls deep review PASS (round 1+2)
2. **Trend correctness**: s0 > s1 > s2 monotonic on every metric
3. **No leakage in the strict sense**: cold drugs in s1/s2 genuinely unseen at
   train; OOV ddi_types excluded; train fail-fast guard active
4. **The s0 0.97 macro_auc is NOT a bug**: it reflects the K-way task
   formulation (positives-only, with assumed-pair-exists prior), combined
   with warm-pair memorization. Real generalization signal is s2 macro_auc 0.68.

## 7. Caveats

1. **n_classes=152 vs paper's 86** — our project taxonomy is finer.
   Affects all per-class metrics (top1/macro_f1 harder; macro_auc less so
   since it's avg-of-binary-AUC). To match paper directly, would need to
   collapse to 86 most-common types.
2. **OOV exclusion silently drops 1+119+32 = 152 cold-start pairs** from
   evaluation. Real numbers would be slightly lower if these were counted as
   incorrect rather than skipped.
3. **macro_f1 0.06 on s2** indicates the model essentially fails at
   per-class precision/recall on truly cold drugs. macro_auc 0.68 says it
   can still rank correct class above incorrect on average, but can't
   reliably pick top-1. Indicates near-uniform output distribution on cold.
4. **Single seed (42)** — no std deviation. Particularly relevant for s2 with
   only n=1887 positives; bootstrap CIs would be informative.
5. **Task asymmetry vs binary**: MC excludes negative pairs from eval (each
   eval row is a known positive whose type is being predicted). Binary
   evaluates 1:1 pos+neg. The MC formulation gives the model a "free pass" on
   the existence question — paper's high reported numbers benefit from this.
6. **Cross-baseline reference still missing**: should run `emergnn_mc` /
   `tiger_mc` (and `*_mc` for any K-way capable baseline) on identical splits
   to know if HDN-DDI's MC s2 macro_auc 0.68 is normal or weak.

## 8. Reproduce command

```bash
python Code/scripts/run_baseline.py \
    --baseline hdn_ddi_mc \
    --kg-source drugbank \
    --seed 42 \
    --epochs 100 \
    --tag hdn_ddi_mc_seed42_full
```

Wall time on this machine (WSL2 + CUDA): ~29 min for 100 epochs full run
(faster than binary because no per-epoch neg-sampling overhead).
Pkl cache HIT — no rebuild needed (built fresh by the binary run earlier).
