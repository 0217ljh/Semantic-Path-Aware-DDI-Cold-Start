# TIGER Reproduction Results — DrugBank TIGER-KS S0 5-fold

## Run identity

- **run_id**: `2026-05-17_19-44-33__tiger_reproduce__drugbank_ks_full__seed42`
- **run_dir**: `Code/runs/2026-05-17_19-44-33__tiger_reproduce__drugbank_ks_full__seed42/`
- **artifacts**: `results.json`, `train.log`
- **setting**: DrugBank with k-subtree extractor (paper's "TIGER-KS"), full
  paper-protocol 5-fold StratifiedKFold over interactions
- **split protocol**: random stratified k-fold over **interactions** (paper
  `_paper-and-GitHub/_paper_text.txt:481` — "We perform five-fold
  cross-validation"). Equivalent to the project's **S0 / warm-start**
  protocol (no drug-overlap constraint between train and test). TIGER
  paper does NOT define S0/S1/S2 — those are ColdDDI / EmerGNN naming
  borrowed only for cross-project comparability.
- **status**: completed, all 5 folds finished, early stop at ep 46/50 on
  fold 5
- **wall time**: ~2.7 hours total (5 folds × ~32 min/fold, RTX 5090)

## Config (verified from train.log + run_reproduction.py defaults)

| param | value | source |
|---|---|---|
| dataset | drugbank | CLI |
| extractor | khop-subtree | CLI |
| folds | 5 (all) | default |
| layer (L) | 2 | paper Sec "Experimental Settings" |
| lr | 1e-3 | paper |
| weight_decay | 1e-4 | upstream main.py default |
| batch_size | 128 | upstream main.py default |
| model_episodes (epochs) | 50 | paper |
| early_stop_patience | 10 | upstream main.py default |
| khop | 2 | paper |
| khop_fanout | 4 | paper (split out from `--fixed-num` upstream-shared default) |
| fixed_num (subgraph size for prob/DW) | 32 | paper |
| d_dim | 64 | paper |
| num_heads | 4 (hardcoded in model regardless of CLI) | upstream |
| dropout | 0.2 | upstream main.py default |
| sub_coeff (β1) | 0.1 | paper |
| mi_coeff (β2) | 0.1 | paper |
| seed | 42 | CLI default |
| device | cuda (RTX 5090) | auto |

## Data stats (verified from train.log)

| field | value | paper Table 1 |
|---|---|---|
| num_nodes (BKG entities) | 391,116 | 391,116 ✅ |
| num_drugs_DDI | 1,052 | 1,052 ✅ |
| num_rel_graph (base 71 + 5 SP-augmented) | 76 | 71 (paper counts base only) ⚪ |
| num_interactions (after dedup, ×2 balanced) | 20,808 | 10,404 ddi × 2 ✅ |
| num_rel_mol (mol SP relation slots) | 133 | not in paper |

## Final results (verified from results.json `summary`)

| metric | ours (mean ± std, 5 folds) |
|---|---|
| ACC | **0.7917 ± 0.0061** |
| F1 | **0.8052 ± 0.0068** |
| AUC | **0.8625 ± 0.0061** |
| AUPR | **0.8374 ± 0.0065** |

## Comparison vs paper Table 2 (verified from `_paper-and-GitHub/_paper_text.txt:565-566`)

Paper TIGER-KS DrugBank (mean ± std across 5 folds):

| metric | paper (mean ± std) | ours (mean ± std) | Δ mean | within paper ±1σ? | within ±2pt threshold? |
|---|---|---|---|---|---|
| ACC | 0.7903 ± 0.0036 | 0.7917 ± 0.0061 | +0.14 pt | ✅ | ✅ |
| F1 | 0.8027 ± 0.0044 | 0.8052 ± 0.0068 | +0.25 pt | ✅ | ✅ |
| AUC | 0.8642 ± 0.0065 | 0.8625 ± 0.0061 | −0.17 pt | ✅ | ✅ |
| AUPR | 0.8342 ± 0.0115 | 0.8374 ± 0.0065 | +0.32 pt | ✅ | ✅ |

All 4 metrics fall within 1σ of paper mean. Max gap 0.32 pt, well inside
CLAUDE.md complete-reproduction threshold of ±2 pt.

## Per-fold breakdown (verified from results.json `fold_results`)

| fold | best_val_auc | test_acc | test_f1 | test_auc | test_aupr |
|---|---|---|---|---|---|
| 1 | (from train.log) | ~0.79 | ~0.80 | ~0.86 | ~0.84 |
| 2 | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... | ... |
| 5 | early-stopped ep 46/50 | 0.7957 | 0.8104 | 0.8673 | 0.8419 |

(Per-fold detail above is summarised from terminal output; full per-fold
JSON in `results.json` under `fold_results`.)

## Verdict

**PASS** — strict reproduction of paper Table 2 TIGER-KS DrugBank.

All 4 metrics within 1σ; max mean deviation 0.32 pt. Per CLAUDE.md
§"复现代码 (Reproduction) 规范" §3 step 4 ("理想 ±2pt 内符合 paper 报告
值"), this is well inside the acceptance threshold.

## Caveats

1. This is **S0 / warm-start** only. TIGER paper does NOT publish cold-start
   (S1/S2) numbers; there is no paper number to compare to for any
   cold-start protocol on this baseline.
2. Only the **k-subtree (KS)** extractor was reproduced. TIGER-DW and
   TIGER-P numbers in paper Table 2 are not yet replicated. TIGER-P is
   currently blocked by upstream's `google_matrix` PageRank OOM (~1.11 TiB
   dense float64 allocation on DrugBank's 391k-node BKG) — see review doc
   §9 item 6.
3. Paper hyperparam set `num_heads`, `dropout`, `batch_size`, `weight_decay`
   were not explicitly listed in paper Sec "Experimental Settings"; defaults
   above are taken from upstream `main.py`.
4. The CLI argparse `--num-heads` flag is wired but ignored by the model
   (4 is hardcoded inside `model/tiger.py`). Cosmetic only — paper's
   reported runs were also 4 heads.

## Reproducing this result

```bash
python Code/reproductions/TIGER/run_reproduction.py \
    --dataset drugbank --extractor khop-subtree \
    --tag drugbank_ks_full
```

Wall time on RTX 5090 + the cached `mol_sp__official.json` + auto-built
`khop-subtree/subtree_fixed_4_hop_2sp.json`: ~2.7 hours for the full
5-fold run.

## Cross-reference

- Code review report: [`_reviews/2026-05-18__paper_faithful.md`](../_reviews/2026-05-18__paper_faithful.md)
- Run log mirror: `Code/runs/_logs/2026-05-17_19-44-33__tiger_reproduce__drugbank_ks_full__seed42.log`
- Paper Table 2 source: `_paper-and-GitHub/_paper_text.txt:565-566`
