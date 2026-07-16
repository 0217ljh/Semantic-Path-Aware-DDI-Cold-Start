# EmerGNN Reproduction Review — Round 4 (FINAL PASS)

- **Date**: 2026-05-18 (round 4, supersedes initial FAIL audit at
  `2026-05-18__paper_faithful.md`)
- **Primary reviewer**: Claude (sonnet-4.5)
- **Independent reviewer**: codex (gpt-5-codex), 4 rounds
- **Triggered by**: 用户 "进入循环处理"
- **Outcome**: ✅ codex round-4 verdict = **PASS**

---

## 4-round progress timeline

| Round | Verdict | Change |
|---|---|---|
| 1 | FAIL | initial audit; 5 priority fixes identified (cross-folder import, layout, missing __official, missing builders, README/result overclaims) |
| 2 | FAIL | R1 layout migration + R2 __official refactor done; 3 P1/P2 items still open |
| 3 | PASS_WITH_NITS | 2 builders + README + _results polish; 2 cosmetic nits flagged |
| 4 | ✅ **PASS** | builder rename + review §11 added |

---

## Final state vs CLAUDE.md §"复现代码 (Reproduction) 规范" checklist

| Spec requirement | Status |
|---|---|
| §1 layout: `_results/`, `_reviews/`, `_Original-Dataset/`, `_paper-and-GitHub/` | ✅ all 4 with `_` prefix |
| §1 layout: `_paper-and-GitHub/<full-git-clone>/` | ✅ `LARS-research__EmerGNN/` real git clone (depth=1) |
| §2 `__official` / `__mine` / legacy detect-order | ✅ `data_loader._resolve_necessary` with stderr `USING ...: <path>` log |
| §2 shipped artefacts under `necessary/__official.<ext>` | ✅ 8 files (4 DrugBank + 4 TWOSIDES) migrated |
| §3 step 5: builder per artefact | ✅ `build_DB_molecular_feats.py` (full reverse-engineer) + `build_vocab.py` (escape-valve documented per §3 step 5 line 374) |
| §3 step 6: no auto-build on reproduction side | ✅ `_resolve_necessary` errors out if neither official nor mine present |
| §5 line 435: code independent of `<full-git-clone>/` | ✅ verified no imports from clone path |
| §"Baseline 规范" line 447-448 "严禁 import 互调" | ✅ `grep "from baseline"` returns only one comment-line match (no real imports) |
| Hyperparam fidelity (lr/n_dim/L/wd/scheduler/seed) | ✅ all match paper |
| `_results/` overclaim wording | ✅ fixed (`partial-validation signal` instead of `validated`) |
| README overclaim wording | ✅ fixed (`structural correspondence confirmed` instead of `verified line-by-line`) |

---

## Files created / modified (4 rounds total)

### New files
| Path | Purpose |
|---|---|
| `model.py` | local copy of `EmerGNN` base class (was: `baseline/emergnn/model.py`) |
| `model_mc.py` | local copy of `EmerGNN_MC` subclass (was: `baseline/emergnn/multi_cls/model.py`) |
| `kg_builder.py` | local copy (was: `baseline/emergnn/kg_builder.py`) |
| `morgan_features.py` | local copy (was: `baseline/emergnn/morgan_features.py`) |
| `shuffle_utils.py` | local copy (was: `baseline/emergnn/shuffle_utils.py`) |
| `_paper-and-GitHub/LARS-research__EmerGNN/` | full git clone of upstream repo (depth 1) |
| `_Original-Dataset/{DrugBank,TWOSIDES}/data/necessary/` | per-dataset `necessary/` subdirs |
| `_Original-Dataset/necessary/build_DB_molecular_feats.py` | reverse-engineered Morgan-FP builder + `--verify-against` |
| `_Original-Dataset/necessary/build_vocab.py` | documented-escape-valve vocab JSON builder |
| `_reviews/2026-05-18__paper_faithful.md` | initial FAIL audit |
| `_reviews/2026-05-18__paper_faithful_round4_PASS.md` | THIS file |

### Modified
| Path | Change |
|---|---|
| `run_reproduction.py` | replaced `from baseline.emergnn.multi_cls.model import EmerGNN_MC` with local `from model_mc import EmerGNN_MC`; removed unused project-root sys.path needs |
| `data_loader.py` | added `_resolve_necessary` + `NECESSARY_DIR`; switched `load_vocab` + `load_morgan_features` to use it; updated `REPO_DATA_DIR` to point to `_Original-Dataset` |
| `model_mc.py` | replaced `from baseline.emergnn.model import EmerGNN` with `from model import EmerGNN`; docstring noting independent-copy semantics |
| `_bench_speed.py`, `_diagnose_mem.py`, `_diagnose_mem2.py` | replaced `from baseline.emergnn.*` imports with local paths |
| `README.md` | removed "verified line-by-line" overclaim; documented new layout |
| `_results/2026-05-17__S2_1_seed0_partial.md` | "**validated**" → "**partial-validation signal**" + caveat block |

### Renamed (layout migration)
- `Original-Dataset/` → `_Original-Dataset/`
- `paper-and-github/` → `_paper-and-GitHub/`
- (`paper-and-github/repo_snapshot/` had no equivalent here — EmerGNN never had a snapshot, just PDF + URL)

### Files NOT modified (intentionally preserved per CLAUDE.md "代码修改规则")
- README.md content beyond the overclaim line
- `_results/2026-05-17__S2_1_seed0_partial.md` data tables

---

## Paper-contribution preservation

After round-1 migration, all 4 paper contributions now live IN the
reproduction folder (not imported from baseline):

| # | Contribution | File:line | Status |
|---|---|---|---|
| C1 | Cold-start emerging-DDI protocol (S1/S2 splits) | `data_loader.load_setting` | ✅ |
| C2 | Path-based pairwise drug repr via L-hop flow GNN | `model.py` (EmerGNN class) | ✅ local copy |
| C3 | Bidirectional edges + self-loops | `shuffle_utils.build_edge_lists_from_triplets` | ✅ |
| C4 | `shuffle_train` per-epoch inductive simulation | `data_loader.shuffle_train` | ✅ |
| C5 | Attention-weighted RESCAL-like aggregation | `model.py` + `model_mc.py` | ✅ local copy |

---

## Codex round-4 verdict (verbatim)

> 核对结果：
> - `_Original-Dataset/necessary/` 现在只有 `build_DB_molecular_feats.py`
>   和 `build_vocab.py`，未见 `build_morgan_features.py`。
> - 全局搜索未发现旧文件名 `build_morgan_features.py` 残留。
> - `build_DB_molecular_feats.py` 顶部 CLI 示例已使用 `python build_DB_molecular_feats.py`。
> - `_reviews/2026-05-18__paper_faithful.md` 已新增 §11，并明确记录三项缺失
>   upstream pins：DrugBank XML version、HetioNet release、relation-text ordering。
> - `build_vocab.py` 现在引用的 "Builders for vocab JSONs" section 已实际存在。
> - 未发现本轮修复引入的新回归。
>
> FINAL VERDICT: PASS

---

## Reproduction command (post-PASS canonical usage)

```bash
# Single setting (paper-faithful S2 cold-start, both drugs new)
python Code/reproductions/EmerGNN/run_reproduction.py \
    --setting S2_1 --n-epoch 100 --tag s2_1_full

# Switch to S1 (one drug new):
python ... --setting S1_1 --tag s1_1_full

# Reuse upstream Morgan-FP cache (auto-detected at necessary/__official):
# nothing extra — _resolve_necessary picks it up

# Build local Morgan-FP cache from scratch (for sanity-check):
python Code/reproductions/EmerGNN/_Original-Dataset/necessary/build_DB_molecular_feats.py \
    --smiles-file <smiles CSV> \
    --out Code/reproductions/EmerGNN/_Original-Dataset/DrugBank/data/necessary/DB_molecular_feats__mine.pkl \
    --verify-against Code/reproductions/EmerGNN/_Original-Dataset/DrugBank/data/necessary/DB_molecular_feats__official.pkl
```

---

## Future work (non-blocking)

1. **Full 5-seed × 100-epoch reproduction**: currently only 1 partial
   S2_1 run (val-only, ep 38/100, val_f1=0.2487 ≈ paper 0.250±0.028).
   For a complete Table-1 PASS evidence chain, run all 5 settings ×
   full 100 epochs × test eval. Estimated 5 days on RTX 5090.
2. **`build_DB_molecular_feats.py` bit-identity spot-check**: run
   `--verify-against` on 10 random drugs from `DB_molecular_feats__official.pkl`
   to confirm Morgan-FP bit-identical (or document RDKit version drift).
3. **Baseline-side review re-check**: now that reproduction has
   independent model copies, the existing
   `Code/baseline/emergnn/_reviews/2026-05-17__baseline.md` claims that
   baseline "inherits from reproduction" become stale — both sides now
   keep INDEPENDENT copies of the model. Re-state the baseline review's
   architectural assumptions.
4. **vocab JSON full reverse-engineering**: deferred per §11 escape valve.
   Would require pinning DrugBank XML version + HetioNet release +
   relation-text ordering.

---

## ✅ FINAL VERDICT: PASS

After 4 rounds (1 layout migration + 1 __official refactor + 1 builders
+ 1 polish), `Code/reproductions/EmerGNN/` fully conforms to CLAUDE.md
§"复现代码 (Reproduction) 规范" (lines 312-441). All 5 paper contributions
now live within the reproduction folder. Cross-folder independence from
`baseline/emergnn/` restored. `__official` artefact convention applied
to all 8 shipped files. 2 reverse-engineered builders shipped (+
documented escape valve for vocab JSONs per spec §3 step 5).

Codex round-4 explicitly confirms PASS with no further nits or regressions.
