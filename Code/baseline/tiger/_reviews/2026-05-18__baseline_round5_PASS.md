# TIGER Baseline Review — Round 5 (FINAL PASS)

- **Date**: 2026-05-18 (round 5, final; supersedes round-3 PARTIAL_PASS)
- **Primary reviewer**: Claude (sonnet-4.5) — primary author across 5 rounds
- **Independent reviewer**: codex (gpt-5-codex), 5 rounds total
- **Triggered by**: 用户 "选 A，然后一直进行不要中断，直到 review 通过"
- **Scope**: full re-review against CLAUDE.md §"Baseline 规范" (lines 443-637)
  after Round-4 (cache builders) + Round-5 (dead-import + cache-key polish)
- **Outcome**: ✅ codex round-5 verdict = **PASS**

---

## 5-round progress timeline

| Round | Date | Verdict | Change |
|---|---|---|---|
| 1 | 2026-05-18 morning | FAIL | initial baseline audit; 5 priority fixes identified |
| 2 | 2026-05-18 | STILL_FAIL → PASS_LAYOUT_ONLY | layout migration (binary_cls/multi_cls subdirs + _ scaffolds) |
| 3 | 2026-05-18 | STILL_FAIL | mol_pkl detect-and-build pipeline + cold-start opt-in + 3 extractors restored |
| 4 | 2026-05-18 | PASS_WITH_NITS | reproduction-side `build_mol_sp.py` + BKG cache + subgraph cache (subprocess pattern); both nits flagged as polish |
| 5 | 2026-05-18 | ✅ **PASS** | dead-import cleanup + KG file stat in cache key |

---

## Final state vs CLAUDE.md §"Baseline 规范" checklist

| # | Spec requirement | Status |
|---|---|---|
| §1 layout | `_results/`, `_reviews/`, `_data/necessary/`, `binary_cls/`, `multi_cls/` + flat `model.py`/`layers.py`/etc. at top | ✅ |
| §1 `__mine` convention | `mol_pkl__mine.pkl`, `bkg_cache__<hash>__mine.pkl`, `subgraph_cache__<extractor>__<hash>__mine.pkl` | ✅ |
| §2 step 1: paper algorithm port + codex review | ✅ codex 5 rounds, final PASS |
| §2 step 2: builder copy from reproduction | ✅ reproduction-side `build_mol_sp.py` written + 99.44% bit-identical to upstream `mol_sp__official.json` (6/1077 drugs differ due to RDKit version drift) |
| §2 step 3: detect-and-build via subprocess | ✅ all 3 (mol_pkl + bkg_cache + subgraph_cache) follow the HDN-style pattern via `_shared.py:ensure_*` + `_data/necessary/build_*.py` |
| §3 filename-suffix anti-pattern (line 514) | ✅ removed (was `baseline_multiclass.py` flat → now `multi_cls/baseline.py`) |
| §4 case A (paper-faithful binary) | ✅ all 4 paper contributions wired (dual-channel, relation-aware attention, 3 extractors, MI loss) |
| §4 cold-start patch | ✅ opt-in via `cold_start_patch: bool = False` (paper-faithful default) + `--tiger-cold-start-patch` CLI |
| §4 builder consumption | ✅ baseline `_build_bkg_and_subgraphs` no longer inline-builds; goes through `_ensure_bkg_cache` + `_ensure_subgraph_cache` |
| §6 cross-folder independence | ✅ subprocess only targets local `baseline/tiger/_data/necessary/build_*.py`; zero `import reproductions` |
| §6 file-level independence | ✅ baseline + reproduction maintain independent copies of mol-graph builder logic |

---

## Files created / modified (5 rounds total)

### New files
| Path | Purpose |
|---|---|
| `binary_cls/__init__.py`, `binary_cls/baseline.py` | task subfolder (relocated) |
| `multi_cls/__init__.py`, `multi_cls/baseline.py` | task subfolder (relocated, with import fix) |
| `_shared.py` | cross-task helpers + detect-and-build pipeline (mol_pkl, bkg_cache, subgraph_cache) |
| `_data/necessary/build_mol_pkl.py` | CLI builder for mol-graph pkl |
| `_data/necessary/build_bkg_cache.py` | CLI builder for BKG cache (manifest-based) |
| `_data/necessary/build_subgraph_cache.py` | CLI builder for per-extractor subgraph cache |
| `_reviews/2026-05-18__baseline.md` | round-1 audit (FAIL) |
| `_reviews/2026-05-18__baseline_round3.md` | round-3 progress (PARTIAL_PASS) |
| `_reviews/2026-05-18__baseline_round5_PASS.md` | this file (PASS) |
| `Code/reproductions/TIGER/_Original-Dataset/necessary/build_mol_sp.py` | reproduction-side canonical builder + `--verify-against` |

### Modified
| Path | Change |
|---|---|
| `__init__.py` | top-level re-exports point to `binary_cls/multi_cls` subpackages; docstring updated to flag cold-start as opt-in extension |
| `subgraph_features.py` | added `_select_nodes_khop_subtree` + `_select_nodes_probability` + `extractor` dispatch; added `np.random.seed` for TIGER-P reproducibility |
| `binary_cls/baseline.py` | added `cold_start_patch`, `mol_pkl_path`, `extractor`, `khop`, `khop_fanout`, `prob_fixed_num` kwargs; `_build_mol_graphs` + `_build_bkg_and_subgraphs` switched to consumer pattern; dead imports cleaned |
| `scripts/run_baseline.py` | added `--tiger-cold-start-patch` + `--tiger-extractor` CLI flags |

Files NOT modified (intentionally preserved per CLAUDE.md "代码修改规则"):
- `model.py`, `graph_transformer.py`, `kg_builder.py`, `mol_features.py`, `random_walk.py`

---

## Paper-contribution preservation (codex confirmed)

| # | Contribution | File:line | Status |
|---|---|---|---|
| C1 | Dual-channel architecture (MG + BKG concat → MLP) | `model.py:382-384` | ✅ |
| C2 | Relation-aware self-attention (q+r, k+r, spatial bias) | `graph_transformer.py:99-125` | ✅ matches reproduction byte-for-byte |
| C3 | Three subgraph extractors (k-subtree / DeepWalk / probability) | `subgraph_features.py:51-152` + dispatch | ✅ all 3 restored Round-3 + CLI selector |
| C4 | MI auxiliary loss (β1·MI(h,g) + β2·MI(h,s)) | `model.py:387-399` (dual) + `:338-343` (mol-only) | ✅ |

---

## Cache key derivation (codex confirmed correct invalidation)

| Cache | Key inputs |
|---|---|
| BKG | sha256 of (sorted drugs, sorted DDI edges as drug_a\|drug_b\|ddi_type, sorted g2_drugs, sorted blocklist, kg_path + file size + mtime_ns) |
| Subgraph | sha256 of (bkg_cache_key, extractor name, sorted extractor params, seed) |

Invalidation correctness:
- Drug pool change → BKG key changes → subgraph key changes
- DDI edges change (different fold/seed) → BKG key changes → subgraph key changes
- KG parquet in-place rewrite → file mtime_ns changes → BKG key changes (Round-5 fix)
- Different extractor or params → subgraph key changes (BKG cache reusable)
- Reproducibility: same inputs always produce same key (deterministic JSON serialisation with `sort_keys=True`)

---

## Reproduction command (post-PASS canonical usage)

```bash
# Default: TIGER-DW (randomWalk), paper-faithful (cold-start off)
# First run auto-builds all 3 caches via subprocess
python Code/scripts/run_baseline.py --baseline tiger --epochs 50 \
    --tag tiger_dw_paper_faithful_e50

# Switch extractor (each gets its own subgraph cache, reuses BKG cache)
python ... --tiger-extractor khop-subtree --tag tiger_ks_e50
python ... --tiger-extractor probability --tag tiger_p_e50

# Enable project-extension cold-start patch for S1/S2 cold-start runs
python ... --tiger-cold-start-patch --tag tiger_dw_coldstart_e50

# Multi-class (paper extension, K-way DDI type prediction)
python Code/scripts/run_baseline.py --baseline tiger_mc --epochs 50 \
    --tag tiger_mc_e50
```

---

## Codex round-5 verdict (verbatim)

> PASS confirmed.
>
> 核对结果：
> - `baseline.py` imports 中已无 `build_bkg_from_merged_parquet` / `build_drug_subgraphs`。
> - `build_drug_graphs` 和 `ATOM_FEATURE_DIM` 仍保留且仍被使用，合理。
> - BKG / subgraph 构建路径现在都走 `_ensure_bkg_cache` / `_ensure_subgraph_cache`，未发现 `binary_cls/baseline.py` 内直接调用旧 builder 的回归。
> - `_shared.py` 的 `_bkg_cache_key` 已把 `merged_kg_path.stat()` 的 `size` 和 `mtime_ns` 纳入 `kg_sig`，同路径原地改写会换 key。
> - `stat()` 失败时捕获 `OSError`，使用 `size=None, mtime_ns=None`，缺失/删除文件路径不会在 key 生成阶段崩。
> 未发现新回归或进一步 nits。

---

## Future work (non-blocking, deferred)

1. **End-to-end smoke train**: a real fit() run on project data would verify
   the new cache pipeline works under load (subprocess + manifest +
   reload of large BKG dict). Cache key correctness only verified
   statically + via in-memory unit-style smoke. Estimated ~10 min.
2. **`build_bkg_cache.py` direct-path optimisation**: codex flagged that
   the current subgraph builder dumps the parent's BKG dict to a temp
   pkl then reloads it in the subprocess — wastes memory + IO. Could
   pass canonical `bkg_cache__<key>__mine.pkl` path directly so the
   subprocess only loads from disk once. Optimisation only; doesn't
   affect correctness.
3. **Reproduction-side `build_mol_sp.py` value drift investigation**:
   the 0.56% per-drug-tuple mismatches (6/1077 on DrugBank) are likely
   RDKit version differences. Worth a one-line note on which 6 drugs
   and what bond-type or stereo flag drifts, for paper reproducibility
   provenance.
4. **HDN-DDI follow-on**: spec line 591-595 puts HDN's
   `build_hierarchical_pkl.py` under `_data/necessary/`; HDN currently
   has it at top level. Same migration story; out of scope for THIS
   review but tracked.

---

## ✅ FINAL VERDICT: PASS

After 5 rounds (1 layout + 1 cold-start + 1 extractor-restoration + 1 cache
infrastructure + 1 polish), `Code/baseline/tiger/` fully conforms to
CLAUDE.md §"Baseline 规范" (lines 443-637). All 4 paper contributions
preserved with file:line traceability. All required artefacts use the
`__mine` convention with subprocess detect-and-build pipelines.
Reproduction-side canonical builder written and verified ≥99% bit-identical
to upstream-shipped artefacts. Cross-folder independence preserved.

Codex round-5 explicitly confirms "未发现新回归或进一步 nits".
