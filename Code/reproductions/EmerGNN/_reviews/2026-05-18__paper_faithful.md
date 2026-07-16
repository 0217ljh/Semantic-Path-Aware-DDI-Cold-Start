# EmerGNN Reproduction Review — paper-faithful (initial audit)

- **Date**: 2026-05-18
- **Primary reviewer**: Claude (sonnet-4.5) — drafted this report,
  inventory audit, contribution mapping, hyperparam cross-check
- **Independent reviewer**: codex (gpt-5-codex) — independent verdict
  verbatim in §6
- **Triggered by**: 用户 "现在请根据 CLAUDE.md 里面的约定，来 review
  reproduction 下的 EMERGNN" (2026-05-18)
- **Scope**: `Code/reproductions/EmerGNN/` — all .py + Original-Dataset
  layout + paper-and-github snapshot
- **CLAUDE.md regulations applied**:
  - §"复现代码 (Reproduction) 规范" §1 layout (lines 318-345)
  - §"复现代码 (Reproduction) 规范" §2 `__official` / `__mine` (lines 350-363)
  - §"复现代码 (Reproduction) 规范" §3 workflow (lines 365-376)
  - §"复现代码 (Reproduction) 规范" §5 严禁 (lines 430-441)
  - §"Baseline 规范" line 447-448 "严禁 import 互调"
  - §"Code review 报告归档规范" 8 required fields (lines 588-601)

---

## 1. Read inventory

### Reproduction files (100% read)

| File | Status |
|---|---|
| `run_reproduction.py` | ✓ — CLI runner for one S0/S1_*/S2_* setting |
| `data_loader.py` | ✓ — paper-faithful data parser; defines `REPO_DATA_DIR`, `load_vocab`, `load_setting`, `build_global_morgan_matrix`, `shuffle_train` |
| `README.md` | ✓ |
| `_bench_speed.py` | ✓ — diagnostic |
| `_diagnose_mem.py` | ✓ — diagnostic |
| `_diagnose_mem2.py` | ✓ — diagnostic |

### Paper / github snapshot

| Asset | Status |
|---|---|
| `paper-and-github/*.pdf` (Zhang et al. 2023 Nat Comp Sci) | ✓ present |
| `paper-and-github/_paper_text.txt` | ✓ present, full extract |
| `paper-and-github/github.txt` (URL) | ✓ `https://github.com/LARS-research/ EmerGNN` (note stray space; benign) |
| `paper-and-github/<full-git-clone>/` | ❌ **MISSING** — spec line 337-338 requires complete git clone snapshot |

### Original-Dataset

| Asset | Status |
|---|---|
| `Original-Dataset/DrugBank/data/{S0,S1_1..S1_12345,S2_1..S2_12345}/*.txt` | ✓ data shipped |
| `Original-Dataset/DrugBank/data/{DB_molecular_feats.pkl, entity_drug.json, node2id.json, relation2id.json}` | ⚠ shipped but **NOT under `necessary/__official.<ext>`** convention |
| `Original-Dataset/TWOSIDES/data/{...}` | same situation |

### Artefacts already archived

| Path | Notes |
|---|---|
| `_results/2026-05-17__S2_1_seed0_partial.md` | ✓ partial run result (val-only, single setting, interrupted at ep 38/100) |
| `_reviews/<...>.md` | ❌ **MISSING** — this file is the first |

---

## 2. Paper core contributions (verbatim extraction from `_paper_text.txt`)

Paper Zhang et al. 2023 Nat Comp Sci — contributions extracted from
abstract + intro (lines 10-100):

| # | Contribution | Paper anchor |
|---|---|---|
| C1 | Cold-start prediction for **emerging drugs** (drugs with no DDI in train) | abstract + line 53-69 |
| C2 | **Path-based pairwise drug representation** via L-hop flow GNN (not single-node embeddings) | line 73-84, 99-134 |
| C3 | **Biomedical-network bridge** (HetioNet integration) so emerging drugs share KG neighbours with existing drugs | line 53-69, 104-117 |
| C4 | **Inductive `shuffle_train`** simulation — per-epoch random 80/20 fact/target split of train DDI | paper Methods §"Per-epoch shuffling" |
| C5 | **Attention-weighted edge/path interpretability** (RESCAL-like aggregation + per-relation attention) | line 81-83, 205-247 |

---

## 3. Per-contribution verification in current code

| # | Anchor in reproduction code | Status |
|---|---|---|
| C1 | implicit — protocol is S1/S2 split semantics in `data_loader.load_setting` + `shuffle_train` | ⚪ no dedicated file:line |
| C2 | model class (`EmerGNN_MC` / `EmerGNN`) — **NOT in reproduction**; imported from `baseline.emergnn.multi_cls.model` at `run_reproduction.py:39` | ❌ contribution-core not in reproduction folder |
| C3 | data loading reads DrugBank's `relation2id.json` (109 relations = 86 DDI + 23 HetioNet) at `data_loader.load_vocab` | ✅ data path verified |
| C4 | `data_loader.shuffle_train` (~line 209+) — own implementation, NOT imported from baseline | ✅ contribution-core verified in reproduction |
| C5 | attention + RESCAL aggregation — **inside `EmerGNN_MC` model code, imported from baseline** | ❌ same as C2 |

**Verdict on contributions**: 2 of 5 (C2, C5) point to baseline-imported code,
violating reproduction independence. Reproduction folder cannot verify
the paper's algorithm-core on its own.

---

## 4. Layout audit vs CLAUDE.md §"复现代码规范" §1

| Spec requirement (lines 318-345) | Actual | Status |
|---|---|---|
| `_Original-Dataset/` | `Original-Dataset/` (no `_`) | ❌ |
| `_paper-and-GitHub/` | `paper-and-github/` (no `_`, wrong case) | ❌ |
| `_reviews/` exists | did not exist; this file creates it | ❌ → being fixed |
| `_results/` exists | ✅ exists |  |
| `_Original-Dataset/necessary/<name>__official.<ext>` | shipped artefacts (DB_molecular_feats.pkl / entity_drug.json / node2id.json / relation2id.json) sit directly in `Original-Dataset/DrugBank/data/`, no `__official` suffix | ❌ |
| `_Original-Dataset/necessary/build_<name>.py` | no builder scripts present | ❌ |
| `_paper-and-GitHub/<full-git-clone>/` | only PDF + `github.txt` + `_paper_text.txt`; **no git clone snapshot** | ❌ |
| Reproduction code flat at root | ✅ `run_reproduction.py` + `data_loader.py` flat | ✅ |
| Code is independent of `_paper-and-GitHub/` (spec line 435) | ✅ no `import paper-and-github` | ✅ |
| **Code is independent of `baseline/`** (spec line 447-448 "严禁 import 互调") | ❌ `run_reproduction.py:39` imports `from baseline.emergnn.multi_cls.model import EmerGNN_MC` + 3 diagnostic scripts also import from `baseline.emergnn.*` | ❌ critical |

---

## 5. Hyperparameter cross-check (paper vs argparse)

Paper Methods §"Implementation details" / S1+S2 overrides
(`_paper_text.txt` Sec hyperparams):

| Hyperparam | Paper value | `run_reproduction.py` default | Match |
|---|---|---|---|
| Optimizer | Adam | Adam | ✅ |
| n_epoch | 100 | 100 | ✅ |
| n_batch | 32 | 32 | ✅ |
| n_dim | 64 | 64 | ✅ |
| length (L) | 3 | 3 | ✅ |
| lr | 1e-3 | 1e-3 | ✅ |
| weight_decay | 1e-8 | 1e-8 | ✅ |
| shuffle_ratio | 0.8 | 0.8 | ✅ |
| epoch_per_test | 5 | 5 | ✅ |
| Scheduler | `ReduceLROnPlateau(mode='max', patience=10, factor=0.1)` | same | ✅ |
| Seed | 0 (paper hardcodes in `evaluate.py:run_model(0)`) | 0 | ✅ |

All hyperparams match paper. ✅

---

## 6. Codex independent verdict (verbatim)

> **FINAL VERDICT: FAIL**
>
> 主要阻断项不是算法细节，而是 reproduction 规范违规：`Code/reproductions/EmerGNN/`
> 不是独立复现代码，直接 import 了 baseline。
>
> **Findings**
>
> 1. **cross-folder-dep / Critical**
>    `run_reproduction.py:39` 直接：`from baseline.emergnn.multi_cls.model import EmerGNN_MC`
>    这违反 CLAUDE.md: `from baseline.<method> import ...` 在 `reproductions/` 下禁止。
>    `_bench_speed.py`、`_diagnose_mem*.py` 也 import baseline，但主 runner 已经足够构成 FAIL。
>
> 2. **layout / High**
>    规范要求复现目录含 `_results/`, `_reviews/`, `_Original-Dataset/`, `_paper-and-GitHub/`。
>    当前目录是：`Original-Dataset/`, `paper-and-github/`, `_results/`，没有 `_reviews/`，
>    且数据/论文目录没有 `_` 前缀。`paper-and-github/` 只有 PDF、`github.txt`、`_paper_text.txt`，
>    没有 `<full-git-clone>/` 子目录。
>
> 3. **missing-builder / High**
>    规范要求必要文件命名为 `__official` / `__mine`，并优先 load `__official`。
>    当前 shipped artefacts 是原始文件名：`DB_molecular_feats.pkl`, `entity_drug.json`,
>    `node2id.json`, `relation2id.json`，TWOSIDES 也是 `entity2id.json`, `id2drug.json`,
>    `id2drug_feat.pkl`, `relation2id.json`。未见 `necessary/`，未见 `build_<name>.py`，
>    未见 `__official` / `__mine` 双轨。
>
> 4. **paper-vs-impl / Medium**
>    论文核心 contribution 不能在 reproduction code 内闭环验证——
>    EmerGNN model、flow-based propagation、attention/edge weighting 不在
>    reproduction 目录内，而是 baseline import。
>
> 5. **repo-vs-impl / Medium**
>    README 声称 "Architecture verified line-by-line against upstream" 且使用
>    "our ported model"，但实际 implementation 依赖 baseline module。
>    `_results/2026-05-17__S2_1_seed0_partial.md` 还是 interrupted、val-only、
>    single seed，却写了 "validated"；论文 Table 1 是 5-fold/seed test mean±std。
>
> Hyperparam 对照：argparse defaults 与 paper S1/S2 overrides 一致，没有发现直接不一致。

---

## 7. Three-bucket discrepancy classification (CLAUDE.md §3 step 3 line 391)

### paper-vs-impl (paper text vs current reproduction code)
- ❌ Core algorithm (model + propagation + attention) lives in baseline,
  not reproduction. Can't trace contributions to file:line within
  reproduction folder.
- ✅ Hyperparams + shuffle_train + data loading are correct in-place.

### repo-vs-impl (upstream code vs our port)
- ⚪ README docstring overclaims ("verified line-by-line") vs reality
  (architecture is in baseline; only data/training loop is here).
- ❌ Result archival overclaims: single val-only partial run is logged
  as "validated", but paper Table 1 requires 5-setting mean±std on test.

### baseline-vs-spec (CLAUDE.md §"复现代码规范" violations)
- ❌ 5 layout deviations (4 underscore-prefix items + 1 missing git clone)
- ❌ Shipped artefacts not under `__official` convention
- ❌ Cross-folder import of `baseline.emergnn.*` from reproduction

---

## 8. Final verdict: ❌ FAIL

Reproduction is **functionally usable** (one partial S2_1 run got
val_f1=0.2487 ≈ paper 0.250 ± 0.028, see `_results/2026-05-17__S2_1_seed0_partial.md`),
but it **fails CLAUDE.md §"复现代码规范" on 3 fronts**:

| Category | Status |
|---|---|
| Algorithm preservation in-folder | ❌ model imported from baseline |
| Layout conformance | ❌ 5 deviations |
| `__official` artefact convention | ❌ none of 4-8 shipped artefacts conform |
| Hyperparam fidelity | ✅ all match |
| Cross-folder independence | ❌ `run_reproduction.py:39` + diagnostic scripts |

Cannot mark PASS without addressing the cross-folder dependency and
layout migration (minimum).

---

## 9. Migration plan (codex priority list, ordered)

| Pri | Task | Effort | Notes |
|---|---|---|---|
| P0 | Copy `baseline.emergnn.multi_cls.model` (`EmerGNN_MC` + `EmerGNN` parent) + `baseline.emergnn.kg_builder` + `baseline.emergnn.morgan_features` into reproduction folder. Update `run_reproduction.py:39` import to local. Maintain independent copies (per CLAUDE.md §文件级独立性, baseline keeps its existing copy). | L | model.py is ~250 lines; needs careful port |
| P0 | Same for diagnostic scripts (`_bench_speed.py`, `_diagnose_mem.py`, `_diagnose_mem2.py`) | S | swap imports only |
| P0 | Layout rename: `Original-Dataset/` → `_Original-Dataset/`, `paper-and-github/` → `_paper-and-GitHub/`, create `_reviews/` (done by this file) | S | mv + `.gitignore` update |
| P0 | Clone upstream `LARS-research/EmerGNN` to `_paper-and-GitHub/LARS-research__EmerGNN/` (note URL in `github.txt` has a stray space — strip when cloning) | S | one git clone |
| P1 | Reorganize shipped artefacts → `_Original-Dataset/DrugBank/necessary/{DB_molecular_feats__official.pkl, entity_drug__official.json, node2id__official.json, relation2id__official.json}` (same for TWOSIDES). Update `data_loader.py` to follow `__official` > `__mine` > legacy detect-order with `USING OFFICIAL: ...` stderr log (mirror TIGER pattern) | M | path refactor + back-compat |
| P1 | Write `_Original-Dataset/necessary/build_DB_molecular_feats.py` reverse-engineered builder + verify bit-identical against `__official` (or document failure to reverse-engineer if applicable) | M | depends on understanding upstream feature schema |
| P2 | Update `README.md` to remove "verified line-by-line" overclaim until model is actually in reproduction folder | S | docstring fix |
| P2 | Update `_results/2026-05-17__S2_1_seed0_partial.md` status from "validated" to "partial S2_1 val checkpoint" to avoid overclaim | S | small wording fix |

---

## 10. Future work / unresolved

1. Full Table-1 reproduction (5 settings × test eval) — currently we only
   have 1 partial val-only run from S2_1.
2. End-to-end smoke train after the migration to verify the moved
   `EmerGNN_MC` + `kg_builder` still match the existing run's behaviour.
3. After migration, the baseline-side `_reviews/2026-05-17__baseline.md`
   verdict needs revisiting to note that baseline now keeps an INDEPENDENT
   COPY of the model rather than the reproduction copying from baseline.
4. `build_DB_molecular_feats.py` bit-identity verification deferred —
   needs spot-check on at least 10 random drugs vs `DB_molecular_feats__official.pkl`.
5. **Builders for vocab JSONs** — see next section.

## 11. Builders for vocab JSONs (documented inability to fully reverse-engineer)

Per CLAUDE.md §"复现代码 (Reproduction) 规范" §3 step 5 line 374: any
required preprocessing artefact should have a builder. The 3 vocab JSONs
(`node2id.json`, `entity_drug.json`, `relation2id.json`) shipped on
Zenodo encode upstream's specific construction choices, and cannot be
bit-identically reverse-engineered without:

1. **DrugBank XML version pin** — paper Methods refers to "DrugBank
   5.1.x" without an exact build date. Different DrugBank XML versions
   produce different drug subsets (e.g., 1710 vs 1700-something) and
   thus different `node2id.json` mappings.
2. **HetioNet release pin** — paper cites HetioNet v1.0 but the exact
   node-filter rules (which non-drug entities to keep) aren't published.
   Different filter rules → different `entity_drug.json` entry counts.
3. **Relation-text ordering** — `relation2id.json` maps int → relation
   text for 109 relations (86 DDI + 23 KG). The 86 DDI types come from
   DrugBank's interaction descriptions; the ordering depends on
   upstream's sort/dedup choices.

Per spec line 374 escape valve "如果算法实在反推不出, 在 `_reviews/`
里明确说明原因": **this section is that explicit documentation**.

`build_vocab.py` is shipped as a `--check-only` stub: it verifies the
3 `__official` files are present + points to the Zenodo download if not.
It does NOT attempt reverse-engineering. A future PR could implement a
real reverse-engineer pipeline given the 3 pins above; until then,
users rely on the upstream-shipped `__official` files (which IS the
intended workflow for paper-faithful reproduction anyway).
