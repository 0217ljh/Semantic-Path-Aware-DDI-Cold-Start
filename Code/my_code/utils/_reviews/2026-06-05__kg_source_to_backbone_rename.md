# `kg_source` → `backbone_kg_source` Codebase-Wide Rename

- **Date**: 2026-06-05
- **Primary reviewer**: Claude (sonnet-4.5) — executed the rename.
- **Independent reviewer**: codex (default model via `mcp__codex__codex`).
- **Triggered by**: 用户 — "这个参数还是澄清一下比较好：kg_source='drugbank'。比如你可以改成 backbone_kg_source=merged, ... 不然容易混淆"
- **Stop condition**: codex round 2 PASS_WITH_NITS. Round 1 NOT_PASS due to TIGER regression. Round 2 confirmed regression fixed; remaining nits are stale help/example strings (rewritten cosmetic-only).

---

## 1. Motivation

The legacy parameter name `kg_source` (defined on `EmerGNNTAGBaseline` parent
in `multimode.py`) was misleading. Despite the name, it only controls the
EmerGNN backbone's path-flow entity vocabulary. The PMP mediator cache, the
v1.1 cluster cache, and the v1.5+ score function ALL use the merged KG
regardless. The same name "kg_source" implied a model-wide setting,
which it is not. v1.6 PMP runs verified this point: `--kg-source=drugbank`
selects the backbone vocab but the v1.6 score function bypasses the
backbone entirely, so the parameter has no effect on AUC for v1.5+.

## 2. Scope (locked per codex design round)

### Allowlist (40 .py files)

Parent + inheritance chain (8 files):
- `Code/my_code/models/screen1_tag_init/multimode.py` (parent)
- `Code/my_code/models/screen1_tag_init/run_screen1.py`
- `Code/my_code/models/screen_s2_iter/{run_nodedup,run_cacr,run_s2_anchor}.py`
- `Code/my_code/models/screen_s2_v2_meetnode/{run_mnah,run_stage3,precompute_meet_features}.py`
- `Code/my_code/models/screen3_meet_in_middle/run_screen3.py`

V3 multimodal family (14 files):
- `Code/my_code/models/screen_s2_v3_multimodal/{run_v3_pmp,v3_pmp_trainer,run_v3_gated_fusion,v3_gated_fusion_trainer,v3_meet_mask_trainer,precompute_meet_mediators,run_v3_meet_mask,run_v3_llm_edge,run_v2i4,run_v2i4d,run_v2i4hn,run_v2llm,run_v2res,run_v2}.py`

PMP source (2 files):
- `Code/my_code/models/pmp_v1/{precompute_pmp_cache,pmp_trainer}.py`

EmerGNN baseline (6 files):
- `Code/baseline/emergnn/{_per_mode,_shared}.py`
- `Code/baseline/emergnn/{binary_cls,multi_cls,deprecate/binary_cls_legacy_buggy}/baseline.py`
- `Code/baseline/emergnn/_data/necessary/build_kg_setup_cache.py`

CLI / front-doors (9 files):
- `Code/scripts/run_pmp_v1{,_1,_2,_3,_4,_5,_6}.py` (7 files)
- `Code/scripts/run_baseline.py`
- `Code/scripts/diagnose_emergnn_s0s1_collapse.py`

### Excluded (NOT renamed — third-party API or schema)

- `Code/baseline/tiger/*` (3 files) — Tiger has its OWN `kg_source` param
  (accepts "merged" or "none" / mol-only), independent of EmerGNN backbone.
- `Code/scripts/run_nbfnet.py` — NBFNet has its own local `kg_source`
  variable for graph construction (accepts "merged" or "drugbank"), unrelated
  to EmerGNN backbone parameter.
- `Code/my_code/utils/run_logger.py` — CSV schema column name `kg_source` in
  `Code/runs/_logs/index.csv` is preserved for historical row compatibility
  (per codex round 0 recommendation). Only the file's docstring was updated
  to clarify the schema-vs-param asymmetry.

## 3. Three rename categories applied

### 3a. Bulk identifier rename (40 files)

Word-boundary regex `\bkg_source\b` → `backbone_kg_source` applied via
Python script at `/tmp/refactor_kg_source.py`. Covered:
- Parameter declarations: `kg_source: str = "drugbank"` → `backbone_kg_source: str = "drugbank"`
- Attribute access: `self.kg_source` → `self.backbone_kg_source`
- Keyword arguments: `kg_source="drugbank"` → `backbone_kg_source="drugbank"`
- Variable access via argparse: `args.kg_source` → `args.backbone_kg_source`
- Internal log strings + docstrings: `kg_source=...` → `backbone_kg_source=...`

Total renames: 130+ occurrences across the 40 files.

### 3b. `set_meta()` kwarg key preservation (7 sites)

After bulk rename, a post-process regex reverted the FIRST argument key inside
`set_meta(...)` calls back to `kg_source`, while keeping the VALUE renamed.

Example:
```python
# Before bulk rename:
rl.set_meta(kg_source=args.kg_source, epochs=args.epochs, ...)
# After bulk rename (raw, BROKEN):
rl.set_meta(backbone_kg_source=args.backbone_kg_source, epochs=args.epochs, ...)
# After post-process fix (CORRECT):
rl.set_meta(kg_source=args.backbone_kg_source, epochs=args.epochs, ...)
```

Affected files (7 sites):
- `run_screen1.py:110`, `run_nodedup.py:69`, `run_cacr.py:75`,
  `run_s2_anchor.py:68`, `run_mnah.py:108`, `run_v2.py:115`, `run_baseline.py:179`.

Rationale: `RunLogger.set_meta(**kwargs)` writes each kwarg as a column to
`index.csv`. Preserving the kwarg key `kg_source` keeps the CSV column name
unchanged, maintaining historical row compatibility (per codex).

### 3c. CLI flag dual-name alias (7 sites)

`p.add_argument("--kg-source", ...)` rewritten to:
```python
p.add_argument(
    "--backbone-kg-source", "--kg-source",
    dest="backbone_kg_source",
    choices=[...], default="drugbank", help="..."
)
```

Both flags now accepted at the CLI; both populate `args.backbone_kg_source`.
Old `--kg-source` is a soft-deprecated alias that should be removed in a
future cleanup pass once any saved shell scripts have been migrated.

Affected: `run_baseline.py:106`, `run_v2.py:84`, `run_screen1.py:89`,
`run_mnah.py:78`, `run_s2_anchor.py:52`, `run_nodedup.py:51`, `run_cacr.py:51`.

## 4. Codex round 1 (NOT_PASS) — critical regression

Codex caught a critical TypeError bug at `run_baseline.py:333,336`:

```python
# After bulk rename — INCORRECT:
if args.backbone_kg_source == "none":
    kwargs["backbone_kg_source"] = "none"   # passed to TIGER constructor
    kwargs["mol_only"] = True
else:
    kwargs["backbone_kg_source"] = "merged"
    kwargs["merged_kg_path"] = str(MERGED_KG)
```

TIGER's `TIGERBaseline.__init__` signature at
`Code/baseline/tiger/binary_cls/baseline.py:111` still accepts `kg_source`,
not `backbone_kg_source`. Passing `backbone_kg_source=` into TIGER would
TypeError at construction.

Fix applied (now correct):
```python
if args.backbone_kg_source == "none":
    # TIGER has its OWN `kg_source` parameter (independent of EmerGNN
    # backbone). Tiger accepts "merged" or "none" (mol-only). Do NOT
    # rename the kwarg key — Tiger's API is not part of this refactor.
    kwargs["kg_source"] = "none"
    ...
```

The CLI variable `args.backbone_kg_source` (correctly renamed for our CLI)
provides the VALUE; the kwarg key into TIGER stays as TIGER's own
`kg_source` API.

## 5. Codex round 2 (PASS_WITH_NITS)

> "TIGER branch is fixed correctly. EmerGNN branch is still using the renamed
> API correctly. Repo sweep did not uncover another live constructor/API
> regression from the bulk rename. The remaining `kg_source` usages fall
> into expected buckets: TIGER own API, logging/schema, NBFNet, legacy
> help strings (functional via alias)."
>
> Cosmetic nits (non-blocking):
> - Several help/doc strings still presented `--kg-source` in examples,
>   e.g. `run_baseline.py:12`, `run_s2_anchor.py:12`, `run_mnah.py:7`.
>
> "VERDICT: PASS_WITH_NITS"

Round 2 cosmetic nits fixed by post-script: 4 stale `--kg-source` references
in usage docstrings rewritten to `--backbone-kg-source` (in `run_baseline.py`,
`run_s2_anchor.py`, `run_mnah.py`). Argparse alias declarations
(intentional `--kg-source` for back-compat) were not touched.

## 6. Verification (post-fix)

### Static
- `grep -rn '\bkg_source\b' Code/my_code/models/screen* Code/baseline/emergnn Code/scripts/run_pmp_v1*.py`:
  22 lines remaining. Breakdown:
  - 5 lines: intentional `set_meta(kg_source=args.backbone_kg_source, ...)` (CSV schema)
  - 6 lines: historical `_results/*.md` and `_reviews/*.md` (CLAUDE.md "不覆盖历史")
  - 6 lines: `.pyc` stale bytecode (regenerates on next import)
  - 5 stray = subset of the above (some files have multiple lines)
- `grep -rn backbone_kg_source Code --include='*.py'`: 117 occurrences.

### Smoke imports
- `EmerGNNTAGBaseline.__init__` signature contains `backbone_kg_source`, NOT `kg_source` ✓
- v1.5 / v1.6 trainer MRO unchanged; `backbone_kg_source` flows through `**kwargs` cascade ✓
- `baseline/emergnn/_per_mode.py` signature: `backbone_kg_source` OK, old `kg_source` removed ✓

### CLI smoke
- `python Code/scripts/run_baseline.py --help` shows
  `--backbone-kg-source {drugbank,merged,none}, --kg-source {drugbank,merged,none}` ✓
- Both flags populate `args.backbone_kg_source` correctly via explicit `dest=`.

### NOT verified (deferred to user's next training run)
- 1-epoch smoke training (no GPU executed in this refactor session).
- TIGER baseline binary-cls / multi-cls runs (no GPU executed).

## 7. Files touched summary

| Action | Files |
|---|---|
| Bulk rename (40) | All allowlist .py files |
| Argparse alias (7) | run_baseline.py, run_v2.py, run_screen1.py, run_mnah.py, run_s2_anchor.py, run_nodedup.py, run_cacr.py |
| set_meta() preservation (7) | Same as argparse files (where set_meta is called) |
| TIGER regression fix (1) | run_baseline.py |
| run_logger.py docstring (1) | run_logger.py |
| Cosmetic doc nits (3) | run_baseline.py, run_mnah.py, run_s2_anchor.py |

## 8. Future cleanup (not done in this pass)

- Remove `--kg-source` legacy alias from CLI after a transition period (say
  1 month). Search-replace `--backbone-kg-source` only.
- Consider renaming the index.csv column `kg_source` → `backbone_kg_source`
  as a separate schema migration. Will require dual-write or version bump.

## 9. Stop condition

Codex round 2 returned PASS_WITH_NITS with one critical fix verified. Cosmetic
nits resolved by post-script. No round 3 executed.
