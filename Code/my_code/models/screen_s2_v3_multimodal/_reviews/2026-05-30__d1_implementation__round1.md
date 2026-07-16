# CP-2 Implementation Review — D1 (LLM mechanistic fields → new KG edge types)

## Metadata

- **Date**. 2026-05-30
- **Scope**. CP-2 implementation review for Round 4 direction D1, per `Notes/Log/round4_backbone_diff_plan.md` §8 and design doc `Notes/Log/d1_llm_edge_design.md`. CP-1 PASS_WITH_NITS at `_reviews/2026-05-30__d1_design__round1.md`.
- **Primary reviewer**. Claude (claude-opus-4-7), drafted the code + this report.
- **Independent reviewer**. codex (codex MCP default model). Codex thread: `019e7b1b-058b-7ac0-9e4e-501804b9176d`.
- **Triggered by**. CP-1 PASS → implement → CP-2 codex gate.
- **Final verdict**. **PASS_WITH_NITS** at codex round 2.
- **Decision**. Proceed to seed42 100-epoch main run with `tag=d1_main_seed42`. Three Round 2 minors (stale "cardinality preserved" wording in file docstring, "exactly preserves" wording before vs after dedup, fixed-point diagnostic undercounting same-drug perm) were already fixed in the same edit pass before this archive.

## Reviewed code (all new files, byte-independent from forbidden files)

| File | Purpose |
|---|---|
| `Code/my_code/models/screen_s2_v3_multimodal/precompute_llm_edges.py` | Offline builder — turns `i4_typed_sets.json` into `Code/data/_cache/llm_edges/llm_drug_edges__seed42_drugbank.parquet`. Applies sanitizer-2 (13 phrases × 5 free-text fields, byte-equal to `v2i4_trainer.py:131-135`) and R3 drug-name audit (whole-word boundary, 800-pool restricted, 4× length cap) using `Code/data/KG/drugbank/filtered/id2name.json` (1900 names). |
| `Code/my_code/models/screen_s2_v3_multimodal/v3_llm_edge_trainer.py` | Trainer — `class _PerModeEmerGNN_V3LLMEdge(_PerModeEmerGNN_V2I4)`. Overrides `_setup_graph` to inject LLM edges after the parent does drugbank 5-bucket setup. Implements K1/K2/K3/K4 controls. |
| `Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py` | CLI entrypoint — mirrors `run_v2i4.py`. Adds `--d1-edges-parquet`, `--d1-disable`, `--d1-shuf-token`, `--d1-rand-token`, `--d1-drop-i4-head`. Reports per-channel AUC + covered-vs-uncovered S2 breakdown to `results.json`. |

## Smoke run (1 epoch, seed42, canonical D1)

Run dir: `Code/runs/2026-05-30_18-54-25__run_v3_llm_edge__d1_smoke_1ep__seed42/`.

- **Injection summary**: `n_edges_input=23003, n_edges_dropped_out_of_pool=558, n_llm_edges_injected=22445, n_new_relations=10, n_old_relations=5, n_new_token_nodes=7632, n_ent_before=5633, n_ent_after=13265, n_base_rel_before=5, n_base_rel_after=15`.
- **Numerics**: val_combined=0.6873, val_emer=0.6912, val_aux=0.4553 (1 epoch). test_s2 AUC=0.6892, emergnn=0.6948, count_only=0.4682, i4_only=0.4432. Purpose: validate the chain wires end-to-end, not measure lift.
- **Coverage breakdown** (expected pattern at canonical 100-epoch: both > one > neither): at 1 epoch already showing `both_covered: 0.7015 (n=2506), one_covered: 0.6675 (n=1212), neither_covered: 0.6680 (n=120)`.
- **Parquet sha**: `b3c524a26dfb48f3` (matches third-run whole-word audit output).

## Contribution list (CP-2 from design doc §1, verified at code level)

| # | Claim | File:line evidence | Verdict |
|---|---|---|---|
| C1 | Backbone is meaningfully changed; new edges reach `combined_logit` only through path-flow propagation. | `v3_llm_edge_trainer.py:_setup_graph` calls super first, then `_inject_llm_edges` (skipped if `d1_disable`). Injection updates `_entity2id` / `_kg_triplets` / `_n_ent` / `_n_base_rel` / `_kg_entity_set` / `_edge_src,_dst,_rel`. Parent fit reads `_n_base_rel` and `_n_ent` to construct `EmerGNN` at `mnah_trainer.py:329-336`. Smoke confirms expansion `5→15, 5633→13265`. | PASS |
| C2 | No cold-start leakage; per-drug intrinsic fields only; sanitizer-2 reused; explicit drug-name audit. | Sanitizer constants at `precompute_llm_edges.py:64` (SANITIZER_PHRASES) and `precompute_llm_edges.py:71` (FREE_TEXT_FIELDS) byte-equal to v2i4 lines 131-133 and 134-135. R3 audit at `precompute_llm_edges.py:152` (loader) / `:135` (split-pool) / `:177` (whole-word + 4× cap rule). Builder reads NO DDI table (only i4_typed_sets.json, id2name.json, split drug-id list). | PASS |
| C3 | Empirical worst-case via K4 parity smoke + inherited readout head + §4 hard-stop. | K4 path at `v3_llm_edge_trainer.py:80` (`if self.d1_disable: return ...` before injection). K3 freeze at `v3_llm_edge_trainer.py:291` (raw_beta_i4=-50, requires_grad=False). | PASS (numeric K4 parity to be verified during CP-3 staging) |

## Codex independent verdict — full timeline

### Round 1 (verdict: NOT_PASS)

> **Verdict: NOT_PASS**
>
> 1 major issue blocks CP-2. No critical issues found in the canonical D1 injection path.
>
> **Findings**
>
> - **major - K1 shuf-token does not preserve per-drug cardinality.**
>   `v3_llm_edge_trainer.py:228` claims "Cardinality per drug preserved", but implementation builds each drug's full `(token_node_id, relation)` list, permutes those lists across drugs, and assigns another drug's list to `d` at lines 239-250. If drug A had 5 edges and receives drug B's 30-edge list, A now has 30 edges. `drop_duplicates()` at line 252 can further alter counts. This means K1 breaks both binding and per-drug edge-count/missingness structure, so it is not the intended clean "binding-only" control.
>
> - **minor - Morgan padding log is wrong.**
>   `v3_llm_edge_trainer.py:100` prints after `morgan_mat = padded`, so the smoke log says `morgan_mat padded from 13265 to 13265` instead of `5633 to 13265`. The padding itself is correct at lines 95-99; only the diagnostic is wrong.

(Codex also confirmed each of the contribution-list claims independently from code reads — see thread for full text.)

### Round 2 (verdict: PASS_WITH_NITS)

After K1 rewritten to edge-level drug-column permutation + morgan padding log corrected:

> **Verdict: PASS_WITH_NITS**
>
> The two blocking issues from the prior review are resolved.
>
> **Verified**
>
> - K1 is now edge-level drug-column permutation, not list-level swap. `v3_llm_edge_trainer.py:268` creates a permutation of `n_edges`; line 269 copies the original `drug_id` column; line 270 reassigns only `drug_id`; `(token_node_id, relation)` stay attached to each row. This destroys the original drug → own-token binding in the intended sense.
>
> - K1 dedup is explicit and logged. `v3_llm_edge_trainer.py:273` dedups on `(drug_id, token_node_id, relation)`, and lines 287-294 report before/after edge count, dropped duplicates, and cardinality drift.
>
> - Morgan padding diagnostic is fixed. `v3_llm_edge_trainer.py:96` captures `n_old_rows` before assigning `morgan_mat = padded`, and lines 101-103 now print the true `old -> new` row count.
>
> **Minor Nits**
>
> - **minor - stale top-level K1 description still says cardinality preserved.** `v3_llm_edge_trainer.py:13` still says `cardinality preserved`, but the new implementation explicitly does not preserve per-drug cardinality. Update this line to match the new docstring.
>
> - **minor - "exactly preserves" claims are only true before dedup.** `v3_llm_edge_trainer.py:245` says per-relation, per-token in-degree, and per-token/relation counts are preserved exactly. If line 273 drops duplicate rows, those counts can change for the dropped token/relation. This is acceptable if dedup loss is logged and small, but the wording should say "before dedup; after dedup, modulo dropped duplicate rows."
>
> - **minor - fixed-point diagnostic undercounts same-drug reassignment.** `v3_llm_edge_trainer.py:271` counts `perm[i] == i`, but because `drug_id` has repeated values, a row can still be reassigned to the same drug with `perm[i] != i`. For a more meaningful diagnostic, count `(original_drugs[perm] == original_drugs).sum()`.
>
> No critical or major issues remain for CP-2.

## Issues + resolutions

| Severity | Issue | File:Line | Resolution |
|---|---|---|---|
| major (R1) | K1 shuf-token gave drug A drug B's full token list, blowing up per-drug cardinality. | `v3_llm_edge_trainer.py:_mutate_shuf_token` (R1 implementation, lines 228-260) | Rewrote to edge-level drug-column permutation. Now preserves (before dedup) total edge count, per-relation count, per-token in-degree, per-(token, relation) count. Acknowledges per-drug cardinality not preserved (logged drift). **Resolved at Round 2.** |
| minor (R1) | Morgan padding diagnostic wrong (prints `13265→13265` instead of `5633→13265`). | `v3_llm_edge_trainer.py:100` (R1) | Captured `n_old_rows` before reassignment. **Resolved at Round 2.** |
| minor (R2) | Stale "cardinality preserved" in file-level docstring. | `v3_llm_edge_trainer.py:13` (R2) | Updated to reflect "drug→own-token binding destroyed; per-drug cardinality NOT preserved by design". **Resolved at Round 2 fix-up.** |
| minor (R2) | "exactly preserves" claims true only pre-dedup. | `v3_llm_edge_trainer.py:245` (R2 docstring inside _mutate_shuf_token) | Added "before dedup" qualifier to each "exactly" claim + general note that post-dedup the counts hold modulo dropped duplicate rows. **Resolved at Round 2 fix-up.** |
| minor (R2) | `n_fixed = (perm == arange).sum()` undercounts because drug-id has many repeats. | `v3_llm_edge_trainer.py:271` (R2) | Switched to `n_same_drug_after_perm = (original_drugs[perm] == original_drugs).sum()`. Log message updated to "rows landing back on same drug". **Resolved at Round 2 fix-up.** |

## Unresolved / followup flagged

- **K4 parity numeric verification deferred to CP-3 staging**. Codex CP-2 confirmed K4 path is structurally correct (`d1_disable=True` returns super output before injection); the per-epoch loss diff vs v2i4 ≤ 1e-4 needs an actual K4 + v2i4 5-epoch back-to-back run, which is part of the run-phase work, not the implementation review.
- **R3 audit conservatism**. Codex CP-2 round 2 explicitly endorsed keeping the audit conservative (whole-word, 800-pool, 4× cap) without an endogenous-compound allowlist yet. The known residual risk is the synonym-gap caveat from CP-1 (primary names only; brand names / synonyms not in audit set). Revisit only if CP-3 shows mechanism quality is hurt.
- **K1 cardinality drift** is by design (edge-level binding breaker has to relax cardinality preservation). Documented + logged.

## Next step

Run seed42 100-epoch canonical D1 main: `python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py --epochs 100 --tag d1_main_seed42 --seed 42`. Check combined AUROC against hard-stop rule (round4 plan §6 + d1_llm_edge_design.md §4). Run K4 parity smoke (`--d1-disable --epochs 5 --tag d1_k4_parity`) in parallel.
