# CP-1 Design Review — D1 (LLM mechanistic fields → new KG edge types)

## Metadata

- **Date**. 2026-05-30
- **Scope**. CP-1 design review for Round 4 direction D1, per `Notes/Log/round4_backbone_diff_plan.md` §8.
- **Primary reviewer**. Claude (claude-opus-4-7), drafted the design + this report.
- **Independent reviewer**. codex (gpt-5-codex unavailable on this account; codex MCP default model). Codex thread: `019e7b09-93a3-7a91-8e24-f438c6f08096`.
- **Triggered by**. User asked to push round 4 D1 through CP-1 → CP-2 → CP-3 codex gates with each PASS (critical=0, major=0) before advancing.
- **Final verdict**. **PASS_WITH_NITS** at codex round 2.
- **Decision**. Proceed to implementation (CP-2 task). Two minor wording issues (stale "Three controls" phrase + `W_r` cite imprecision) were fixed in the same docs at design-doc round 2 before this report was archived.

## Reviewed artifacts

- `Notes/Log/d1_llm_edge_design.md` (D1 design draft).
- `Notes/Log/round4_d1_kg_source_deviation.md` (KG-source deviation note, recorded before drafting).
- Implicitly: `Notes/Log/round4_backbone_diff_plan.md`, `Code/my_code/models/screen_s2_v3_multimodal/v2i4_trainer.py`, `Code/my_code/models/screen_s2_v2_meetnode/mnah_trainer.py`, `Code/baseline/emergnn/_per_mode.py`, `Code/baseline/emergnn/model.py`, `Code/baseline/emergnn/kg_builder.py`, `Code/baseline/emergnn/kg_builder_merged.py`, `Code/my_code/models/screen_s2_v3_multimodal/run_v2i4.py`, `Code/data/_cache/llm_pharma/i4_typed_sets.json`, `Code/data/coldddi_legacy/800drug/seed42.pkl`, `Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json`.

## Contribution list (CP-1 from design doc §1)

| # | Claim | Verified at code level? |
|---|---|---|
| C1 | Backbone is meaningfully changed (KG input space grows by 10 relations + ~7800 nodes); EmerGNN propagation code untouched. | Yes (codex Q1 PASS, Round 1) |
| C2 | No cold-start leakage via new edges; per-drug intrinsic LLM fields only; sanitizer reused; explicit token-vs-drug-name audit (now using `id2name.json`). | Directional PASS after R3 fix (codex Q2, Round 2) |
| C3 | Empirical guard (not architectural) against falling below 0.78: inherited readout head as soft floor + K4 parity smoke + §4 hard-stop. Original architectural worst-case claim withdrawn. | PASS (codex Round 2, after Round 1 NOT_PASS) |

## Codex independent verdict — full timeline

### Round 1 (verdict: NOT_PASS)

Codex original output:

> **Verdict: NOT_PASS**
>
> Major issues present, so it does not meet round4 §8.4 PASS criteria.
>
> **Issues**
>
> - **major**: C3 worst-case bound is not valid.
>   `d1_llm_edge_design.md:17` claims adding LLM edges "degenerates cleanly back to v2i4" if new relations stay near init. That does not follow from the current EmerGNN implementation. Increasing `n_base_rel` from 5 to 15 changes `all_rel`, relation attention output size, relation embedding tables, self-loop relation id, and forward relation ids for the original KG. See `model.py:77`, `model.py:137`, and `kg_builder.py:166`. Even "unused/noisy" new edges perturb message passing, and the baseline graph is not functionally identical after relation-space expansion.
>
> - **major**: R3 drug-name leakage audit is specified against the wrong file.
>   `d1_llm_edge_design.md:149` says `drug_smiles__seed42.csv` is the 1900-drug name list. It is not; it has columns `drugbank_id,smiles`. A token-vs-drug-name audit cannot be implemented from that file as written. Also, "token equals drug name" is too weak; it should check normalized containment / aliases / brand or generic names, at least for free-text fields.
>
> - **major**: K2 rand-token control is not guaranteed to destroy shared-token routing as stated.
>   `d1_llm_edge_design.md:113` says collisions from 8000 synthetic tokens are negligible. With ~23k drug-token edges, independent draws from 8000 will create many collisions. If K2 is meant to remove bridges, tokens should be unique per edge or per drug-token slot, or sampled with a construction that explicitly prevents cross-drug sharing.
>
> **Claims**
>
> - **C1: mostly holds.** Injecting drug-token triples into `self._kg_triplets`, rebuilding edge lists, and expanding `n_ent`/`n_base_rel` is a real backbone/input-graph change. The hook point is plausible: `_setup_graph` exists at `_per_mode.py:218`, model allocation consumes updated `n_ent/n_base_rel` at `_per_mode.py:366`. Caveat: the same LLM substrate still reaches `combined_logit` through inherited v2i4 i4 readout, so backbone novelty must be judged by `auc_emergnn` and controls, not by combined alone.
>
> - **C2: not fully proven, but design is directionally OK after fixing R3.** The fields are per-drug and sample cache content is intrinsic-looking; v2i4 sanitizer exists at `v2i4_trainer.py:131`. But the proposed drug-name audit is currently not implementable from the cited file, so C2 cannot pass as written.
>
> - **C3: does not hold.** The i4 head may provide empirical fallback, but there is no architectural worst-case guarantee that combined AUROC cannot fall below 0.7804 after graph/relation-space expansion.
>
> **Q1-Q10**
>
> Q1: Yes, it changes the backbone. Not a readout-only hack if implementation only injects through KG triplets. But combined metric remains confounded by inherited i4 head; require `auc_emergnn` lift.
>
> Q2: Intrinsic-field premise is reasonable, but R3 must be fixed. Current name-audit path is wrong and equality-only matching is too weak.
>
> Q3: No. C3 is false for the reasons above.
>
> Q4: K1 is broadly capacity-preserving. K2 needs collision-free construction. Add a stronger cheap control: `D1 edges present but all llm relation ids mapped to a single random relation` or `drop-new-edges in D1 code path` to verify exact v2i4 parity.
>
> Q5: 19.4% uncovered is acceptable for one experiment because G1/G2 coverage is roughly even per deviation note, but report covered-only vs uncovered-only S2 metrics if possible.
>
> Q6: Clean override is fine. New `v3_llm_edge_trainer.py` inheriting and overriding `_setup_graph` does not modify forbidden files.
>
> Q7: Most paths checked out: v2i4 uses `kg_source="drugbank"` at `run_v2i4.py:83`, run metrics match `0.7803698`, `auc_emergnn=0.7405175`. But the drug-name CSV claim is wrong, and C3's parameter-count/worst-case statement is unverified/incorrect.
>
> Q8: Fix length=3 for first D1 for apples-to-apples. Sweep length only after seed42 canonical clears the hard-stop.
>
> Q9: The story is supportable if `auc_emergnn` lifts and controls drop. Add per-relation/token-degree diagnostics now; attention weights are useful but not required for CP-1.
>
> Q10: Add an explicit D1 "drop-new-edges" parity smoke test. It should instantiate the D1 trainer with D1 disabled or edge parquet empty and verify metrics/logits are close to v2i4 under the same seed, otherwise C1/C3-style claims remain hard to interpret.

### Round 2 (verdict: PASS_WITH_NITS)

After design-doc edits addressing all 3 majors + 2 recommendations:

> **Verdict: PASS_WITH_NITS**
>
> The three previous major issues are closed.
>
> C3 now correctly drops the architectural guarantee and replaces it with empirical guardrails: inherited readout floor, K4 parity smoke, and the §4 hard-stop. K2 is now collision-free and tests bridge removal cleanly. R3 now uses a real name source (`id2name.json`) and stronger token audit logic. Covered-vs-uncovered reporting is a useful CP-3 diagnostic and does not introduce a new concern.
>
> **Minor Issues**
>
> - **minor**: stale wording in controls/questions.
>   `d1_llm_edge_design.md:122` still says "Three controls" although K4 was added. `d1_llm_edge_design.md:218` still phrases Q3 as if the old C3 worst-case bound is under review, and `d1_llm_edge_design.md:220` lists only K1/K2/K3.
>
> - **minor**: C3 code citation wording is slightly imprecise.
>   `d1_llm_edge_design.md:21` says `model.py:77` allocates `W_r ~ randn(all_rel, n_dim)`. The relevant relation-space facts are `all_rel = 2*n_base_rel+1` at `model.py:59`, relation embeddings at `model.py:78`, attention output at `model.py:84`, and self-loop id at `kg_builder.py:175`. The argument is correct, but the `W_r` label should be fixed.
>
> No critical or major issues remain under round4 §8.4.

## Issues + resolutions

| Severity | Issue | File:Line | Resolution |
|---|---|---|---|
| major (R1) | C3 architectural worst-case bound is false. | `d1_llm_edge_design.md` C3 paragraph | Rewrote C3 to drop architectural guarantee; added K4 empirical parity control + cited inherited readout floor + §4 hard-stop as empirical guards. **Resolved at Round 2.** |
| major (R1) | R3 drug-name audit pointed at wrong file (`drug_smiles__seed42.csv` has no name column). | `d1_llm_edge_design.md` R3 paragraph | Rewrote R3: source-of-truth now `Code/data/KG/drugbank/filtered/id2name.json` (verified 1900 drugs); audit logic upgraded from equality to 3-rule normalized containment with explicit synonym caveat. **Resolved at Round 2.** |
| major (R1) | K2 rand-token control would preserve shared-token bridges due to 23k draws over 8k vocab. | `d1_llm_edge_design.md` K2 paragraph | Rewrote K2 to be collision-free per-edge unique (`rand_<edge_idx>`); explicitly contrasted purpose vs K1. **Resolved at Round 2.** |
| recommended (R1, Q4/Q10) | Need explicit parity smoke for v2i4. | new K4 in §3 | Added K4 control: `--d1-disable` flag, 5-epoch seed42, mean abs diff per-epoch loss vs v2i4 ≤ 1e-4. **Added at Round 2.** |
| recommended (R1, Q5) | Need covered-vs-uncovered S2 breakdown. | §2.4 channel reporting | Added covered/one/neither buckets with per-bucket AUC + expected directional pattern. **Added at Round 2.** |
| minor (R2) | "Three controls" wording stale post-K4. | `d1_llm_edge_design.md:122` (former) | Rephrased to "Four controls (K1–K3 main + K4 parity smoke)". **Resolved at Round 2 fix-up.** |
| minor (R2) | Q3 + Q4 phrasings stale post-C3 rewrite. | `d1_llm_edge_design.md` §7 Q3 / Q4 | Rewrote both Q3 and Q4 to reflect updated C3 + four-control structure. **Resolved at Round 2 fix-up.** |
| minor (R2) | C3 code-cite `W_r` label wrong; correct lines are `model.py:59` `all_rel`, `model.py:78` `rel_kg`, `model.py:84` `attn_relation`, `kg_builder.py:175` self-loop id. | `d1_llm_edge_design.md` §1 C3 second paragraph | Updated cite to enumerate the actual lines + symbols. **Resolved at Round 2 fix-up.** |

## Unresolved / future-work flagged

- **v1 audit synonym gap (R3 explicit caveat)**. Primary-name-only audit (1900 names) does not cover DrugBank brand names / synonyms (e.g. "Coumadin" for warfarin). If CP-3 result review surfaces residual leakage suspicion, v2 audit must source synonyms from `Code/data/KG/drugbank/drugbank_with_mechanisms.csv` (141 MB) or a synonyms extract. Flagged to CP-2 to surface during code review.
- **Open-vocab token explosion mitigation deferred**. Per §5 R1, capping `therapeutic_class`/`primary_targets`/`pd_effects`/`toxicity_mechanisms`/`clearance` to top-K most-frequent tokens is held in reserve as a hyperparam axis for the 0.775–0.785 sweep branch of the hard-stop rule, not exercised by canonical D1.
- **D1 on merged KG followup**. The KG-source deviation note explicitly defers `d1_merged` to a follow-up experiment after drugbank-side D1 verifies. The follow-up requires a fresh "v2i4 main on merged KG" anchor to keep the comparison single-change.

## Next step

Move to CP-2: implement builder + trainer + run script + parity smoke; trigger codex CP-2 implementation review when implementation is ready. Reviews + results both co-locate to `Code/my_code/models/screen_s2_v3_multimodal/_reviews/` and `_results/` per CLAUDE.md / round4 plan §8.5.
