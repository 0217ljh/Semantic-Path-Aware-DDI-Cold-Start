# Round 4 D1 — KG-source deviation note (drugbank 5-bucket, not merged)

**Date**. 2026-05-30
**Author**. Claude (claude-opus-4-7) — context inherited from `round4_backbone_diff_plan.md`
**Status**. Recorded BEFORE writing the D1 design doc, per round4 plan §6 "任何 deviate from this plan before acting, first open a .md in Notes/Log/".

---

## What the plan said

`Notes/Log/round4_backbone_diff_plan.md:229`:
> 当前 0.78+ 用的是哪一份. v2i4 trainer 的 KG 输入是 merged KG（178,029 nodes）。**D1 新增 LLM edge 时直接在 merged KG 上扩展**，不要回退到 drugbank-only。

## What is actually true

Verified from source code and the canonical 0.7804 run's results.json:

- Entry script `Code/my_code/models/screen_s2_v3_multimodal/run_v2i4.py:85` constructs the trainer with `kg_source="drugbank"`. There is no CLI flag to override this in run_v2i4.py.
- `Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json:2-11` records the config — no `kg_source` field, no `merged_kg_path` field; the trainer therefore used the hardcoded default `kg_source="drugbank"` from run_v2i4.py:85.
- `Code/baseline/emergnn/_per_mode.py:283-315` shows the `"drugbank"` code path: builds the KG via `build_kg_from_kb(kb, drug_id_list)` where `kb = _kg_to_kb_dict(kg)` extracts the legacy 5-bucket schema (`enzymes / targets / transporters / carriers / pathways`) from `train.kg`. The relation count is `N_BASE_REL = 5` (`Code/baseline/emergnn/kg_builder.py:N_BASE_REL`).
- The merged KG path (kg_source="merged", 59 relations, 178,029 nodes) exists in `_per_mode.py:244-268` and `baseline/emergnn/_data/necessary/build_kg_setup_cache.py:73-147`, but it is NOT exercised by run_v2i4.py and was NOT used by the 0.7804 baseline.

So the plan §7.2 claim "v2i4 trainer 的 KG 输入是 merged KG" is incorrect. The 0.7804 baseline uses the DrugBank 5-bucket KG (5 relation types).

## Why this matters for D1

D1 = "LLM mechanistic fields as new KG edge types injected into backbone path-flow".

If I follow the plan literally (extend merged KG):
- The base graph the model sees changes simultaneously with the LLM-edge change. The combined AUROC of the new D1 model is then no longer comparable to 0.7804, because the KG source itself changed. I'd need a new "v2i4 main on merged KG" baseline run to do the comparison — two confounded changes at once, not a clean evidence test.
- Faster runs become slower (178k nodes + ~7M edges vs ~10k entities + much smaller edge count).

If I deviate (extend drugbank 5-bucket KG):
- Apples-to-apples lift vs 0.7804 baseline. Single-change ablation is clean.
- Smaller graph fits the existing batch_size=32 / length=3 budget without surprise OOM or epoch-time blowup.
- The drugbank 5-bucket already has `enzymes / targets / transporters / carriers / pathways`. LLM fields add **role-typed sub-distinctions** (CYP substrate vs inhibitor vs inducer) and **downstream-effect evidence** (`pd_effects`, `toxicity_mechanisms`, `clearance`, `therapeutic_class`) that the 5-bucket schema does NOT carry. So D1-on-drugbank still genuinely adds evidence, not duplication.

## Decision

**D1 first pass targets the drugbank 5-bucket KG (kg_source="drugbank") to preserve the 0.7804 anchor.** This is the deviation from plan §7.2 / §7.3 instruction "在 merged KG 上扩展".

D1 on merged KG remains an option for a follow-up experiment (likely tagged `d1_merged`) AFTER the drugbank-side D1 is verified — at that point we'd run both a "v2i4 main on merged" anchor and a "v2i4 + D1 on merged" treatment for the comparison to stay clean.

## Other verified facts that shape D1 (recorded once here so the design doc can reference)

- **LLM-cache coverage gap**. Verified 2026-05-30:
  - `i4_typed_sets.json` covers 1530 of the 1900 DrugBank pool.
  - 800-drug split: 645/800 = 80.6% covered. G1 (seen): 515/640 = 80.5%. G2 (unseen, drives S2): 129/159 = 81.1%, with **30 of 159 G2 drugs uncovered**.
  - Uncovered drugs will contribute no LLM-typed edges in D1's extended KG. They fall back to whatever the drugbank-5-bucket already carries for them (which is what 0.7804 had). So D1 cannot hurt them; it can only improve covered drugs. Coverage gap is roughly even across G1 and G2, so the cold-start probe is not disproportionately weakened.
- **LLM field vocabularies (sample run on i4_typed_sets.json, 2026-05-30)**:
  - Closed-vocab (small): `cyp_substrate` 24 tokens, `cyp_inhibitor` 12, `cyp_inducer` 8, `transporter_substrate` 10, `transporter_inhibitor` 6 — total ~60 new nodes.
  - Open-vocab (large): `therapeutic_class` 1000, `primary_targets` 1115, `pd_effects` 2404, `toxicity_mechanisms` 2731, `clearance` 561 — total ~7800 new nodes.
  - Total per-field token sums (drug→token edges): cyp_substrate 1244, cyp_inhibitor 552, cyp_inducer 81, transporter_substrate 330, transporter_inhibitor 38, therapeutic_class 3640, primary_targets 2688, pd_effects 5847, toxicity_mechanisms 5664, clearance 3143. Sum = **23,227 drug→token directed edges** added.
  - Existing drugbank 5-bucket KG edge count is on the order of ~10k–50k (to be verified at build time); the LLM-edge addition is therefore the same order of magnitude as the base graph, which is a meaningful structural change (good for novelty story, but increases the budget for OOM / training-time regressions and needs to be checked at smoke-run).
- **0.7804 baseline config snapshot (verified)**: `length=3, n_dim=64, batch_size=32, lr=1e-3, weight_decay=1e-8, shuffle_train_mode="S2", shuffle_ratio=0.8, kg_source="drugbank"`. Run: `Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/`.
- **Two sanitizers already in v2i4_trainer.py:128-141** that drop tokens containing interaction language ("interact", "coadminist", "combined with", "concomitant", "avoid with", "contraindicated", "with inhibitor", "with inducer", "increase levels", "decrease levels", "co-medic", "co-prescri"). D1 builder must reuse this exact list to keep cold-start leakage protection consistent across the readout-side i4 head (v2i4) and the new backbone path (D1).

## What this note does NOT change

- Hard-stop rule (combined < 0.775 stop, 0.775–0.785 limited sweep, ≥0.785 multi-seed). Anchor 0.7804 unchanged.
- Codex 3-gate flow (CP-1 design → CP-2 implementation → CP-3 results), each with PASS gate (critical=0, major=0). Reviews co-located at `Code/my_code/models/screen_s2_v3_multimodal/_reviews/`.
- "Do not modify existing files" rule. v2i4_trainer.py, mnah_trainer.py, baseline/emergnn/* stay untouched. All D1 code is new files.
