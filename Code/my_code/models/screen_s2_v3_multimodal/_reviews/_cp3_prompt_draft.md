# CP-3 codex prompt draft (autonomous staging, fill K2/K3 numbers when runs finish)

This is the actual prompt to send to `mcp__codex__codex` once all four runs
(D1 main + K1 + K2 + K3) have results.json. Numbers in `<<...>>` placeholders
will be replaced by the summarize_d1_controls.py output.

---

```
CP-3 results review for Round 4 D1 (LLM-typed fields injected as new KG edge
types into EmerGNN backbone). CP-1 PASS_WITH_NITS at
`Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-05-30__d1_design__round1.md`
(codex thread 019e7b09). CP-2 PASS_WITH_NITS at
`Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-05-30__d1_implementation__round1.md`
(codex thread 019e7b1b).

Per round4 plan §8.3 CP-3 template. All numbers below are verified from each
run's results.json (file paths in the table).

v2i4 anchor (canonical, run_id 2026-05-29_22-02-27): combined 0.7804,
emergnn 0.7405, count_only 0.6688, i4_only 0.6003, NLL 1.4723.

Four-cell falsification result:

| branch       | D1 main | K1 shuf | K2 rand | K3 drop | v2i4 anchor | Δ vs anchor (D1/K1/K2/K3) |
| ------------ | ------- | ------- | ------- | ------- | ----------- | ------------------------- |
| combined     | 0.7770  | 0.7827  | <<K2_combined>> | <<K3_combined>> | 0.7804 | -0.33/+0.23/<<K2dc>>/<<K3dc>> pp |
| emergnn      | 0.7533  | 0.7495  | <<K2_emergnn>>  | <<K3_emergnn>>  | 0.7405 | +1.28/+0.90/<<K2de>>/<<K3de>> pp |
| count_only   | 0.6818  | 0.7236  | <<K2_count>>    | <<K3_count>>    | 0.6688 | +1.30/+5.48/<<K2dn>>/<<K3dn>> pp |
| i4_only      | 0.5301  | 0.5365  | <<K2_i4>>       | <<K3_i4>>       | 0.6003 | -7.01/-6.37/<<K2di>>/<<K3di>> pp |
| NLL          | 1.2446  | 1.5792  | <<K2_nll>>      | <<K3_nll>>      | 1.4723 | (Δ as decimal: -0.23/+0.11/<<K2dn>>/<<K3dn>>) |

Coverage breakdown (D1 main): both_covered 0.7727 (n=1308+1198), one_covered
0.7807 (n=556+656), neither_covered 0.8179 (n=55+65). Score-distribution
analysis (Notes/Log/round4_d1_seed42_observation.md) showed the neither_covered
AUC is dominated by a lower negative-score mean (0.848 vs 0.890 in
both_covered), so the apparent inversion is a score-separation artifact, not a
real D1-evidence failure.

Run dirs (each has results.json):
- D1 main: Code/runs/2026-05-30_19-05-21__run_v3_llm_edge__d1_main_seed42__seed42
- K1 shuf: Code/runs/2026-05-31_16-08-02__run_v3_llm_edge__d1_shuf_token_seed42__seed42
- K2 rand: Code/runs/<<K2_run_id>>
- K3 drop: Code/runs/<<K3_run_id>>

Design §3 falsification predictions (for reference):

- K1 shuf-token: combined drops ≥ 1.5 pp from D1 main; emergnn drops back to
  ≤ 0.7405. If K1 does NOT drop, lift is capacity-driven, not semantics-driven.
- K2 rand-token (globally-unique per-edge token IDs, zero cross-drug bridges):
  combined drops more than K1; emergnn returns to ≈ 0.7405. If K2 does NOT
  drop, KG-density alone (not bridges) is driving lift.
- K3 drop-i4-head: combined ≥ 0.7804 (anchor) IF backbone genuinely absorbed
  the LLM signal. If K3 < 0.7804, the inherited i4 readout was carrying
  whatever D1 lift exists, not the backbone.

K4 parity (structural) verified in Notes/Log/round4_d1_k4_observation.md
(numerical parity claim withdrawn, structural verified).

Q1 (lift attribution). With the 4-cell table above, walk through each control
and tell me: does D1's +1.28 pp backbone-branch lift come from LLM-evidence
pair-specific semantics, or is it a capacity / KG-density / model-recalibration
artifact? Cite specific Δ-values.

Q2 (K1 surprise). K1 combined is HIGHER than D1 main by 0.57 pp. This violates
the design §3 prediction (combined should drop ≥ 1.5 pp). Two reads in the
observation note (Notes/Log/round4_d1_seed42_observation.md): (a) capacity
confound — bindings don't matter, density does; (b) seed noise (K4 showed ~3 pp
between two structurally identical runs). With K2 + K3 in hand, which read is
more consistent with the data?

Q3 (K2 mechanism). K2 destroys cross-drug bridges (every edge gets a unique
synthetic token). If K2 combined and emergnn drop substantially below K1, the
mechanism is bridge-routing. If K2 ≈ K1, the mechanism is per-token degree /
local KG density. Tell me which.

Q4 (K3 fusion saturation). D1 main saw i4_only collapse from 0.6003 to 0.5301
(−7 pp). K3 freezes β_i4=0. Did combined increase when i4 readout is removed?
If yes, the i4 readout was actively HURTING combined in D1 main (noise-injection
into additive fusion). If no, the i4 readout's contribution was non-negative.

Q5 (capacity vs novelty). Even if Q1 returns "capacity confound", is the
architectural change still paper-publishable as "KG structural augmentation via
auto-generated pharma-typed entities lifts cold-start AUC"? Or does the
capacity attribution kill the novelty claim entirely?

Q6 (NLL movement). D1 main improved NLL by −0.23 (1.47 → 1.24). K1 made NLL
WORSE (+0.11 vs anchor). What does this dissociation tell us? Cross-ref
Notes/Log/paper_writeup.md §3.X CACR.

Q7 (multi-seed go/no-go). Given the 4-cell table, is it worth burning 2 more
GPU-hours on seed=43 + seed=44 multi-seed runs to nail down whether the small
positive Δ (+0.57 pp combined for K1) is signal or noise? OR should we declare
the result negative and pivot to D2 / D3 instead?

Q8 (paper section). Is this evidence base sufficient to draft a paper section
(Notes/Log/paper_writeup.md Section 3.W1)? If yes, suggest the
one-sentence headline summary. If no, what additional run is required?

verdict ∈ {PASS, PASS_WITH_NITS, NOT_PASS, FAIL}. Each issue critical / major /
minor + run_id / file:line where applicable. PASS = 0 critical + 0 major per
round4 plan §8.4. The codex CP-1/CP-2 PASS criteria did NOT require D1 to
actually be a positive lift, so a NOT_PASS at CP-3 ONLY for the original D1
thesis is acceptable — the design + implementation are still correct, the
mechanism just wasn't what we hypothesized.
```

---

## Trigger checklist before sending

- [ ] All four runs have results.json. Verify via `summarize_d1_controls.py`.
- [ ] Fill all `<<...>>` placeholders with verified numbers.
- [ ] Choose codex sandbox: `read-only` (no edits expected during CP-3).
- [ ] Send via `mcp__codex__codex` with `model` left to MCP default.
- [ ] Archive verdict + back-and-forth to
      `_reviews/2026-05-30__d1_results__round1.md`.
