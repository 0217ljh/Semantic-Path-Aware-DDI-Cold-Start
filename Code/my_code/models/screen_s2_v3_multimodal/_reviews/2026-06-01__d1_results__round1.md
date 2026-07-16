# CP-3 Results Review — D1 (LLM mechanistic fields → new KG edge types)

## Metadata

- **Date**. 2026-06-01
- **Scope**. CP-3 results review for Round 4 direction D1, per round4 plan §8.3 CP-3 template. CP-1 PASS_WITH_NITS at `_reviews/2026-05-30__d1_design__round1.md`; CP-2 PASS_WITH_NITS at `_reviews/2026-05-30__d1_implementation__round1.md`.
- **Primary reviewer**. Claude (claude-opus-4-7), drafted this report + ran analysis.
- **Independent reviewer**. codex (codex MCP default model). Codex thread: `019e847b-1d88-7750-9bd0-6a6a227c7706`.
- **Final verdict**. **NOT_PASS** for the original D1 mechanism claim. CP-1/CP-2 stand; hypothesis failed under controls as the CP process was designed to detect.
- **Decision**. Archive D1 as a negative/diagnostic subsection in `paper_writeup.md`. SKIP D1 multi-seed. PIVOT to D2 (meeting-node-aware propagation), starting fresh CP-1 there.

## Four-cell falsification result (verified)

All numbers from each run's results.json (file paths below). Δ in percentage points (pp).

| branch     | D1 main | K1 shuf | K2 rand | K3 drop | v2i4 anchor | Δ vs anchor (D1/K1/K2/K3) |
|------------|---------|---------|---------|---------|-------------|---------------------------|
| combined   | 0.7770  | 0.7827  | 0.7778  | 0.7779  | 0.7804      | −0.33/+0.23/−0.25/−0.25   |
| emergnn    | 0.7533  | 0.7495  | 0.7477  | 0.7430  | 0.7405      | +1.28/+0.90/+0.72/+0.25   |
| count_only | 0.6818  | 0.7236  | 0.6950  | 0.7221  | 0.6688      | +1.30/+5.48/+2.62/+5.34   |
| i4_only    | 0.5301  | 0.5365  | 0.5731  | 0.4747  | 0.6003      | −7.01/−6.37/−2.72/−12.55  |
| NLL        | 1.2446  | 1.5792  | 1.5274  | 1.5397  | 1.4723      | −0.228/+0.107/+0.055/+0.067 |

Coverage breakdown (D1 main only): both_covered 0.7727 (n_pos=1308, n_neg=1198), one_covered 0.7807 (n_pos=556, n_neg=656), neither_covered 0.8179 (n_pos=55, n_neg=65). Score-distribution analysis in `Notes/Log/round4_d1_seed42_observation.md` shows the neither_covered AUC is dominated by lower negative-score mean (0.8477 vs 0.8898 in both_covered), a score-separation artifact, not a real evidence failure.

Run dirs (each has results.json):
- D1 main: `Code/runs/2026-05-30_19-05-21__run_v3_llm_edge__d1_main_seed42__seed42`
- K1 shuf: `Code/runs/2026-05-31_16-08-02__run_v3_llm_edge__d1_shuf_token_seed42__seed42`
- K2 rand: `Code/runs/2026-05-31_22-17-53__run_v3_llm_edge__d1_rand_token_seed42__seed42`
- K3 drop: `Code/runs/2026-05-31_22-15-57__run_v3_llm_edge__d1_drop_i4head_seed42__seed42`

## Design §3 falsification predictions vs observed

- K1 expected combined drop ≥ 1.5 pp + emergnn ≤ 0.7405. **OBSERVED**: combined drop = −0.56 pp (went UP); emergnn = 0.7495. **FAIL**.
- K2 expected combined drop more than K1 + emergnn ≤ 0.7405. **OBSERVED**: combined drop = −0.08 pp (barely changed); emergnn = 0.7477. **FAIL**.
- K3 expected combined ≥ 0.7804. **OBSERVED**: combined = 0.7779. **FAIL**.

## Codex independent verdict (round 1; PASTED VERBATIM per CLAUDE.md §"Code review 报告归档规范" point 6)

> **Verdict: NOT_PASS** for the original D1 mechanism claim. Not FAIL: the runs and controls are useful, and CP-1/CP-2 still stand. But D1 does not pass CP-3 as "LLM pair-specific semantic evidence injected into the backbone."
>
> **Q1. Lift Attribution**
> D1 main's +1.28 pp emergnn lift cannot honestly be attributed to pair-specific LLM semantics. K1 destroys drug-token binding and still has emergnn 0.7495, +0.90 pp over anchor. K2 destroys cross-drug token bridges and still has emergnn 0.7477, +0.72 pp over anchor. That means the backbone improvement survives both semantic binding destruction and bridge destruction. The load-bearing mechanism is not "drug A and drug B share meaningful LLM-derived entities"; it is some shuffle-invariant effect of adding typed edges/entities/relations to the KG.
>
> **Q2. Capacity Confound Seriousness**
> Capacity/KG-density is the leading explanation, but not the only possible one. A third possibility is marginal-structure signal: per-drug relation counts, field coverage, token-degree distribution, altered relation-table size, changed initialization dynamics, or regularization effects from forcing EmerGNN through a larger typed graph. K2 preserving edge count and relation labels while removing sharing is especially damaging to the semantic-bridge story: if K2 still lifts, even cross-drug LLM token bridges are not necessary. So the honest claim is "D1 perturbs the KG/backbone in a useful way," not "D1 routes semantic LLM evidence."
>
> **Q3. Backbone vs Readout**
> K3 shows the i4 readout is not carrying D1's combined result. Freezing `β_i4≈0` gives combined 0.7779 versus D1 main 0.7770, effectively unchanged. The i4 head should be framed as redundant or destabilized under D1, not as the source of lift. The D1 backbone appears to absorb or displace whatever weak LLM-readout signal existed, while the standalone `i4_only` head collapses from 0.6003 to 0.5301. That is not evidence that i4 became useless globally; it is evidence that in the D1 training regime, the readout head loses discriminative ranking power and does not improve fusion.
>
> **Q4. NLL Movement**
> The NLL improvement in D1 main is real and is the healthiest D1 signal, but it is dissociated from AUROC. Combined AUC slightly regresses while NLL improves sharply, matching the CACR lesson: calibration/log-likelihood can improve without ranking improvement. K1 is important because it reverses this: K1 combined rises to 0.7827, but NLL worsens to 1.5792. So the paper story should say D1 sharpened probability calibration only when the original LLM-token binding was preserved, even though that did not translate into ranking lift. That is a secondary result, not a main DDI ranking contribution.
>
> **Q5. Paper Story Salvage**
> D1 is publishable only as a negative/diagnostic subsection, not as a positive method. A fair headline would be: "Auto-generated pharma-typed KG augmentation improves backbone behavior and calibration in some settings, but falsification controls show the ranking lift is not attributable to pair-specific LLM semantics." I would not include D1 as a claimed model improvement. It can be valuable beside CACR as a "what failed and why" result, especially if the paper emphasizes mechanistic falsification rather than leaderboard gain.
>
> **Q6. Multi-Seed Go/No-Go**
> Do not spend GPU on D1/K1/K2/K3 multi-seed for the original claim. The observed D1-vs-control differences are around 0.3-0.6 pp, while K4 exposed a much larger single-seed/no-seed noise concern in short runs. Multi-seed might estimate the tiny differences more cleanly, but it cannot rescue the core falsification: K1 and K2 did not directionally collapse toward anchor. Unless the paper specifically needs confidence intervals for the negative result, stop here.
>
> **Q7. Pivot Decision**
> D1 failed because the mechanism is mostly shuffle-invariant. That points away from more global KG-density augmentation and toward explicitly pair-conditional mechanisms. D2 is the best next pivot among D2/D3/D4 because meeting-node-aware propagation injects pair-specific mediator masks directly into the backbone, and its controls can test mediator identity versus cardinality. D4 is also conceptually aligned if relation routing is pair-conditional. D3 alignment/init is weaker as the next step because initialization effects may again become global and shuffle-invariant unless the loss is tightly pair-conditioned.
>
> **Q8. CP-1 / CP-2 Retroactive Verdict**
> Do not retroactively modify CP-1 or CP-2. They validated design integrity, no obvious leakage, real backbone modification, and proper controls. Those were correct. The hypothesis failed under the controls, which is exactly what the CP process was supposed to reveal. The only process improvement I would add is to require deterministic seeding before parity/noise-floor claims, because K4's numeric parity target was too strong without `torch.manual_seed` and `np.random.seed`.
>
> Final recommendation: mark D1 CP-3 as **NOT_PASS**, archive it as a useful negative/calibration result, skip D1 multi-seed, and pivot to CP-1 for D2 with pair-specific shuffle/random-mediator controls built in from the start.

## Issues + actions

| Severity | Issue | Action |
|---|---|---|
| critical | D1's original mechanism claim ("LLM mechanistic fields convey pair-specific semantic evidence to the backbone via path-flow") is falsified by K1+K2. The +1.28pp emergnn lift survives shuffling and bridge-removal — it is shuffle-invariant, therefore not pair-specific. | Withdraw the original D1 paper claim. Reframe as negative/diagnostic subsection beside CACR (paper_writeup.md §3.X analogue). |
| major | All three controls (K1/K2/K3) failed their design §3 falsification predictions. The combined-AUC deltas are ≤ 0.6 pp at single seed; below the seed-noise floor exposed by K4. | Skip multi-seed for D1; will not rescue the verdict. |
| minor | `i4_only` AUC collapsed from 0.6003 to 0.5301 in D1 main and to 0.4747 in K3 (β_i4 frozen). Standalone i4 head lost discriminative ranking power under D1 training regime. | Note in writeup. The i4 readout's reduced discriminativeness is consistent with the codex finding that the head is destabilized but not load-bearing in D1. |
| process minor | CP-1's K4 numerical parity claim ("mean abs diff ≤ 1e-4 per-epoch loss") was too strict; could not be met without explicit `torch.manual_seed`. Already withdrawn in `Notes/Log/round4_d1_k4_observation.md`. | Add deterministic-seeding requirement to future CP-1 templates that include parity claims. |

## Followup pending (autonomous loop continues)

- ✅ Archive D1 result in `paper_writeup.md` Section 3.X-style negative/diagnostic subsection (next).
- ⏭ Start fresh CP-1 for D2 using `Notes/Log/d2_meet_mask_design_sketch.md` as the design draft, refined to include shuffle-mediator + random-mediator controls from the start (per codex Q7 guidance).
- ⏸ Skip D1 multi-seed (codex Q6).
- ⏸ Defer D3 (alignment InfoNCE init) and D4 (pair-conditional relation routing) until D2 result known.
