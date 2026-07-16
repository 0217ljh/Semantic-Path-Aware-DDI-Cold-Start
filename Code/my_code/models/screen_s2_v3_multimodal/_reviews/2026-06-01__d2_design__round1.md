# CP-1 Design Review — D2 (Meeting-Node-Aware Propagation)

## Metadata

- **Date**. 2026-06-01
- **Scope**. CP-1 design review for Round 4 direction D2, per round4 plan §8.3. Triggered by D1 CP-3 NOT_PASS verdict (`_reviews/2026-06-01__d1_results__round1.md`).
- **Primary reviewer**. Claude (claude-opus-4-7), drafted the design + this report.
- **Independent reviewer**. codex (codex MCP default model). Codex thread: `019e8482-0fb7-7542-ab5e-43c9fae00a9d`.
- **Final verdict**. **PASS_WITH_NITS** at codex round 3 (after round 1 NOT_PASS with 5 majors + 1 minor, round 2 NOT_PASS with 2 majors + 2 minors).
- **Decision**. Proceed to CP-2 implementation. Three round-3 minor fixes (K1 "EXACTLY preserved" qualifier, §8 Q4 wording sync, K2a fallback ladder) already applied before this archive.

## Reviewed artifacts

- `Notes/Log/d2_meet_mask_design.md` (full design, replaces earlier `Notes/Log/d2_meet_mask_design_sketch.md`).
- `Notes/Log/round4_d1_seed42_observation.md` (D1 falsification: K1/K2/K3 all failed → motivation for D2's pair-conditional pivot).
- `_reviews/2026-06-01__d1_results__round1.md` (CP-3 verdict on D1; Q7 explicitly recommended D2).
- Code touched at the line level: `baseline/emergnn/model.py:116-210`, `mnah_trainer.py:329-372`, `v2i4_trainer.py`, `precompute_meet_features.py:81-185`, `refine-logs/LEAKAGE_AUDIT.txt:22, 33`.

## Contribution list

| # | Claim | Verified at design level? |
|---|---|---|
| C1 | Backbone change is pair-conditional and non-readout-only (mediator mask depends on a_b AND b_b jointly; bonus flows only through path-flow). | PASS (codex Q1) |
| C2 | No cold-start leakage; D2 builder applies same n1/n2 drug-drug-edge filter as MNAH; cardinality + leakage audit run at builder time (NOT transitively inherited from 22-d count audit). | PASS (codex round 2) |
| C3 | Two structural-parity guards (`--d2-disable`, `--d2-freeze-alpha-meet`) + one runtime guard (§6 hard-stop). `load_best_model_at_end` NOT cited as a safety guard. | PASS (codex round 2) |

## Codex independent verdict — round-by-round timeline

### Round 1 (NOT_PASS, 5 majors + 1 minor)

> Major 1: C3 overclaims with `load_best_model_at_end`.
> Major 2: C2 does not transitively inherit 22-d count audit (D2 uses union+cross-hop, not n1∩n1 + n2∩n2).
> Major 3: Audit "no drug-drug edges" was misleading — safety comes from builder filtering, not from mask1 itself (4855 het:CrC remain).
> Major 4: K1/K2 unclean falsifiers (K1 "approximately preserves cardinality"; K2 uniform random samples wrong-kind/unreachable entities).
> Major 5: R1 cardinality threshold contradicted between sections.
> Minor: Cache universe should match MNAH's Cartesian-pair builder, not just observed pairs.

### Round 2 (NOT_PASS, 2 majors + 2 minors)

> Major 1: Stale §3.5 CLI still exposed old "preserve cardinality on average" K1 and old uniform-random K2; control contract internally contradictory.
> Major 2: R1 threshold appeared 3 times in 3 different forms.
> Minor 1: Cold-start safety wording mis-states builder reads.
> Minor 2: LEAKAGE_AUDIT.txt line references off (14/35 vs actual 22/33).

### Round 3 (PASS_WITH_NITS, 3 minors)

> The blocking contradictions are resolved. The design is now clean enough to move into implementation/CP-2.
>
> Remaining nits:
> - K1 says "per-pair cardinality EXACTLY preserved" globally, but singleton-bucket pairs fall back to ±1 cardinality. Should clarify.
> - §8 Q4 still has old "approximately preserved" / "uniform random" wording in the question prompt.
> - K2a should define fallback behavior when no same-kind ±20% degree candidate exists.
>
> No critical or major issues remain. CP-1 can proceed as PASS_WITH_NITS under round4 §8.4.

## Issues + resolutions

All issues from round 1, 2, 3 + resolution status. Round-3 minor fixes were applied BEFORE archiving this report.

| Severity | Issue | Resolution | Resolved at |
|---|---|---|---|
| major (R1) | C3 overclaim — `load_best_model_at_end` cannot restore initial v2i4 state | Removed; replaced with `--d2-disable` + `--d2-freeze-alpha-meet` + §6 hard-stop | Round 2 |
| major (R1) | C2 — 22-d count audit does NOT transitively cover D2 (union+cross-hop expanded) | Added explicit D2-specific leakage + cardinality audit at builder time; cited differing mediator universe | Round 2 |
| major (R1) | Audit safety claim — builder filtering, not mask1, must be cited | Rewrote C2 to credit `src_is_drug and not dst_is_drug` filter at `precompute_meet_features.py:97-104`; documented 4855 residual het:CrC edges | Round 2 |
| major (R1) | K1/K2 unclean falsifiers | K1 → cardinality-bucketed derangement; K2 → split into K2a (degree+kind-matched, fallback ladder) + K2b (uniform harsher) | Round 2 + minor 3 in Round 3 |
| major (R1) | R1 threshold contradictions | Single threshold contract: `mean > 50` OR `p95 > 200` OR `max > 1000` triggers top-K filtering, repeated at all 3 sites with cross-references | Round 2 |
| minor (R1) | Cache universe doesn't match MNAH's | Fixed: D2 builder cache covers train + val_s2 + test_s2 + negatives + train_negatives epoch_0, per `precompute_meet_features.py:167-185` pattern | Round 2 |
| major (R2) | Stale §3.5 CLI semantics conflicted with §4 control definitions | §3.5 rewritten: `--d2-disable / --d2-freeze-alpha-meet / --d2-shuf-mediators / --d2-rand-mediators-kind-matched / --d2-rand-mediators-uniform / --d2-alpha-init / --deterministic / --d2-shuffle-seed`; old CLI explicitly removed | Round 3 |
| major (R2) | R1 threshold appeared in 3 contradictory forms | Single threshold (mean>50 / p95>200 / max>1000) propagated across §3.2 step 4, §3.2 step 5, §6 R1 with cross-references | Round 3 |
| minor (R2) | "Cold-start safety" wording mis-states builder reads | Reworded to clarify split-table reads are pair-universe enumeration only (drug_a_id, drug_b_id columns) | Round 3 |
| minor (R2) | LEAKAGE_AUDIT.txt line references off | Fixed 14→22 (drug-drug edges) and 35→33 (mean mediator count) | Round 3 |
| minor (R3) | K1 "EXACTLY preserved" overclaim re singleton fallback | Qualified: "exact for bucket>1 derangement; ±1 may apply for singleton-bucket merged fallback, count logged" | Round 3 |
| minor (R3) | §8 Q4 stale wording | Rewrote to reference K1 / K2a / K2b / K3 / K4 by the round-2-final definitions | Round 3 |
| minor (R3) | K2a fallback when no same-kind ±20% degree candidate | Added fallback ladder: ±20% → ±50% → any-degree same kind → K2b uniform fallback, with per-level count logged | Round 3 |

## Unresolved / followup flagged

- **CP-2 must run a builder cardinality probe before declaring PASS**. If `mean n_mediators > 50` OR `p95 > 200` OR `max > 1000`, the §6 one-sweep allowance is pre-committed to top-K filtering (preliminary K=20).
- **Deterministic seeding is on by default in D2** (`--deterministic`). This sharpens K4 numerical parity to within 1e-4 (vs D1 which had to withdraw the parity claim per `Notes/Log/round4_d1_k4_observation.md`).
- **Capacity confound re-emergence remains the biggest R4 risk**. D2 adds only L=3 floats so it can't be raw capacity, but the mask could act as content-agnostic node-level regularization. K1 + K2a + K2b together must rule this out.

## Next step

CP-2 implementation: builder + EmerGNN subclass + trainer + run script (Task #13). Then CP-2 codex implementation review (Task #14). Then D2 runs + CP-3 (Task #15).
