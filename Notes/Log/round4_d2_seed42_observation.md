# Round 4 D2 — seed42 main result observation (pre-CP-3 staging)

**Date**. 2026-06-01
**Status**. Preliminary observation. CP-3 codex review will produce the formal verdict; this is intermediate analysis Claude is doing while K1 + K3 controls run.

## Run identity

- run_id. `2026-06-01_15-46-08__run_v3_meet_mask__d2_main_seed42__seed42`
- Verified config from results.json: seed=42, epochs=100, batch_size=32, length=3, n_dim=64, kg_source=drugbank, mediator_parquet=`Code/data/_cache/meet_mediators/meet_mediators__seed42_drugbank__topk20.parquet` (Path B, sha16=ac2b6b345f4264dd), d2_alpha_init=0.0, deterministic=True.
- Wall time. To verify from results.json.

## Headline numbers (vs v2i4 anchor 2026-05-29_22-02-27)

| Branch | D2 main | v2i4 anchor | Δ |
|---|---|---|---|
| combined | **0.7780** | 0.7804 | **−0.24 pp** |
| emergnn (backbone) | **0.7344** | 0.7405 | **−0.61 pp** |
| count_only (i2) | 0.7172 | 0.6688 | +4.84 pp |
| i4_only (i4 readout) | 0.5506 | 0.6003 | −4.97 pp |
| NLL | 1.3227 | 1.4723 | −0.15 |

## Cardinality-stratified breakdown (the key diagnostic)

Per `run_v3_meet_mask.py:128` quartile binning, the test_s2 pair set splits into 2 quartile buckets (the high-zero-fraction distribution collapses the lower quartiles together):

| Bucket | n_mediators range | n_pos | n_neg | AUC |
|---|---|---|---|---|
| q1 (mask empty) | [0, 0] | 900 | 1643 | **0.6767** |
| q2 (mask non-empty, 1-18) | [1, 18] | 1019 | 276 | **0.7218** |

So in pairs where D2's mask provides a signal (q2, ~30% of pairs), AUC is 0.7218 — notably higher than the empty-mask cohort. **In pairs where the mask is empty (~70% of pairs), AUC is 0.6767** — D2 cannot contribute. The overall combined of 0.7780 is the weighted average over these two regimes.

Compare this to v2i4 anchor 0.7804 which scores uniformly (no per-pair D2 mechanism). D2 trades signal asymmetry against v2i4's uniform 0.7804.

## Hard-stop verdict

LIMITED SWEEP per round4 plan §6. combined 0.7780 is in [0.775, 0.785) band — above the 0.775 hard stop but below the 0.785 multi-seed bar.

Action launched (per `Notes/Log/round4_d2_run_plan.md`):
- K1 shuf-mediators (`d2_shuf_mediators_seed42`) — falsification
- K3 freeze-alpha (`d2_freeze_alpha_seed42`) — architectural parity sanity

K2b deferred unless verdict shifts to ≥ 0.785 after K1.

## Key observations

**1. emergnn-branch REGRESSED**. D2's stated goal was to lift emergnn-branch via pair-conditional propagation bonus. Observed: emergnn dropped from 0.7405 (v2i4) to 0.7344 (D2 main, −0.61pp). This is the inverse of the desired direction. Possible reads:

- The alpha_meet bonus (learned to ~0.147 for layer 0, ~−0.033 for layer 1, 0.0 for layer 2 per last-epoch log) shifts the propagation in a way that helps q2 pairs but hurts q1 pairs uniformly, lowering the overall backbone-branch AUC.
- 70% of training pairs have empty masks; the model may have learned to use the bonus path opportunistically for q2 but neglected backbone quality on q1.
- Layer-0 alpha_meet = +0.147 is non-trivial; layer-1 alpha_meet = −0.033 is small but nonzero; layer-2 alpha_meet = 0.0 (stayed at init). The model is using the bonus mostly at the first layer, then suppressing it slightly at layer 1.

**2. count_only branch JUMPED +4.84pp**. The 22-d count readout head got much more discriminative. The model rebalanced toward the count head when D2 introduced backbone-branch perturbation. This is similar to the D1 main behavior but in the opposite direction (D1 had count_only +1.30pp; D2 has +4.84pp).

**3. i4_only branch DROPPED −4.97pp**. Same as D1 — the LLM-pair-feature readout becomes less discriminative under the new propagation regime. Possibly the model can predict i4-style mechanism overlap better via the count head + bonus path, so the i4 MLP "unlearned" some discriminative function.

**4. NLL dropped −0.15**. Calibration improved (similar to D1's −0.23). Consistent with the project's CACR-pattern observation that adding architectural prior can sharpen calibration without lifting ranking AUC.

## What K1 + K3 will tell us

K1 expected behavior (if D2 mechanism is pair-conditional):
- combined drops ≥ 1.5pp (vs D2 main)
- emergnn-branch drops back to ~v2i4 anchor's 0.7405 or lower

If K1 doesn't drop, D2 inherits D1's capacity-confound diagnosis. Given the empty-mask 70% reality, K1's binding-destruction should especially hurt the q2 cohort (where D2 has signal), so K1 should be detectable.

K3 expected behavior (if architecture is sane):
- combined ≈ v2i4 anchor 0.7804 (mathematical equivalence — alpha_meet=0 zeros the bonus path)
- The actual difference will tell us how much the "extra L=3 alpha_meet floats" affect the model's gradient dynamics even when frozen at 0

## Open question for CP-3

The 30% mask-coverage reality (Path B intrinsic) means D2 can only contribute to a minority of S2 pairs. Is the q2-cohort lift (0.7218 vs 0.6767 contrast) paper-publishable as "D2 helps when mediator structure exists, neutral when it doesn't"? Or does the combined-level regression invalidate that story?

Codex CP-3 will weigh in once all 4-cell data is available (D2 main + K1 + K3; K2b deferred to FULL CONTROLS unless verdict shifts).
