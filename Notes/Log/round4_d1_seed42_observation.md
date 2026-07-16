# Round 4 D1 — seed42 main result observation (pre-CP-3 staging)

**Date**. 2026-05-30
**Status**. Preliminary observation note. CP-3 codex review will produce the formal verdict; this is intermediate analysis Claude is doing while K1 control runs.

## Run identity

- run_id. `2026-05-30_19-05-21__run_v3_llm_edge__d1_main_seed42__seed42`
- Config (verified from results.json): seed=42, epochs=100, batch_size=32, length=3, n_dim=64, kg_source=drugbank, edges=`Code/data/_cache/llm_edges/llm_drug_edges__seed42_drugbank.parquet` (sha16=b3c524a26dfb48f3).
- Injection summary: `n_edges_input=23003, n_edges_dropped_out_of_pool=558, n_llm_edges_injected=22445, n_new_relations=10, n_old_relations=5, n_new_token_nodes=7632, n_ent_before=5633, n_ent_after=13265, n_base_rel_before=5, n_base_rel_after=15`.
- Wall time. 3204s = 53.4 min (vs v2i4 anchor 1848s = 30.8 min; ~1.7× slowdown matches the design §5 R5 risk projection).

## Headline numbers (vs v2i4 anchor 2026-05-29_22-02-27)

| Branch | D1 main | v2i4 anchor | Δ |
|---|---|---|---|
| combined | 0.7770 | 0.7804 | −0.33 pp |
| emergnn (backbone) | **0.7533** | 0.7405 | **+1.28 pp** |
| count_only (i2 readout) | 0.6818 | 0.6688 | +1.30 pp |
| i4_only (i4 readout) | 0.5301 | 0.6003 | **−7.01 pp** |
| AUPRC | 0.7841 | 0.7922 | −0.81 pp |
| NLL | 1.2446 | 1.4723 | −0.228 (calibration better) |

Coverage-stratified:
| Bucket | D1 main AUC | n_pos | n_neg |
|---|---|---|---|
| both_covered | 0.7727 | 1308 | 1198 |
| one_covered | 0.7807 | 556 | 656 |
| neither_covered | 0.8179 | 55 | 65 |

## Hard-stop verdict

LIMITED SWEEP per round4 plan §6. combined 0.7770 is in [0.775, 0.785) — above the 0.775 stop but below the 0.785 multi-seed bar. Action: K1 shuf-token launched (run_id pending, `tag=d1_shuf_token_seed42`); will not multi-seed until controls justify it.

## Key observations (pre-CP-3)

**The backbone DID absorb the LLM signal**. The emergnn-branch lift of +1.28 pp (0.7405 → 0.7533) is the load-bearing evidence that the 22,445 new drug→token edges are flowing through EmerGNN's path-flow propagation and lifting the backbone's pair score. This is the architectural-novelty signal we set out to produce. It hits below the design-doc target of ≥ +1.5 pp (auc_emergnn ≥ 0.755) but is non-trivially positive.

**Combined did NOT lift; arguably regressed slightly**. combined 0.7770 < v2i4 0.7804 by 0.33 pp. The architectural change is real but the additive fusion saturated against the inherited v2i4 readout heads. This is the same family of phenomenon documented in `Notes/Log/paper_writeup.md` §3.Z Stage 2: "branch lift but additive logit fusion translates only a fraction of the branch-level gain into combined ranking improvement". There it was +4.55 pp aux branch → +0.52 pp combined. Here it is +1.28 pp backbone branch → −0.33 pp combined. The fusion ceiling sign is even worse for D1 because the LLM substrate is now reaching `combined_logit` through TWO paths (the new backbone integration + the inherited i4 readout head), and the optimizer rebalances them in a way that does not produce net gain.

**The i4 readout head collapsed**. i4_only AUC dropped from 0.6003 to 0.5301 (−7.01 pp), the largest single-branch shift. This is consistent with the optimizer reallocating decision capacity from the readout-side i4 head to the backbone branch once the backbone could route LLM evidence on its own. Note: i4_only AUC is rank-equivalent to i4_logit alone (scaling by softplus(β_i4) does not change ranks), so this is the i4 MLP itself becoming less discriminative on pair features, not just β_i4 shrinking. The 13-d pair features are still informative (v2i4 demonstrated 0.6003), so the readout MLP "unlearned" a useful function in favor of letting the backbone do the work. Adding a per-epoch `β_i4` log to the trainer would have made this diagnosis sharper; flagged as a CP-2-deferred minor.

**Count readout lifted too** (+1.30 pp), unrelated to D1 directly. Probably an indirect effect of the modified entity vocabulary and shared optimizer state with the now-stronger backbone. Not a confound for D1's main claim.

**NLL fell sharply** (1.4723 → 1.2446, Δ −0.228). Cross-reference to `paper_writeup.md` §3.X CACR — there we saw NLL ↓ without AUROC ↑ and concluded that calibration is decoupled from ranking. Same here. The NLL improvement is a useful secondary finding but not a primary contribution.

**Coverage breakdown is inverted at small n; explained by negative-score distribution shift.** Apparent pattern: both_covered 0.7727 < one_covered 0.7807 < neither_covered 0.8179. But the score-distribution breakdown (verified 2026-05-30 from `test_s2_scores.npz`) reveals this is a score-separation artifact, not a real D1 evidence failure.

| bucket | mean(pos) ± std | mean(neg) ± std | separation | n_pos | n_neg | AUC |
|---|---|---|---|---|---|---|
| both_covered | 0.9504 ± 0.058 | 0.8898 ± 0.062 | 0.061 | 1308 | 1198 | 0.7727 |
| one_covered | 0.9359 ± 0.059 | 0.8715 ± 0.059 | 0.064 | 556 | 656 | 0.7807 |
| neither_covered | 0.9299 ± 0.061 | **0.8477 ± 0.064** | **0.082** | 55 | 65 | 0.8179 |

The neither_covered AUC is higher because its NEGATIVE score mean is lower (0.8477 vs 0.8898 in both_covered), not because positives are scored better. Pairs in neither_covered are pairs where both drugs are LLM-cache-uncovered, i.e. less-studied drugs. These drugs have smaller KG neighborhoods (the LLM substrate concentrates on well-characterized drugs), and the model assigns negatives among them a cleaner low score because there is simply less spurious evidence to score on. The both_covered AUC of 0.7727 and one_covered AUC of 0.7807 are statistically indistinguishable from each other and from v2i4's combined 0.7804.

So coverage breakdown does NOT falsify D1's evidence story; it also does not validate the "both > one > neither" prediction. The bucket-level AUCs are roughly flat (~0.78) on the well-powered buckets (both/one) and ~0.82 on the noisy small bucket (neither, CI half-width ≈ 0.07 at p=0.82). CP-3 should treat the coverage table as **uninformative for or against D1**, not as a falsification.

The original two reads I outlined need updating: Two possible reads:

1. The LLM substrate's pair-specific information is already substantially captured by the inherited v2i4 readout heads (counts + 13-d i4 head), so adding LLM-as-edges does not add fresh ranking gradient for the pairs the readout already serves. The backbone branch lift is real but it just shifts which path carries the signal, not how much signal reaches the combined logit.
2. The LLM substrate is itself weakly informative on this dataset's S2 split — the +1.28 pp backbone lift is modest, near the noise floor we observed in K4 parity (~3 pp variation between two structurally-identical 5-epoch runs).

K1 shuf-token will adjudicate: if K1 collapses the backbone branch back near 0.7405, the +1.28 pp lift is real LLM evidence; if K1 stays near 0.7533, the lift is from capacity / KG-density and the architectural-novelty claim weakens.

## Open questions for CP-3

- Q. Is the backbone-branch lift of +1.28 pp paper-worthy given the combined regression?
- Q. Does the coverage-inverted pattern (both ≯ one ≯ neither) falsify the D1 claim? Or is n=120 on neither_covered too small to draw a firm conclusion?
- Q. Should we pursue a Stage-3-style fix that **forces the combined fusion to use the backbone evidence instead of the inherited i4 head** (e.g. drop i4 head from the very start as a permanent ablation, not just K3 control)?
- Q. What hyperparam sweep would round4 plan §6 allow (we have one sweep budgeted at this band)? Candidates: (a) per-open-vocab field cap K=200 most-frequent tokens (the design doc §5 R1 mitigation), (b) length=4 (longer path budget for new edges to reach further drugs), (c) higher β_i4 init to keep readout from collapsing.
- Q. Is the NLL improvement an independent contribution worth a paragraph (cross-reference §3.X CACR), or just a sidebar?

## What this does NOT change

- D1's architectural backbone integration mechanism is operationally validated (backbone-branch +1.28 pp confirms LLM edges are flowing through path-flow).
- The codex CP-1 and CP-2 verdicts both stand. CP-2 specifically said combined-only metric is confounded by inherited i4 readout, so backbone-branch + control runs are the load-bearing evidence — that framing aligns with what we observe.
- The K4 parity observation in `Notes/Log/round4_d1_k4_observation.md` is independent of this.
- Hard-stop rule was respected.

## K1 shuf-token result (2026-05-31, verified)

run_id `2026-05-31_16-08-02__run_v3_llm_edge__d1_shuf_token_seed42__seed42`.

| Branch | K1 (shuf-token) | D1 main | v2i4 anchor | K1 vs D1 main | K1 vs anchor |
|---|---|---|---|---|---|
| combined | **0.7827** | 0.7770 | 0.7804 | **+0.57 pp** | +0.23 pp |
| emergnn | 0.7495 | 0.7533 | 0.7405 | −0.38 pp | +0.90 pp |
| count_only | 0.7236 | 0.6818 | 0.6688 | +4.18 pp | +5.48 pp |
| i4_only | 0.5365 | 0.5301 | 0.6003 | +0.64 pp | −6.38 pp |

Coverage: both 0.7833, one 0.7758, neither 0.8190 (essentially the same shape as D1 main).

**D1 FALSIFIED by its own design §3 criterion.** Design doc §3 K1 prediction:

> Expected: combined AUROC drops by ≥ 1.5 pt below the canonical D1 run. emergnn-branch AUC drops at least back to the v2i4 anchor's 0.7405 (or lower). If shuf-token does NOT drop, the canonical D1 lift was likely capacity-driven, not semantics-driven → NOT_PASS at CP-3.

Actual: K1 combined is **higher** than D1 main by +0.57 pp; K1 emergnn-branch is barely different from D1 main (−0.38 pp), nowhere near falling back to the v2i4 anchor's 0.7405. So the backbone-branch lift in D1 main is invariant to drug→token binding. This means it is NOT carrying pair-specific LLM evidence; it is carrying KG-density / capacity / generic structure invariant to the binding.

Three reads (CP-3 will need to discriminate):

1. **Pure-capacity confound.** The 22,445 new drug→token edges enrich the KG by ~5x relative to drugbank-5-bucket's edge count, and the EmerGNN backbone uses this density to learn better representations (perhaps via deeper neighborhood coverage during path-flow). Shuffling drug→token binding preserves density and per-token degree, so the capacity stays available. Under this read, D1 has architectural novelty (yes the backbone uses new edges) but the novelty is content-blind — the LLM information itself is not the value-add. Reviewer story: "we add KG structural augmentation via auto-generated 7k+ pharma-typed entities; it lifts the backbone +1.28 pp; but the substrate content is not load-bearing."

2. **Seed noise.** The K4 parity observation showed ~3 pp variation in 5-epoch test AUC between two structurally identical runs. K1 vs D1 main differ by 0.57 pp combined / 0.38 pp emergnn — that is well within K4-observed noise. So the result is "consistent with no effect" at single-seed resolution. We would need ≥ 3 seeds to distinguish "tiny capacity lift survives shuffle" from "no lift at all".

3. **Wrong falsification target.** The design's K1 (shuffle drug→token binding while preserving token-level structure) does not actually break the most plausible loss-of-signal path. If the backbone uses each token node's degree as a feature (not the specific drugs attached to it), then K1 is a no-op for the backbone's signal. K2 (rand-token, every edge gets a globally unique synthetic token) would be the genuine bridge-destroyer. Without K2 we cannot rule out "K1 was a poor control choice".

Combining: the most honest read at 1 seed is **D1 is unlikely to support a strong novelty claim** under the original frame, but the result is not so bad that the D1 backbone integration is useless. CP-3 should call this NOT_PASS for the original D1 thesis but flag K2 + multi-seed as the way to make the call rigorous.

## K3 drop-i4-head (in-flight 2026-05-31 22:15, parallel with K1 just finished)

Tag `d1_drop_i4head_seed42`. With β_i4 frozen at ≈0, the readout-side i4 head is dead and the combined logit becomes `emergnn + β_count × count_logit`. Two possible outcomes:
- K3 combined < D1 main (lower or similar): rejecting the i4 readout costs nothing; the readout was already noise; the i4 evidence path is genuinely doing nothing in either D1 or v2i4 readout.
- K3 combined > D1 main: removing the readout's noise rescues combined and possibly lifts past the 0.785 multi-seed bar. Would change the paper story to "the i4 readout was actively harmful once the backbone absorbed the same substrate; D1 replaces the readout with backbone integration".

## Decisions pending K3 result

- If K3 combined ≥ 0.785: launch K2 (rand-token, the proper bridge breaker), then multi-seed (seed=43,44). Trigger CP-3 with the "D1-replaces-i4-readout" framing.
- If K3 combined < 0.785: launch K2 as the final disambiguating control. Trigger CP-3 with the strong "K1 falsified semantic claim; D1 has architectural side-effects but no semantic value-add at 1 seed" framing.
- In either case, the design doc §3 K1 prediction failed; this is a NOT_PASS-leaning input for CP-3 codex.

## Format target

`Notes/Log/paper_writeup.md` Section 3.W1 (forthcoming): D1 will be reported as either (a) the negative-result subsection alongside CACR (mirroring §3.X structure) if K3 also fails to clear the bar, or (b) a refined positive-result subsection with the "drop i4 readout" framing if K3 succeeds.
