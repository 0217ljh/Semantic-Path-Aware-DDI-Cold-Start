# CP-3 codex prompt draft for D2 (fill placeholders after main + controls done)

This is the actual prompt to send to `mcp__codex__codex` once D2 main + K1 + K2b + K3 have results.json. Numbers in `<<...>>` placeholders will be replaced by the `summarize_d2_controls.py` output.

---

```
CP-3 results review for Round 4 D2 (Meeting-node-aware propagation; Path B
drugbank 5-bucket mediator universe). 

CP-1 PASS_WITH_NITS at `Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-06-01__d2_design__round1.md`
(codex thread 019e8482; 3 rounds: 5 majors + 1 minor → 2 majors + 2 minors → 3 minors → PASS_WITH_NITS).
CP-2 PASS_WITH_NITS at `Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-06-01__d2_implementation__round1.md`
(codex thread 019e84b0; 4 rounds: save/load gradual fix).

**Critical context** for CP-3:
- CP-2 disclosed Path A → Path B pivot (`Notes/Log/round4_d2_vocab_mismatch_discovery.md`).
  Original CP-1 design §3.2 used merged-KG union+cross-hop mediators; smoke
  revealed 98% of merged-KG mediator IDs are not in drugbank 5-bucket trainer
  vocab. Path B builds n1 directly over drugbank 5-bucket bipartite KG. n2 is
  empty by schema.
- Path B cardinality: mean=0.57, median=0, p95=3, p99=5, max=20 post-topK,
  **zero_fraction = 70.2%** (70% of pairs have no shared 5-bucket entity).
- D1 (the prior direction this is pivoting from) CP-3 NOT_PASS for its
  mechanism claim. K1+K2 falsified D1's "LLM evidence routes through path-flow"
  hypothesis. D2 is the explicit pair-conditional response.

All numbers from each run's results.json (file paths in table).

v2i4 anchor (canonical): combined 0.7804, emergnn 0.7405, count_only 0.6688,
i4_only 0.6003, NLL 1.4723.

**Four-cell falsification result** (verified):

| branch       | D2 main | K1 shuf | K2b rand | K3 frz-α | v2i4 anchor | Δ vs anchor (D2/K1/K2b/K3) |
|--------------|---------|---------|----------|----------|-------------|-----------------------------|
| combined     | <<D2c>> | <<K1c>> | <<K2bc>> | <<K3c>>  | 0.7804      | <<D2dc>>/<<K1dc>>/<<K2bdc>>/<<K3dc>> pp |
| emergnn      | <<D2e>> | <<K1e>> | <<K2be>> | <<K3e>>  | 0.7405      | <<D2de>>/<<K1de>>/<<K2bde>>/<<K3de>> pp |
| count_only   | <<D2n>> | <<K1n>> | <<K2bn>> | <<K3n>>  | 0.6688      | <<D2dn>>/<<K1dn>>/<<K2bdn>>/<<K3dn>> pp |
| i4_only      | <<D2i>> | <<K1i>> | <<K2bi>> | <<K3i>>  | 0.6003      | <<D2di>>/<<K1di>>/<<K2bdi>>/<<K3di>> pp |
| NLL          | <<D2l>> | <<K1l>> | <<K2bl>> | <<K3l>>  | 1.4723      | (Δ as decimal) |

Mediator-cardinality-stratified S2 AUC (D2 main):
- Q1_zero (mediator count = 0, ~70% of pairs): AUC=<<Q1>> n=<<Q1n>>
- Q2_low (count 1-3): AUC=<<Q2>> n=<<Q2n>>
- Q3_mid (count 4-10): AUC=<<Q3>> n=<<Q3n>>
- Q4_high (count >10): AUC=<<Q4>> n=<<Q4n>>

Run dirs:
- D2 main: <<D2_run_id>>
- K1 shuf: <<K1_run_id>>
- K2b rand: <<K2b_run_id>>
- K3 frz-α: <<K3_run_id>>

Design §3-§4 falsification predictions:
- K1 shuf-mediators (cardinality-bucketed derangement): combined drops ≥ 1.5pp;
  emergnn drops to ≤ 0.7405. Test: "drug → shared-mediator identity matters".
  If K1 doesn't drop → identity doesn't matter; D2 same capacity story as D1.
- K2b rand-uniform (each pair's mediator IDs replaced by uniform random non-drug
  entity indices): combined drops MORE than K1. Test: "mediator IDs being IN
  the propagation graph at all is necessary".
- K3 freeze-alpha (alpha_meet[l]=0 frozen): combined ≈ v2i4 anchor 0.7804
  (mathematical equivalence; bonus path is zero by construction). Sanity test
  that K4 structural parity claim holds.

Note: K2a (degree+kind-matched) was deferred to round 2 per CP-2 disclosure;
K2b uniform is the harshest control available now.

Q1 (lift attribution). 4-cell table 看 D2 lift 在 emergnn-branch 是不是 pair-conditional
mediator signal (好) 还是又是 capacity/density artifact (D1 重复)? K1 should be the load-bearing.

Q2 (capacity ruled out). K3 freeze-alpha is the architectural-equivalence sanity.
是否真的 ≈ v2i4 anchor? Verify (within ±0.5pp).

Q3 (mediator-cardinality stratification). D2 main 是否在 Q1_zero (没 mediator 的 70%
pairs) 上跟 v2i4 anchor 表现相同 (because mask all-zero, no D2 contribution there)?
是否在 Q3_mid + Q4_high 上比 anchor 更好 (because D2 has signal there)?
This is the Path B sparsity check — if D2 doesn't help in mediator-rich pairs, the entire
backbone-injection idea is invalid.

Q4 (NLL vs AUROC dissociation). 跨 D1 paper §3.X CACR + D1 main 的 framing 看 D2 是否
重复 calibration-without-ranking pattern.

Q5 (D1 capacity confound recurrence). D1 codex Q1 attributed +1.28pp emergnn lift to
shuffle-invariant KG-density. D2's design philosophy: pair-conditional mask should be
shuffle-sensitive. Is K1's behavior consistent with that prediction? If K1 doesn't drop,
D2 has the same shuffle-invariance problem (capacity confound).

Q6 (paper story). 如果 D2 PASS, paper section can claim:
"meeting-node-aware propagation adds pair-conditional evidence injection into the
backbone, with shuf-mediator + rand-uniform controls validating that the mediator
IDENTITY (not just cardinality / KG density) drives the lift". Is the evidence base
sufficient for this claim? If yes, suggest one-sentence headline.

Q7 (multi-seed go/no-go). Should we burn GPU on seed=43+44 multi-seed for D2?

Q8 (D3/D4 pivot if D2 also fails). Round4 plan §4 has D3 (alignment InfoNCE → backbone
init embedding) and D4 (pair-conditional relation routing). Given D1 and possibly D2
both fail in the capacity/shuffle-invariance category, what's the right next direction?

verdict ∈ {PASS, PASS_WITH_NITS, NOT_PASS, FAIL}. PASS = 0 critical + 0 major per
round4 §8.4. Per-issue file:line + run_id where applicable.
```

---

## Pre-CP-3 Claude-side checklist (before sending)

- [ ] d2_main_seed42 completed, results.json verified.
- [ ] Apply hard-stop rule via summarize_d2_controls.py: combined < 0.775 → STOP; 0.775 ≤ combined < 0.785 → LIMITED SWEEP (K1 only); ≥ 0.785 → all controls.
- [ ] If verdict ≥ LIMITED SWEEP: launch K1, K3 in parallel (GPU has room).
- [ ] If verdict = PASS: also launch K2b.
- [ ] After controls finish: fill placeholders, send to codex.
- [ ] Archive at `_reviews/2026-06-XX__d2_results__round1.md`.
