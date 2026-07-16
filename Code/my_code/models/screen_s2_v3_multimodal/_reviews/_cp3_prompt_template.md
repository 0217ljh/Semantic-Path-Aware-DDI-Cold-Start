# CP-3 codex prompt skeleton (fill placeholders after runs complete)

This is a scratch staging file (NOT a review report). It lives under `_reviews/`
so it co-locates with the actual CP-3 report that will be archived as
`2026-05-30__d1_results__round1.md` after the codex verdict comes back.

---

## Prompt template

```
CP-3 results review for Round 4 D1 (LLM-typed fields injected as new KG edge
types into EmerGNN backbone). CP-1 PASS_WITH_NITS at
`_reviews/2026-05-30__d1_design__round1.md` (codex thread 019e7b09); CP-2
PASS_WITH_NITS at `_reviews/2026-05-30__d1_implementation__round1.md`
(codex thread 019e7b1b).

Per round4 plan §8.3 CP-3 template. v2i4 anchor (canonical):
- run_id: 2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42
- combined AUROC 0.7804, emergnn-branch 0.7405, count-only 0.6688, i4-only 0.6003

D1 main run:
- run_id: <FILL d1_main_seed42 run_id>
- combined AUROC <FILL>, emergnn-branch <FILL>, count-only <FILL>, i4-only <FILL>
- coverage breakdown: both_covered <FILL>, one_covered <FILL>, neither_covered <FILL>
- injection summary: <FILL n_llm_edges, n_new_relations, n_ent_after>

Control runs:
- K1 shuf-token (run_id <FILL>): combined <FILL>, emer <FILL>, expected drop ≥ 1.5pp
- K2 rand-token (run_id <FILL>): combined <FILL>, emer <FILL>, expected drop ≥ K1
- K3 drop-i4-head (run_id <FILL>): combined <FILL>, emer <FILL>, expected ≥ 0.78 if backbone really absorbed signal

K4 parity observation already logged in Notes/Log/round4_d1_k4_observation.md:
the originally claimed "≤ 1e-4 mean abs diff per-epoch loss" is withdrawn (model
init non-determinism), but structural K4 parity verified by reading
`v3_llm_edge_trainer.py:78-89` (--d1-disable returns parent _setup_graph output
before any injection).

Q1 (lift attribution). Is the (D1 main − K1 shuf-token) gap ≥ 1.5pp? Does the
emergnn-branch lift specifically come from the LLM-edge injection, not from
inherited v2i4 readout? Verify against control runs.

Q2 (capacity confound). Compare D1 main vs K2 rand-token. K2 preserves edge
count and per-relation density but breaks cross-drug bridges. If K2 ≈ K1 ≈ v2i4
anchor, capacity is not the explanation. If K2 > K1, capacity contributes some
of the lift, and the lift fraction attributable to LLM-evidence specifically is
(D1 main − K2) rather than (D1 main − K1).

Q3 (backbone vs readout). Compare D1 main vs K3 drop-i4-head. If K3 still
≥ 0.78, backbone genuinely absorbed signal. If K3 falls back near v2i4 0.7804
or lower, the lift in D1 main was carried by the inherited i4 readout head not
the new backbone edges (architectural novelty claim weakens).

Q4 (NLL vs AUROC dissociation). Did NLL move in same direction as AUROC?
(Per `Notes/Log/paper_writeup.md` Section 3.X CACR lesson: NLL improvement
without AUROC improvement is a calibration artifact, not a ranking gain.)

Q5 (covered-vs-uncovered). Verify the expected pattern both_covered >
one_covered > neither_covered. If neither_covered shows AUC ≈ both_covered,
that means lift is NOT explained by LLM evidence — it's a KG-rebuild or capacity
artifact. Important falsification signal.

Q6 (paper-story alignment). Does the result support the round4 plan §9
paper-story sentence: "我们识别了 LLM-distilled mechanistic edges 作为 KG 上
missing 的 evidence layer, 把它当成新 relation type 直接进 path-flow"? Yes or
no, with file:line / run_id evidence.

Q7 (multi-seed go/no-go). If combined ≥ 0.785 + K1 dropped ≥ 1.5pp + K2 dropped
≥ K1 + K3 still ≥ 0.78, recommend proceeding to seed=43 + seed=44 multi-seed
validation. Otherwise specify exactly which condition failed.

Q8 (paper section ready?). Is the evidence base sufficient to draft a Section
3.W1 in `Notes/Log/paper_writeup.md`? If yes, suggest the one-sentence headline
summary.

verdict ∈ {PASS, PASS_WITH_NITS, NOT_PASS, FAIL}. Each issue critical / major /
minor + run_id / file:line where applicable. PASS = 0 critical + 0 major per
round4 plan §8.4.
```

---

## Local pre-CP-3 checklist (Claude side, before sending the prompt)

- [ ] d1_main_seed42 completed, results.json written, `analyze_d1_main_result.py` verdict captured.
- [ ] If verdict ≥ LIMITED SWEEP: K1 run launched + completed.
- [ ] If verdict = PASS: K1 + K2 + K3 runs all completed.
- [ ] All control runs have their `results.json` read; values filled into the prompt.
- [ ] Coverage breakdowns extracted for D1 main + each control.
- [ ] Hard-stop rule explicitly invoked (combined < 0.775 stops the pipeline before CP-3).
