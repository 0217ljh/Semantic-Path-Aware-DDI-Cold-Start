# Story design: "evidence accessibility mismatch" (position × propagation)

2026-07-01. codex-reviewed (thread 019f1f09). How to PROVE that position-agnostic
AND-intersection captures balanced-deep shared mediators that propagation/flow GNNs
(EmerGNN) underutilize. Refines Claim ① of [[spmn_v2_theory_anchor]].

## Verdict: full 3-layer "per-position decline" story is NOT cleanly provable here
Reasons: l_max=3 → only 3 centrality levels (1,1)/(1,2)/(2,2); ~all supports (2,2)-dominated
(least-central quartile already 73% (2,2)); EmerGNN observed only via final scores; pair
difficulty entangled with support composition. Current per-pair quartile gap (AND−EmerGNN
+0.041→+0.102) is WEAK: crude mixture summary, compressed axis, both models drop at center
(only the GAP grows → dismissible as "central pairs are just harder").

## REFRAME the thesis (codex, defensible)
NOT "EmerGNN fails from over-smoothing" (unprovable from black-box scores), NOT "monotone
per-position decline" (too fragile in noisy real data). INSTEAD:
> "In cold-start DDI, decisive evidence often appears as shared BALANCED-DEEP mediators.
> Position-agnostic intersection captures such evidence once both drugs REACH it, whereas
> propagation-based pairwise scoring UNDERUTILIZES it as transport distance grows."
Only needs: (1) central witnesses are informative, (2) AND captures them, (3) EmerGNN
under-responds to them RELATIVE to proximal witnesses. No beautiful gradient required.

## Highest-value experiment (do this ONE if only one)
**Semi-synthetic planted-witness recovery on the REAL KG:** start from negative/ambiguous
pairs, ADD one matched decisive shared mediator at a CONTROLLED position (1,1)/(1,2)/(2,2)
(and synthetic-only (3,3) as a deeper stress point), everything else matched (degree ranges,
background support count, label prevalence, distractors). Re-run BOTH models; measure Δscore /
recovery AUROC vs witness depth. AND should recover ~depth-invariantly once reachable;
EmerGNN should decay with depth. Kills "harder pairs" (controlled), gives a depth-response
curve, and can exceed l_max=3 synthetically. Cost: needs EmerGNN inference on edited graphs
(baseline infra exists).

## If observational only (cheaper): within-pair local witness knockout
For a pair, DELETE (or add) ONE shared mediator at known position via a MINIMAL local edit
(break only the shortest a→m / b→m witness path, preserve degree/profile); re-run both;
record per-pair Δscore; model Δscore ~ position + strength + support + difficulty. Test
whether |Δscore| declines toward center for EmerGNN but not AND. WITHIN-PAIR → kills the
difficulty confound. Far stronger than coarse global "leave-position-out" (which mutilates
the graph = distribution shift, not evidence-use).

## Fix the current weak slice: MATCHED strata, not raw quartiles
Match pairs across position strata on total support size, AA mass, degree(a,b), label
balance, hardness proxy → then compare AND vs EmerGNN. Removes difficulty confound.
Also: "single/near-pure-position" strata (support share at one position ≥0.9) give clean
per-model (incl black-box) numbers if sample allows.

## Layer 1 (premise, not headline): intrinsic per-position discriminativity
For each position c: per-pair feature = Σ_{mediators at c} 1/log(deg) (AA, hub-robust);
matched univariate AUROC / conditional logistic vs label. Report count/AA/RA for robustness.
Proves informativeness, NOT model accessibility → premise figure only.
KEY metric for layer 2: "CAPTURE EFFICIENCY" = model AUROC at c RELATIVE to intrinsic
discriminativity at c → separates "evidence weaker at c" from "model uses c worse".
Don't call ours "flat" without this normalization.

## Figure hierarchy (codex)
F1 intrinsic discriminativity by position (premise) · F2 matched-strata AND-vs-EmerGNN gap
· **F3 counterfactual intervention response by position (strongest mechanism figure)**.
Headline = model gap under controlled evidence composition, NOT model-free discriminativity.

## l_max
Keep main benchmark l_max=3. Use l_max=4 / synthetic ONLY as a mechanism stress-test with
deeper planted witnesses (flooding: shared set ~40% of graph at ℓ≥3 → not a main substrate).

相关:[[spmn_v2_theory_anchor]] · [[project_semantic_path_ddi]]
