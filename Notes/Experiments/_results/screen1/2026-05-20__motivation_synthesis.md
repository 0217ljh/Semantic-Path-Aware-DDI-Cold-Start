# Motivation Synthesis (E1b + E1c + E7) — 2026-05-20

Three motivation experiments executed within Screen 1's autonomous loop.
Each gates a different downstream screen.

## E1c — Node Readability (gates i4 / Screen 1)

**Gate**: ≥80% of biomedical-relevant nodes have human-readable names.

| Bucket | Count | Share |
|---|---|---|
| Total nodes | 178,029 | 100% |
| Biomedical-relevant kinds | 128,564 | 72.2% |
| **Of relevant: readable** | **105,123** | **81.8%** ✅ PASS |

→ Screen 1 (TAG init) feasible. Variant D PubMedBERT encoder covers
174,882 / 178,029 = 98.2% of nodes with non-empty text.

## E1b — PK/PD Path Endpoint Asymmetry (gates i1 / Screen 2 + 5)

**Gate**: PK pairs concentrate in molecular layer; PD pairs in effect-system layer.

| | PK_layer (mol) | PD_layer (effect) | Other |
|---|---|---|---|
| PK pairs (n=3K) | **52.4%** | 44.0% | 3.6% |
| PD pairs (n=3K) | 32.7% | **55.6%** | 11.7% |

Chi-square omnibus χ²=320.60, dof=2, **p=2.41e-70**.

**Verdict**: statistically PASS but effect-size asymmetric:
- PD → PD-layer signal STRONG (22.9pt margin)
- PK → PK-layer signal MODERATE (8.4pt margin)

→ Screen 5 ⑤a (PK/PD subgraph split): motivated for PD side, motivated-with-caveat for PK side.
→ Screen 2 (PK/PD path prior): same caveat — PK prior may be weaker than PD prior.

## E7 — LR-over-Meeting-Node Baseline (gates i2 / Screen 3)

**Gate**: LR over 24 shared-mediator features achieves test_s2 AUC within
3pt of vanilla GCN K=2 (used here as informal benchmark since GCN baseline
on this data wasn't run; we compare to EmerGNN multimode instead).

| Method | KG access | test_s2 AUC | 95% bootstrap CI |
|---|---|---|---|
| EmerGNN multimode (seed 42) | drugbank | 0.7462 | — |
| LR-count (24 log1p features) | **merged (E7 v1)** | 0.7573 | [0.7411, 0.7726] |
| LR-binary (24 indicators) | merged (E7 v1) | 0.7384 | [0.7217, 0.7543] |
| **LR-count (matched access, E7 v2)** | **drugbank** | **0.7311** | **[0.7153, 0.7468]** |
| LR-binary (E7 v2) | drugbank | 0.7182 | [0.7030, 0.7330] |
| HDN-DDI binary | molecules only | 0.6201 | — |
| TIGER binary (no cold-start patch) | drugbank | 0.5842 | — |

**Critical refinement (codex round 5 + E7 v2)**: E7 v1's "LR ties EmerGNN" was
partly a KG-access artifact. With matched drugbank-only KG access, LR-count
drops to 0.7311 (vs merged 0.7573) — a 2.6pt loss. EmerGNN beats LR-count
by 1.5pt under matched access, with overlapping CI.

**Verdict (post codex round 5)**: Under matched drugbank-only KG access
(E7 v2), LR-count test_s2 AUC = **0.7311** vs EmerGNN multimode **0.7462**.
EmerGNN has a modest 1.5pt point-estimate advantage with overlapping CIs
(EmerGNN within LR's 95% CI [0.7153, 0.7468]) — **statistical tie**.
LR-count still **substantially beats** HDN-DDI (+11.1pt) and TIGER
(+14.7pt).

→ Screen 3 (meet-in-middle pooling) has **MODERATE motivation**: 24
simple hand-crafted meeting-node features approach but do not match
EmerGNN's flow under matched access. The 2.6pt loss going from merged KG
to drugbank-only KG suggests the **richer KG annotations (SE, Disease,
Phenotype, Anatomy)** that drugbank lacks contribute significantly — i.e.
node-feature richness matters as much as architectural change for cold-start.

## Combined implications for Screen 1+ execution priority

| Screen | Motivation strength | Notes |
|---|---|---|
| **1 (TAG init)** | E1c PASS + indirect E7 v1 vs v2 signal | **HIGHEST PRIORITY**: 2.6pt LR-count gain from merged KG suggests semantic node features carry significant cold-start signal. Variant D testing this directly. |
| 3 (Meet-in-Middle) | E7 PASS-with-caveat — moderate | Junction signal is real but does not exceed flow under matched KG. Architecture change adds value, magnitude TBD. |
| 5 ⑤a (PK/PD subgraph) | E1b PASS Cramer's V=0.22 (small-medium) | PD-side clear (margin 23pt); PK-side modest (margin 8pt) |
| 2 (PK/PD prior) | E1b PASS-with-caveat | Same as 5 ⑤a |

**Revised priority** (data-driven, post codex round 5):
1. ✅ **Screen 1 variant D (running)** — i4 hypothesis test (PRIMARY claim)
2. → Screen 1 variant E (queued, auto-launches after D) — i4 control
3. → Screen 5 ⑤a (PK/PD subgraph + merged KG) — moderate motivation
4. → Screen 3 (meet-in-middle) — MODERATE motivation (not strong as initially read)
4. Screen 2 — likely dominated by Screen 5; defer

## Files

- E1c: `Notes/Experiments/_results/screen1/2026-05-20__e1c_node_readability.md`
- E1b: `Notes/Experiments/_results/screen1/2026-05-20__e1b_pkpd_endpoint.md`
- E7:  `Notes/Experiments/_results/screen1/2026-05-20__e7_lr_meeting.md`
- This synthesis: `Notes/Experiments/_results/screen1/2026-05-20__motivation_synthesis.md`
