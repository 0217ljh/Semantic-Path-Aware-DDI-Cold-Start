# PD core-effect depth in the merged KG (2026-06-07)

## Goal
DDI mechanisms have a named **core effect** (QTc prolongation, CNS depression, bleeding, ...).
For pharmacodynamic (PD) DDIs we ask: in the merged biological KG, **at what undirected
hop-depth does each drug of the pair reach its DDI's core-effect node, and do both drugs
reach it?** This tests whether the discriminative effect sits at 1-hop or deeper, and whether
the effect node is a hub or a specific node. Motivation: justify a learnable depth-spanning
semantic attention to the core effect (rather than a fixed 1-hop readout).

Script: `Code/scripts/analyze_pd_core_effect_depth.py`
Run dir: `Code/runs/2026-06-07__pd_core_effect_depth/`
Env: WSL conda `project_1`.

## Method (summary)
- **STEP 1 PD bucket.** Keep `ddi_type` rows containing any of: `risk or severity`,
  `activities`, `efficacy`, `cns depression`, `qtc`, `hypertension`, `hypotensive`,
  `sedative`, `adverse effects`. -> 279,799 PD rows, 187 distinct PD types.
- Core effect via regex (priority): `risk or severity of (.+?) can be`, then
  `(?:increase|decrease) the (.+?) activities` (+ ` [activity]`). -> **178/187** distinct PD
  types get a core effect (9 generic efficacy/absorption types do not). 153 distinct core
  effects, 233,558 samples carry a mapped core string.
- **STEP 2 curated mapping** (top-30 core effects by sample mass). Auto substring matching is
  unreliable, so each effect is mapped by a hand-verified synonym to an **exact** KG effect-node
  name (kinds {Side Effect, effect/phenotype, Symptom, disease, Disease}), preferring
  Side Effect / effect/phenotype over disease. Generic effects are excluded. Full audit table:
  `Code/runs/2026-06-07__pd_core_effect_depth/core_effect_node_mapping.csv`.
- **STEP 3 sample.** seed 42, 100 PD rows whose core effect is mapped (169,438 eligible rows).
- **STEP 4 depth.** Undirected, binary, self-loop-free graph over all 178,029 KG nodes from
  ~7.1M edges (symmetrized, deduped). Depth-limited BFS (L=4) from each drug to its
  effect node. dA / dB in {1,2,3,4, unreachable}.

## Curated mapping (27 mapped / 30; 3 excluded)
Source: `core_effect_node_mapping.csv`. Top rows (core_effect -> node_name, kind, degree):

| core_effect | n_samples | node_name | kind | degree |
|---|---:|---|---|---:|
| cns depression | 37967 | CNS depression NOS | Side Effect | 44 |
| adverse effects | 36843 | **(excluded, generic)** | | |
| qtc prolongation | 25598 | Prolonged QT interval | effect/phenotype | 108 |
| antihypertensive [activity] | 15057 | Hypertension | Side Effect | 453 |
| methemoglobinemia | 14340 | Methemoglobinemia | effect/phenotype | 33 |
| hypertension | 13484 | Hypertension | Side Effect | 453 |
| hypotensive [activity] | 8858 | Hypotension | Side Effect | 529 |
| cns depressant [activity] | 7117 | CNS depression NOS | Side Effect | 44 |
| bleeding | 4554 | Haemorrhage | Side Effect | 317 |
| nephrotoxicity | 4433 | Nephrotoxicity | Side Effect | 32 |
| tachycardia | 4355 | Tachycardia | Side Effect | 501 |
| hyperkalemia | 3354 | Hyperkalemia | effect/phenotype | 166 |
| arrhythmogenic [activity] | 2921 | Arrhythmia | Side Effect | 365 |
| serotonin syndrome | 2640 | Serotonin syndrome | Side Effect | 36 |
| myopathy, rhabdomyolysis, ... | 2455 | Rhabdomyolysis | Side Effect | 115 |
| gastrointestinal irritation | 2419 | **(excluded, no clean node)** | | |
| hyperglycemia | 2341 | Hyperglycemia | effect/phenotype | 322 |
| sedative [activity] | 2300 | Sedation | Side Effect | 13 |
| hypoglycemia / hypoglycemic | 2288/1978 | Hypoglycemia | effect/phenotype | 352 |
| gastrointestinal bleeding | 1882 | Gastrointestinal haemorrhage | Side Effect | 200 |
| bradycardic [activity] | 1835 | Bradycardia | Side Effect | 310 |
| hypokalemia | 1676 | Hypokalaemia | Side Effect | 225 |
| hypotension / orthostatic | 1345/891 | Hypotension | Side Effect | 529 |
| neuromuscular blocking [activity] | 1239 | Neuromuscular block prolonged | Side Effect | 4 |
| anticoagulant [activity] | 1227 | Haemorrhage | Side Effect | 317 |
| immunosuppressive [activity] | 929 | **(excluded, no single node)** | | |

**Excluded** (generic / not localizable to a single node): `adverse effects`,
`gastrointestinal irritation`, `immunosuppressive [activity]`, plus `neurotoxic [activity]`
(in dict, not in top-30 row shown).

## STEP 4 results (n = 100 sampled pairs, L = 4)
All numbers from `summary.json`.

**Per-drug depth to its DDI's core-effect node** (dA = first listed drug, dB = second):

| depth | dA | dB |
|---|---:|---:|
| 1-hop | 32 | 16 |
| 2-hop | 34 | 21 |
| 3-hop | 34 | 58 |
| 4-hop | 0 | 2 |
| unreachable (>4) | 0 | 3 |

- dA: **32% at 1-hop, 68% at 2-4 hops, 0% unreachable.**
- A reaches its effect: **100%**; B reaches: **97%**.
- **Both drugs reach the core effect within L=4: 97%.** Both at exactly 1-hop: only **8%**.
- Both-reach cumulative by depth: <=1: 8, <=2: 25, <=3: 95, <=4: 97. The mass lands at depth 3.
- Top joint (dA,dB) cells: (3,3)=20, (1,3)=19, (2,3)=19, (3,2)=9, (1,1)=8, (2,2)=7.

**Core-effect node degree** (14 distinct nodes used in the sample): min 13, p25 44,
median **108**, mean 222, p75 453, max **529**. The high-mass effects (Hypotension 529,
Tachycardia 501, Hypertension 453, Arrhythmia 365, Hypoglycemia 352, Haemorrhage 317) are
clear **hubs**; only a few (Sedation 13, Neuromuscular block 4, Nephrotoxicity 32,
Methemoglobinemia 33) are specific.

## Caveats (honest)
- Curation is **top-30 by mass only**; the long tail of 153 core effects is not mapped.
- 3-4 generic effects are **excluded** (no faithful single node); their large sample mass
  (esp. `adverse effects` 36,843, `cns depression` is mapped but to "CNS depression NOS").
- Depth is **shortest-path** (BFS), undirected, binary; it ignores relation type, direction,
  and path multiplicity. A 1-hop reach does not mean a *mechanistically meaningful* edge.
- Several effects map to **hub** nodes (e.g. Hypotension deg 529); short depth to a hub is
  partly a hub artifact, not necessarily DDI-specific signal.
- **Single sample, single seed (42), n=100, L=4.** Asymmetry between dA and dB reflects the
  fixed drug ordering in DrugBank rows, not biology.

## Takeaway
The core effect is **mostly NOT at 1-hop**: only 32% of first-listed drugs and 16% of
second-listed reach it in one hop, and **both drugs share the 1-hop effect in just 8% of
pairs**, whereas 97% reach it within 4 hops with the mass at depth 3. The effect nodes are
predominantly **hubs** (median degree 108). This supports a **learnable depth-spanning
attention to the core effect** rather than a fixed 1-hop readout, while flagging that the
hub nature of these nodes must be controlled for.

## Artifacts
- `Code/scripts/analyze_pd_core_effect_depth.py`
- `Code/runs/2026-06-07__pd_core_effect_depth/core_effect_node_mapping.csv`
- `Code/runs/2026-06-07__pd_core_effect_depth/per_pair_depths.csv` (+ `.parquet`)
- `Code/runs/2026-06-07__pd_core_effect_depth/summary.json`
