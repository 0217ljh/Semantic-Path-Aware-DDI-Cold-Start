# Screen 1 Autonomous Research Run — INDEX

**Run date**: 2026-05-20
**Duration commitment**: 10 hours, agent active throughout
**Status when reading this**: training and waiting; see autonomous_run_status.md for live snapshot

## Quick links

### Where the action happens
- 🏃 **Variant D training (PRIMARY)**: `Code/runs/2026-05-20_17-14-20__run_screen1__screen1_D_P1_seed42_full__seed42/`
  - Live log: `Code/runs/_screen1_D_P1_full.log`
  - Results (when done): `<run_dir>/results.json`
- ⏸️ **Variant E (QUEUED, auto-launches when D completes)**: log will appear at `Code/runs/_screen1_E_P1_full.log`

### Reports (all under `Notes/Experiments/_results/screen1/`)
1. [`INDEX.md`](INDEX.md) — this file
2. [`2026-05-20__autonomous_run_status.md`](2026-05-20__autonomous_run_status.md) — live status snapshot
3. [`2026-05-20__e1c_node_readability.md`](2026-05-20__e1c_node_readability.md) — i4 motivation (PASS, 81.8% relevant readable)
4. [`2026-05-20__e1b_pkpd_endpoint.md`](2026-05-20__e1b_pkpd_endpoint.md) — i1 motivation (PASS, p=2.4e-70)
5. [`2026-05-20__e7_lr_meeting.md`](2026-05-20__e7_lr_meeting.md) — i2 motivation (STRONG: LR ties EmerGNN)
6. [`2026-05-20__motivation_synthesis.md`](2026-05-20__motivation_synthesis.md) — combined motivation findings
7. [`2026-05-20__final_synthesis.md`](2026-05-20__final_synthesis.md) — Screen 1 final report (D + E results when done)
8. [`2026-05-20__screening_decision_tree.md`](2026-05-20__screening_decision_tree.md) — what to do when you see results

### Code (all under `Code/my_code/models/`)
- `screen1_tag_init/` — TAG init for EmerGNN flow (Screen 1, full impl)
  - Entry: `run_screen1.py`
  - Aggregator: `aggregate_results.py`
- `screen3_meet_in_middle/` — meet-in-middle pooling (Screen 3, full impl, CPU smoke PASS)
  - Entry: `run_screen3.py`
- `screen5_pkpd_subgraph/` — PK/PD dual subgraph + VME (Screen 5, code skeleton, trainer NotImplementedError until rel-vocab audit)

### Motivation experiments (under `Code/my_code/`)
- `exp_e1b_pkpd_endpoint/run_e1b.py` — PK/PD path endpoint asymmetry
- `exp_e7_lr_meeting/run_e7.py` — LR-over-meeting-node baseline

### Caches (large, gitignored, under `Code/data/KG/_merged_kg/_cache/screen1_tag_init/`)
- `d_full_text__pubmedbert.pt` (268 MB)
- `d_name_only__pubmedbert.pt` (268 MB)
- `e_shuffled_text__pubmedbert.pt` (268 MB)
- `f_typename_only__pubmedbert.pt` (268 MB)
- `deepwalk_d64__seed42.pt` (variant C, in-progress generation)
- `node_text_stats.parquet` (E1c stats by node)

## TL;DR for the user

1. **Open** [`2026-05-20__final_synthesis.md`](2026-05-20__final_synthesis.md) — the headline document
2. **Check** that `Code/runs/_screen1_D_P1_full.log` has reached "results saved" line
3. **Run** `python Code/my_code/models/screen1_tag_init/aggregate_results.py` to see comparison table
4. **Read** [`2026-05-20__screening_decision_tree.md`](2026-05-20__screening_decision_tree.md) for next-action mapping based on what D/E show

## Headline findings (preliminary)

- **E7 LR-meeting baseline**: LR-count over 24 hand-crafted shared-mediator features achieves test_s2 AUC = **0.7573**, statistically ties EmerGNN multimode 0.7462. **Strong motivation for Screen 3 meet-in-middle.**
- **E1b PK/PD endpoint**: χ² = 320.60, p = 2.41e-70 — paradigm asymmetry IS real. But PD-side (22.9pt margin) is much stronger than PK-side (8.4pt margin). Screen 5 ⑤a is motivated but expect moderate gain.
- **E1c node readability**: 81.8% of biomedical-relevant nodes have readable names → TAG init feasible.
- **Variant D early convergence**: D val_s0 ep 1 = 0.8967 vs A val_s0 ep 1 = 0.8697 (+2.7pt). D appears to converge faster.

## Codex review pass log

| Round | Subject | Verdict |
|---|---|---|
| 1 | Screen 1 skeleton | NEEDS_FIX → fixed 2 CRITICAL + 7 WARN |
| 2 | Screen 1 post-fix | PASS |
| 3 | Screen 3 + 5 code | NEEDS_FIX → fixed 2 CRITICAL (sub-graph, junction cache) |
| 4 | Screen 3 + 5 post-fix | PASS (Screen 3) + WARNs noted |
| 5 (final) | Screen 5 subgraph post-WARN-fix | PASS + caveats |

## Where to get help

- Each screen's `README.md` has launch commands + outstanding-work list
- `first_step_plan.md` (project root) has the full screening plan with variant matrices

## Why this was an autonomous run

User instruction (16:30): "立刻开始 10 小时连续推进" with codex review after each big milestone. Agent worked autonomously, conducting:
- 5 codex review iterations
- 7 CPU smoke tests
- 1 GPU smoke test (variant A, 1 epoch)
- 1 full training run (variant D, in progress)
- 3 motivation experiments
- 0 user touch-points (autonomous from start until variant D completes)
