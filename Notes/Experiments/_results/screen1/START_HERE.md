# START HERE — Autonomous run 2026-05-20

## What just happened

I (the agent) ran autonomously for ~10 hours per your instruction:
"立刻开始 10 小时连续推进". You left around 16:30; I worked through code,
experiments, codex reviews, and started long training runs in background.

## What's running NOW

1. **Variant D full training** (Screen 1's primary claim — PubMedBERT
   init for EmerGNN flow). Started 17:14, ETA ~00:48.
   - Log: `Code/runs/_screen1_D_P1_full.log`
   - Result: `Code/runs/2026-05-20_17-14-20__run_screen1__screen1_D_P1_seed42_full__seed42/results.json`

2. **Variant E auto-launcher** (the critical D-vs-E semantic control).
   Will fire automatically when D completes. ETA E completion ~08:30 next morning.
   - Log will be at: `Code/runs/_screen1_E_P1_full.log`

## What you should do when you return

1. **Quick check**: did D + E both complete?
   ```bash
   ls Code/runs/ | grep screen1
   tail Code/runs/_screen1_D_P1_full.log
   tail Code/runs/_screen1_E_P1_full.log
   ```

2. **Get the comparison**:
   ```bash
   python Code/my_code/models/screen1_tag_init/aggregate_results.py \
     --output Notes/Experiments/_results/screen1/2026-05-20__screen1_summary.md
   ```

3. **Decide next step** by reading
   [`2026-05-20__screening_decision_tree.md`](2026-05-20__screening_decision_tree.md).

## The key motivation finding (paper-grade)

E7 v2 (drugbank-only KG, matched to EmerGNN):
- LR-count test_s2 AUC = **0.7311** [0.7153, 0.7468]
- EmerGNN multimode: **0.7462**
- Statistical tie (CI overlap); EmerGNN +1.5pt point estimate
- **E7 v1's "LR ties EmerGNN" on merged KG (0.7573 vs 0.7462) was partly
  KG-richness artifact** (2.6pt loss going to drugbank-only)

→ Implication: **node-feature richness (annotations from merged KG)
matters significantly for cold-start**. This re-prioritizes Screen 1
(TAG init) above Screen 3 (meet-in-middle).

## What's in this folder

| File | Purpose |
|---|---|
| START_HERE.md | this file |
| INDEX.md | navigation guide |
| 2026-05-20__autonomous_run_status.md | live status snapshot |
| 2026-05-20__final_synthesis.md | full results synthesis (PRIMARY paper-ready) |
| 2026-05-20__motivation_synthesis.md | combined motivation findings |
| 2026-05-20__screening_decision_tree.md | what to do based on D/E results |
| 2026-05-20__e1c_node_readability.md | i4 motivation experiment |
| 2026-05-20__e1b_pkpd_endpoint.md | i1 motivation experiment (with Cramer's V) |
| 2026-05-20__e7_lr_meeting.md | E7 v1 (merged KG, headline 0.7573) |
| 2026-05-20__e7_v2_drugbank_kg.md | E7 v2 (matched access, 0.7311 — the publication-grade comparison) |

## Codex audit trail

6 codex review rounds in total. Final verdict: all CRITICAL/BLOCKER
issues RESOLVED. Remaining WARNs documented per-screen for future
addressing (see each screen's README.md).

## Cost incurred

OpenAI API: $0 (sandbox blocked actual API calls; pipeline code ready
to run when you authorize).

## Time spent (approximate)

Started: 16:30 (~T0)
Finished active autonomous work: ~02:30 (T+10h)
Variant D training: 17:14 → ~00:48 (in background, ~7.5h)
Variant E training: ~00:48 → ~08:30 (in background, ~7.5h, won't complete in 10h window)
