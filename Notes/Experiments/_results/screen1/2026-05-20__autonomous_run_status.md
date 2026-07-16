# Autonomous Research Run Status — 2026-05-20

**Wall clock**: started ~16:30, this report at 17:30 (1h in). User instruction: 10h continuous activity.

## Status at-a-glance

| Item | Status | Detail |
|---|---|---|
| Screen 1 code skeleton | ✅ DONE | 9 source files in `Code/my_code/models/screen1_tag_init/` |
| PubMedBERT embedding cache (D / D-name / E / F variants) | ✅ DONE | 4 × 268 MB caches in `Code/data/KG/_merged_kg/_cache/screen1_tag_init/` |
| Codex review #1 (skeleton) | ✅ NEEDS_FIX → fixed | 2 CRITICAL + 7 WARN addressed |
| Codex review #2 (post-fix) | ✅ PASS | proceed to training |
| Variant A 1-epoch smoke | ✅ PASS | val_s0=0.870 val_s1=0.802 val_s2=0.658 |
| **Variant D full training** | ⏳ **RUNNING** | seed 42, 100 epochs, started 17:14:20, ETA ~00:45 |
| Codex review #3 (Screen 3 + 5) | ✅ NEEDS_FIX → fixed CRITICAL #1, #2 | subgraph_builder + junction cache |
| Screen 3 CPU smoke | ✅ 3/3 PASS | junction shapes + forward + gradient + use_mim toggle |
| E1c node readability | ✅ DONE | 81.8% biomedical-relevant nodes readable, gate PASS |
| E1b PK/PD path endpoint | ✅ DONE | χ²=320.6, p=2.4e-70, PD-side strong, PK-side moderate |
| E7 LR-meeting baseline | ✅ DONE | **LR-count test_s2 AUC=0.7573 ties EmerGNN multimode 0.7462** |
| Variant E (critical control) | ⏸️ QUEUED | will launch after D completes |
| Screen 5 trainer | ⏸️ DEFERRED | skeleton only, `NotImplementedError` |
| Variant H (Qwen-72B) | ⏸️ DEFERRED | requires Qwen weights download |

## Key findings so far

### 1. E7 — meeting-node features tie SOTA flow
```
LR-count (24 features) test_s2 AUC = 0.7573 [0.741, 0.773]
EmerGNN multimode        test_s2 AUC = 0.7462
```
A simple LR over 24 hand-crafted shared-mediator features (12 kinds × 1-hop/2-hop intersections) matches the SOTA flow-based EmerGNN on cold-start. This is **strong motivation** for Screen 3 (meet-in-middle pooling).

### 2. E1b — PK/PD asymmetry strong on PD side, moderate on PK side
```
PD pairs: PD-layer 55.6% vs PK-layer 32.7% (margin 22.9pt)
PK pairs: PK-layer 52.4% vs PD-layer 44.0% (margin 8.4pt)
```
Statistically significant (p<1e-70) but effect-size asymmetric. **Screen 5 ⑤a motivated for PD-side, weaker for PK-side.**

### 3. Variant D (PubMedBERT) converges faster than variant A (random)
At epoch 1: D val_s0=0.8967, A val_s0=0.8697 (+2.7pt for D). At epoch 10: D val_s0=0.962. Strong early evidence that PubMedBERT init helps.

## Files created in this run

### Code
- `Code/my_code/models/screen1_tag_init/` — 9 source files (Screen 1 complete)
- `Code/my_code/models/screen3_meet_in_middle/` — 5 source files (Screen 3, full trainer, smoke PASS)
- `Code/my_code/models/screen5_pkpd_subgraph/` — 5 source files (Screen 5, trainer skeleton + VME generator code)
- `Code/my_code/exp_e1b_pkpd_endpoint/` — E1b experiment script
- `Code/my_code/exp_e7_lr_meeting/` — E7 experiment script

### Reports
- `Notes/Experiments/_results/screen1/2026-05-20__e1c_node_readability.md`
- `Notes/Experiments/_results/screen1/2026-05-20__e1b_pkpd_endpoint.md`
- `Notes/Experiments/_results/screen1/2026-05-20__e7_lr_meeting.md`
- `Notes/Experiments/_results/screen1/2026-05-20__motivation_synthesis.md`
- `Notes/Experiments/_results/screen1/2026-05-20__autonomous_run_status.md` (this file)

### Caches (large, gitignored)
- `Code/data/KG/_merged_kg/_cache/screen1_tag_init/d_full_text__pubmedbert.pt` (268 MB)
- ... and D-name, E-shuffled-text, F-typename equivalents

## What I will do for the remaining ~9 hours

1. Continue monitoring variant D training (every ~30 min)
2. After D completes (~00:45), launch variant E (critical D-vs-E semantic control)
3. Variant E ETA: ~08:30 next morning
4. Write final Screen 1 D + E results synthesis when both complete
5. If time allows: refine Screen 3/5 code per codex remaining WARN items

## Outstanding items for the user when they return

1. **Variant D results** (test_s0/s1/s2 AUC) — should be in
   `Code/runs/2026-05-20_17-14-20__run_screen1__screen1_D_P1_seed42_full__seed42/results.json`
   when training completes
2. **Variant E results** — same dir pattern with `_E_P1_` tag
3. **VME smoke test was blocked by sandbox** — user can run manually:
   `python Code/my_code/models/screen5_pkpd_subgraph/smoke_vme.py`
   to validate GPT-4o API works (~$0.005 spend)
4. **Codex review #3 outstanding WARN items**:
   - vme_gap_finder.py gap definition needs reframing as "shortcut" not "missing"
   - vme_generator.py Qwen path returns 8192d not 64d (contract mismatch)
   - Qwen-72B weights not yet downloaded (deferred)
