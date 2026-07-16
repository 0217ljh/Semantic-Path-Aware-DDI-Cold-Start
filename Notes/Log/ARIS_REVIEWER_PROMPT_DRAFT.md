# ARIS Round 1 Phase E — Reviewer Agent prompt draft

To be sent to Reviewer Agent (general-purpose Agent, codex unavailable) once R1 main + R3 main complete. Placeholders `<<...>>` filled from results.json.

---

```
[REVIEWER AGENT — analyze-iterate skill Phase E]

You are dispatched in PI's autonomous loop. Project: `/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start`. Read CLAUDE.md.

Round 1 of analyze-iterate, attempt 2 of D2 architecture (5-attempt budget). Two proposals were implemented in parallel: R1 (gated replacement fusion) and R3 (PMP replacement readout). Per CLAUDE.md §"Code review 报告归档规范", you are the Primary reviewer; Independent reviewer field MUST be "未调用 (原因: codex MCP gpt-5.3-codex 不支持 ChatGPT 账户, 2026-06-02 verified)".

## Verified anchor
v2i4 anchor (`Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json`):
combined 0.7804, emergnn 0.7405, count_only 0.6688, i4_only 0.6003, NLL 1.4723.

## R1 result (single seed=42, deterministic, 100 epochs)
- run_id: <<R1_run_id>>
- results.json: <<R1_results_path>>
- combined: <<R1_combined>>
- emergnn-branch: <<R1_emergnn>>
- count-branch: <<R1_count>>
- i4-branch: <<R1_i4>>
- NLL: <<R1_nll>>
- final gate_means: <<R1_gate_means>>
- final gate_entropy: <<R1_gate_entropy>>

## R3 result (single seed=42, deterministic, 100 epochs)
- run_id: <<R3_run_id>>
- results.json: <<R3_results_path>>
- combined: <<R3_combined>>
- emergnn-branch: <<R3_emergnn>>
- m_present_mean (training): <<R3_m_present_mean>>
- NLL: <<R3_nll>>

## Theorist predictions (from THEORIST_VERDICT.md)
- R1 expected combined 0.785 ± 0.004 single seed; ≥0.785 = PASS for fusion-reform
- R3 P(combined ≥ 0.79) = 28%; P(combined ≥ 0.785) = 65%
- F1 ceiling without fusion reform: ~0.785 single seed
- K4 observation noted ~3pp 5-epoch single-seed noise (smaller at 100 epoch but still 0.3-0.5pp)

## Required checks (each must have a verified YES/NO answer)

### Check 1: improvement vs anchor
- Is R1 combined > 0.7804 anchor? By how many pp?
- Is R3 combined > 0.7804 anchor? By how many pp?
- Is the delta larger than the K4-observed noise floor (~0.5pp)?

### Check 2: fairness
- Same split (800-drug seed42), same eval (test_s2 1919 pos / 1919 neg, verify n_pos / n_neg in results.json), same epoch budget (100)?
- Both used --deterministic (torch.manual_seed + cuda + np)?
- R1 / R3 use the same v2i4 normalizer fit on train pos + neg? Verify by inspecting trainer code at v3_gated_fusion_trainer.py + v3_pmp_trainer.py if needed.

### Check 3: cherry-pick / variance concern
- Both runs are SINGLE-SEED. Theorist gave seed-plan = 3 seeds {42, 7, 1234} for PASS confirmation. So even if combined ≥ 0.785, single-seed claim must NOT be taken as paper-ready. Verdict should be "IMPROVED (single-seed pending multi-seed)" not "IMPROVED (final)".

### Check 4: leakage check
- Did R1's gate_mlp see any label-leaked features? gate_input is pair_feat (13-d LLM mechanism overlap) — same as v2i4 i4 input, already audited at v2i4_trainer.py:128-141 sanitizer-2. PASS unless new leakage path.
- R3's m_pool = morgan FPs of mediator entities. Mediators come from Path B drugbank 5-bucket KG (DDI-edge-free by construction). m_present_bit is purely structural (n_mediators > 0). PASS unless mediator parquet had leakage.

### Check 5: signs of mechanism actually working
- R1 gate entropy at end of training: if < 0.1 nats → gate collapsed to one branch → mechanism didn't differentiate → mark SUSPICIOUS even if combined > anchor.
- R3 combined > R1 combined when m_present_mean > 0.5? (Test: does R3 benefit from non-empty mediator pairs?)

### Check 6: fusion-saturation diagnosis re-check
The Theorist's F1 hypothesis says fusion structure IS the ceiling. Critical sanity:
- If R1 or R3 combined > 0.7843 (the K3 freeze-α D2 result from 2026-06-02_00-52-02 = "v2i4 + deterministic"), the fusion reform genuinely beat the deterministic-seeded v2i4. Otherwise the lift is just deterministic seeding compensation.
- Compare combined - K3 baseline (0.7843) explicitly.

## Verdict format

Provide:
1. **R1 verdict**: IMPROVED / NO_CHANGE / DEGRADED / SUSPICIOUS — with each Check 1-6 line.
2. **R3 verdict**: same format.
3. **Round 1 net verdict** for the analyze-iterate loop:
   - **PROGRESS** if either R1 or R3 combined > 0.7843 by ≥ 0.005 (single seed). Recommend multi-seed (seeds 7 + 1234) of the winner + K1/K2/K3 controls.
   - **PARTIAL** if either ≥ 0.7804 (anchor) but ≤ 0.7843 (K3 deterministic baseline). Suggests fusion reform helps marginally; consider R2 in attempt 4.
   - **NO_PROGRESS** if both ≤ 0.7804. Need to pivot to R2 or new architecture.
   - **REGRESS** if either < 0.775 (hard-stop band). Immediate review of implementation for bugs.
4. **Attempt budget**: D2 architecture used 1+2=3 of 5 attempts. Remaining 2 attempts go to:
   - If PROGRESS: multi-seed validation + controls
   - If PARTIAL: try R2 (hypernet) as attempt 4
   - If NO_PROGRESS: archive D2 architecture + transition to D3 (alignment InfoNCE init, prereq verified)
5. **Score 1-10** the current state of the architectural-novelty-with-numeric-lift goal.

## Archive
Write `Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-06-02__r1_r3_results__round1.md` with full review. Primary reviewer: claude-opus-4-7 ARIS Reviewer Agent. Independent reviewer: 未调用 (codex MCP unavailable for ChatGPT account, 2026-06-02 verified).
```
