# Screen 1 D vs E — Final Verdict (i4 hypothesis)

**Date**: 2026-05-21 (autonomous run completed)
**Variants compared**:
- **D**: PubMedBERT [CLS] of full text (drug profile + KG name) → P1 random projection 64d
- **E**: PubMedBERT [CLS] of within-(canonical_kind, text_source)-shuffled FULL TEXT → same projection
- Both: EmerGNN multimode, 3 sub-models s0/s1/s2, seed 42, 100 epochs, drugbank KG

## Results table

| Split | D (real text) | E (shuffled text) | Δ (D − E) | vs upstream 0.7462 (s2) |
|---|---|---|---|---|
| test_s0 (warm) | **0.9894** | 0.9856 | +0.38pt | n/a |
| test_s1 (1-cold) | **0.8322** | 0.8264 | +0.58pt | n/a |
| **test_s2 (cold)** | **0.7289** | **0.7347** | **−0.58pt** | −1.73pt / −1.15pt |

Fit times: D=7.11h, E=6.88h.

## Gate Verdict

Gate (per first_step_plan.md §4.5): D > E by ≥ 2pt test_s2 AUC, paired bootstrap CI excluding 0.

Observed: D − E = **−0.58pt on test_s2**. Gate **FAILED**.

**i4 hypothesis (PubMedBERT semantic content drives cold-start improvement)
NOT confirmed at seed 42.**

## Interpretation

1. **Warm s0**: D narrowly beats E (+0.38pt). PubMedBERT text might help when source/target drugs are both in training pool.
2. **Half-cold s1**: D narrowly beats E (+0.58pt). Same direction.
3. **Cold s2 — THE TEST**: E wins by 0.58pt. Real text creates spurious confidence on incorrect drug-drug similarities, or shuffled-text noise makes the init more "neutral" and easier to fine-tune.
4. **Both D and E lose to upstream on s2** (0.7289 / 0.7347 vs 0.7462). Original EmerGNN multimode (Morgan FP + learned 'E') is better than ANY of our TAG variants on test_s2.

## Implications for screening priority

### Screen 1 (TAG init) — DOWNGRADED
i4 NOT confirmed. Single-seed; multi-seed + frontier-LM (Qwen H) + D-name + projection P2 are unexplored.

### Screen 3 (Meet-in-Middle) — NOW HIGHEST PRIORITY
E7 v2: LR-count meeting-node features tie EmerGNN under matched KG access. Bake meeting-node signal into flow readout has potential to exceed upstream.

### Screen 5 ⑤a (PK/PD subgraph) — MAINTAIN
E1b: Cramer's V = 0.22, PD-side strong. Run on merged KG.

### Screen 2 — DROP
Dominated by Screen 5 ⑤a.

## Next concrete step

Run Screen 3 R1 (J1 junction):
```bash
python -u Code/my_code/models/screen3_meet_in_middle/run_screen3.py \
  --init-variant A --projection NONE \
  --junction-type J1 --max-junctions 32 \
  --shuffle-train-mode S2 --epochs 100 --seed 42 \
  --tag screen3_R1_J1_random_S2_seed42
```

Note: random init (A) as anchor since variant D underperformed.
Gate: test_s2 AUC > 0.7462 with paired bootstrap CI excluding 0.

## Files

- D results: `Code/runs/2026-05-20_17-14-17__run_screen1__screen1_D_P1_seed42_full__seed42/results.json`
- E results: `Code/runs/2026-05-21_00-22-23__run_screen1__screen1_E_P1_seed42_full__seed42/results.json`
