# Screen 1 Decision Tree — When User Returns

After the autonomous run, the user will have these results to interpret:
- Variant D full training (PubMedBERT [CLS] of full text, projected via P1=random)
- Variant E full training (FULL TEXT shuffled within canonical kind, P1)
- Possibly variant A (random init) if started after D

The decision tree below maps result patterns to next actions.

## Decision tree

### Case 1: D > E by ≥ 2pt test_s2 AUC (paired bootstrap CI excludes 0)

**Interpretation**: real biomedical text semantics matter beyond text-length/template.
**Decision**: i4 confirmed → proceed to Screen 3 with init_variant=D + junction.

Suggested launches:
```bash
# Screen 3 R0 (terminal-only anchor with TAG init D)
python -u Code/my_code/models/screen3_meet_in_middle/run_screen3.py \
    --init-variant D --projection P1 --use-mim False \
    --shuffle-train-mode S2 --epochs 100 --seed 42 \
    --tag screen3_R0_D_S2_seed42

# Screen 3 R1 (J1 junction + TAG init D)
python -u Code/my_code/models/screen3_meet_in_middle/run_screen3.py \
    --init-variant D --projection P1 --junction-type J1 \
    --shuffle-train-mode S2 --epochs 100 --seed 42 \
    --tag screen3_R1_J1_D_S2_seed42
```

Compare test_s2: if R1 > R0 by ≥ 1pt with CI excluding 0, MIM has architectural value.

### Case 2: D ≈ E (no meaningful gap, or D < E)

**Interpretation**: PubMedBERT text semantics do NOT matter for cold-start in this backbone.
**Decision**: i4 NOT confirmed → consider:
- Variant D-name (drug name only) vs E — does just-name beat shuffled-text?
- Variant F (kind label only) — does even type-name semantic info help?
- Variant H (Qwen-72B) — does richer LM matter?
- Or: drop Screen 1, focus on Screen 3 (which doesn't depend on TAG init quality)

Suggested launches:
```bash
# Variant D-name comparison
python -u Code/my_code/models/screen1_tag_init/run_screen1.py \
    --variant D-name --projection P1 --epochs 100 --seed 42 \
    --tag screen1_Dname_P1_seed42_full

# Variant F (kind-only)
python -u Code/my_code/models/screen1_tag_init/run_screen1.py \
    --variant F --projection P1 --epochs 100 --seed 42 \
    --tag screen1_F_P1_seed42_full
```

### Case 3: D ≫ Anchor (e.g. test_s2 AUC > 0.78)

**Interpretation**: PubMedBERT init dramatically improves cold-start.
**Decision**: Strong i4 result — paper can lead with TAG init as a major contribution.
- Run additional variants (B, C, D-name) to characterize what part of the gain comes from
- Run Screen 3 R1 with init=D to see if MIM is additive

### Case 4: Both D and E ≪ Anchor (e.g. test_s2 AUC < 0.72)

**Interpretation**: TAG init pipeline has a systematic issue (alignment, projection, etc.)
**Decision**: Debug before drawing conclusions.
- Check `[per_mode_tag] external_init alignment` log lines — should show ~88% hit
- Check `[screen1] init shape` — should be (178029, 64) with 174,882 nonzero rows for D
- Check variant A (random init via our pipeline) — should approximate upstream EmerGNN multimode
  (0.7462 on test_s2). If A diverges by >2pt, the wrapper has a bug not in upstream

## Quick sanity matrix (for fast eyeballing)

After variant D + E complete, run:
```bash
python Code/my_code/models/screen1_tag_init/aggregate_results.py \
    --output Notes/Experiments/_results/screen1/2026-05-20__screen1_summary.md
```

Expected pattern in summary table:
```
Variant   Proj    Epochs   Fit(h)   s0 AUC    s1 AUC    s2 AUC
(anchor)  -       -        -        0.9895    0.8328    0.7462
D         P1      100      ~7.5     ?         ?         ?       <- primary
E         P1      100      ~7.5     ?         ?         ?       <- control
```

## When to launch Screen 5

Per E1b motivation:
- PD-side asymmetry is strong (22.9pt margin) → ⑤a should help PD pairs
- PK-side asymmetry is moderate (8.4pt margin) → ⑤a may not help PK pairs much

So Screen 5 ⑤a (S3) should be launched, but with realistic expectation that
gain may be limited. ⑤b (S4 with GPT-4o VME) should NOT be launched until:
1. Screen 5 trainer fully implemented (currently NotImplementedError)
2. VME gap definition redesigned (current is "shortcut" not "missing")
3. S3 result shows non-zero gain (otherwise VME has nothing to add to)

## When to launch Screen 2

Screen 2 (PK/PD path prior) is likely DOMINATED by Screen 5 ⑤a (PK/PD structural
split). Run Screen 2 only if Screen 5 fails to deliver any gain — Screen 2 is a
lighter alternative.

## When to STOP screening

Stop if:
- Screen 3 R1 (junction) provides ≥ 2pt over EmerGNN anchor on test_s2
- AND variant D provides ≥ 2pt over variant A (TAG init verified)
- Then combine: Screen 3 R1 + variant D init = method proposal

This is the goal state. Write up paper with these two ingredients.
