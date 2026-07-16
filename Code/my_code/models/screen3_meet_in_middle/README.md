# Screen 3 — Meet-in-Middle Pooling

**Hypothesis (i2)**: reading hidden states at junction nodes (1-hop shared
neighbors of source / target) provides extra cold-start signal beyond
EmerGNN's terminal-only readout.

**Backbone**: EmerGNN_TAG (Screen 1's TAG-init EmerGNN, feat='X' with
external init). Forward is overridden to ALSO gather hidden states at
junction nodes and merge them into the score head.

**Motivation from E7**: 24 hand-crafted meeting-node features (LR-count)
achieves test_s2 AUC = 0.7573, **statistically tying EmerGNN multimode
0.7462**. This is strong prior that the junction signal is high-value.

## Variants (per first_step_plan.md §4.6)

| ID | --use-mim | --junction-type | Role |
|---|---|---|---|
| R0 | False | — | Terminal-only anchor (= Screen 1 EmerGNN_TAG) |
| R1 | True | J1 | Primary (no hyperparameter) |
| R2 | True | J3-PK | Mechanism: molecular layer |
| R3 | True | J3-PD | Mechanism: effect-system layer |
| R4 | True | J3-both | Combined mechanism layers |
| R6 | True | Jr | Random control (same K, scrambled identity) |
| R7 | True | J7 | 2-hop junction (deeper variant) |

## Gates (vs R0 on test_s2 AUC, paired bootstrap CI)

| Comparison | Threshold | Interpretation |
|---|---|---|
| R1 > R0 | ≥ +1pt | Junction adds real signal |
| R1 > R6 | ≥ +1.5pt | Gain from structural selection, not added capacity |
| R4 > R1 | ≥ +0.5pt | Mechanism awareness adds incremental value |

## Files

- `__init__.py`
- `junction_finder.py` — J1 / J3-PK / J3-PD / J3-both / J7 / Jr builders + adjacency utilities
- `emergnn_mim.py` — `EmerGNN_MIM(EmerGNN_TAG)`, overrides forward + Wr head (6*n_dim)
- `_per_mode_mim.py` — full per-mode trainer (re-implements upstream shuffle_train loop with junction injection + pair-level junction cache for performance)
- `run_screen3.py` — CLI entry
- `smoke_test_cpu.py` — CPU-only sanity test (3/3 PASS as of 2026-05-20)

## Status (2026-05-20)

- Code: WRITTEN, fixed per Codex round 3 + 4 reviews
- CPU smoke: 3/3 PASS (junction shapes, forward+grad, use_mim toggle)
- GPU smoke: NOT yet run (variant D training holds GPU during this autonomous session)
- Full training: NOT yet run

## How to launch (when GPU available)

Example R1 primary run:
```bash
wsl bash -c "source /home/lakestar_ljh/miniconda3/bin/activate project_1 && \
  cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && \
  python -u Code/my_code/models/screen3_meet_in_middle/run_screen3.py \
    --init-variant D --projection P1 \
    --junction-type J1 --max-junctions 32 \
    --shuffle-train-mode S2 \
    --epochs 100 --seed 42 \
    --tag screen3_R1_J1_D_seed42_full"
```

Note: only one shuffle_train_mode at a time (S2 evaluates on test_s2). For
full multimode (s0+s1+s2 sub-models), additional orchestration is needed.

## Codex review history

- **Round 3 (post code writeup)**: NEEDS_FIX → 2 CRITICAL fixed
  - Subgraph builder PK/PD by non-drug endpoint
  - Junction precompute cache (no per-batch rebuild)
- **Round 4 (post-fix)**: PASS with WARN-level items (cache not cleared between fits, Jr seed not pair-deterministic, CrC edge leakage potential — Screen 5 issue)
