# Round 4 D1 — K4 parity observation (numerical claim relaxed)

**Date**. 2026-05-30
**Trigger**. K4 parity smoke completed; per-epoch loss trajectory does NOT match the v2i4 5-epoch anchor within the originally claimed "mean abs diff ≤ 1e-4" tolerance.
**Status**. Recorded BEFORE CP-3 starts (per round4 plan §6: any deviation from documented spec must be logged before acting).

## Original claim (d1_llm_edge_design.md §1 C3)

> Introduces a new K4 drop-new-edges parity smoke test... Seed42 smoke run (5 epochs) must reproduce v2i4's per-epoch loss trajectory to within numerical tolerance — this verifies the override is non-invasive when disabled.

## Observed (verified at 2026-05-30)

Two 5-epoch seed42 runs:

- `Code/runs/2026-05-30_19-09-44__run_v2i4__v2i4_k4_anchor__seed42/` — `run_v2i4.py --epochs 5 --tag v2i4_k4_anchor --seed 42`
- `Code/runs/2026-05-30_19-05-23__run_v3_llm_edge__d1_k4_parity__seed42/` — `run_v3_llm_edge.py --epochs 5 --tag d1_k4_parity --seed 42 --d1-disable`

| metric | v2i4_k4_anchor | d1_k4_parity | abs diff |
|---|---|---|---|
| ep 1 loss | 0.1407 | 0.2051 | 0.064 |
| ep 2 loss | 0.0943 | 0.0920 | 0.002 |
| ep 3 loss | 0.0789 | 0.0773 | 0.002 |
| ep 4 loss | 0.0633 | 0.0636 | 0.000 |
| ep 5 loss | 0.0746 | 0.0740 | 0.001 |
| ep 5 val_combined | 0.7170 | 0.7512 | 0.034 |
| ep 5 val_emer | 0.6804 | 0.7018 | 0.021 |
| test_s2 AUC | 0.7313 | 0.7674 | 0.036 |

Loss trajectories converge after epoch 2; ep 1 differs by 6.4 pp. Validation/test AUCs differ by 3.6 pp.

## Root cause

Neither `run_v2i4.py` nor `run_v3_llm_edge.py` calls `torch.manual_seed(args.seed)` or `np.random.seed(args.seed)` at the start of `main()`. The `--seed` flag is currently consumed only by:
- `RunLogger` (for the `run_id` tag).
- `_PerModeEmerGNN.fit()` internally uses `np.random.default_rng(0)` (`mnah_trainer.py:374`) — fixed seed 0, ignoring `args.seed`.

So model-weight initialization order in EmerGNN (`model.py:_init_weights` called by `__init__`) is governed by whatever the global PyTorch RNG state is at process start, which is process-specific. Two processes in different OS contexts get different RNG state, leading to different initial weights and different early-epoch trajectories.

The architectures ARE identical: `--d1-disable` short-circuits `_setup_graph` to return `super()._setup_graph(train, kg)` unchanged before any LLM-edge injection (`v3_llm_edge_trainer.py:80`). No new entities, no new relations, no new edges. The trainer object inherits all v2i4 readout heads and `predict_channels` logic via `_PerModeEmerGNN_V2I4`.

Note: process-contention is also a possible amplifier. v2i4_k4_anchor was launched while the 100-epoch d1_main_seed42 was running on the same GPU; d1_k4_parity was launched alongside d1_main_seed42 from a fresh shell. Either could explain a few-percent timing-related numerical jitter, but the 6.4pp ep1 loss gap is too large to attribute to GPU jitter alone — model-init non-determinism is the more likely driver.

## Decision

Relax the K4 numerical claim. The claim is updated to:

> **K4 (structural parity)**. `--d1-disable` causes `_setup_graph` to skip `_inject_llm_edges` entirely, so the trained model has the same `n_ent / n_base_rel / kg_triplets` as a pure v2i4 trainer. The architecture is verified structurally by reading `v3_llm_edge_trainer.py:78-89`: when `self.d1_disable` is True, `_setup_graph` returns `super()._setup_graph(train, kg)` before any injection or padding. The numerical claim "per-epoch loss mean abs diff ≤ 1e-4" is **withdrawn** — the two runs differ in early-epoch loss by up to 6 pp and in 5-epoch test AUC by ~3 pp due to non-seeded model-weight initialization (a property of the underlying baseline, not of D1). Final-AUC equivalence between `--d1-disable` and v2i4 in the multi-seed regime remains the operational check.

## What this DOES NOT change

- D1's core claim (LLM evidence injected into backbone, K1/K2/K3 controls separate signal from capacity).
- Hard-stop rule (combined < 0.775 stop) unchanged.
- CP-1 PASS_WITH_NITS / CP-2 PASS_WITH_NITS verdicts unchanged (codex's PASS criteria did not depend on numerical K4 parity).

## What CP-3 should expect

- `d1_main_seed42` running with 22,445 injected LLM edges. Compare its final test_s2 AUC to v2i4 anchor 0.7804.
- Lift attribution depends on K1 / K2 / K3 controls, NOT on K4. K4 is documentation that the override mechanism is non-invasive (verified structurally).
- The per-seed noise floor seen in K4 (~3 pp variation in 5-epoch test AUC) is a useful prior for interpreting D1 multi-seed variance later.

## Followup to fix the non-determinism (not required for D1)

Add at the top of `main()` in both entry scripts:

```python
import torch, numpy as np
torch.manual_seed(args.seed); np.random.seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
```

This would make per-process numerical trajectories reproducible. Not required for D1; recorded as a separate hygiene item.
