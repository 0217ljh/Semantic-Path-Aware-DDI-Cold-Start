# Round 4 D2 — Run plan (ready-to-fire when main finishes)

**Status**. d2_main_seed42 in flight (run_id `2026-06-01_15-46-08__run_v3_meet_mask__d2_main_seed42__seed42`).
ETA ~50 min from launch (15:46 + ~50 min ≈ 16:36).

**Background**. D1 was capacity-driven; CP-3 NOT_PASS. D2 design adds pair-conditional propagation bonus on per-pair meeting-mediator masks. CP-1 + CP-2 PASS_WITH_NITS after multi-round fixes. Path B vocab pivot disclosed.

**Hard-stop rule** (round4 plan §6): combined < 0.775 → STOP. 0.775 ≤ combined < 0.785 → LIMITED SWEEP (K1 only). ≥ 0.785 → FULL CONTROLS.

## Verdict-conditional launch matrix (ready to copy-paste)

### If `combined ≥ 0.785` (FULL CONTROLS path):

Launch K1 + K2b + K3 all in parallel (GPU has room — verified RTX 5090 with 25GB free during D1):

```
# K1 shuf-mediators (cardinality-bucketed derangement, "binding destroyed")
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_shuf_mediators_seed42 --seed 42 --deterministic --d2-shuf-mediators"

# K2b rand-uniform (zero shared-mediator bridges)
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_rand_uniform_seed42 --seed 42 --deterministic --d2-rand-mediators-uniform"

# K3 freeze-alpha (alpha_meet=0, mathematical v2i4 parity)
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_freeze_alpha_seed42 --seed 42 --deterministic --d2-freeze-alpha-meet"
```

After all three complete (~50 min in parallel): run `Code/scripts/summarize_d2_controls.py`, fill CP-3 prompt placeholders, trigger codex.

### If `0.775 ≤ combined < 0.785` (LIMITED SWEEP path):

Launch K1 only (one shuffle control + one hyperparam sweep budget per §6). The §6 sweep slot should go to either K3 (architectural parity sanity) or `alpha_meet_init=0.1` (escape zero-init).

```
# K1 only
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_shuf_mediators_seed42 --seed 42 --deterministic --d2-shuf-mediators"

# K3 parity sanity (recommended sweep)
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_freeze_alpha_seed42 --seed 42 --deterministic --d2-freeze-alpha-meet"
```

### If `combined < 0.775` (STOP path):

Do NOT launch any controls. Open `Notes/Log/round4_d2_failure.md` with full diagnostic, then proceed directly to D3 design (alignment InfoNCE).

D3 scoping prerequisites:
- `train_alignment_infonce.py` already exists in this dir, produces `Code/data/_cache/molecular_aligned_infonce.npz` with `z_m` per drug
- D3 mechanism: use `z_m` as initial drug node embedding in EmerGNN (concat or add to morgan features)
- D3 controls: shuffled-mol (permute `z_m` across drugs), random-mol (gaussian), no-init ablation

## Multi-seed (post-CP-3 PASS only)

```
# seed 43
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_main_seed43 --seed 43 --deterministic"

# seed 44
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_main_seed44 --seed 44 --deterministic"
```
