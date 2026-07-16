# ARIS Round 1 — R1/R3 Control Launch Plan (verdict-conditional, ready-to-fire)

**Status**. R1 main + R3 main 100-epoch runs in background (R1 task `buwzusppe`, R3 task `b9ibos4mx`). Monitor `bt0ytugw0` armed.

**Hard-stop and band rules** (round4 plan §6 + Theorist):

| R1 combined | Verdict | Action |
|---|---|---|
| < 0.775 | STOP | Implementation bug suspected; debug |
| 0.775–0.785 | LIMITED SWEEP | K1 shuf-pair-feat only |
| ≥ 0.785 (single seed) | FULL CONTROLS | K1 + K2 + K3, then multi-seed |

Same bands apply to R3.

## R1 K-controls (if combined ≥ 0.7825, +0.21pp above K3 deterministic baseline):

```bash
# K1: shuffle pair_feat — destroys gate's pair-specific input
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_gated_fusion.py --epochs 100 --tag r1_K1_shuf_pair_feat_seed42 --seed 42 --deterministic --r1-shuf-pair-feat 2>&1 | tail -30"

# K2: freeze gate uniform [1/3, 1/3, 1/3] — collapses to uniform-average additive
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_gated_fusion.py --epochs 100 --tag r1_K2_uniform_seed42 --seed 42 --deterministic --r1-freeze-gate-uniform 2>&1 | tail -30"

# K3: gate = [1, 0, 0] frozen — emergnn-only baseline
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_gated_fusion.py --epochs 100 --tag r1_K3_emergnn_only_seed42 --seed 42 --deterministic --r1-zero-gate-emergnn-only 2>&1 | tail -30"
```

**Expected** (R1 mechanism real):
- K1 combined ≤ R1 main − 1.5pp (gate sees noise, can't differentiate)
- K2 combined ≈ R1 main − 0.5–1.0pp (no pair-conditional gating but still has 3 branches additive)
- K3 combined ≈ v2i4 emergnn AUC (no readout signal at all)

## R3 K-controls:

```bash
# K1: shuffle m_pool — destroys mediator-pool's pair-specific input
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_pmp.py --epochs 100 --tag r3_K1_shuf_pool_seed42 --seed 42 --deterministic --r3-shuf-mediator-pool 2>&1 | tail -30"

# K2: zero m_pool — drops mediator-pool input from MLP
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_pmp.py --epochs 100 --tag r3_K2_zero_pool_seed42 --seed 42 --deterministic --r3-zero-m-pool 2>&1 | tail -30"

# K3: additive init — initialize MLP_score to mimic softplus-additive
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_pmp.py --epochs 100 --tag r3_K3_additive_init_seed42 --seed 42 --deterministic --r3-additive-init 2>&1 | tail -30"
```

**Expected** (R3 mechanism real):
- K1 combined ≤ R3 main − 1.0–1.5pp (m_pool ID information destroyed)
- K2 combined ≤ R3 main − 0.5–1.0pp (no m_pool contribution at all)
- K3 combined > R3 main if "additive sum is actually better" → would invert the paper claim

## Multi-seed plan (if either R1 or R3 hits ≥ 0.785 PASS gate and K-controls fail back):

Seeds: 7, 1234. Three commands per winning method:

```bash
# R1 multi-seed (only if R1 won)
... --tag r1_main_seed7 --seed 7 --deterministic
... --tag r1_main_seed1234 --seed 1234 --deterministic

# R3 multi-seed (only if R3 won)
... --tag r3_main_seed7 --seed 7 --deterministic
... --tag r3_main_seed1234 --seed 1234 --deterministic
```

## D2 attempt budget accounting

Round 1 (this round): 2 attempts spent (R1 + R3) on top of D2-attempt-1 = **3 of 5 used**.

If R1+R3 NO_PROGRESS → attempt 4 = R2 (hypernet), attempt 5 = final pivot or D3-architecture transition.

If R1 OR R3 PROGRESS → attempt 4 = K-controls (cheap, single-seed) + multi-seed seeds {7, 1234}. Attempt 5 = combined R1+R3 if both promise, OR CP-3 writeup.

## Theorist warnings to watch for

- R1 gate collapse: if entropy < 0.05 nats at end → gate became constant → mechanism dead even if combined > anchor. **Mark SUSPICIOUS.**
- R3 70% empty mediator pairs: q1 (mediator=0) AUC will likely stay ~0.68 like D2 main. R3's lift must come from q2 (non-empty mediator) pairs. **If q1 AUC drops below D2 main's 0.6767, the m_pool replacement is hurting empty-mediator pairs.**

## GPU concurrency

R1 + R3 main running parallel now. If K-controls launched after main completes: K1 + K2 + K3 of one method = 3 jobs parallel + the other method's controls = 6 total. Per past observations RTX 5090 25GB can handle 3-4 parallel without OOM. Will batch in waves of 3.
