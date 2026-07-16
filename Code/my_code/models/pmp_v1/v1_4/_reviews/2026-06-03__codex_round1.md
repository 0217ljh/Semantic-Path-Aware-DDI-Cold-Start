# PMP v1.4 (Layer-2-only ablation) — Codex Review Report

- **Date**: 2026-06-03
- **Primary reviewer**: Claude (sonnet-4.5) — drafted v1.4 as a new package
  under `Code/my_code/models/pmp_v1/v1_4/`.
- **Independent reviewer**: codex (default model via `mcp__codex__codex`).
- **Triggered by**: 用户 "可以了,可以写1.3和1.4的代码了,记住codex的review,以及正确的落盘".
  Goal: measure Layer 2 (per-cluster within-pool) standalone cold-start AUC
  when trained in isolation, complementing v1.2's joint-train +
  zero-out-branch diagnostic.
- **Stop condition**: round-1 PASS, no findings. No round 2 needed.

---

## 1. Architectural delta from v1.2 (drop cluster path)

```
v1.2: score = MLP_score(concat[z_cluster_pair, z_within])
v1.4: score = MLP_score(z_within)            # cluster path DROPPED entirely
```

- Within-pool (phi_mlp, within_attn, p_a * p_b weighting) is byte-identical to
  v1.1 / v1.2.
- `W_T`, `cluster_attn_W/w` REMOVED.
- `mlp_pair_cluster` REMOVED.
- Pair-conditional H_a / H_b construction REMOVED.
- Per-drug per-cluster mediator tensor construction NOT used at training time
  (and not even built, because v1.4 inherits v1.1 directly, NOT v1.2).
- `mlp_score` input dim changes from `hidden + d` (160) to `d` (32).
- p_a, p_b = softmax(affinity) computed directly inside v1.4 `forward()`, NOT
  through any cluster path call.

EmerGNN bypass: UNCHANGED from v1.1.
Cache reuse: PMP cache v1, v1.1 cluster cache v2 (no rebuild).
Inheritance: v1.4 trainer inherits `_PerModeEmerGNN_PMP_v1_1` directly.

## 2. Locked v1.4 spec

```
Embeddings (cluster-path components REMOVED):
  mediator_embed (n_mediators, d)         Xavier
  type_embed     (n_types, type_dim)
  rel_embed      (n_rels+1, rel_dim)
  phi_mlp (d + type_dim + 2*rel_dim -> hidden -> d)
  within_attn_W (d, d), within_attn_w (d, 1)
  mlp_score (d -> hidden -> 1)

Per-cluster within-pool (same as v1.1 / v1.2):
  p_a = softmax(c_a),  p_b = softmax(c_b)        # used as joint relevance weight
  for k in 0..K-1:
    M_k = members(k) ∩ N_<=2(a) ∩ N_<=2(b)
    phi(m) = MLP([mediator_embed[m], type_embed[k],
                  rel_embed[a->m], rel_embed[m->b]])
    z_within^k = AttnPool({phi(m) for m in M_k}) or 0
  z_within = sum_k (p_a[k] * p_b[k]) * z_within^k

Score:
  score = MLP_score(z_within)
  loss  = BCE(sigmoid(score), y)
```

## 3. Read inventory

| File | R1 reviewed |
|---|---|
| `Code/my_code/models/pmp_v1/v1_4/__init__.py` | ✓ |
| `Code/my_code/models/pmp_v1/v1_4/cluster_head.py` | ✓ |
| `Code/my_code/models/pmp_v1/v1_4/pmp_v1_4_trainer.py` | ✓ |
| `Code/scripts/run_pmp_v1_4.py` | ✓ |

For comparison (already approved, not re-reviewed):
- v1.2, v1.1, parent v1, MNAH parent, `_format_epoch_log` hook in MNAH.

## 4. Round 1 codex verdict (verbatim, v1.4-relevant excerpts)

> No findings.
>
> Key checks passed:
> - v1.4 correctly drops the cluster path: no `W_T`, no cluster-path helper,
>   and `p_a/p_b = softmax(affinity_*)` are computed directly inside
>   `forward()` before the within-pool in cluster_head.py:145. The
>   within-pool logic matches the approved v1.1/v1.2 formulation.
> - v1.4 trainer inherits v1.1 directly, uses
>   `_build_per_cluster_mediator_tensors`, never touches
>   `_build_per_drug_per_cluster_tensors` or v1.2-only cache state, and
>   returns branch slots in the requested order
>   `(combined, zeros, combined_copy)` in _predict_branches_v1_4
>   (pmp_v1_4_trainer.py:129).
> - Both trainers preserve the EmerGNN bypass by overriding _combined_logit
>   directly and not calling super()._combined_logit; v1.4 also overrides
>   _format_epoch_log with the requested labels at pmp_v1_4_trainer.py:211.
> - Run script is wired correctly: run_pmp_v1_4.py:78 does NOT have
>   --cluster-max-mediators-per-drug (which is v1.2/v1.3-only). Instantiates
>   the right trainer, passes cache args through, and writes results.json
>   plus test_s2_scores.npz.
>
> Parameter-count sanity also matches the spec by layer-shape arithmetic:
> - v1.4 = 2,924,801
>
> Residual risk: this was a static code review only. I did not run
> training/inference in this sandbox.
>
> VERDICT: PASS

## 5. Stop condition

Per user's review-loop spec: stop when codex says correct OR three rounds done.
Round 1 = PASS with no findings. No round 2 executed.

## 6. Files created

- `Code/my_code/models/pmp_v1/v1_4/__init__.py`
- `Code/my_code/models/pmp_v1/v1_4/cluster_head.py`
- `Code/my_code/models/pmp_v1/v1_4/pmp_v1_4_trainer.py`
- `Code/scripts/run_pmp_v1_4.py`
- `Code/my_code/models/pmp_v1/v1_4/_reviews/2026-06-03__codex_round1.md` (this report)

v1.2 / v1.1 files untouched.

## 7. Ready to run

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_pmp_v1_4.py --epochs 100 --seed 42 --n-dim 32 --tag pmp_v1_4_ndim32_seed42"
```

Expected runtime: ~0.5–0.7h (cluster path dropped; single pool pass per batch,
similar to v1.1 single intersection pool). v1.1 cluster cache v2 reused; no
rebuild needed.

## 8. Param count

| ndim=32 | Total |
|---|---|
| **v1.4** | **2,924,801** |
| (v1.2 reference) | 2,977,489 |
| (v1.1 reference) | 2,976,785 |
| (v1.3 reference) | 2,960,881 |

## 9. Expected comparison

- v1.4 vs v1.2 within_only (0.7344): trains-alone vs trained-with-cluster-zero-out
  - v1.4 > 0.7344 -> joint v1.2 training has cluster branch stealing gradient
  - v1.4 ~= 0.7344 -> within-pool's capacity is the bottleneck
- v1.4 vs v1.1 (0.7751): cleaner within-only baseline
  - v1.1 has dead `cluster_embed (12*d)` + dead EmerGNN backbone params taking
    optimizer slots / gradient noise. v1.4 has none of these.
  - If v1.4 > 0.7751 -> dead params in v1.1 were dragging the within-pool down
  - If v1.4 ~= 0.7751 -> dead params are inert (just wasted memory)
- v1.4 vs v1.2 combined (0.7745): can layer 2 alone match the full model
  - v1.4 close to 0.7745 -> layer 1 is contributing nothing in v1.2 joint
  - v1.4 substantially below 0.7745 -> layer 1 adds something in joint, even if
    cluster_only alone was 0.7016
