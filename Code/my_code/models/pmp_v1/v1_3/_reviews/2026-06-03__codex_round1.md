# PMP v1.3 (Layer-1-only ablation) — Codex Review Report

- **Date**: 2026-06-03
- **Primary reviewer**: Claude (sonnet-4.5) — drafted v1.3 as a new package
  under `Code/my_code/models/pmp_v1/v1_3/`.
- **Independent reviewer**: codex (default model via `mcp__codex__codex`).
- **Triggered by**: 用户 "可以了,可以写1.3和1.4的代码了,记住codex的review,以及正确的落盘".
  Goal: measure Layer 1 (pair-conditional cluster path) standalone cold-start
  AUC when trained in isolation, complementing v1.2's joint-train +
  zero-out-branch diagnostic.
- **Stop condition**: round-1 PASS, no findings. No round 2 needed.

---

## 1. Architectural delta from v1.2 (drop within branch)

```
v1.2: score = MLP_score(concat[z_cluster_pair, z_within])
v1.3: score = MLP_score(z_cluster_pair)        # within-pool DROPPED entirely
```

- Cluster path (W_T, H_a/H_b, psi, mlp_pair_cluster) is byte-identical to v1.2.
- `phi_mlp`, `within_attn_W/w` REMOVED.
- `within_cluster_pool_forward()` REMOVED.
- Intersection mediator tensor construction (`_build_per_cluster_mediator_tensors`)
  NOT called in v1.3 trainer (the helper still exists in v1.1 parent but is
  unused).
- `mlp_score` input dim changes from `hidden + d` (160) to `hidden` (128).

EmerGNN bypass: UNCHANGED from v1.1 / v1.2.
Cache reuse: PMP cache v1, v1.1 cluster cache v2 (no rebuild).

## 2. Locked v1.3 spec

```
Embeddings:
  mediator_embed (n_mediators, d)         Xavier
  type_embed     (n_types, type_dim)
  rel_embed      (n_rels+1, rel_dim)
  W_T            (n_clusters, n_clusters) Xavier
  cluster_attn_W (d, d), cluster_attn_w (d, 1)
  mlp_pair_cluster (4d + 2*rel_dim -> hidden -> hidden)
  mlp_score (hidden -> hidden -> 1)

Pair-conditional cluster repr (same as v1.2):
  H_a[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(a)})
  H_b similarly.

Cluster path (same as v1.2):
  p_a = softmax(c_a),  p_b = softmax(c_b)
  T = softmax(W_T)
  S_ij = p_a[i] * T[i,j] * p_b[j]
  alpha_ij = S_ij / sum_uv S_uv
  psi(i,j,a,b) = concat([H_a^i, H_b^j, |H_a^i - H_b^j|, H_a^i * H_b^j,
                         r_a[:, i, :], r_b[:, j, :]])
  h_path = sum_{ij} alpha_ij * psi
  z_cluster_pair = MLP_pair_cluster(h_path)

Score:
  score = MLP_score(z_cluster_pair)
  loss  = BCE(sigmoid(score), y)
```

## 3. Read inventory

| File | R1 reviewed |
|---|---|
| `Code/my_code/models/pmp_v1/v1_3/__init__.py` | ✓ |
| `Code/my_code/models/pmp_v1/v1_3/cluster_head.py` | ✓ |
| `Code/my_code/models/pmp_v1/v1_3/pmp_v1_3_trainer.py` | ✓ |
| `Code/scripts/run_pmp_v1_3.py` | ✓ |

For comparison (already approved, not re-reviewed):
- v1.2, v1.1, parent v1, MNAH parent, `_format_epoch_log` hook in MNAH.

## 4. Round 1 codex verdict (verbatim)

> No findings.
>
> Key checks passed:
> - v1.3 keeps the v1.2 cluster-path machinery intact and only narrows
>   `mlp_score` to `hidden -> hidden -> 1`: `W_T`, pair-conditional `H_a/H_b`,
>   `psi`, `alpha`, and `mlp_pair_cluster` all match the approved v1.2
>   structure in cluster_head.py:63 versus v1_2/cluster_head.py:74.
> - v1.3 trainer wiring is correct: it inherits v1.2, builds per-drug
>   per-cluster tensors for both drugs, does not build intersection tensors,
>   calls the 8-arg v1.3 forward, and consumes the 2-tuple output in
>   _combined_logit (pmp_v1_3_trainer.py:71) and _predict_branches_v1_3
>   (pmp_v1_3_trainer.py:130).
> - Both trainers preserve the EmerGNN bypass by overriding _combined_logit
>   directly and not calling super()._combined_logit; v1.3 also overrides
>   _format_epoch_log with the requested labels at pmp_v1_3_trainer.py:219.
> - Run script is wired correctly: run_pmp_v1_3.py:98 exposes
>   --cluster-max-mediators-per-drug. Instantiates the right trainer, passes
>   cache args through, and writes results.json plus test_s2_scores.npz.
>
> Parameter-count sanity also matches the spec by layer-shape arithmetic:
> - v1.3 = 2,960,881
>
> Residual risk: this was a static code review only. I did not run
> training/inference in this sandbox.
>
> VERDICT: PASS

## 5. Stop condition

Per user's review-loop spec: stop when codex says correct OR three rounds done.
Round 1 = PASS with no findings. No round 2 executed.

## 6. Files created

- `Code/my_code/models/pmp_v1/v1_3/__init__.py`
- `Code/my_code/models/pmp_v1/v1_3/cluster_head.py`
- `Code/my_code/models/pmp_v1/v1_3/pmp_v1_3_trainer.py`
- `Code/scripts/run_pmp_v1_3.py`
- `Code/my_code/models/pmp_v1/v1_3/_reviews/2026-06-03__codex_round1.md` (this report)

v1.2 / v1.1 files untouched.

## 7. Ready to run

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_pmp_v1_3.py --epochs 100 --seed 42 --n-dim 32 --tag pmp_v1_3_ndim32_seed42"
```

Expected runtime: ~0.8–1.0h (slightly less than v1.2's 1.17h because
within-pool is dropped). v1.1 cluster cache v2 reused; no rebuild needed.

## 8. Param count

| ndim=32 | Total |
|---|---|
| **v1.3** | **2,960,881** |
| (v1.2 reference) | 2,977,489 |
| (v1.1 reference) | 2,976,785 |

## 9. Expected comparison

- v1.3 vs v1.2 cluster_only (0.7016): trains-alone vs trained-with-within-zero-out
  - v1.3 > 0.7016 -> joint v1.2 training has within branch stealing gradient
  - v1.3 ~= 0.7016 -> cluster path's capacity is the bottleneck, not gradient interference
- v1.3 vs v1.2 combined (0.7745): can layer 1 alone match the full model
  - v1.3 close to 0.77 -> layer 2 contributes little marginal info
  - v1.3 substantially below 0.77 -> layer 2 is doing real work in v1.2
