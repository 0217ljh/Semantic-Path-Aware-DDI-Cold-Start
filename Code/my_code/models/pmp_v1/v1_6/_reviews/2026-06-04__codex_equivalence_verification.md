# PMP v1.6 — Equivalence Verification Report

- **Date**: 2026-06-04
- **Primary reviewer**: Claude (sonnet-4.5) — implemented v1.6 as PyG MessagePassing rewrite of v1.5 A.
- **Independent reviewer**: codex (default model via `mcp__codex__codex`).
- **Triggered by**: User asked to design GNN-equivalent of v1.5 A with codex verification.
- **Stop condition**: codex VERDICT = `PASS_EQUIVALENT_WITH_CAVEATS`, smoke test max abs diff = 2.98e-08 < 1e-5. No further rounds.

---

## 1. Goal

Re-implement v1.5 A as a PyG `MessagePassing`-based GNN with strict
mathematical equivalence to the v1.5 A forward output, for ANY input and
any shared learnable parameter values, within float32 epsilon.

Purpose:
- Sanity check the GNN-paradigm framing of v1.5 A (not a new architecture).
- Provide a PyG-derived base for future GNN-specific extensions (R-GCN typed
  message kernels, DiffPool, HGT-style heterogeneous attention).
- Paper positioning: "PMP-v1.6 = GNN-form formulation of the v1.5 A
  analytical pipeline."

## 2. Smoke test (empirical)

Identical seed init for shared learnable params, random synthetic batch
(B=6, K=12, d=16, n_mediators=200), `eval()` mode:

```
score    max abs diff: 0.00e+00   (bitwise identical)
z_evi    max abs diff: 2.98e-08   (float32 epsilon from scatter vs loop sum order)
alpha    max abs diff: 0.00e+00
attn_a   max abs diff: 0.00e+00
attn_b   max abs diff: 0.00e+00
```

PASS at the 1e-5 threshold.

## 3. Codex VERDICT (verbatim)

> VERDICT: PASS_EQUIVALENT_WITH_CAVEATS
>
> Assumptions required:
> - Compare against the actual canonical v1.5 A regime in this repo, where
>   `lambda_content == 0` and the `g_mlp` last layer is zeroed, so Step 1
>   reduces to pure `base_logit`.
> - Judge forward equivalence in `eval()` mode, or ignore dropout randomness
>   entirely. Train-mode bitwise matching is not guaranteed by shared seed
>   alone.
> - Inputs are the valid inherited intersection tensors from the trainer
>   (`med_idx_per_cluster`, `rel_a_per_cluster`, `rel_b_per_cluster`,
>   `mask_per_cluster`, `type_id_per_cluster` all consistent and non-`None`).
>
> Under those assumptions, the PyG rewrite is a real mathematical rewrite of
> v1.5 A's live forward path, and the smoke-test result is consistent with
> the code.

All three caveats are satisfied by design and by the v1.5 A canonical lock
(see `v1_5/_reviews/2026-06-04__official_v1_5_lock.md`).

## 4. Codex equivalence findings (8 items, per-claim)

| # | Severity | Claim | Verified by codex |
|---|---|---|---|
| 1 | MAJOR | Step 1 equivalence depends on `lambda_content==0` | YES — true by design in canonical v1.5 A |
| 2 | MINOR | Step 1 base_logit + flatten-softmax identical | YES |
| 3 | MINOR | Step 2 `phi_in` concat order [mediator, type, rel_a, rel_b] matches | YES |
| 4 | MINOR | Step 2 within-softmax groups (per-(b,k)) match | YES (each `K*b+k` index group = one v1.5 A softmax group) |
| 5 | MINOR | Step 2 aggregation (sum) matches up to float32 epsilon | YES (2.98e-08 observed = expected scatter-vs-loop accumulation order eps) |
| 6 | MINOR | Empty-cluster handling matches (both yield zero) | YES |
| 7 | MINOR | Layout `K*b+k` row-major + `view(B, K, d)` correct | YES |
| 8 | MAJOR | Train-mode dropout NOT guaranteed bitwise equivalent | YES — this is an expected caveat, not a code bug; eval-mode is the canonical comparison |
| 9 | MINOR | Compatibility shim inputs (`med_a/b_per_cluster`) properly ignored | YES |

## 5. Architecture mapping (v1.5 A -> v1.6)

| v1.5 A operation | v1.6 operation | Equivalent? |
|---|---|---|
| `affinity_a/b` from cache | identical | ✓ |
| `T = softmax(W_T, dim=-1)` | identical | ✓ |
| `base_logit = log p_a + log T + log p_b` | identical | ✓ |
| `alpha = softmax over (i,j) of base_logit` | identical (view/softmax/view) | ✓ |
| `attn_a^k = alpha.sum(dim=2)` | identical | ✓ |
| `attn_b^k = alpha.sum(dim=1)` | identical | ✓ |
| `for k: pool over M_k^{a∩b}` | `MessagePassing.propagate` with bipartite edges | ✓ |
| `phi(m) = MLP([med_embed[m], type_embed[k], rel(a->m), rel(m->b)])` | `message(x_j, edge_attr) = phi_mlp(concat[x_j, edge_attr])` | ✓ |
| Per-(b,k) attention softmax over phi(m) values | `pyg_softmax(scores, dst_cluster_global_idx, num_nodes=K*B)` | ✓ |
| `z_within^k = sum_m alpha_m * phi(m)` | `scatter_add_(0, dst_idx, alpha * phi)` | ✓ (float32 eps) |
| `z_evidence_a = sum_k attn_a^k * z_within^k` | identical einsum | ✓ |
| `z_evidence_b = sum_k attn_b^k * z_within^k` | identical einsum | ✓ |
| `score = MLP_score(concat[z_a, z_b])` | identical | ✓ |

## 6. What's in v1.6 that's NOT in v1.5 A

Nothing. v1.6 is a strict subset of v1.5 A architecturally: it omits
v1.5 A's dead-code components (`cluster_attn_W/w`, `g_mlp`,
`lambda_content`, `H_a/H_b` computation). Since those are dead in v1.5 A
canonical regime (lambda_content=0), omitting them in v1.6 does not
change the output.

Param count:
- v1.5 A:    2,931,683 (includes ~2.6K dead-code params)
- v1.6:  ~2,929,000 (estimate, minus dead-code components)

## 7. Bipartite graph construction

The pair-specific subgraph has:
- Mediator "edge slot" nodes: one per `(pair_b, cluster_k, m in M_k^{a∩b}_b)` triple
- Cluster nodes: 12 per pair, layout `cluster_global = K * b + k` for batched view

Edge per (b, k, m in M_k^{a∩b}_b):
- source slot index: `0..E-1` (each edge gets unique source slot, NO mediator
  deduplication — mediator m may appear in multiple edges if it's in
  multiple pairs' intersections)
- target cluster index: `K * b + k`
- edge_attr: `concat[type_embed[k], rel_embed[rel(a->m)], rel_embed[rel(m->b)]]`
- source feature: `mediator_embed[m]`

Empty clusters / pairs with no valid intersection mediators contribute no
edges, so the corresponding target cluster receives zero from `scatter_add_`,
matching v1.5 A's "skip cluster with no any_valid" behavior.

## 8. Files created

- `Code/my_code/models/pmp_v1/v1_6/__init__.py` — spec docstring
- `Code/my_code/models/pmp_v1/v1_6/cluster_head.py` — PMPv1_6_Module + WithinClusterMP (PyG `MessagePassing`)
- `Code/my_code/models/pmp_v1/v1_6/pmp_v1_6_trainer.py` — trainer (inherits v1.5)
- `Code/scripts/run_pmp_v1_6.py` — CLI
- `Code/my_code/models/pmp_v1/v1_6/_reviews/2026-06-04__codex_equivalence_verification.md` (this file)

v1.5 A files: UNTOUCHED.

## 9. Run command

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_pmp_v1_6.py --epochs 100 --seed 42 --n-dim 32 --tag pmp_v1_6_ndim32_seed42"
```

Expected result:
- AUC ~ 0.7774 ± 0.002 (matches v1.5 A within reproducibility noise)
- AUPRC ~ 0.7943 ± 0.005
- fit time: ~1.0-1.3h (similar to v1.5 A)

Note: PyG `MessagePassing` introduces some overhead vs the dense loop in
v1.5 A. Expected ~10-20% slowdown. Trade-off accepted for the cleaner
extensibility.

## 10. Paper claim

After both runs (v1.5 A and v1.6) match within reproducibility noise,
paper can claim:

> "PMP-v1.6 is a GNN-form formulation of the v1.5 A analytical pipeline,
> verified mathematically equivalent (forward output diff < 1e-5 under shared
> parameter values) and empirically matched within seed-noise on cold-start
> S2 (AUC = X.X ± 0.X)."

This gives a clean two-line story:
1. The v1.5 A analytical formulation is theoretically clean.
2. The v1.6 form provides a base for GNN-specific extensions
   (typed messages, hierarchical pooling, cross-pair attention, etc.).

## 11. Next steps unlocked

With v1.6 in place, future GNN-specific evolutions can extend
`WithinClusterMP` or replace it with PyG built-ins:

- **v1.6-GNN-R**: replace `phi_mlp` with R-GCN-style typed transform (per-cluster Linear)
- **v1.6-GNN-HGT**: replace within-pool with Heterogeneous Graph Transformer head
- **v1.6-GNN-cross-cluster**: add cluster-to-cluster message passing on top of attention marginals
- **v1.7-GNN-flat**: dissolve the explicit cluster-cluster attention and let GAT learn cluster-cluster edges end-to-end

These would all be `class Foo(WithinClusterMP)` extensions, requiring minimal
new code.
