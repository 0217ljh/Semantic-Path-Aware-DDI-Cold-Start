# PMP v1.2 Initial Implementation — Codex Review Report (Round 1 + Round 2)

- **Date**: 2026-06-03
- **Primary reviewer**: Claude (sonnet-4.5) — drafted v1.2 from scratch as a
  new package under `Code/my_code/models/pmp_v1/v1_2/`, applied codex round-1
  fix, ran codex round-2 verification.
- **Independent reviewer**: codex (default model via `mcp__codex__codex`).
- **Triggered by**: 用户 "写代码,并遵循codex的检查流程,落盘到正确位置".
  Designed in conversation as the minimal architectural change vs v1.1 that
  realizes "Layer 1 coarse type aggregation, pair-conditional" (the original
  vision of dividing the KG into special path clusters with mediator identity
  preserved into cluster path).
- **Stop condition**: round-2 PASS_WITH_NITS, sole nit (stale docstring) was
  fixed inline. Codex confirmed no functional issues. No round 3 needed.

---

## 1. Architectural delta from v1.1 (single change)

```
v1.1:  e_i = cluster_embed.weight[i]                            (fixed, shared)
v1.2:  e_i = AttnPool({mediator_embed[m] : m ∈ members(i) ∩ N(a)})   (pair-conditional)
```

Symmetric for `e_j` with drug b. Everything else (within-pool, T = softmax(W_T),
relation context r_a / r_b, MLP_pair_cluster, MLP_score, EmerGNN bypass) is
unchanged. Cluster cache schema v2 is reused as-is.

This realizes the user's "Layer 1 pair-conditional, Layer 2 within-cluster
fine pool" vision discussed before implementation.

## 2. Locked v1.2 spec (Algorithm 1)

```
Embeddings (cluster_embed REMOVED; cluster_attn ADDED):
  mediator_embed (n_mediators, d)        Xavier
  type_embed     (n_types, type_dim)
  rel_embed      (n_rels+1, rel_dim)
  W_T            (n_clusters, n_clusters) Xavier; T = softmax(W_T) at forward
  cluster_attn_W (d, d), cluster_attn_w (d, 1)         # NEW in v1.2
  phi_mlp, within_attn_W, within_attn_w                # unchanged (v1.1)
  mlp_pair_cluster, mlp_score                          # unchanged shapes (v1.1)

Pair-conditional cluster repr (per cluster k, per drug):
  H_a[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(a)})
  H_b[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(b)})
  Empty M_k(a) -> H_a[:, k, :] = 0.

Cluster path:
  p_a = softmax(c_a), p_b = softmax(c_b)
  T = softmax(W_T)
  S_ij    = p_a[i] * T[i, j] * p_b[j]
  alpha_ij = S_ij / sum_uv S_uv
  e_i      = H_a[:, i, :]                # CHANGED vs v1.1
  e_j      = H_b[:, j, :]                # CHANGED vs v1.1
  psi(i,j,a,b) = concat([e_i, e_j, |e_i - e_j|, e_i * e_j,
                         r_a[:, i, :], r_b[:, j, :]])
  h_path   = sum_{ij} alpha_ij * psi
  z_cluster_pair = MLP_pair_cluster(h_path)

Within-cluster pool (UNCHANGED from v1.1):
  for k in 0..K-1:
    M_k = members(k) ∩ N_<=2(a) ∩ N_<=2(b)
    z_within^k = AttnPool({phi(m) for m in M_k}) or 0
  z_within = sum_k (p_a[k] * p_b[k]) * z_within^k

Score:
  z_pair = concat([z_cluster_pair, z_within])
  score  = MLP_score(z_pair)
  loss   = BCE(sigmoid(score), y)
```

### User explicit confirmations baked in

- Pair-conditional cluster repr uses ONLY `mediator_embed` (no type/rel fusion inside H_a^k).
- Separate `cluster_attn_W / w` (not shared with within_attn) to keep two pools distinct.
- `cluster_embed` parameter is GONE (not just unused).
- `members(k) ∩ N(a)` covers both 1-hop and 2-hop with 1-hop precedence on dedup.
- Per-(drug, cluster) cap default = 64 with deterministic ascending PMP-id truncation.
- Empty M_k(a) -> zero vector for that cluster slot (not skipped, not fallback to cluster_embed).

## 3. Read inventory

| File | R1 reviewed | R2 reviewed |
|---|---|---|
| `Code/my_code/models/pmp_v1/v1_2/__init__.py` | ✓ | ✓ |
| `Code/my_code/models/pmp_v1/v1_2/cluster_head.py` | ✓ | ✓ |
| `Code/my_code/models/pmp_v1/v1_2/pmp_v1_2_trainer.py` | ✓ | ✓ |
| `Code/scripts/run_pmp_v1_2.py` | ✓ | ✓ |
| `Code/my_code/models/pmp_v1/v1_1/*` | reference only | reference only |
| `Code/my_code/models/pmp_v1/pmp_trainer.py` | reference only | reference only |
| `Code/my_code/models/pmp_v1/precompute_pmp_cache.py` | reference only | — |

## 4. Round 1 issues + applied fixes

| # | Severity | Issue | File:line | Fix | Verified |
|---|---|---|---|---|---|
| 1 | MAJOR | `_load_feature_cache()` built `_drug_per_cluster_mediators` only from `set(self._pmp_n1.keys()) \| set(self._pmp_n2.keys())`. PMP cache builder only materializes keys for drugs with at least one 1-hop or 2-hop mediator. A cluster-cache-resident drug with empty PMP neighborhood would have no entry, and `_build_per_drug_per_cluster_tensors` would raise `KeyError` instead of producing K empty lists per spec (empty M_k(a) -> H_a[:, k, :] = 0). | `pmp_v1_2_trainer.py:109`, batch lookup at `pmp_v1_2_trainer.py:231`. | Seed `per_drug` from the UNION of `self._pmp_n1.keys()`, `self._pmp_n2.keys()`, and `self._cluster_drug2id` keys. Inner loop uses `.get(d, [])` so a drug with no n1 / n2 entries keeps K empty lists naturally. Added `n_empty_drugs` diagnostic counter. | codex round-2 confirmed at `pmp_v1_2_trainer.py:119` (union seeding), `:128` `:132` (`.get` fallback), `:262` (M_max_k = max(.., 1) handles empty), `:175` (counter in log). |

## 5. Round 1 NOT_PASS verdict (verbatim from codex)

> **MAJOR** pmp_v1_2_trainer.py:109, pmp_v1_2_trainer.py:231
> Spec / expected behavior: `Empty M_k(a) -> H_a[:, k, :] = 0` for any cluster.
> If a drug has no PMP mediators at all, that should degenerate to all-zero
> `H_a`, not an exception.
> Actual implementation: `_load_feature_cache()` only builds
> `_drug_per_cluster_mediators` for `set(self._pmp_n1.keys()) | set(self._pmp_n2.keys())`.
> The PMP cache builder only materializes keys for drugs with at least one
> 1-hop or 2-hop mediator (precompute_pmp_cache.py:177, precompute_pmp_cache.py:210).
> A cluster-cache-resident drug with empty PMP neighborhood therefore has no
> entry, and `_build_per_drug_per_cluster_tensors()` raises `KeyError` instead
> of returning K empty lists / zero slots.
> Fix recommendation: Seed `per_drug` for all relevant drugs, ideally every
> `self._cluster_drug2id` entry, with `[[] for _ in range(K)]`, then populate
> from `n1/n2` when present.
>
> ...
>
> **VERDICT**: `NOT_PASS`
>
> Must fix for round 2:
> 1. Make per-drug cluster tensors support drugs with empty PMP neighborhoods
>    by emitting all-empty/all-zero cluster slots instead of raising `KeyError`.

## 6. Round 2 PASS_WITH_NITS verdict (verbatim)

> No must-fix findings in the round-2 fix.
>
> 1. Confirmed: `_load_feature_cache()` now seeds from the union of
>    `self._pmp_n1.keys()`, `self._pmp_n2.keys()`, and `self._cluster_drug2id`
>    keys (pmp_v1_2_trainer.py:119).
> 2. Confirmed: each seeded drug starts with `per_cluster = [[] for _ in range(K)]`,
>    and both PMP loops use `.get(d, [])`, so a drug with no n1 and/or no n2
>    entries keeps K empty lists rather than failing.
> 3. Confirmed: `_build_per_drug_per_cluster_tensors()` handles the
>    K-empty-lists case correctly. For an empty cluster list, M_max_k is forced
>    to 1, med_t and mask_t remain all zeros for that row, and no write occurs
>    because n == 0 short-circuits. Downstream, AttnPool safely leaves those
>    rows at zero via any_valid, zero-row softmax protection, and final zeroing
>    in cluster_head.py:165. That matches the spec.
> 4. Confirmed: the diagnostic log now includes n_empty_drugs at
>    pmp_v1_2_trainer.py:175.
> 5. New-issue check: I do not see a functional regression from seeding with
>    cluster_drug2id. cluster_drug2id is the validated cluster-cache drug
>    universe; adding entries for cluster-cache drugs that have zero PMP
>    neighbors is the correct behavior; those entries just carry K empty lists
>    and produce zero cluster representations.
>
> Nits:
> - The docstring/comment in `_build_per_drug_per_cluster_tensors()` still
>   says the per-drug cache is keyed by the union of only PMP n1/n2 keys;
>   after this fix that statement is stale.
>
> VERDICT: PASS_WITH_NITS

## 7. Round 2 nit fix (post-PASS, behaviorally inert)

Stale docstring in `_build_per_drug_per_cluster_tensors` rewritten to reflect
the union-with-cluster_drug2id seeding (the actual code post-fix). No code
behavior changes. Applied at `pmp_v1_2_trainer.py:227-233`.

## 8. Architectural invariants verified by codex round 2

| Invariant | Status |
|---|---|
| `H_a[:, k, :] = AttnPool({mediator_embed[m] : m ∈ members(k) ∩ N(a)})`, mediator_embed only (no type/rel fusion). | ✓ `cluster_head.py:135-175` (`_pair_conditional_cluster_repr`). |
| `e_i` broadcasts along K-rows from H_a, `e_j` along K-cols from H_b. | ✓ `cluster_head.py:241-242`. |
| H_a uses drug-a's per-cluster tensors, H_b uses drug-b's; NOT swapped or shared. | ✓ `pmp_v1_2_trainer.py` `_combined_logit:288-294` passes a_list / b_list to the helper independently. |
| Separate `cluster_attn_W/w` (not shared with within_attn). | ✓ `cluster_head.py:90-92`. |
| `cluster_embed` parameter removed (not just unused). | ✓ `cluster_head.py:78` — no nn.Embedding for cluster_embed exists. |
| 1-hop precedence on dedup, type-id bucketing, ascending sort before cap. | ✓ `pmp_v1_2_trainer.py:125-159`. |
| Mask-based padding semantics (pad slot id 0 is not relied on as identifier). | ✓ `cluster_head.py:160-167` (mask drives softmax, not index values). |
| Cache schema reused v1.1 v2 without modification. | ✓ `pmp_v1_2_trainer.py:32-33` imports `DEFAULT_CLUSTER_CACHE` from v1.1. No new persisted fields. |
| Trainer wiring: `_build_aux_head`, `_combined_logit`, `_predict_branches_v1_2`. | ✓ all overridden; `_predict_branches` delegates to v1_2 variant. |
| CLI: `cluster_max_mediators_per_drug=0` -> None (cap disabled). | ✓ `run_pmp_v1_2.py:121-124`. |
| Cold-start invariant: score uses no drug-id embedding from EmerGNN backbone. | ✓ structural — same as v1.1, only cluster path + within-pool feed `mlp_score`. |
| Drugs with empty PMP neighborhood produce well-defined zero H without error. | ✓ codex round 2 confirmed. |

## 9. Param count comparison (ndim=32)

| Param group | v1.1 | v1.2 | Delta |
|---|---|---|---|
| mediator_embed (90855, 32) | 2,907,360 | 2,907,360 | 0 |
| cluster_embed (12, 32) | 384 | — | -384 |
| cluster_attn_W (32, 32) + bias | — | 1,056 | +1,056 |
| cluster_attn_w (32, 1, no bias) | — | 32 | +32 |
| W_T (12, 12) | 144 | 144 | 0 |
| type_embed (12, 8) | 96 | 96 | 0 |
| rel_embed (60, 8) | 480 | 480 | 0 |
| phi_mlp 56→128→32 | 11,424 | 11,424 | 0 |
| within_attn_W, _w | 1,088 | 1,088 | 0 |
| mlp_pair_cluster | 35,072 | 35,072 | 0 |
| mlp_score | 20,737 | 20,737 | 0 |
| **TOTAL** | **2,976,785** | **2,977,489** | **+704** |

Net change. **+704 parameters (+0.024%)** for the architectural lift.

## 10. Stop condition

Per user's review-loop spec: stop when codex says correct OR three rounds done.
Round 1 = NOT_PASS (1 MAJOR, all else clean). Round 2 = PASS_WITH_NITS (1
docstring nit, fixed inline). No round 3 executed.

## 11. Files created

- `Code/my_code/models/pmp_v1/v1_2/__init__.py`
- `Code/my_code/models/pmp_v1/v1_2/cluster_head.py`
- `Code/my_code/models/pmp_v1/v1_2/pmp_v1_2_trainer.py`
- `Code/scripts/run_pmp_v1_2.py`
- `Code/my_code/models/pmp_v1/v1_2/_reviews/2026-06-03__codex_round1_round2.md` (this report)

v1.1 files untouched per "不得删除或重构已有函数/类/文件" rule.

## 12. Ready to run

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_pmp_v1_2.py --epochs 100 --seed 42 --n-dim 32 --tag pmp_v1_2_ndim32_seed42"
```

Expected runtime: ~0.8h on the same GPU as v1.1 ndim=32 (the per-cluster pool
loop runs twice, once for a_list and once for b_list, vs v1.1's single
intersection pool — bigger constant factor but same asymptotic). Cluster cache
v2 from v1.1 is reused; no rebuild needed.
