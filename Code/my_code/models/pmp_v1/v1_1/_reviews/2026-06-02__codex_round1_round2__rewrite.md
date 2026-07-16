# PMP v1.1 Rewrite — Codex Review Report (Round 1 + Round 2)

- **Date**: 2026-06-02
- **Primary reviewer**: Claude (sonnet-4.5) — drafted the v1.1 overwrite,
  applied codex round-1 fixes, ran codex round-2 verification.
- **Independent reviewer**: codex (default model via `mcp__codex__codex`).
- **Triggered by**: 用户 "这是 1.1，你别改错了" 指出之前的 v1.1 实现 silently
  简化了用户设计，要求重写整个 v1.1 为完整 cluster path + per-cluster
  within-pool + EmerGNN replace 版本。
- **Stop condition**: round-2 PASS satisfies the user's rule (codex 认为
  代码正确). No round 3 needed.

---

## 1. Why this is a rewrite, not an iteration

The first v1.1 implementation (committed earlier the same session) used a
simplified scalar-per-cluster propagation that diverged from the locked
spec. Codex round 1 of that first attempt PASS_WITH_NITS, but the user then
identified four substantive deltas from the original intent. This rewrite
replaces the cluster module, trainer, and cache builder to match the spec
exactly. The earlier review report
`2026-06-02__codex_round1_round2.md` is retained per "do not delete history"
rule.

## 2. Locked v1.1 spec (Algorithm 1, do not change without re-review)

```
Embeddings:
  mediator_embed (n_mediators, d), Xavier init
  cluster_embed (12, d), init from member mediator mean (trainable after)
  type_embed (n_types, type_dim)
  rel_embed (n_rels_plus_special, rel_dim)
  W_T (12, 12) Xavier init; T = softmax(W_T) at forward (LEARNABLE)

Cluster path (explicit pairwise, single transition):
  p_a = softmax(c_a); p_b = softmax(c_b)
  r_a = drug_cluster_rel_dist[a] @ rel_embed.weight    (B, K, rel_dim)
  r_b similarly
  S_ij    = p_a[i] * T[i, j] * p_b[j]
  alpha_ij = S_ij / sum_uv S_uv
  psi(i,j,a,b) = concat([
    e_i, e_j, |e_i - e_j|, e_i * e_j, r_a[:, i, :], r_b[:, j, :]
  ])
  h_path = sum_{ij} alpha_ij * psi
  z_cluster_pair = MLP_pair_cluster(h_path)

Per-cluster within-pool:
  for k in 0..11:
    M_k = members(cluster_k) ∩ N_<=2(a) ∩ N_<=2(b)
    z_within^k = AttnPool({phi(m) for m in M_k}) or zero
      phi(m) = MLP([mediator_embed[m], type_embed[k], rel_embed[a->m], rel_embed[m->b]])
  z_within = sum_k (p_a[k] * p_b[k]) * z_within^k

Score (REPLACES EmerGNN backbone entirely):
  z_pair = concat([z_cluster_pair, z_within])
  score = MLP_score(z_pair)
  loss = BCE(sigmoid(score), y)
Training protocol: shuffle_train(mode='S2') inherited.
```

User confirmations.
- cluster_embed init = mean(member mediator embeds) only, no type bias
- per-cluster within-pool aggregation = scalar sum sum_k z_within^k
- EmerGNN replaced entirely (no residual fusion)
- T learnable via softmax(W_T)
- psi includes relation context r_a_to_i and r_j_to_b
- cluster_embed init prior preservation = none (free training)

## 3. Read inventory

| File | Reviewed by codex (R1) | Reviewed by codex (R2) |
|---|---|---|
| `Code/my_code/models/pmp_v1/v1_1/cluster_head.py` | ✓ | ✓ |
| `Code/my_code/models/pmp_v1/v1_1/pmp_v1_1_trainer.py` | ✓ | ✓ |
| `Code/my_code/models/pmp_v1/v1_1/precompute_cluster_cache.py` | ✓ | ✓ |
| `Code/scripts/run_pmp_v1_1.py` | ✓ | ✓ |
| `Code/scripts/precompute_pmp_v1_1_cluster_cache.py` (existing CLI wrapper) | ✓ | ✓ |
| `Code/my_code/models/pmp_v1/v1_1/__init__.py` | ✓ | ✓ |

## 4. Round 1 issues + applied fixes

| # | Severity | Issue | File:line | Fix | Verified |
|---|---|---|---|---|---|
| 1 | CRITICAL | `drug_cluster_affinity` was computed as a sum of relation incidences, multi-counting mediators with multiple typed edges. Spec requires `log1p(|N_<=2(d) ∩ cluster_k|)` = log of UNIQUE mediator count per cluster. | `precompute_cluster_cache.py` pass-2 loop, originally summing `drug_cluster_rel_dist[d_idx].sum(axis=1)`. | Track per-cluster unique mediator set during pass 2 (`cluster_unique_mediators[k]`), 1-hop and 2-hop both add to the set, then `affinity[d_idx, k] = log1p(len(cluster_unique_mediators[k]))`. `drug_cluster_rel_dist` continues to use per-edge counts (correct for relation distribution semantics). | codex round-2 confirmed at `precompute_cluster_cache.py:208–241`. |
| 2 | MAJOR | Trainer never validated `type2id` / `kind_order` / `rel2id` consistency between PMP cache and v1.1 cluster cache. Would silently misread cluster IDs and relation channels if the two caches were built from different KG parquets. | `pmp_v1_1_trainer.py` `_load_feature_cache`. | Re-load the PMP cache to extract these fields (v1's loader does not stash them as attributes), then validate equality with v1.1 cluster cache before any further use. Raise `RuntimeError` with rebuild instructions on mismatch. Conditional on fields being present (allows older PMP caches without `kind_order`). | codex round-2 confirmed at `pmp_v1_1_trainer.py:138–188`. |
| 3 | MINOR | Docstring claimed `_predict_branches` was overridden, but it was not. Training val log emitted meaningless `sigmoid(0) ≈ 0.5` values for the parent MNAH `emergnn` / `aux` branches. | `pmp_v1_1_trainer.py`. | Added a thin `_predict_branches` override that delegates to `_predict_branches_v1_1`. The training-val dict keys still say `emergnn` / `aux` (parent baked in), but the values now reflect cluster-only / within-only AUC. | codex round-2 confirmed at `pmp_v1_1_trainer.py:465–467`. |

## 5. Codex independent verdicts (verbatim)

### Round 1 — NOT_PASS

> **Critical**
> - `precompute_cluster_cache.py:208` `drug_cluster_affinity` is computed
>   from summed relation counts, not unique mediator membership. … Fix:
>   build per-cluster unique mediator sets from `n1_set ∪ n2_med_set` and
>   compute affinity from those cardinalities, then rebuild the v2 cache.
>
> **Major**
> - `pmp_v1_1_trainer.py:145` … never validates `type2id`, `kind_order`,
>   `rel2id` between PMP and v1.1 cluster caches. … Fix: load/store PMP
>   `type2id` and `rel2id`, validate exact equality with the cluster cache.
>
> **Minor**
> - `pmp_v1_1_trainer.py:35` says `_predict_branches` is overridden, but it
>   is not.
> - `__init__.py` is docstring-only and exports nothing.
> - A few imports are unused.
>
> **VERDICT** `NOT_PASS`
>
> **Must fix for round 2**
> 1. Fix `drug_cluster_affinity` to count unique mediators per cluster, …
> 2. Add strict cache-consistency validation for `type2id/kind_order/rel2id`
>    between PMP and v1.1 cluster caches, and stop assuming `cluster k ==
>    type id k` unless validated.

### Round 2 — PASS

> **Round 1 Fix Check**
> 1. `CRITICAL` drug-cluster affinity unique-count semantics:
>    `CONFIRMED_FIXED`. `precompute_cluster_cache.py:208` now builds
>    `cluster_unique_mediators` as a per-cluster set, … `drug_cluster_rel_dist`
>    still increments per edge / per 2-hop mediator event …, so the fix is
>    scoped correctly.
> 2. `MAJOR` cache-consistency validation: `CONFIRMED_FIXED`.
>    `pmp_v1_1_trainer.py:138` reopens the PMP cache, extracts `type2id` /
>    `kind_order` / `rel2id`, then compares them to the v1.1 cluster cache
>    and raises `RuntimeError` on mismatch.
> 3. `MINOR` `_predict_branches` override: `CONFIRMED_FIXED`.
>    `pmp_v1_1_trainer.py:465` overrides `_predict_branches` and delegates
>    directly to `_predict_branches_v1_1` …
>
> **New Issues** — No new CRITICAL or MAJOR issues found in the reviewed
> files.
>
> **Verdict** `PASS`. No round-3 fixes required.

## 6. Acknowledged minor non-blocking items

These were called out in round 1 minor but are non-behavioral and were not
re-fixed in round 2 to keep the loop tight.

- `__init__.py` is docstring-only (no `__all__` re-export). Canonical
  imports work through the explicit module paths documented in the docstring.
- A couple of unused imports in `cluster_head.py` (`numpy as np`) and
  `pmp_v1_1_trainer.py` (`torch.nn.functional as F`, `roc_auc_score`).
  These do not affect behavior.

## 7. Architectural invariants verified

| Invariant | Status |
|---|---|
| `cluster_embed[k]` initialized from mean of `mediator_embed[m]` for `m ∈ members(k)`. | ✓ `cluster_head.py:143–164`, called from `pmp_v1_1_trainer.py:244–260`. |
| `T = softmax(W_T)` at every forward. | ✓ `cluster_head.py:170–172`. |
| `alpha_ij` normalized over (i, j) to sum to 1 per batch sample. | ✓ `cluster_head.py:209–233`. |
| `psi` includes `e_i`, `e_j`, `|e_i - e_j|`, `e_i * e_j`, `r_a_to_i`, `r_j_to_b`. | ✓ `cluster_head.py:209–233`. |
| `z_within = sum_k (p_a[k] * p_b[k]) * z_within^k`. | ✓ `cluster_head.py:282–306`. |
| Final score is `MLP_score(concat[z_cluster_pair, z_within])` only. | ✓ `cluster_head.py:333–341`. |
| `_combined_logit` returns the v1.1 score with NO EmerGNN backbone term. | ✓ `pmp_v1_1_trainer.py:409–419`. |
| EmerGNN backbone params remain in optimizer but receive zero gradient. | ✓ structural — backbone forward output not added to score. |
| Cluster cache schema v2 validated; v1 cache schema rejected. | ✓ `pmp_v1_1_trainer.py:160–165`. |
| PMP cache `type2id` / `kind_order` / `rel2id` consistency check. | ✓ `pmp_v1_1_trainer.py:169–188`. |
| Drug coverage check rejects training with drugs missing from cluster cache. | ✓ `pmp_v1_1_trainer.py:223–235`. |
| Cold-start invariant: score function input contains no drug-id embedding from EmerGNN backbone. | ✓ structural — v1.1 forward consumes only `affinity_a`, `affinity_b`, `rel_dist_a`, `rel_dist_b`, and per-cluster mediator tensors. |

## 8. Stop condition

Per user's review-loop spec: stop when codex says correct OR three rounds
done. Codex round 2 returned `PASS` with no new issues. Round 3 NOT executed.
