# PMP v1.5 (sequential cluster-attention + content residual) — Codex Design Round + Review Report

- **Date**: 2026-06-03
- **Primary reviewer**: Claude (sonnet-4.5) — drafted v1.5 as a new package
  under `Code/my_code/models/pmp_v1/v1_5/`.
- **Independent reviewer / design consultant**: codex (default model via
  `mcp__codex__codex`). Codex provided the architectural recommendation
  ("Option B with content residual") AND the static code review.
- **Triggered by**: 用户 ablation results showed each v1.2 layer trained alone
  significantly outperformed its joint-training diagnostic, indicating
  gradient competition over the shared mediator_embed. v1.5 redesigns the
  architecture as sequential (Layer 1 attention -> Layer 2 evidence) with a
  prior + content residual attention. User: "用这个Codex Option B，写的是
  注意之前跟你说的规范".
- **Stop condition**: round 1 PASS with zero findings (the design round
  already established the spec; code matches the spec). No round 2 needed.

---

## 1. Design discussion (codex round 0)

### Background numbers user shared with codex

| Variant | AUC |
|---|---|
| v1.1 full | 0.7751 |
| v1.2 full (parallel concat fusion) | 0.7745 |
| v1.2 cluster_only diagnostic | 0.7016 |
| v1.2 within_only diagnostic | 0.7344 |
| v1.3 (L1 only, standalone train) | **0.7414** |
| v1.4 (L2 only, standalone train) | **0.7703** |

Critical observation: each layer standalone outperformed its joint-train
diagnostic by ~3.6 pt. Joint training was creating gradient competition over
the shared mediator_embed, and the parallel fusion was leaving ~3 pt on the
table.

### Three options codex evaluated

- **A**: Drop H_a / H_b. Pure affinity attention `alpha = softmax(p_a * T * p_b)`. Simplest, but loses cold-start identity signal in alpha.
- **B**: Content residual on alpha logit. `log alpha = log p_a + log T + log p_b + lambda * g(H_a, H_b)` with `g` small + zero-init last layer + `lambda` init 0.
- **C**: Inject H_a / H_b into phi(m). Each mediator phi sees cluster context. Risks reintroducing gradient competition pathology of v1.2.

### Codex verdict (verbatim)

> My recommendation is **Option B**, but in a constrained form:
> e_{ij} = log p_a[i] + log T[i,j] + log p_b[j] + lambda * g(H_a^i, H_b^j)
> alpha = softmax over (i, j) of e_{ij}
> where `g` is a small additive content residual, not a full unconstrained
> replacement for `p_a * T * p_b`. That keeps the user's `T = softmax(W_T)`
> cluster-to-cluster routing as the backbone, while letting the pair-conditional
> mediator summaries actually decide when a given (i, j) pair is unusually
> relevant for this drug pair.
>
> I would rank the options for v1.5 as B > A >> C.
>
> [...]
>
> Bottom line:
> If the target is to beat both v1.2 = 0.7745 and v1.4 = 0.7703, the
> highest-upside design that still respects your constraints is:
> - Choose B
> - but implement it as prior + content residual, not as a fully free alpha MLP
> - do not add any separate z_cluster_pair
> - do not inject H_a/H_b into phi for v1.5
>
> Cited supporting literature: GATv2 (How Attentive are GATs?), Hierarchical
> Question-Image Co-Attention, Hierarchical Attention Networks, Dynamic
> Coattention Networks.

User chose Option B.

## 2. Locked v1.5 spec (Algorithm 1)

```
Embeddings:
  mediator_embed (n_mediators, d)
  type_embed     (n_types, type_dim)
  rel_embed      (n_rels+1, rel_dim)
  W_T            (n_clusters, n_clusters)
  cluster_attn_W (d, d), cluster_attn_w (d, 1)        for H_a / H_b pool
  g_mlp (3d -> content_hidden -> 1)                    content residual; LAST LAYER ZERO-INIT
  lambda_content (scalar nn.Parameter)                 INIT 0.0; learnable
  phi_mlp (d + type_dim + 2*rel_dim -> hidden -> d)    within-pool
  within_attn_W (d, d), within_attn_w (d, 1)
  mlp_score (2*d -> hidden -> 1)

Step 1 (sequential, content-residual attention):
  H_a[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(a)})
  H_b[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(b)})

  log_p_a = log_softmax(c_a)
  log_p_b = log_softmax(c_b)
  log_T   = log_softmax(W_T, dim=-1)
  base_logit[i, j] = log_p_a[i] + log_T[i, j] + log_p_b[j]

  content_input[i, j] = concat[H_a^i, H_b^j, H_a^i ⊙ H_b^j]
  content_logit[i, j] = g_mlp(content_input)

  total_logit = base_logit + lambda_content * content_logit
  alpha       = softmax over JOINT (i, j) axes of total_logit

  attn_a^k = sum_j alpha[k, j]
  attn_b^k = sum_i alpha[i, k]

Step 2 (attention-marginal weighted within-pool):
  for k in 0..K-1:
    z_within^k = AttnPool(phi(m) for m in members(k) ∩ N(a) ∩ N(b))
                 with phi(m) = MLP([mediator_embed[m], type_embed[k],
                                    rel_embed[a->m], rel_embed[m->b]])
  z_evidence_a = sum_k attn_a^k * z_within^k   (B, d)
  z_evidence_b = sum_k attn_b^k * z_within^k   (B, d)
  z_evidence   = concat[z_evidence_a, z_evidence_b]   (B, 2d)

Score:
  score = MLP_score(z_evidence)
  loss  = BCE(sigmoid(score), y)
```

Key invariants:
- `lambda_content` init 0.0 + `g_mlp[-1]` weight & bias zeroed -> alpha at
  init equals pure-affinity softmax (Option A behavior). Model gradually
  learns to incorporate content-conditional attention.
- mediator_embed has ONE coherent gradient path: through phi -> z_stack ->
  score. H_a / H_b multiply by `lambda_content` (init 0) in alpha, so at init
  no gradient flows back to mediator_embed via H_a / H_b. As lambda grows
  during training, a SCALAR gradient signal flows through alpha into H_a/H_b,
  but it modulates the SAME final mlp_score, not a sibling branch.

## 3. Smoke test (pre-codex)

All critical invariants verified empirically by `wsl python -c '...'`:

```
score shape: (4,)
z_evidence shape: (4, 16)
alpha shape: (4, 12, 12)
alpha sum per sample: tensor([1.0000, 1.0000, 1.0000, 1.0000])
lambda_content init: 0.0 (should be 0)
g_mlp last weight max abs: 0.0 (should be 0)
g_mlp last bias max abs: 0.0 (should be 0)
init alpha matches pure affinity softmax: True
trainer MRO: ['_PerModeEmerGNN_PMP_v1_5', '_PerModeEmerGNN_PMP_v1_2',
              '_PerModeEmerGNN_PMP_v1_1', '_PerModeEmerGNN_PMP',
              '_PerModeEmerGNN_MNAH']
```

The "init alpha matches pure affinity softmax: True" check confirms Option B
correctly degenerates to Option A at init.

## 4. Read inventory (codex round 1)

| File | Reviewed |
|---|---|
| `Code/my_code/models/pmp_v1/v1_5/__init__.py` | ✓ |
| `Code/my_code/models/pmp_v1/v1_5/cluster_head.py` | ✓ |
| `Code/my_code/models/pmp_v1/v1_5/pmp_v1_5_trainer.py` | ✓ |
| `Code/scripts/run_pmp_v1_5.py` | ✓ |

For comparison (already approved, not re-reviewed):
- v1.2, v1.4, v1.1, parent v1, MNAH `_format_epoch_log` hook.

## 5. Codex round 1 verdict (verbatim)

> No findings.
>
> The v1.5 implementation matches the locked spec on the points you asked to
> review:
> - Step 1 logit composition is correct in cluster_head.py:233: log_p_a,
>   log_p_b, and log_T use log_softmax; base_logit is built as
>   log_p_a.unsqueeze(2) + log_T.unsqueeze(0) + log_p_b.unsqueeze(1);
>   content_input is concat[H_a, H_b, H_a*H_b]; total_logit = base_logit
>   + lambda_content * content_logit; joint normalization is done by
>   flattening to (B, K*K) and softmaxing at cluster_head.py:254.
> - Marginals are correct at cluster_head.py:259: attn_a = alpha.sum(dim=2)
>   is the row marginal, attn_b = alpha.sum(dim=1) is the column marginal.
> - Option B initialization constraints are correctly enforced at
>   cluster_head.py:108 and cluster_head.py:121: the last nn.Linear in
>   g_mlp has both weight and bias zeroed, and lambda_content is
>   nn.Parameter(torch.zeros(1)).
> - The H_a / H_b pool is the same v1.2 pair-conditional cluster-attention
>   pattern, including safe masking and empty-cluster-to-zero behavior.
> - The within-pool is structurally the v1.1/v1.2/v1.4 logic, with the only
>   intended change being stack-first then outer weighting.
> - Trainer wiring is correct: _build_aux_head passes content_hidden;
>   _combined_logit builds both per-drug-per-cluster tensors and intersection
>   tensors, then calls the 5-output forward in the correct order;
>   _predict_branches is overridden to the single-branch convention;
>   validation and epoch logging include lambda_content. MNAH's fit loop
>   still calls the override hook.
> - Cache reuse and inheritance are correct: v1.5 inherits v1.2, so
>   cluster_max_mediators_per_drug behavior comes through unchanged; no new
>   cache schema is introduced.
> - CLI wiring is correct: correct trainer, content_hidden, 0 -> None cap
>   mapping, results.json, test_s2_scores.npz, and lambda_content_final.
> - Edge cases are handled safely: empty drug neighborhood yields zero H_a/H_b
>   rows without NaNs; empty within-cluster intersections yield zero
>   z_within^k; pad index 0 is masked out before attention and never relied
>   on semantically.
>
> Parameter-count sanity also matches your estimate: the module definition
> implies 2,931,683 parameters for the stated configuration.
>
> Residual risk: this was a static review only. I did not run training or
> gradient checks in this pass.
>
> VERDICT: PASS

## 6. Stop condition

Round 1 PASS with zero findings. No round 2 executed.

## 7. Files created

- `Code/my_code/models/pmp_v1/v1_5/__init__.py`
- `Code/my_code/models/pmp_v1/v1_5/cluster_head.py`
- `Code/my_code/models/pmp_v1/v1_5/pmp_v1_5_trainer.py`
- `Code/scripts/run_pmp_v1_5.py`
- `Code/my_code/models/pmp_v1/v1_5/_reviews/2026-06-03__codex_design_round0_review_round1.md` (this report)

v1.1 / v1.2 / v1.3 / v1.4 files untouched.

## 8. Param count

| ndim=32, hidden=128, content_hidden=16 | Total |
|---|---|
| **v1.5** | **2,931,683** |
| (v1.4 reference) | 2,924,801 |
| (v1.2 reference) | 2,977,489 |
| (v1.1 reference) | 2,976,785 |

Net add over v1.4: ~6,882 params (W_T 144 + cluster_attn 1,088 + g_mlp 1,553
+ lambda 1 + larger mlp_score input 64 vs 32 -> +4,096). Modest.

## 9. Ready to run

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_pmp_v1_5.py --epochs 100 --seed 42 --n-dim 32 --tag pmp_v1_5_ndim32_seed42"
```

Expected runtime: ~1.0–1.2h (similar to v1.2; per-drug per-cluster + within
intersection both built, plus 12×12 content_logit MLP).

## 10. What to look for in results

| Reference | AUC | Question |
|---|---|---|
| v1.2 (parallel concat) | 0.7745 | v1.5 should beat this if sequential design is better |
| v1.4 (within only) | 0.7703 | v1.5 should beat this if content-aware attention adds info |
| max(v1.3, v1.4) | 0.7703 | v1.5 > this means sequential beats both layers alone |

Diagnostic `lambda_content_final` (the learned scalar at end of training):
- ~0.0 -> content residual stayed off; model preferred pure affinity. v1.5
  effectively reduces to "v1.4 with T-routed marginal weights".
- > 0 (small but nonzero) -> model learned to incorporate content-conditional
  routing. Expect AUC > v1.4.
- Large value -> content dominates affinity. Watch for overfit.
