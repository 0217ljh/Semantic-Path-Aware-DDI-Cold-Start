# PMP v1.5 — Official Lock Report

- **Date locked**: 2026-06-04
- **Locked by**: 用户决定 ("现在的这个模型可以记录为正式的1.5版本了")
- **Architecture**: sequential cluster-attention -> within-pool, content residual dead by design (lambda_content init = 0, g_mlp[-1] zero-init).

This file marks v1.5 as the official current-best architecture for the
Semantic-Path-Aware DDI Cold-Start project. Future architectural changes
should create v1.6+ in a new directory, NOT modify v1.5.

---

## 1. Two-run reproducibility (ndim=32, seed=42)

| Run | Run dir | AUC | AUPRC | NLL | fit time | λ_final |
|---|---|---|---|---|---|---|
| 1 | `Code/runs/2026-06-03_22-34-16__run_pmp_v1_5__pmp_v1_5_ndim32_seed42__seed42/` | 0.7764 | 0.7911 | — | 1.11h | 0.0 |
| 2 | `Code/runs/2026-06-04_*__run_pmp_v1_5__pmp_v1_5_A_ndim32_seed42_rerun__seed42/` | 0.7783 | 0.7975 | — | 1.13h | 0.0 |
| **avg** | — | **0.7774** | **0.7943** | — | 1.12h | 0.0 |

Single-seed delta: +0.0019 AUC / +0.0064 AUPRC across the two runs.
Interpretation: CUDA non-determinism noise band is roughly ±0.002 AUC
and ±0.005-0.006 AUPRC for this configuration. Multi-seed (3-5 seeds)
remains required for paper.

## 2. Comparison to other variants (ndim=32, seed=42)

| Variant | AUC | AUPRC | Notes |
|---|---|---|---|
| v1.1 ndim=32 | 0.7751 | 0.7865 | full v1 architecture |
| v1.2 ndim=32 | 0.7745 | 0.7762 | parallel concat fusion |
| v1.3 (cluster path only) | 0.7414 | 0.7475 | Layer 1 standalone |
| v1.4 (within only) | 0.7703 | 0.7673 | Layer 2 standalone |
| **v1.5 avg** | **0.7774** | **0.7943** | **official** |
| (v1.5 content-fix Option B test) | 0.7547 | 0.7568 | dead-init fix broke things; reverted |

v1.5 - v1.2 = **+0.29 pt AUC, +1.81 pt AUPRC**.

## 3. Architecture (locked)

```
Step 1 — cluster-cluster attention (scalar prior only):
  p_a = softmax(typed mediator count vector c_a)
  p_b = softmax(c_b)
  T   = softmax(W_T)                         # (K, K) learnable
  α[i,j] ∝ p_a[i] * T[i,j] * p_b[j]          # softmax over (i, j)
  attn_a^k = sum_j α[k, j]                   # row marginal (B, K)
  attn_b^k = sum_i α[i, k]                   # col marginal (B, K)

Step 2 — within-cluster pool over intersection mediators:
  for k in 0..K-1:
    M_k^{a∩b} = members(k) ∩ N_<=2(a) ∩ N_<=2(b)
    z_within^k = AttnPool({phi(m) for m in M_k^{a∩b}}) or 0
    phi(m) = MLP([mediator_embed[m], type_embed[k],
                  rel_embed[a->m], rel_embed[m->b]])

Step 3 — attention-marginal weighted aggregation:
  z_evidence_a = sum_k attn_a^k * z_within^k       # (B, d)
  z_evidence_b = sum_k attn_b^k * z_within^k       # (B, d)
  z_evidence   = concat[z_evidence_a, z_evidence_b]  # (B, 2*d)

Step 4 — score:
  score = MLP_score(z_evidence)
  loss  = BCE(sigmoid(score), y)
```

### Architectural invariants

- **Single gradient path to mediator_embed**: only through phi -> within
  -> score. The cluster_attn pool computes H_a/H_b but they feed g_mlp,
  whose output is multiplied by `lambda_content = 0`; chain rule gives
  zero gradient back through cluster_attn into mediator_embed.
- **No drug-id-keyed learnable embedding** enters score. Cold-start safe
  by construction.
- **Cluster-level information** is encoded only by scalar-valued objects
  (p_a, T, p_b, alpha, marginals). No content-based attention.
- **Documented dead code**: cluster_attn_W/w, g_mlp, lambda_content remain
  in the module structurally to keep the v1.5 B-iso (stop-grad) experiment
  one edit away. They contribute zero parameters' worth of learning but
  ~2.6K parameters of memory.

### Param count (ndim=32, hidden=128, content_hidden=16)

Total: 2,931,683 parameters. Breakdown:
- mediator_embed (90855, 32) = 2,907,360 (97.8%)
- mlp_pair_cluster + mlp_score + phi_mlp + within_attn + cluster_attn +
  W_T + g_mlp + lambda_content + type_embed + rel_embed = 24,323 (0.8%)

## 4. Why v1.5 wins (architectural mechanism)

### v1.2 had parallel branches sharing mediator_embed
```
mediator_embed ─┬─► cluster_attn ─► H_a/H_b ─► psi ─► z_cluster_pair ─┐
                │                                                       │
                └─► phi ─► z_within ────────────────────────────────────┤
                                                                        ▼
                                                                   concat → MLP_score
```
Two gradient paths into mediator_embed -> gradient competition -> joint
training hurts both branches (each lost ~3.6 pt vs standalone training).

### v1.5 has one effective gradient path
```
mediator_embed ─► phi ─► z_within ─► z_evidence ─► MLP_score
                                          ↑
                                   attn_a, attn_b (scalar
                                   weights from W_T routing)
```
mediator_embed has one job. cluster_attn pool branch is dead (lambda=0).
W_T learns routing on its own without polluting mediator_embed.

## 5. What v1.5 does NOT include (intentional, deferred to v1.6+)

- Molecular features (SMILES, Morgan, BRICS)
- LLM-distilled mediator semantics
- PubMedBERT text features
- Cross-cluster mixing between z_within^k vectors (only at attention
  weighting)
- Stop-grad isolation experiment for content residual
- Multi-seed averaging / deterministic flags

## 6. Critical reproducibility note

Single-seed runs of v1.5 differ by ~±0.002 AUC / ±0.005 AUPRC across
reruns due to CUDA non-deterministic operations. For paper:

- **Quote v1.5 ndim=32 AUC ~ 0.777**, with reported range across
  the 2 reruns as needed (cannot meaningfully resolve the 3rd decimal).
- **Multi-seed (3-5 seeds) is REQUIRED** before submission.
- To tighten reproducibility add:
  ```python
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  ```
  Trade-off: 10-30% training slowdown.

## 7. Known follow-up work

| Branch | Status | Description |
|---|---|---|
| Multi-seed v1.5 | TODO | seed=43, 44, 45 to get mean ± std for paper |
| Deterministic flags | TODO | add torch flags for stable reruns |
| v1.5 B-iso (stop-grad) | TODO | test content residual with detached mediator_embed |
| v1.6 cross-cluster mixing | TODO | transformer or T-driven mixing across z_within^k |
| v1.7 GNN-formulation | TODO | rewrite v1.5 as common-neighbor pooling GNN for paper positioning |
| v2.x C2 (LLM enrichment) | TODO | add LLM-distilled mediator semantics |
| v2.x C3 (molecular) | TODO | add molecular side fusion |

## 8. Files (canonical paths)

Code:
- `Code/my_code/models/pmp_v1/v1_5/__init__.py` (spec docstring)
- `Code/my_code/models/pmp_v1/v1_5/cluster_head.py` (PMPv1_5Module)
- `Code/my_code/models/pmp_v1/v1_5/pmp_v1_5_trainer.py`
- `Code/scripts/run_pmp_v1_5.py`

Review reports (this directory):
- `2026-06-03__codex_design_round0_review_round1.md` — initial codex design discussion + PASS code review
- `2026-06-04__dead_init_fix.md` — discovery of the dead-init deadlock, the
  failed Option B fix experiment (AUC dropped to 0.7547), and the revert.
- `2026-06-04__official_v1_5_lock.md` (this file) — official lock as v1.5.

Run records (Code/runs/):
- `2026-06-03_22-34-16__run_pmp_v1_5__pmp_v1_5_ndim32_seed42__seed42/`
- `2026-06-04_*__run_pmp_v1_5__pmp_v1_5_A_ndim32_seed42_rerun__seed42/`

Memory:
- `~/.claude/projects/D--My-Research/memory/project_pmp_v1_5_official.md`
  (Memory index entry pointing here)

## 9. The "Option B" experiment (documented dead end)

For posterity: between run 1 and run 2 of v1.5 A, we attempted to "fix"
the dead-init by setting `lambda_content = 1.0` initial (Codex's Option B
proposal). This activated the content residual path and let H_a/H_b flow
gradients back into mediator_embed via cluster_attn pool. Result:

```
v1.5 Option B (lambda init = 1.0): AUC 0.7547, lambda_final = 1.29
```

This is **WORSE** than v1.5 A (0.7764) by ~2.2 pt AUC. The conclusion:

> Activating content-aware cluster attention (which requires
> mediator_embed gradient via cluster_attn pool) reintroduces the gradient
> competition pathology that v1.5 was designed to avoid.

The Option B code path was reverted. Files left in module:
`cluster_attn_W/w`, `g_mlp`, `lambda_content` — all retained for the
future stop-grad ablation (v1.5 B-iso) where this conclusion can be
revisited with isolation safeguards.

Lesson for paper write-up: v1.5's parsimony (cluster_attn-free routing)
is principled, not a happy accident. Document this.

## 10. Lock acknowledgment

This file makes v1.5 the official current-best architecture. Any future
modification to v1.5/ requires acknowledgment that we are touching the
locked baseline. Architectural ideas should fork to v1.6/ instead.
