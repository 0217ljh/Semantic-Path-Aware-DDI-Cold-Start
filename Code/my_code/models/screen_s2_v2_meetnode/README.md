# v2 — Meeting-Node Auxiliary Head on EmerGNN (MNAH)

**Date**: 2026-05-21
**Status**: design (pre-codex-review)
**Baseline target**: anchor 0.7458, CACR2 0.7472 → goal ≥ 0.755 (≥+0.9pt)

## Motivation

Current S2 anchor: 0.7458. Best CACR variant: 0.7472 (+0.14pt, within noise).
Hyperparameter tuning of CACR plateaued. Per user workflow protocol, switch to
**insight-level innovation**.

User's locked insights (Notes/Settings/Insights/):

- **i1** (PK/PD asymmetry): PK uses molecular layer (83.2%), PD uses effect layer (75.8%).
- **i2** (meeting-node anchor): DDI signal lives at shared mediators, not endpoints. E7 proved 22-dim shared-mediator LR-count reaches 0.7171 on S2 — beats vanilla GCN K=2 (0.680) by +3.7pt with NO learnable GNN at all.
- **i3** (attention insufficient): Internal attention can't inject external priors; need pair-conditional selection.
- **i4** (node-name text semantics): PubMedBERT(name) carries cold-start-stable signal.

**Joint position from \_SYNTHESIS.md**: no method combines (1) meeting-node anchoring + (2) PK/PD-aware reasoning + (3) node-name text priors — all evaluated rigorously under S2 cold-start.

## Hypothesis

EmerGNN flow and meeting-node count signals are **complementary**:

- EmerGNN captures multi-hop relational reasoning via flow GNN
- 22-dim shared-mediator counts capture simple co-occurrence statistics that flow-aggregation may dilute via multi-layer averaging
- Late-fused, the two should combine for additive gain

## Stage 1 design (minimal-viable v2)

Architecture:

```
EmerGNN backbone (frozen-arch, full re-train) ──▶ logit_emergnn
                                                       │
22-dim shared-mediator counts (pre-computed) ──▶ MLP ─▶ logit_aux
                                                       │
                                       sigmoid(α·logit_emergnn + β·logit_aux)
```

Where:

- `α`, `β` are learnable scalars init at 1.0 each
- MLP is 22 → 32 → 1 with ReLU, dropout=0.2
- Loss = BCE(combined_sigmoid, y)
- 22 features = E7 features: 11 node kinds × {1-hop, 2-hop} shared-mediator counts (log1p)

Features are **pre-computed once** (deterministic KG operation, no learnable
component) and cached. Per i2 doc: M-retrieval is a "deterministic KG set
operation, not subject to GNN depth limitations."

## Why this is i2 (not just feature engineering)

The 22-dim mediator features are exactly the **meeting-node statistics** that i2
predicts should carry DDI signal. E7 isolated proves they do (0.7171 on S2).
Adding them as an explicit auxiliary signal to EmerGNN is the **first**
combination of:

- EmerGNN flow (endpoint-anchored path reasoning) +
- Explicit meeting-node readout (i2 architectural prior)

If MNAH > EmerGNN by ≥ +1pt, this proves the meeting-node anchor carries
**independent signal** that flow GNN doesn't fully capture — direct empirical
support for i2 as architectural contribution, not just analytical lens.

## What this does NOT yet include (deferred to v3+)

- PubMedBERT text init (i4) — adds engineering complexity, defer
- PK/PD dual-channel routing (i1) — needs ddi_type labels at training, defer
- Pair-conditional node-instance selection (i3 full form) — defer to v4

Stage 1 is intentionally minimal: prove the meeting-node SIGNAL is additive
before scaling the architectural complexity.

## Implementation plan

1. Pre-compute 22-dim features for all (drug_a, drug_b) pairs in
   train + val_s2 + test_s2, cache to parquet.
2. Subclass `_PerModeEmerGNN` → `_PerModeEmerGNN_MNAH`:
   - `__init__`: load 22-dim feature cache, init aux MLP, init α/β
   - `_score(head, tail)`: combine emergnn_logit + aux_mlp(features[head,tail])
   - `fit()`: standard BCE on combined logits, same shuffle/neg pipeline
3. New run script: `run_mnah.py` (mirrors `run_s2_anchor.py`)
4. Tag: `mnah_v2stage1_drugbank_seed42`
5. Run 100 ep, compare to anchor 0.7458

## Expected outcomes

| Outcome | Interpretation |
|---|---|
| MNAH ≥ 0.755 (+0.9pt) | **i2 confirmed**: meeting-node carries additive signal; proceed to v2 Stage 2 (i4 PubMedBERT) |
| MNAH 0.748–0.755 (~+0.5pt) | Partial confirm; check feature importance, scale aux weight, retry |
| MNAH < 0.748 | Negative: either fusion is redundant or implementation bug; revisit i2 implementation form |

## Failure modes / risks

| Risk | Mitigation |
|---|---|
| 22-dim features computed on KG that includes train DDI → leakage | Use KG_only (no DDI edges); E7 already did this |
| Aux MLP gradient overwhelms EmerGNN signal | Init α=β=1.0; monitor logit magnitudes |
| Numerical instability from large mediator counts | log1p transform (E7 used this) |
| EmerGNN sub-optimum from shorter training (combined loss harder to optimize) | Allow 100 ep, learning rate same as anchor |

## Codex round-11 review applied (FINAL design)

PASS items (kept as-is):
- Late logit fusion at logit level (not representation)
- Full 22-dim features (not 12-dim)
- Stage 1 staging (no PubMedBERT, no PK/PD yet)
- BCE on combined logit, no separate aux supervision

FIX items (apply before implementation):
1. **Leakage guard**: Compute 22-dim features on KG-ONLY graph (NO DDI edges),
   identical to E7 protocol. NOT from EmerGNN's post-`_setup_graph` graph.
2. **Cache provenance**: cache key = `{dataset}_{seed}_s2_kgonly_v1_feathash`
3. **Asymmetric scalars**: Fix `α = 1.0` for EmerGNN logit. Only β is learnable:
   `combined_logit = emergnn_logit + softplus(raw_beta) * aux_logit`
   with `raw_beta` init s.t. `softplus(raw_beta) ≈ 1.0` (i.e., raw_beta=0.541)
4. **Train-only normalization**: fit feature mean/std on TRAIN pairs only,
   apply to val/test. NOT global normalization.
5. **Pair canonicalization**: sort `(drug_a, drug_b)` lexicographically to
   ensure (a,b) and (b,a) give identical features.
6. **Per-branch logging**: at each eval, log:
   - emergnn-only AUC (using α·logit_emergnn)
   - aux-only AUC (using β·logit_aux)
   - fused AUC (combined)
   - current β value
   This lets us diagnose dominance / collapse.

DEFERRED (future stages):
- Representation fusion → v2 Stage 2
- 12-dim 1-hop ablation → after Stage 1 confirms 22-dim works
- PubMedBERT(name) → v2 Stage 3 (after MNAH proven)
- PK/PD dual-channel routing → v3

## Implementation plan (revised)

1. **Pre-compute features** (offline script `precompute_meet_features.py`):
   - Load merged KG, NO DDI edges (KG-only)
   - For each pair in train+val_s2+test_s2: compute 22-dim log1p counts
   - Canonical sort drug ids before computing
   - Cache to parquet: `Code/data/_cache/meet_feat_drugbank_seed42_kgonly_v1.parquet`
2. **Fit train-only normalizer** (in trainer init): compute train mean/std
3. **`mnah_trainer.py`**: subclass `_PerModeEmerGNN`:
   - Load feature cache + normalizer
   - aux_mlp: 22 → 32 → 1, ReLU, dropout=0.2
   - raw_beta: nn.Parameter init=0.541 (so softplus ≈ 1.0)
   - `_score(head, tail)`: combine emergnn_logit + softplus(raw_beta)*aux_logit
   - `fit()`: standard BCE on combined; print 3 AUCs + beta every epoch
4. **`run_mnah.py`**: mirror `run_s2_anchor.py`
5. **Tag**: `mnah_v2s1_drugbank_seed42`

## Multi-seed protocol (per codex risk)

Run seed=42 first. If MNAH > anchor by ≥ +0.5pt, also run seed=43 + seed=44 to
verify. Single-seed +0.14 (CACR2) is within noise; need consistent gain across
seeds before claiming success.
