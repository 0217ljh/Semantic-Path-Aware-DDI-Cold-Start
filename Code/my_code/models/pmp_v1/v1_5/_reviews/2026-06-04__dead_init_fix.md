# PMP v1.5 — Dead-Init Deadlock Fix Report

- **Date**: 2026-06-04
- **Primary reviewer**: Claude (sonnet-4.5)
- **Independent reviewer**: codex (default model via `mcp__codex__codex`)
- **Triggered by**: v1.5 ndim=32 seed=42 finished with `lambda_content_final = 0.0` (exactly), suggesting the content residual never updated during training. Investigation showed a dead-init deadlock between `lambda_content` and `g_mlp[-1]`.

---

## 1. Bug discovery

v1.5 ndim=32 seed=42 final test_s2:
```
AUC=0.7764  AUPRC=0.7911  fit=1.11h  lambda_content_final=0.0
```

The architecture still beat v1.2 (0.7745) and v1.4 (0.7703) despite the dead content residual — i.e. v1.5 was effectively running as **Option A** (pure affinity attention + sequential within-pool marginal weighting), not Option B (content-residual-aware attention).

## 2. Root cause: simultaneous zero-init of two multiplicatively-coupled parameters

v1.5 cluster_head.py `__init__` had:
```python
# g_mlp last linear weights AND bias zeroed
with torch.no_grad():
    last_linear = self.g_mlp[-1]
    last_linear.weight.zero_()
    last_linear.bias.zero_()

# AND lambda_content zeroed
self.lambda_content = nn.Parameter(torch.zeros(1))
```

The forward composition is:
```
total_logit = base_logit + lambda_content * g_mlp(content_input)
```

At init:
- `g_mlp[-1]` zero -> `content_logit = g_mlp(content_input) ≡ 0`
- `lambda_content = 0` -> `lambda * content_logit ≡ 0`
- `total_logit = base_logit`, alpha = softmax(base_logit). Option A behavior at init.

Gradient flow at init:
- `∂loss/∂lambda_content = sum (∂loss/∂total_logit) * content_logit = 0` (since content_logit ≡ 0). Lambda CAN'T move.
- `∂loss/∂g_mlp[-1].weight = ∂loss/∂content_logit * lambda * pre_last_input = 0` (since lambda = 0). g_mlp[-1] CAN'T move.
- Earlier g_mlp layers (Linear(3d, 16)) flow back through g_mlp[-1].weight which is zero -> their gradient is also 0.

The result: **ALL g_mlp parameters AND lambda_content stay frozen at their init values for the entire training**. PyTorch autograd cannot escape this deadlock because all gradient paths to these parameters are multiplicatively gated by other parameters that are also zero.

## 3. Codex's verbatim verification

> Yes: with `g_mlp[-1].weight = 0`, `g_mlp[-1].bias = 0`, and `lambda_content = 0`, the residual branch is exactly dead at step 0 and stays dead. `content_logit` is identically zero, so `dL/dlambda = sum(dL/dtotal_logit * content_logit) = 0`. And because `lambda = 0`, `dL/d(content_logit) = lambda * dL/dtotal_logit = 0`, so the last linear gets zero grad; because its weight is also zero, earlier `g_mlp` layers get zero grad too. PyTorch autograd does not "escape" this on its own.
>
> Best fix recommendation: `lambda_content = 1.0`, `g_mlp[-1]` zero-init. It gives you exact Option A at init and a clean unlock. `lambda = 0.1` also works, just with 10x smaller initial gradient into `g_mlp`. Initializing `g_mlp[-1]` small and keeping `lambda = 0` does not work; the branch is still multiplied by zero, so it remains dead. Initializing both small nonzero also works, but you lose the exact "pure affinity at init" property.

Codex also flagged a subtle note for future consideration:
> `lambda_content` is unconstrained, so it can go negative and invert the meaning of the content residual. That may be fine, but if you intend it as a nonnegative strength parameter, use something like `softplus(raw_lambda)` instead of a free scalar.

We do NOT change to softplus for now (the inversion may be a useful expressive capacity), but flagged for review if training shows lambda going negative.

## 4. Applied fix (single line)

In `cluster_head.py:124`:
```diff
- self.lambda_content = nn.Parameter(torch.zeros(1))
+ self.lambda_content = nn.Parameter(torch.ones(1))
```

Plus extended docstring explaining the deadlock and the fix rationale.

## 5. Empirical fix verification

`wsl python` smoke test confirmed:
```
lambda_content init: 1.0 (should be 1.0)                                    ✓
g_mlp[-1] weight max abs: 0.0 (should be 0)                                 ✓
g_mlp[-1] bias max abs: 0.0 (should be 0)                                   ✓
init alpha still matches pure affinity (Option A at init): True             ✓
After backward:
  g_mlp[-1] weight grad max abs: 2.65e-05  (should be > 0 — unlocked)       ✓
  g_mlp[-1] bias grad max abs:   8.73e-11  (should be > 0 — unlocked)       ✓
  lambda_content grad max abs:   0.0       (should be 0 — still gated)      ✓
DEADLOCK BROKEN: g_mlp[-1] can now learn direction.
```

Note the lambda_content grad is still 0 immediately after init (expected — content_logit is zero before g_mlp[-1] moves). After one or two SGD steps, g_mlp[-1] becomes nonzero, content_logit becomes nonzero, and lambda begins receiving gradient.

## 6. Expected re-run behavior

With the fix:
- `lambda_content_final` should be nonzero (positive or negative).
- If `lambda_content_final` ≈ 0 (e.g., < 0.01 magnitude): content residual was found to be unhelpful by the model; Option A and Option B converge to the same AUC.
- If `lambda_content_final` is meaningfully nonzero AND AUC > 0.7764: Option B's content-aware routing is genuinely helping.
- If `lambda_content_final` is large and AUC drops or training is unstable: content residual is overfit; consider reducing learning rate or adding regularization.

## 7. Files touched

- `Code/my_code/models/pmp_v1/v1_5/cluster_head.py` — one-line fix + extended docstring at the `lambda_content` declaration. No other changes; `g_mlp[-1]` zero-init kept.
- `Code/my_code/models/pmp_v1/v1_5/_reviews/2026-06-04__dead_init_fix.md` (this file).

No new code review report needed — codex separately verified the analysis and the fix above.

## 8. Re-run command

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_pmp_v1_5.py --epochs 100 --seed 42 --n-dim 32 --tag pmp_v1_5_ndim32_seed42_fix"
```

(Tag includes `_fix` to distinguish from the pre-fix run.)

Original pre-fix run:
- `Code/runs/2026-06-03_22-34-16__run_pmp_v1_5__pmp_v1_5_ndim32_seed42__seed42/`
- Result: AUC=0.7764 / AUPRC=0.7911 / lambda_content_final=0.0 (Option A behavior despite Option B code).
