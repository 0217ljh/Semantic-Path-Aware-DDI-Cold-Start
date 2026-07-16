"""Stage 4 / DDI-LoRA adapter - configuration (the M_A.M_B typed semantic adapter).

Shapes/symbols follow the paper Method (tex 337-443):
  T      = number of mediator types t(m) in 1..T
  lam    = lambda, prototypes per type; a = T*lam = adapter rank (<< d)
  d      = H_base / prototype dimension (matches the backbone node-rep dim, or the
           adapter-alone structural H_base dim)
  d_z    = frozen semantic encoding dim (z_m, z_r; e.g. 768 for SapBERT)
Trainable: {U_t, W_A, W_B, M_B(=B_t), gamma_t, pooling attention, MLP+Head}.
Frozen:    {H_base, z_m, z_r, k-means centroids c_{t,k}}.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterConfig:
    n_types: int              # T
    lam: int                  # lambda: prototypes per type
    d: int                    # H_base / prototype dim
    d_z: int                  # frozen semantic dim (z_m/z_r)
    n_out: int                # task head output width (binary -> 1, multi/label -> K)
    mlp_hidden: int = 256     # h, the MLP hidden dim before Head
    rho: float = 0.5          # path length-decay for r_tilde (pi_p propto rho^|p|)
    tau: float = 1.0          # temperature for U_t k-means warm-start (U^0_k = c/tau)
    eps: float = 1e-2         # gamma_t initial residual scale
    eta_entropy: float = 0.0  # entropy-regularizer coefficient (0 = off)
    dropout: float = 0.2
    rel_init_scale: float = 0.1  # W_A/W_B init downscale so the relation arm does not
    #                     start ~sqrt(d_z) larger than the node term U z_m (LayerNorm'd
    #                     r_tilde has norm ~sqrt(d_z) vs unit-norm z_m); keeps U learnable
    use_node: bool = True        # ablation: if False, drop the node term U z_m from beta
    #                     (pathway-only assignment) - lets us prove node vs pathway each help
    use_mb: bool = True          # ablation (M_B): if False, drop the typed-prototype residual
    #                     (n_uvm = h_base only) - tests whether the low-rank M_B directions matter

    @property
    def rank(self) -> int:    # a = T * lam
        return self.n_types * self.lam


__all__ = ["AdapterConfig"]
