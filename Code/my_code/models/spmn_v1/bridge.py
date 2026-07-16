"""SPMN v1 — molecule↔KG bridge (B1-B4): the cross-space weakly-supervised
retrieval that connects molecular fragments to KG mechanism entities.

Per the codex-reviewed design (NOT fragment-protein binding; a learned
fragment-conditioned mechanism-compatibility map, grounded by masked
mechanism-completion pretraining):

  B1  shared projection:  z̃_i = W_f z_i (fragment),  h̃(v) = W_h h(v) (entity)
  B2  typed bilinear compatibility:  s(i,v) = z̃_i^T B_{φ(v)} h̃(v) / sqrt(r)
  B3  drug→entity weight:  w_d(v) = softmax_{v within type τ}( aggscore_d(v) / T ),
        aggscore_d(v) = logsumexp_i s(i,v)        (multi-instance over fragments)
      sparse/selective (low T + entropy reg in loss + top-k at inference;
      anti-hub via frequency-aware negatives in pretraining).
  B4 (in the head):  q_{τ,τ'}(a,b) = Σ_{(u,v) co-path} w_a(u)·w_b(v)

`aggscore` is exposed for the pretraining entity-completion objective; no
drug-id anywhere (inductive: shared W_f, W_h, B_τ + shared entity states).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax


class SPMNBridge(nn.Module):
    def __init__(self, d_frag: int, d_kg: int, n_types: int,
                 r: int = 32, temp: float = 0.5) -> None:
        super().__init__()
        self.r = int(r)
        self.n_types = int(n_types)
        self.temp = float(temp)
        self.W_f = nn.Linear(d_frag, r, bias=False)
        self.W_h = nn.Linear(d_kg, r, bias=False)
        # Typed bilinear B_τ (one r×r per mechanism type).
        self.B = nn.Parameter(torch.empty(n_types, r, r))
        nn.init.xavier_uniform_(self.B)

    def compat_scores(self, z_frags: torch.Tensor, h_ent: torch.Tensor,
                      ent_type: torch.Tensor) -> torch.Tensor:
        """s[i,v] = z̃_i^T B_{φ(v)} h̃(v)/sqrt(r).  z_frags:(F,d_frag),
        h_ent:(E,d_kg), ent_type:(E,) -> (F, E)."""
        zf = self.W_f(z_frags)                       # (F, r)
        he = self.W_h(h_ent)                         # (E, r)
        Bv = self.B[ent_type]                        # (E, r, r)
        he_t = torch.einsum("erc,ec->er", Bv, he)    # B_{φ(v)} h̃(v), (E, r)
        return (zf @ he_t.t()) / math.sqrt(self.r)   # (F, E)

    def entity_weights(self, z_frags: torch.Tensor, h_ent: torch.Tensor,
                       ent_type: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (w, aggscore), both (E,).

        aggscore_d(v) = logsumexp_i s(i,v);  w = softmax of aggscore/T WITHIN
        each entity's type group (selective per mechanism type). If a drug has
        no fragments (F==0), returns zeros."""
        E = h_ent.size(0)
        if z_frags.size(0) == 0 or E == 0:
            z = torch.zeros(E, device=h_ent.device)
            return z, z
        s = self.compat_scores(z_frags, h_ent, ent_type)   # (F, E)
        aggscore = torch.logsumexp(s, dim=0)               # (E,)
        # Per-type softmax (groups = ent_type), temperature-scaled.
        w = pyg_softmax(aggscore / self.temp, ent_type, num_nodes=self.n_types)
        return w, aggscore


__all__ = ["SPMNBridge"]
