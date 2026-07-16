"""HDN-DDI multi-class variant model.

Builds on HDN_DDI: reuses all hierarchical molecular graph blocks + co-attention,
swaps the KGE readout from per-pair scalar (RESCAL with selected `r`) to
per-pair (B, n_classes) output (scores over ALL relations).

Reference: HDN-DDI paper uses triplet-binary formulation natively (each (a, r, b)
is binary). For multi-class we evaluate per-pair argmax over all 86 candidate
relations — which means we need (B, 86) output. We compute this efficiently
in one forward pass via einsum over the existing relation embeddings.

The original RESCAL.forward computes:
    scores[b, i, j] = heads[b, i] @ rel_emb[r[b]] @ tails[b, j].T  -> sum over (i,j) -> (B,)
We extend to:
    scores[b, r, i, j] = heads[b, i] @ rel_emb[r] @ tails[b, j].T -> sum over (i,j) -> (B, R)

Only the final readout changes; everything else verbatim from HDN_DDI.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from baseline.hdn_ddi.models import HDN_DDI


class HDN_DDI_MC(HDN_DDI):
    """HDN-DDI multi-class: forward returns (B, n_rels) scores per pair.

    Inherits the binary HDN_DDI; overrides only forward() to:
      - skip the per-pair `rels` selection (rels argument ignored)
      - compute per-relation scores via einsum over rel_emb.weight
    """

    def forward(self, triples):
        h_data, t_data, _rels_ignored, b_graph = triples
        # ----- 1. Initial norm + conv (verbatim) -----
        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        h_data.x = self.initial_conv(h_data.x, h_data.edge_index)
        t_data.x = self.initial_conv(t_data.x, t_data.edge_index)

        # ----- 2. Blocks (verbatim) -----
        repr_h = []
        repr_t = []
        for block in self.blocks:
            out = block(h_data, t_data, b_graph)
            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)
        repr_h = torch.stack(repr_h, dim=-2)  # (B, n_blocks, F)
        repr_t = torch.stack(repr_t, dim=-2)  # (B, n_blocks, F)
        attentions = self.co_attention(repr_h, repr_t)  # (B, n_blocks, n_blocks)

        # ----- 3. Multi-class readout (override) -----
        # KGE.rel_emb.weight: (n_rels, F*F)
        n_rels = self.KGE.n_rels
        F_dim = self.KGE.n_features
        all_rels = self.KGE.rel_emb.weight  # (R, F²)
        all_rels = F.normalize(all_rels, dim=-1)
        all_rels = all_rels.view(n_rels, F_dim, F_dim)  # (R, F, F)

        heads_n = F.normalize(repr_h, dim=-1)  # (B, n_blocks, F)
        tails_n = F.normalize(repr_t, dim=-1)  # (B, n_blocks, F)

        # einsum: (B, i, F) @ (R, F, F) @ (B, j, F) -> (B, R, i, j)
        # bif, rfg, bjg -> brij
        scores = torch.einsum("bif,rfg,bjg->brij", heads_n, all_rels, tails_n)
        # apply attentions (B, i, j) → broadcast to (B, 1, i, j)
        scores = scores * attentions.unsqueeze(1)
        scores = scores.sum(dim=(-2, -1))  # (B, R)
        return scores
