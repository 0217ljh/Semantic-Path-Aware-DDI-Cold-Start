"""EmerGNN with the ORIGINAL torchdrug ``generalized_rspmm`` kernel (rspmm backend).

This is a paper-faithful re-implementation of the ORIGINAL EmerGNN forward
(``LARS-research/EmerGNN/DrugBank/models.py`` ``enc_ht`` L49-97), which uses the
fused ``generalized_rspmm`` CUDA kernel for bidirectional relational message
passing. It is a SEPARATE model from the pure-PyTorch chunk model
(:class:`baseline.emergnn.model.EmerGNN`) — the chunk model reads the stored KG
indices as ``(edge_src, edge_dst)`` and scatters ``out[dst] += rel * h[src]``,
which swaps the forward/reverse relation-slot labels relative to the original;
this model reproduces the original ``out[row] += rel * h[col]`` flow exactly.

Layer parameters (``rel_kg`` / ``linear`` / ``relation_linear`` / ``attn_relation``
/ ``Wr`` / entity features) are inherited UNCHANGED from
:class:`baseline.emergnn.model.EmerGNN`; only the message-passing kernel and the
forward signature differ (this takes the pre-built sparse KG, mirroring the
original ``enc_ht(head, tail, KG)``).
"""
from __future__ import annotations

import torch

from baseline.emergnn.model import EmerGNN
from baseline.emergnn._rspmm_utils import get_generalized_rspmm


class EmerGNN_RSPMM(EmerGNN):
    """EmerGNN forward via torchdrug ``generalized_rspmm`` (binary head).

    Same ``__init__`` as :class:`EmerGNN` (all layers reused). ``forward`` takes
    the coalesced 3D sparse KG (``(n_ent, n_ent, all_rel)``) instead of dense
    edge lists — the caller (rspmm training core) builds it once per KG change.
    """

    def _propagate_rspmm(
        self,
        source_idx: torch.Tensor,   # (B,)
        source_embed: torch.Tensor,  # (B, n_dim)
        ht_embed: torch.Tensor,      # (B, 2*n_dim)  static per pair, reused both dirs
        kg: torch.Tensor,            # sparse (n_ent, n_ent, all_rel), coalesced
    ) -> torch.Tensor:
        """One directional pass: init hiddens at ``source_idx`` then L layers of
        ``generalized_rspmm``. Returns hiddens (n_ent, B, n_dim). Faithful to the
        original ``enc_ht`` inner loop (models.py:59-74)."""
        rspmm = get_generalized_rspmm()
        B = source_idx.size(0)
        n_ent = self.n_ent
        device = source_embed.device

        hiddens = torch.zeros(n_ent, B, self.n_dim, device=device)
        b_arange = torch.arange(B, device=device)
        hiddens[source_idx, b_arange] = source_embed

        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)                       # (n_ent, B*n_dim)
            # per-(pair, relation) gated relation embedding, conditioned on the
            # STATIC (head, tail) query (attention does not depend on hiddens).
            rel_w = self.attn_relation[l](self.act(self.relation_linear[l](ht_embed)))
            rel_w = torch.sigmoid(rel_w).unsqueeze(2)              # (B, all_rel, 1)
            rel_emb = self.rel_kg[l].weight                        # (all_rel, n_dim)
            relation_input = rel_w * rel_emb                       # (B, all_rel, n_dim)
            relation_input = relation_input.view(B, -1, self.n_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)  # (all_rel, B*n_dim)
            hiddens = rspmm(kg, relation_input, hiddens, sum="add", mul="mul")  # (n_ent, B*n_dim)
            hiddens = hiddens.view(n_ent * B, -1)
            hiddens = self.act(self.linear[l](hiddens))
        return hiddens.view(n_ent, B, self.n_dim)

    def enc_ht(self, head: torch.Tensor, tail: torch.Tensor, kg: torch.Tensor) -> torch.Tensor:
        """Bidirectional propagation -> concatenated pair embedding.
        Mirrors original ``enc_ht`` (models.py:49-97)."""
        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
        B = head.size(0)
        b_arange = torch.arange(B, device=head.device)

        hid_uv = self._propagate_rspmm(head, head_embed, ht_embed, kg)  # (n_ent, B, n_dim)
        tail_hid = hid_uv[tail, b_arange]                              # (B, n_dim)

        hid_vu = self._propagate_rspmm(tail, tail_embed, ht_embed, kg)
        head_hid = hid_vu[head, b_arange]                             # (B, n_dim)

        if self.feat == "E":
            return torch.cat([head_embed, tail_embed, head_hid, tail_hid], dim=-1)
        return torch.cat([head_hid, tail_hid], dim=-1)

    def forward(
        self,
        head: torch.Tensor,   # (B,)
        tail: torch.Tensor,   # (B,)
        kg: torch.Tensor,     # coalesced sparse (n_ent, n_ent, all_rel)
    ) -> torch.Tensor:
        """Return logits (B,). Binary head (``Wr`` out_dim=1)."""
        embed = self.enc_ht(head, tail, kg)
        return self.Wr(embed).squeeze(-1)


__all__ = ["EmerGNN_RSPMM"]
