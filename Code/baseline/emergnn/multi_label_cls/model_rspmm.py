"""EmerGNN multilabel model (TWOSIDES 200-head), rspmm backend.

rspmm twin of :class:`baseline.emergnn.multi_label_cls.model.EmerGNN_ML`. TWOSIDES
has its own relation-slot scheme (``all_rel_slots = 2*all_rel - eval_rel + 1``,
DDI facts on slot 0), so this is a SEPARATE model (per the design review) rather
than a subclass of the DrugBank ``EmerGNN_RSPMM``. It inherits ``EmerGNN_ML``'s
``__init__`` (all layers + fixed 200-logit sigmoid head) UNCHANGED and swaps only
the message-passing kernel: the chunked scatter loop -> torchdrug fused
``generalized_rspmm``, taking the pre-built sparse KG (mirroring the original
TWOSIDES ``enc_ht(head, tail, KG)``, TWOSIDES/models.py:49-97).

feat='E' concat order matches the ML port: ``[head_embed, head_hid, tail_hid,
tail_embed]`` (differs from the DrugBank ``[head_embed, tail_embed, head_hid,
tail_hid]``); feat='M' concat is ``[head_hid, tail_hid]``.
"""
from __future__ import annotations

import torch

from baseline.emergnn._rspmm_utils import get_generalized_rspmm
from baseline.emergnn.multi_label_cls.model import EmerGNN_ML


class EmerGNN_ML_RSPMM(EmerGNN_ML):
    """EmerGNN multilabel forward via torchdrug ``generalized_rspmm``."""

    def _propagate_rspmm(
        self,
        source_idx: torch.Tensor,    # (B,)
        source_embed: torch.Tensor,  # (B, n_dim)
        ht_embed: torch.Tensor,      # (B, 2*n_dim) static, reused both directions
        kg: torch.Tensor,            # sparse (n_ent, n_ent, all_rel_slots), coalesced
    ) -> torch.Tensor:
        """One directional pass via generalized_rspmm. Faithful to the original
        TWOSIDES enc_ht inner loop (uses ``all_rel_slots`` relation slots)."""
        rspmm = get_generalized_rspmm()
        B = source_idx.size(0)
        n_ent = self.n_ent
        device = source_embed.device

        hiddens = torch.zeros(n_ent, B, self.n_dim, device=device)
        b_arange = torch.arange(B, device=device)
        hiddens[source_idx, b_arange] = source_embed

        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)                        # (n_ent, B*n_dim)
            rel_w = self.attn_relation[l](self.act(self.relation_linear[l](ht_embed)))
            rel_w = torch.sigmoid(rel_w).unsqueeze(2)               # (B, all_rel_slots, 1)
            rel_emb = self.rel_kg[l].weight                         # (all_rel_slots, n_dim)
            relation_input = rel_w * rel_emb                        # (B, all_rel_slots, n_dim)
            relation_input = relation_input.view(B, -1, self.n_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)  # (all_rel_slots, B*n_dim)
            hiddens = rspmm(kg, relation_input, hiddens, sum="add", mul="mul")
            hiddens = hiddens.view(n_ent * B, -1)
            hiddens = self.act(self.linear[l](hiddens))
        return hiddens.view(n_ent, B, self.n_dim)

    def forward(
        self,
        head: torch.Tensor,   # (B,)
        tail: torch.Tensor,   # (B,)
        kg: torch.Tensor,     # coalesced sparse (n_ent, n_ent, all_rel_slots)
    ) -> torch.Tensor:
        """Return logits (B, eval_rel)."""
        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
        B = head.size(0)
        b_arange = torch.arange(B, device=head.device)

        hid_uv = self._propagate_rspmm(head, head_embed, ht_embed, kg)
        tail_hid = hid_uv[tail, b_arange]

        hid_vu = self._propagate_rspmm(tail, tail_embed, ht_embed, kg)
        head_hid = hid_vu[head, b_arange]

        if self.feat == "E":
            embed = torch.cat([head_embed, head_hid, tail_hid, tail_embed], dim=-1)
        else:
            embed = torch.cat([head_hid, tail_hid], dim=-1)
        return self.Wr(embed)          # (B, eval_rel)


__all__ = ["EmerGNN_ML_RSPMM"]
