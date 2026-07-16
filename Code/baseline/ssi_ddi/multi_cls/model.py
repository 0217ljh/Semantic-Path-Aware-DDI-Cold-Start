"""SSI-DDI multi-class model — all-relation RESCAL scoring head.

The binary SSI-DDI collapses the paper's multi-relational RESCAL head to a single
"interaction" relation (``rel_total=1``). This variant RESTORES SSI-DDI's original
multi-relational design: ``rel_total = n_classes`` RESCAL relation matrices, and a
forward that scores ALL K relations for a pair in one pass (``(B, K)`` logits), so
the unified multiclass task can apply softmax cross-entropy over the K DDI types.

Reuses the parent :class:`baseline.ssi_ddi.models.SSI_DDI` GAT blocks +
co-attention + RESCAL ``rel_emb`` UNCHANGED; only adds a vectorized all-relation
scoring path. The all-K score for relation r is mathematically identical to the
single-relation RESCAL forward evaluated at r (verified by equivalence test):
    scores[b, r] = sum_{i,j} alpha[b,i,j] * (Hn[b,i] @ Mn[r] @ Tn[b,j])
with Hn/Tn = L2-normalized block embeddings and Mn = L2-normalized (over the
flattened n*n) relation matrices — exactly RESCAL's normalization (layers.py:49-58).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from baseline.ssi_ddi.models import SSI_DDI


class SSI_DDI_MC(SSI_DDI):
    """SSI-DDI with an all-relation multiclass head (``forward_all`` -> (B, n_rels))."""

    def _pair_reprs(self, h_data, t_data):
        """Run the GAT blocks + co-attention (parent forward's body, minus RESCAL).
        Returns (repr_h, repr_t, attentions) with repr_* = (B, n_blocks, kge_dim),
        attentions = (B, n_blocks, n_blocks)."""
        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h, repr_t = [], []
        for i, block in enumerate(self.blocks):
            out1, out2 = block(h_data), block(t_data)
            h_data = out1[0]
            t_data = out2[0]
            repr_h.append(out1[1])
            repr_t.append(out2[1])
            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        attentions = self.co_attention(repr_h, repr_t)
        return repr_h, repr_t, attentions

    def forward_all(self, triples_ht) -> torch.Tensor:
        """(h_data, t_data) -> (B, n_rels) logits over ALL relations."""
        h_data, t_data = triples_ht
        repr_h, repr_t, attentions = self._pair_reprs(h_data, t_data)
        n = self.kge_dim
        # RESCAL normalization (layers.py:49-52): normalize rel matrices over the
        # flattened n*n vector, heads/tails over the feature dim.
        Mn = F.normalize(self.KGE.rel_emb.weight, dim=-1).view(self.rel_total, n, n)
        Hn = F.normalize(repr_h, dim=-1)   # (B, L, n)
        Tn = F.normalize(repr_t, dim=-1)   # (B, L, n)
        # scores[b,r] = sum_{i,j} alpha[b,i,j] * (Hn[b,i] @ Mn[r] @ Tn[b,j])
        scores = torch.einsum("bij,bip,rpq,bjq->br", attentions, Hn, Mn, Tn)
        return scores                       # (B, n_rels)


__all__ = ["SSI_DDI_MC"]
