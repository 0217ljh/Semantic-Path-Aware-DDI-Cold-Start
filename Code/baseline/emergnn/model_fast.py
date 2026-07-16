"""
EmerGNNFast — EmerGNN with CSR sparse-matmul propagation (RTX 5090 / sm_120).

Drop-in subclass of :class:`baseline.emergnn.model.EmerGNN` that overrides ONLY
``_propagate``. The base class replaces torchdrug's fused ``generalized_rspmm``
CUDA kernel with a Python-level chunked ``index_select``/``index_add_`` loop under
``torch.utils.checkpoint`` (2x compute). This subclass instead recasts the same
math as per-relation CSR sparse-dense matmuls (cuSPARSE backend), which on the
5090 is ~1.69x faster than the checkpointed chunk loop at comparable memory, and
is numerically equivalent forward+backward (fp64 exact; fp32 differs only by
summation-order roundoff). See:
  * Code/baseline/emergnn/rspmm_sparsemm.py          (the kernel)
  * Code/scripts/test_rspmm_equivalence.py           (fwd+bwd+gradcheck proof)
  * Code/scripts/bench_rspmm_speed.py                (benchmark)

Everything else (constructor, forward, attention-over-relations, score head,
weights) is inherited unchanged, so EmerGNNFast is a behavioural drop-in for
EmerGNN up to fp32 roundoff.

The per-relation adjacency is a fixed function of the KG edge lists, so it is
built once and cached. shuffle_train reshuffles edge ORDER each epoch but the
coalesced adjacency is order-invariant, so the cache stays valid; the cache is
keyed by a content fingerprint (order-invariant) plus device/dtype/shape so it
invalidates correctly if the graph content, device, or dtype ever changes.

New file; does not modify the existing model.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from baseline.emergnn.model import EmerGNN
from baseline.emergnn.rspmm_sparsemm import (
    build_relation_adjacency,
    generalized_rspmm_sparsemm,
)


class EmerGNNFast(EmerGNN):
    """EmerGNN whose message passing uses per-relation CSR sparse.mm.

    Identical constructor signature and outputs to :class:`EmerGNN` (up to fp32
    summation-order roundoff). Only ``_propagate`` is overridden.
    """

    def _adjacency_fingerprint(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple:
        """Order-invariant, content-sensitive cache key for the KG edge lists.

        Uses sums (order-invariant) so an epoch-to-epoch edge reshuffle does not
        invalidate the cache, but any change to graph content / size / relation
        vocabulary / device / dtype does.
        """
        if edge_rel.numel() == 0:
            rel_max = -1
        else:
            rel_max = int(edge_rel.max().item())
        return (
            int(edge_src.numel()),
            int(edge_src.sum().item()),
            int(edge_dst.sum().item()),
            int(edge_rel.sum().item()),
            rel_max,
            self.n_ent,
            self.all_rel,
            str(edge_src.device),
            str(dtype),
        )

    def _get_cached_adjacency(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        dtype: torch.dtype,
    ) -> List[Optional[torch.sparse.Tensor]]:
        """Build (once) and cache the per-relation CSR adjacency for this KG."""
        fp = self._adjacency_fingerprint(edge_src, edge_dst, edge_rel, dtype)
        cached = getattr(self, "_adj_cache", None)
        if cached is not None and cached[0] == fp:
            return cached[1]
        adj = build_relation_adjacency(
            edge_src, edge_dst, edge_rel, self.n_ent, self.all_rel,
            dtype=dtype, to_csr=True,
        )
        # not an nn.Parameter/buffer — plain attribute so it is not moved by
        # .to()/.cuda() (it is already on the correct device) nor saved in
        # state_dict; it is regenerated on demand from the edge lists.
        self._adj_cache = (fp, adj)
        return adj

    def _propagate(
        self,
        source_idx: torch.Tensor,           # (B,)
        source_embed: torch.Tensor,         # (B, n_dim)
        ht_embed: torch.Tensor,             # (B, 2*n_dim)
        edge_src: torch.Tensor,             # (E,)
        edge_dst: torch.Tensor,             # (E,)
        edge_rel: torch.Tensor,             # (E,)
    ) -> torch.Tensor:
        """L layers of message passing, CSR sparse.mm form.

        Byte-for-byte identical to ``EmerGNN._propagate`` EXCEPT the inner
        chunked message-passing loop is replaced by a single per-relation CSR
        ``generalized_rspmm_sparsemm`` call. The relation-attention computation
        (rel_w / rel_emb / rel_t) and the per-layer ``act(linear(.))`` are copied
        verbatim from the parent so the two paths differ only in the aggregation
        kernel.
        """
        B = source_idx.size(0)
        device = source_embed.device

        hiddens = torch.zeros(self.n_ent, B, self.n_dim, device=device)
        batch_arange = torch.arange(B, device=device)
        hiddens[source_idx, batch_arange] = source_embed

        adj = self._get_cached_adjacency(edge_src, edge_dst, edge_rel, hiddens.dtype)

        for l in range(self.L):
            # (a) attention over relation slots — identical to parent
            rel_w = self.attn_relation[l](F.relu(self.relation_linear[l](ht_embed)))
            rel_w = torch.sigmoid(rel_w)
            rel_emb = self.rel_kg[l].weight
            rel_per_batch = rel_w.unsqueeze(-1) * rel_emb.unsqueeze(0)
            rel_t = rel_per_batch.transpose(0, 1)  # (all_rel, B, n_dim)

            # (b) message passing via per-relation CSR sparse.mm (replaces the
            #     chunked index_select/index_add_ + checkpoint loop)
            new_hiddens = generalized_rspmm_sparsemm(adj, rel_t, hiddens)
            new_hiddens = self.act(self.linear[l](new_hiddens))
            hiddens = new_hiddens
        return hiddens
