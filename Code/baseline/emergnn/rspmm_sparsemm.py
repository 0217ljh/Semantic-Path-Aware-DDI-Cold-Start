"""
Fast pure-PyTorch replacement for the fused ``generalized_rspmm`` kernel.

The original EmerGNN propagation (Zhang et al. 2023) is

    out[t, b, :] = sum over edges (h, t, r)  of  rel_t[r, b, :] * hiddens[h, b, :]

which torchdrug fuses into one CUDA kernel (``functional.generalized_rspmm``).
Our portable port (``model.py::_chunk_compute_and_scatter``) instead materializes
a per-chunk ``(chunk_E, B, D)`` message tensor and scatter-adds it, wrapped in
``torch.utils.checkpoint`` so backward recomputes the message (2x compute).
For an E~3.6M / N~21k KG that Python-level chunk loop + checkpoint is the
bottleneck.

This module recasts the SAME math as a set of relation-wise sparse-dense
matmuls. Because ``rel_t[r, b, :]`` is CONSTANT across all edges of a given
relation slot ``r`` (it does not depend on the endpoints h, t), the sum factors:

    out[t, b, :] = sum_r  rel_t[r, b, :] * ( A_r @ H )[t, b, :]

where ``A_r[t, h]`` counts edges (h, t, r), and H = hiddens reshaped to
(N, B*D). This never builds ``(E, B, D)``; it scales with N*B*D, not E*B*D, and
relies entirely on native autograd (``torch.sparse.mm`` has a backward w.r.t.
its dense operand). No custom kernel, no gradient checkpointing, and it runs on
any device PyTorch 2.x supports (incl. sm_120 / RTX 5090).

Semantics are numerically equivalent to ``model.py::_chunk_compute_and_scatter``
(single full chunk); see ``Code/scripts/test_rspmm_equivalence.py`` for the
forward + backward + gradcheck equivalence proof.

This file is a NEW standalone helper; it does not modify the existing model.
"""
from __future__ import annotations

from typing import List, Optional

import torch


def build_relation_adjacency(
    edge_src: torch.Tensor,   # (E,) long — head h
    edge_dst: torch.Tensor,   # (E,) long — tail t
    edge_rel: torch.Tensor,   # (E,) long — relation slot r
    n_ent: int,
    all_rel: int,
    dtype: torch.dtype = torch.float32,
    to_csr: bool = False,
) -> List[Optional[torch.sparse.Tensor]]:
    """Build per-relation coalesced sparse adjacency matrices.

    Returns a list of length ``all_rel``; entry ``r`` is a coalesced sparse
    tensor ``A_r`` of shape ``(n_ent, n_ent)`` with ``A_r[t, h]`` = number of
    edges (h, t, r), or ``None`` if relation slot ``r`` has no edges.

    Coalescing sums duplicate (t, h) pairs, so repeated edges and self-loops are
    handled exactly as the edge-list scatter-add would handle them.

    With ``to_csr=True`` each matrix is converted to CSR layout, which routes
    ``torch.sparse.mm`` through the cuSPARSE spmm backend — empirically ~1.9x
    faster than COO on the RTX 5090 (see Code/scripts/bench_rspmm_speed.py).

    The adjacency is a fixed function of the KG edge lists, so build it ONCE and
    reuse across every propagation layer / training step.
    """
    device = edge_src.device
    adj: List[Optional[torch.sparse.Tensor]] = []
    for r in range(all_rel):
        mask = edge_rel == r
        if bool(mask.any()):
            t = edge_dst[mask]
            h = edge_src[mask]
            indices = torch.stack([t, h], dim=0)  # rows = dst, cols = src
            values = torch.ones(indices.size(1), device=device, dtype=dtype)
            a_r = torch.sparse_coo_tensor(
                indices, values, size=(n_ent, n_ent)
            ).coalesce()
            if to_csr:
                a_r = a_r.to_sparse_csr()
            adj.append(a_r)
        else:
            adj.append(None)
    return adj


def generalized_rspmm_sparsemm(
    adj: List[Optional[torch.sparse.Tensor]],
    rel_t: torch.Tensor,    # (all_rel, B, D)
    hiddens: torch.Tensor,  # (n_ent, B, D)
) -> torch.Tensor:
    """Relation-wise sparse-matmul form of ``generalized_rspmm`` (sum/mul).

    out[t, b, :] = sum_r  rel_t[r, b, :] * (A_r @ hiddens)[t, b, :]

    Returns the aggregated messages of shape ``(n_ent, B, D)`` BEFORE the
    per-layer ``act(linear(.))`` (i.e. the exact analogue of ``new_hiddens``
    accumulated across chunks in ``model.py::_propagate``).
    """
    n_ent, batch, dim = hiddens.shape
    h2 = hiddens.reshape(n_ent, batch * dim)
    out = torch.zeros_like(hiddens)
    for r, a_r in enumerate(adj):
        if a_r is None:
            continue
        mat = a_r if a_r.dtype == h2.dtype else a_r.to(h2.dtype)
        agg = torch.sparse.mm(mat, h2).reshape(n_ent, batch, dim)
        out = out + agg * rel_t[r].unsqueeze(0)
    return out


def generalized_rspmm_edgelist(
    hiddens: torch.Tensor,    # (n_ent, B, D)
    rel_t: torch.Tensor,      # (all_rel, B, D)
    edge_src: torch.Tensor,   # (E,)
    edge_rel: torch.Tensor,   # (E,)
    edge_dst: torch.Tensor,   # (E,)
    n_ent: int,
) -> torch.Tensor:
    """Reference edge-list form (single full chunk).

    Byte-for-byte the same computation as ``model.py::_chunk_compute_and_scatter``
    with a single chunk covering all edges (pure-PyTorch ``index_add_`` path).
    Used only as the ground-truth reference in the equivalence test; NOT for
    training (it materializes the full ``(E, B, D)`` message tensor).
    """
    h_feat = hiddens.index_select(0, edge_src)  # (E, B, D)
    r_feat = rel_t.index_select(0, edge_rel)    # (E, B, D)
    msg = h_feat * r_feat                        # (E, B, D)
    out = torch.zeros(
        n_ent, hiddens.size(1), hiddens.size(2),
        device=hiddens.device, dtype=hiddens.dtype,
    )
    out.index_add_(0, edge_dst, msg)
    return out
