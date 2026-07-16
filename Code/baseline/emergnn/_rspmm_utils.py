"""Helpers for the torchdrug ``generalized_rspmm`` rspmm backend.

Isolates the torchdrug dependency so the default ``chunk`` path never imports
torchdrug. ``generalized_rspmm`` is torchdrug's fused relational sparse-dense
matmul CUDA kernel; the first call in a process JIT-compiles the extension
(~30s, one-time), subsequent calls are instant.

Convention (verified against the kernel, diff 0.00):
    out[row] += relation[rel] * input[col]      # for every (row, col, rel) nnz
i.e. the sparse tensor's FIRST index is the output/destination node, the SECOND
index is the input/source node. This matches the ORIGINAL EmerGNN graph built in
``LARS-research/EmerGNN/DrugBank/load_data.py`` (``double_triple`` stores
``(t, h, r)`` and ``(h, t, r + n_rel)``), so feeding the same index layout
reproduces the paper's message flow exactly.
"""
from __future__ import annotations

from typing import Callable

import torch

_RSPMM_FN: Callable | None = None


def get_generalized_rspmm() -> Callable:
    """Lazily import + return torchdrug's ``generalized_rspmm`` (JIT-compiles on
    first CUDA use). Kept lazy so the ``chunk`` backend never needs torchdrug."""
    global _RSPMM_FN
    if _RSPMM_FN is None:
        from torchdrug.layers import functional as _f  # noqa: WPS433 (lazy by design)

        _RSPMM_FN = _f.generalized_rspmm
    return _RSPMM_FN


def build_sparse_kg(
    row: torch.Tensor,
    col: torch.Tensor,
    rel: torch.Tensor,
    n_ent: int,
    all_rel: int,
) -> torch.Tensor:
    """Build the coalesced 3D sparse-COO KG ``generalized_rspmm`` consumes.

    Args:
        row: first-index (output/destination node) ids, shape (E,).
        col: second-index (input/source node) ids, shape (E,).
        rel: relation-slot ids in ``[0, all_rel)``, shape (E,).
        n_ent: number of entities (both node axes).
        all_rel: number of relation slots (= ``2 * n_base_rel + 1``).

    The incoming ``(row, col, rel)`` come from the SAME layout as
    ``kg_builder.build_sparse_adj`` / ``edges_as_dense_lists`` (i.e. the sparse
    indices ``idx[0], idx[1], idx[2]``), so ``generalized_rspmm`` on this tensor
    reproduces the original EmerGNN propagation. Values are unit; the tensor is
    coalesced (required by the kernel).
    """
    idx = torch.stack([row.long(), col.long(), rel.long()], dim=0)
    val = torch.ones(idx.shape[1], dtype=torch.float32, device=row.device)
    return torch.sparse_coo_tensor(
        idx, val, size=(n_ent, n_ent, all_rel)
    ).coalesce()


def build_sparse_kg_from_triplets(
    triplets: torch.Tensor,   # (T, 3) int: (head, tail, rel) with rel in [0, n_rel)
    n_ent: int,
    n_rel: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build the ``generalized_rspmm`` sparse KG DIRECTLY from ``(h, t, r)`` triplets,
    mirroring the ORIGINAL EmerGNN ``load_graph`` + ``double_triple``
    (``LARS-research/EmerGNN/DrugBank/load_data.py:128-134,177-185``) byte-for-byte:

        for each (h, t, r):  add nnz (t, h, r)  and  (h, t, r + n_rel)
        add self-loops:      (e, e, 2 * n_rel)  for every entity e
        size = (n_ent, n_ent, 2 * n_rel + 1); unit values; coalesced.

    With ``generalized_rspmm`` (``out[row] += rel[slot] * in[col]``) this yields:
        forward slot r:      out[t] += rel[r]      * in[h]     (h -> t)
        reverse slot r+n:    out[h] += rel[r+n]    * in[t]     (t -> h)
    exactly the paper's message flow. Building from triplets (not from cached
    edge_src/dst/rel) keeps this path independent of the chunk backend's edge
    convention, per the design review.
    """
    tri = torch.as_tensor(triplets, dtype=torch.long)
    if tri.numel() == 0:
        h = t = r = torch.empty(0, dtype=torch.long)
    else:
        h, t, r = tri[:, 0], tri[:, 1], tri[:, 2]
    e = torch.arange(n_ent, dtype=torch.long)
    row = torch.cat([t, h, e])
    col = torch.cat([h, t, e])
    rel = torch.cat([r, r + n_rel, torch.full((n_ent,), 2 * n_rel, dtype=torch.long)])
    idx = torch.stack([row, col, rel], dim=0).to(device)
    val = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        idx, val, size=(n_ent, n_ent, 2 * n_rel + 1)
    ).coalesce()
