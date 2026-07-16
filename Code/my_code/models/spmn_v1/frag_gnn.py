"""SPMN v1 — g_frag: learnable GNN over BRICS fragment atom-subgraphs.

Consumes the atom-subgraphs from :mod:`frag_encoder` and produces one
embedding ``z_i`` per fragment (shared across all drugs -> inductive; no
drug-id). A small GIN over atoms + mean pool. This is the faithful learnable
``g_frag`` (not a fixed fingerprint).

Batching follows PyG convention: atoms of all fragments in a minibatch are
concatenated; ``edge_index`` is offset per fragment; a ``batch`` vector maps
each atom to its fragment, and ``global_mean_pool`` reduces atoms -> fragment.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

from my_code.models.spmn_v1.frag_encoder import (
    ATOM_FEAT_DIM, BOND_FEAT_DIM, FragmentGraph,
)


class FragGNN(nn.Module):
    """GINE over fragment atom-graphs — uses bond-type edge features (GINEConv,
    edge_dim=BOND_FEAT_DIM), so the chemically-meaningful bond types actually
    affect the fragment embedding (not silently dropped)."""

    def __init__(self, d: int = 64, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.d = int(d)
        self.atom_lin = nn.Linear(ATOM_FEAT_DIM, d)
        self.convs = nn.ModuleList()
        for _ in range(int(n_layers)):
            mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            self.convs.append(GINEConv(mlp, edge_dim=BOND_FEAT_DIM))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                n_frags: int) -> torch.Tensor:
        """x:(A,ATOM_FEAT_DIM) edge_index:(2,E) edge_attr:(E,BOND_FEAT_DIM)
        batch:(A,) -> (n_frags, d)."""
        if x.numel() == 0:
            return torch.zeros(n_frags, self.d, device=self.out.weight.device)
        h = self.atom_lin(x)
        for conv in self.convs:
            h = self.dropout(torch.relu(conv(h, edge_index, edge_attr)))
        z = global_mean_pool(h, batch, size=n_frags)   # (n_frags, d)
        return self.out(z)


def collate_fragment_graphs(
    frags: list[FragmentGraph], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Batch a list of FragmentGraph into (x, edge_index, edge_attr, batch, n_frags)."""
    if not frags:
        z = torch.zeros(0, ATOM_FEAT_DIM, device=device)
        return (z, torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, BOND_FEAT_DIM, device=device),
                torch.zeros(0, dtype=torch.long, device=device), 0)
    xs, eis, eas, batch = [], [], [], []
    offset = 0
    for fi, fg in enumerate(frags):
        xs.append(fg.x)
        if fg.edge_index.shape[1] > 0:
            eis.append(fg.edge_index + offset)
            eas.append(fg.edge_attr)
        batch.append(np.full(fg.x.shape[0], fi, dtype=np.int64))
        offset += fg.x.shape[0]
    x = torch.as_tensor(np.concatenate(xs), device=device)
    if eis:
        edge_index = torch.as_tensor(np.concatenate(eis, axis=1), device=device)
        edge_attr = torch.as_tensor(np.concatenate(eas, axis=0), device=device)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.zeros(0, BOND_FEAT_DIM, device=device)
    batch_t = torch.as_tensor(np.concatenate(batch), device=device)
    return x, edge_index, edge_attr, batch_t, len(frags)


__all__ = ["FragGNN", "collate_fragment_graphs"]
