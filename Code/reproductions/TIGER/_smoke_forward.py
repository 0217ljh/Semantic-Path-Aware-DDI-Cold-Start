"""TIGER reproduction smoke-test: build model + run forward+backward on
synthetic mini data (no Google-Drive download needed).

Tests:
  1. TIGER instantiation with mini stats
  2. Forward pass with batch_size=4 synthetic mol+subgraph data
  3. Backward + step
  4. Predict probabilities have correct shape (B,)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric import data as DATA
from torch_geometric.data import Batch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model.tiger import TIGER


def _make_mol_data(n_atoms: int, num_features_drug: int, max_rel_mol: int) -> DATA.Data:
    # Each atom has a one-hot-ish 67-d feature (normalised so sum=1 per atom).
    feats = np.eye(num_features_drug)[np.random.randint(0, num_features_drug, n_atoms)]
    feats = feats / feats.sum(axis=1, keepdims=True)

    # Build a simple chain graph: atom 0 ↔ 1 ↔ 2 ↔ ...
    edges = []
    for i in range(n_atoms - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edges = np.array(edges, dtype=np.int64)

    n_edges = len(edges)
    # sp_value: shortest path = 1 for direct, 2 for two-hop, etc.
    sp_edge_index = edges.copy()
    sp_value = np.ones(n_edges, dtype=np.int64)
    sp_rel = np.random.randint(0, max_rel_mol, n_edges)

    data = DATA.Data(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=torch.tensor(edges.T, dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        sp_edge_index=torch.tensor(sp_edge_index.T, dtype=torch.long),
        sp_value=torch.tensor(sp_value, dtype=torch.float32),
        sp_edge_rel=torch.tensor(sp_rel, dtype=torch.long),
    )
    data.__setitem__("c_size", torch.tensor([n_atoms], dtype=torch.long))
    return data


def _make_subgraph_data(n_nodes: int, num_nodes_vocab: int, max_rel_graph: int, centre: int = 0) -> DATA.Data:
    node_ids = np.random.randint(0, num_nodes_vocab, n_nodes)
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edges = np.array(edges, dtype=np.int64)
    n_edges = len(edges)

    sp_edge_index = edges.copy()
    sp_value = np.ones(n_edges, dtype=np.int64)
    sp_rel = np.random.randint(0, max_rel_graph, n_edges)

    mapping = [False] * n_nodes
    mapping[centre] = True

    data = DATA.Data(
        x=torch.tensor(node_ids, dtype=torch.long),
        edge_index=torch.tensor(edges.T, dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        id=torch.tensor(mapping, dtype=torch.long),
        sp_edge_index=torch.tensor(sp_edge_index.T, dtype=torch.long),
        sp_value=torch.tensor(sp_value, dtype=torch.float32),
        sp_edge_rel=torch.tensor(sp_rel, dtype=torch.long),
    )
    return data


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cpu"

    B = 4
    num_features_drug = 67
    num_nodes = 200
    num_rel_mol = 50
    num_rel_graph = 80
    max_degree_graph = 50
    max_degree_node = 50

    print(f"=== TIGER smoke test (B={B}, device={device}) ===")
    model = TIGER(
        max_layer=2,
        num_features_drug=num_features_drug,
        num_nodes=num_nodes,
        num_relations_mol=num_rel_mol,
        num_relations_graph=num_rel_graph,
        output_dim=64,
        max_degree_graph=max_degree_graph,
        max_degree_node=max_degree_node,
        sub_coeff=0.1,
        mi_coeff=0.1,
        dropout=0.2,
        device=device,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TIGER params: {n_params:,}")

    # Build B drug-pairs, each pair = (mol1, sub1, mol2, sub2)
    items = []
    for i in range(B):
        mol1 = _make_mol_data(n_atoms=8, num_features_drug=num_features_drug, max_rel_mol=num_rel_mol)
        mol2 = _make_mol_data(n_atoms=6, num_features_drug=num_features_drug, max_rel_mol=num_rel_mol)
        sub1 = _make_subgraph_data(n_nodes=10, num_nodes_vocab=num_nodes, max_rel_graph=num_rel_graph)
        sub2 = _make_subgraph_data(n_nodes=12, num_nodes_vocab=num_nodes, max_rel_graph=num_rel_graph)
        mol1.y = torch.tensor([i % 2], dtype=torch.long)
        mol2.y = torch.tensor([i % 2], dtype=torch.long)
        sub1.y = torch.tensor([i % 2], dtype=torch.long)
        sub2.y = torch.tensor([i % 2], dtype=torch.long)
        items.append((mol1, sub1, mol2, sub2))

    batchA = Batch.from_data_list([t[0] for t in items])
    batchB = Batch.from_data_list([t[1] for t in items])
    batchC = Batch.from_data_list([t[2] for t in items])
    batchD = Batch.from_data_list([t[3] for t in items])
    print(f"  batchA x.shape: {batchA.x.shape}, batchB x.shape: {batchB.x.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    optimizer.zero_grad()
    predicts, loss = model(batchA, batchB, batchC, batchD)
    print(f"  forward OK: predicts.shape={predicts.shape}  loss={loss.item():.4f}")
    assert predicts.shape == (B,), f"expected ({B},), got {predicts.shape}"
    loss.backward()
    optimizer.step()
    print(f"  backward + step OK")

    # Inference pass
    model.eval()
    with torch.no_grad():
        predicts2, loss2 = model(batchA, batchB, batchC, batchD)
    print(f"  eval OK: predicts.shape={predicts2.shape}  loss={loss2.item():.4f}")

    print("\nTIGER smoke test PASS")


if __name__ == "__main__":
    main()
