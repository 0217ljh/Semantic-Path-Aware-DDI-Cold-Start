"""Dataset + collate helpers. Mirrors upstream ``utils.py``.

Each :class:`DTADataset` item yields a 4-tuple of PyG ``Data`` objects:
``(drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph)``. The first two
go through the MG channel, the latter two through the BKG channel. Label
``y`` is duplicated onto every ``Data`` for convenience; the model only
reads ``drug1_mol.y``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric import data as DATA
from torch_geometric.data import Batch, InMemoryDataset


class DTADataset(InMemoryDataset):
    def __init__(
        self,
        x=None,
        y=None,
        sub_graph: dict | None = None,
        smile_graph: dict | None = None,
    ) -> None:
        super().__init__()
        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph

    def read_drug_info(self, drug_id, labels):
        # drug_id is keyed as string in the JSON caches.
        c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, _deg = (
            self.smile_graph[str(drug_id)]
        )
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, _deg = (
            self.sub_graph[str(drug_id)]
        )

        data_mol = DATA.Data(
            x=torch.Tensor(np.array(features)),
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            y=torch.LongTensor([labels]),
            rel_index=torch.Tensor(np.array(rel_index, dtype=int)),
            sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0),
            sp_value=torch.Tensor(np.array(sp_value, dtype=int)),
            sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int)),
        )
        data_mol.__setitem__("c_size", torch.LongTensor([c_size]))

        data_graph = DATA.Data(
            x=torch.LongTensor(subset),
            edge_index=torch.LongTensor(subgraph_edge_index).transpose(1, 0),
            y=torch.LongTensor([labels]),
            id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
            rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
            sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
            sp_value=torch.Tensor(np.array(s_value, dtype=int)),
            sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int)),
        )

        return data_mol, data_graph

    def __len__(self) -> int:
        return len(self.drug_ID)

    def __getitem__(self, idx: int):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        labels = int(self.labels[idx])
        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id, labels)
        drug2_mol, drug2_subgraph = self.read_drug_info(drug2_id, labels)
        return drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph


def collate(data_list):
    """4-way batching of (drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph)
    tuples. Each PyG ``Batch`` is then forwarded to a separate channel of
    the TIGER model."""
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    return batchA, batchB, batchC, batchD
