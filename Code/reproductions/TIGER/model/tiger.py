"""TIGER main model (paper Sec "Dual-channel Representation Learning" +
"Drug-Drug Interaction Prediction", Eqs. 6-11).

Direct port of upstream ``model/tiger.py`` (Blair1213/TIGER). Algorithm
preserved byte-for-byte; cosmetic additions only.

Architecture summary:
  * MG channel  : Drug molecular graph atoms → GraphTransformer(type='graph')
                  → mean-pool → ``g_i``  [Eq. 6]
  * BKG channel : Drug subgraph nodes      → GraphTransformer(type='node')
                  → centre-node pick       → ``s_i``  [paper "node-level"]
  * Concat g_i || s_i → MLP fc1 → ``h_i``                [Eq. 7]
  * Concat h_i || h_j → MLP fc2 (2-d logits) → softmax    [Eq. 8]
  * Loss = NLL(label) + β1·BCE(MI(h,g)) + β2·BCE(MI(h,s)) [Eqs. 9-11]
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, Linear
from torch_geometric.utils import degree

from .graph_transformer import GraphTransformer


def init_params(module: torch.nn.Module, layers: int = 2) -> None:
    """Per-paper init: small-std Normal for Linear weights (Xavier-equivalent
    at small scale) and Normal(0, 0.02) for Embedding weights."""
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NodeFeatures(torch.nn.Module):
    """Node-feature encoder: project raw features + add degree embedding
    (paper Eq. 3).

    For MG channel (type='graph'): ``feature_num`` = 67 (atom one-hot dim),
    ``node_encoder`` is a Linear.
    For BKG channel (type='node'): ``feature_num`` = #BKG-nodes, ``node_encoder``
    is an Embedding (lookup by node-id).
    """

    def __init__(
        self,
        degree: int,
        feature_num: int,
        embedding_dim: int,
        layer: int = 2,
        type: str = "graph",
    ) -> None:
        super().__init__()

        if type == "graph":
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = torch.nn.Embedding(feature_num, embedding_dim)

        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim, padding_idx=0)
        self.apply(lambda module: init_params(module, layers=layer))

    def reset_parameters(self) -> None:
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data) -> torch.Tensor:
        from torch_geometric.utils import degree as pyg_degree
        _row, col = data.edge_index
        x_degree = pyg_degree(col, data.x.size(0), dtype=data.x.dtype)
        node_feature = self.node_encoder(data.x)
        node_feature += self.degree_encoder(x_degree.long())
        return node_feature


class TIGER(torch.nn.Module):
    """TIGER — dual-channel relation-aware graph transformer for DDI."""

    def __init__(
        self,
        max_layer: int = 6,
        num_features_drug: int = 78,
        num_nodes: int = 200,
        num_relations_mol: int = 10,
        num_relations_graph: int = 10,
        output_dim: int = 64,
        max_degree_graph: int = 100,
        max_degree_node: int = 100,
        sub_coeff: float = 0.2,
        mi_coeff: float = 0.5,
        dropout: float = 0.2,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()

        print("TIGER Loaded")
        self.device = device

        self.layers = max_layer
        self.num_features_drug = num_features_drug

        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node

        # NOTE upstream uses ``self.mol_coeff = sub_coeff``. Per paper Eq. 11,
        # ``β1`` weights L_MI(h, g) (MG-channel MI) and ``β2`` weights
        # L_MI(h, s) (BKG-channel MI). The ``sub_coeff`` CLI arg maps to β1.
        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout

        self.mol_atom_feature = NodeFeatures(
            degree=max_degree_graph,
            feature_num=num_features_drug,
            embedding_dim=output_dim,
            type="graph",
        )
        self.drug_node_feature = NodeFeatures(
            degree=max_degree_node,
            feature_num=num_nodes,
            embedding_dim=output_dim,
            type="node",
        )

        self.mol_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=output_dim,
            num_heads=4,
            num_rel=num_relations_mol,
            dropout=dropout,
            type="graph",
        )
        self.node_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=output_dim,
            num_heads=4,
            num_rel=num_relations_graph,
            dropout=dropout,
            type="node",
        )

        # fc1: per-drug dual-channel projection (Eq. 7)
        self.fc1 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

        # fc2: pairwise classifier (Eq. 8) → 2 logits (binary DDI)
        self.fc2 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

    def to(self, device):  # type: ignore[override]
        self.mol_atom_feature.to(device)
        self.drug_node_feature.to(device)
        self.mol_representation_learning.to(device)
        self.node_representation_learning.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.disc.to(device)
        self.b_xent.to(device)
        return self

    def reset_parameters(self) -> None:
        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()
        self.mol_representation_learning.reset_parameters()
        self.node_representation_learning.reset_parameters()

    def forward(
        self,
        drug1_mol,
        drug1_subgraph,
        drug2_mol,
        drug2_subgraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Atom + node feature encoding (paper Eq. 3)
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        # Dual-channel representation learning (paper Eq. 4-6)
        mol1_graph_embedding, mol1_atom_embedding, _mol1_attn = self.mol_representation_learning(
            mol1_atom_feature, drug1_mol
        )
        mol2_graph_embedding, mol2_atom_embedding, _mol2_attn = self.mol_representation_learning(
            mol2_atom_feature, drug2_mol
        )

        drug1_node_embedding, drug1_sub_embedding, _drug1_attn = self.node_representation_learning(
            drug1_node_feature, drug1_subgraph
        )
        drug2_node_embedding, drug2_sub_embedding, _drug2_attn = self.node_representation_learning(
            drug2_node_feature, drug2_subgraph
        )

        # Dual-channel concatenation → drug embedding h_i (Eq. 7)
        drug1_embedding = self.fc1(
            torch.concat([drug1_node_embedding, mol1_graph_embedding], dim=-1)
        )
        drug2_embedding = self.fc1(
            torch.concat([drug2_node_embedding, mol2_graph_embedding], dim=-1)
        )

        # Pairwise classifier (Eq. 8) → 2-dim logits
        score = self.fc2(torch.concat([drug1_embedding, drug2_embedding], dim=-1))

        # MI auxiliary losses (Eq. 10): for each drug, MI(h, atom-set) and MI(h, sub-set)
        loss_s_m = self.loss_MI(self.MI(drug1_embedding, mol1_atom_embedding)) + self.loss_MI(
            self.MI(drug2_embedding, mol2_atom_embedding)
        )
        loss_s_d = self.loss_MI(self.MI(drug1_embedding, drug1_sub_embedding)) + self.loss_MI(
            self.MI(drug2_embedding, drug2_sub_embedding)
        )

        predicts_drug = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(predicts_drug, drug1_mol.y.view(-1))

        # Total objective (Eq. 11)
        loss = loss_label + self.mol_coeff * loss_s_m + self.mi_coeff * loss_s_d

        return torch.exp(predicts_drug)[:, 1], loss

    def MI(
        self, graph_embeddings: torch.Tensor, sub_embeddings: list[torch.Tensor]
    ) -> torch.Tensor:
        """Batch-wise MI discriminator inputs: positive=(h, sub), negative=
        (h_shuffled, sub). Shuffle is index-reversal with middle swap (matches
        upstream exactly)."""
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))  # pos
            c_1_list.append(c_1.expand_as(sub))  # neg
        c_0, c_1, sub = (
            torch.cat(c_0_list),
            torch.cat(c_1_list),
            torch.cat(sub_embeddings),
        )
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits: torch.Tensor) -> torch.Tensor:
        """JS-divergence MI estimator as BCE-with-logits between pos/neg
        scores (paper Eq. 10)."""
        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(
            self.device
        )
        return self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))

    def save(self, path: str) -> str:
        save_path = os.path.join(path, self.__class__.__name__ + ".pt")
        torch.save(self.state_dict(), save_path)
        return save_path


class Discriminator(nn.Module):
    """Bilinear discriminator for JS-MI estimator. Inputs:
      * sub:   sub-element embeddings (atoms or BKG-nodes), shape (N, d)
      * c_pl:  positive context (matching drug h),           shape (N, d)
      * c_mi:  negative context (shuffled drug h),           shape (N, d)
    Output: concat(sc_pos, sc_neg) → 2N scalar logits for BCE.
    """

    def __init__(self, n_h: int) -> None:
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m: torch.nn.Module) -> None:
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(
        self,
        c: torch.Tensor,
        h_pl: torch.Tensor,
        h_mi: torch.Tensor,
        s_bias1: Optional[torch.Tensor] = None,
        s_bias2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        return logits
