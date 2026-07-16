"""TIGER core model — dual-channel mode (mol + BKG subgraph).

Port of ``Code-Released/baseline/TIGER/model/tiger.py`` (the dual-channel
default; ``mol_only=False``). Default forward path:

  drug1_mol  ──► mol GraphTransformer ──┐
                                        ├──► fc1 ──► fc2 (binary head)
  drug1_sub  ──► node GraphTransformer ─┘                       ▲
  drug2_mol  ──► mol GraphTransformer ──┐                       │
                                        ├──► fc1 ──── concat ───┘
  drug2_sub  ──► node GraphTransformer ─┘

Cold-start extension (kept from Code-Released, NOT in upstream
Blair1213/TIGER): if a drug appears in ``unseen_ids``, its BKG-subgraph
center-node embedding is REPLACED with
``cold_start_proj(mol_graph_emb) + degree_encoder(center_degree)`` before
the node GraphTransformer runs, so cold-start drugs (g2 in our split
terminology) still get a usable KG-channel representation.

Loss (dual mode):
    loss = nll(log_softmax(score), y)
         + sub_coeff · MI(drug_emb, mol_atom_emb)
         + mi_coeff  · MI(drug_emb, sub_node_emb)

Returns ``(softmax_probs, loss)`` where ``softmax_probs`` has shape
``(B, n_classes)``. For the binary case (``n_classes=2``) the
baseline adapter takes ``[:, 1]`` to get the positive-class score; for
multi-class (``n_classes=86``) the adapter uses the full row. The
loss tensor is ignored at eval time.

Logical equivalence to Blair1213/TIGER warm path (S0) verified:
  - same module composition (NodeFeatures × 2, GraphTransformer × 2,
    fc1, fc2, Discriminator, b_xent)
  - identical concat order in fc1 / fc2
  - same two MI terms (mol-atom + sub-node)
  - upstream uses ``BCELoss(softmax(pred), one_hot)`` which is
    mathematically equivalent to ``NLL(log_softmax(score), y_int)``
    used here
  - fc2 output dim is parameterized by ``n_classes`` (2 for binary,
    86 for DrugBank multi-class). Upstream's ``Linear(512, num_rel)``
    is the same mechanism.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, Linear
from torch_geometric.utils import degree

from baseline.tiger.graph_transformer import GraphTransformer


def _init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NodeFeatures(nn.Module):
    """Atom / BKG-node feature encoder.

    ``type='graph'`` for SMILES atoms (continuous Linear encoder over
    one-hot atom features); ``type='node'`` for BKG nodes (Embedding
    lookup over global node ids).
    """

    def __init__(
        self,
        max_degree: int,
        feature_num: int,
        embedding_dim: int,
        layer: int = 2,
        type: str = "graph",
    ) -> None:
        super().__init__()
        self.type = type
        if type == "graph":
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = nn.Embedding(feature_num, embedding_dim)
        self.degree_encoder = nn.Embedding(max_degree, embedding_dim, padding_idx=0)
        self.apply(lambda m: _init_params(m, layers=layer))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data):
        _, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=torch.long)
        x_degree = x_degree.clamp(max=self.degree_encoder.num_embeddings - 1)
        if self.type == "graph":
            node_feature = self.node_encoder(data.x)
        else:
            node_feature = self.node_encoder(data.x.long())
        node_feature = node_feature + self.degree_encoder(x_degree)
        return node_feature


class Discriminator(nn.Module):
    def __init__(self, n_h: int) -> None:
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        sc_1 = self.f_k(h_pl, c)
        sc_2 = self.f_k(h_mi, c)
        return torch.cat((sc_1, sc_2), 0)


class TIGER(nn.Module):
    """Dual-channel TIGER.

    Args:
        max_layer:           layers per GraphTransformer.
        num_features_drug:   atom feature dim (67 from
                             :mod:`mol_features`).
        num_nodes:           BKG total node count (drugs + entities).
                             Required when ``mol_only=False``.
        num_relations_mol:   relation vocab size for the molecule
                             GraphTransformer (covers bond types +
                             length-offset shortest-path rels).
        num_relations_graph: relation vocab size for the BKG-subgraph
                             GraphTransformer (covers KG rels +
                             length-offset shortest-path rels).
        output_dim:          channel embedding dim.
        max_degree_graph:    cap for atom-degree encoder.
        max_degree_node:     cap for BKG-node-degree encoder.
        sub_coeff:           weight for mol-atom MI loss.
        mi_coeff:            weight for sub-node MI loss (ignored when
                             ``mol_only=True``).
        dropout:             dropout prob.
        device:              ``cuda`` / ``cpu``.
        mol_only:            if ``True`` skip the KG branch entirely
                             (legacy ColdDDI mode).
    """

    def __init__(
        self,
        max_layer: int = 4,
        num_features_drug: int = 67,
        num_nodes: int = 200,
        num_relations_mol: int = 64,
        num_relations_graph: int = 64,
        output_dim: int = 64,
        max_degree_graph: int = 100,
        max_degree_node: int = 100,
        sub_coeff: float = 0.2,
        mi_coeff: float = 0.5,
        dropout: float = 0.2,
        device: str = "cuda",
        mol_only: bool = False,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.device = device
        self.mol_only = mol_only
        self.n_classes = int(n_classes)

        self.layers = max_layer
        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node
        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff

        # Mol channel
        self.mol_atom_feature = NodeFeatures(
            max_degree=max_degree_graph,
            feature_num=num_features_drug,
            embedding_dim=output_dim,
            type="graph",
        )
        self.mol_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=output_dim,
            num_heads=4,
            num_rel=num_relations_mol,
            dropout=dropout,
            type="graph",
        )

        # KG/BKG channel (only when mol_only=False)
        if not mol_only:
            self.drug_node_feature = NodeFeatures(
                max_degree=max_degree_node,
                feature_num=num_nodes,
                embedding_dim=output_dim,
                type="node",
            )
            self.node_representation_learning = GraphTransformer(
                layer_num=max_layer,
                embedding_dim=output_dim,
                num_heads=4,
                num_rel=num_relations_graph,
                dropout=dropout,
                type="node",
            )
            # Cold-start patch projection: mol-graph emb → KG-node space
            self.cold_start_proj = nn.Linear(output_dim, output_dim)

        # Heads
        self.fc1 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_classes),
        )

        # MI components
        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Cold-start patch
    # ------------------------------------------------------------------

    def _patch_unseen_center_nodes(
        self,
        node_feature: torch.Tensor,
        subgraph,
        mol_embedding: torch.Tensor,
        batch_idx: torch.Tensor,
        unseen_ids: set[int],
    ) -> None:
        """Overwrite the center-node row of every sample whose global
        drug-index is in ``unseen_ids`` with the molecule-channel
        embedding projected into the KG-node space plus a degree
        embedding. In-place on ``node_feature``.
        """
        if not unseen_ids:
            return
        dev = node_feature.device
        row, col = subgraph.edge_index[0], subgraph.edge_index[1]
        x_degree = degree(col, subgraph.x.size(0), dtype=torch.long)
        batch_size = mol_embedding.size(0)
        for i in range(batch_size):
            if int(batch_idx[i].item()) not in unseen_ids:
                continue
            center_mask = (subgraph.batch == i) & subgraph.id.bool()
            if not center_mask.any():
                continue
            center_global_idx = center_mask.nonzero(as_tuple=True)[0][0]
            deg_val = x_degree[center_global_idx].clamp(
                0, self.max_degree_node - 1
            )
            z_deg = self.drug_node_feature.degree_encoder(
                deg_val.unsqueeze(0)
            ).squeeze(0)
            node_feature[center_global_idx] = (
                self.cold_start_proj(mol_embedding[i]) + z_deg.to(dev)
            )

    # ------------------------------------------------------------------
    # MI
    # ------------------------------------------------------------------

    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        if len(idx) > 1:
            # safer than upstream's unconditional [-1] swap when batch=2
            mid = len(idx) // 2
            mid_alt = mid + 1 if mid + 1 < len(idx) else mid
            idx[mid] = idx[mid_alt]
        shuffle_embeddings = torch.index_select(
            graph_embeddings, 0, idx.to(graph_embeddings.device)
        )
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))
            c_1_list.append(c_1.expand_as(sub))
        c_0 = torch.cat(c_0_list)
        c_1 = torch.cat(c_1_list)
        sub = torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):
        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat(
            [torch.ones_like(temp), torch.zeros_like(temp)], dim=0
        ).float().to(logits.device)
        return self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        drug1_mol,
        drug1_subgraph,
        drug2_mol,
        drug2_subgraph,
        batch_idx1=None,
        batch_idx2=None,
        unseen_ids: set[int] | None = None,
    ):
        # ── 1. Mol channel ─────────────────────────────────────────
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        mol1_graph_emb, mol1_atom_emb, _ = self.mol_representation_learning(
            mol1_atom_feature, drug1_mol
        )
        mol2_graph_emb, mol2_atom_emb, _ = self.mol_representation_learning(
            mol2_atom_feature, drug2_mol
        )

        if self.mol_only:
            # legacy mol-only path: fc1 concatenates mol with itself
            drug1_emb = self.fc1(torch.cat([mol1_graph_emb, mol1_graph_emb], dim=-1))
            drug2_emb = self.fc1(torch.cat([mol2_graph_emb, mol2_graph_emb], dim=-1))
            score = self.fc2(torch.cat([drug1_emb, drug2_emb], dim=-1))
            loss_s_m = self.loss_MI(self.MI(drug1_emb, mol1_atom_emb)) + self.loss_MI(
                self.MI(drug2_emb, mol2_atom_emb)
            )
            log_probs = F.log_softmax(score, dim=-1)
            loss_label = F.nll_loss(log_probs, drug1_mol.y.view(-1))
            loss = loss_label + self.mol_coeff * loss_s_m
            return torch.exp(log_probs), loss

        # ── 2. KG/BKG channel ──────────────────────────────────────
        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        # Cold-start patch: replace center embedding for unseen drugs
        if (
            unseen_ids is not None
            and len(unseen_ids) > 0
            and batch_idx1 is not None
            and batch_idx2 is not None
        ):
            batch_idx1 = batch_idx1.to(drug1_node_feature.device)
            batch_idx2 = batch_idx2.to(drug2_node_feature.device)
            self._patch_unseen_center_nodes(
                drug1_node_feature,
                drug1_subgraph,
                mol1_graph_emb,
                batch_idx1,
                unseen_ids,
            )
            self._patch_unseen_center_nodes(
                drug2_node_feature,
                drug2_subgraph,
                mol2_graph_emb,
                batch_idx2,
                unseen_ids,
            )

        drug1_node_emb, drug1_sub_emb, _ = self.node_representation_learning(
            drug1_node_feature, drug1_subgraph
        )
        drug2_node_emb, drug2_sub_emb, _ = self.node_representation_learning(
            drug2_node_feature, drug2_subgraph
        )

        # ── 3. Fuse + score ───────────────────────────────────────
        drug1_emb = self.fc1(torch.cat([drug1_node_emb, mol1_graph_emb], dim=-1))
        drug2_emb = self.fc1(torch.cat([drug2_node_emb, mol2_graph_emb], dim=-1))
        score = self.fc2(torch.cat([drug1_emb, drug2_emb], dim=-1))

        # ── 4. MI auxiliary losses (drug ↔ mol-atom, drug ↔ sub-node) ─
        loss_s_m = self.loss_MI(self.MI(drug1_emb, mol1_atom_emb)) + self.loss_MI(
            self.MI(drug2_emb, mol2_atom_emb)
        )
        loss_s_d = self.loss_MI(self.MI(drug1_emb, drug1_sub_emb)) + self.loss_MI(
            self.MI(drug2_emb, drug2_sub_emb)
        )

        log_probs = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(log_probs, drug1_mol.y.view(-1))
        loss = (
            loss_label
            + self.mol_coeff * loss_s_m
            + self.mi_coeff * loss_s_d
        )
        return torch.exp(log_probs), loss

    # ------------------------------------------------------------------
    # Raw-logits path (ADDITIVE — multilabel Case-B, TIGER-scoped)
    # ------------------------------------------------------------------

    def forward_logits(
        self,
        drug1_mol,
        drug1_subgraph,
        drug2_mol,
        drug2_subgraph,
        batch_idx1=None,
        batch_idx2=None,
        unseen_ids: set[int] | None = None,
    ):
        """Run the SAME encoder + dual-channel fusion + ``fc2`` as
        :meth:`forward`, but return RAW LOGITS ``(B, n_classes)`` and the
        auxiliary MI loss (NO ``log_softmax`` / ``nll_loss`` / label loss).

        Added for the TWOSIDES multilabel Case-B adaptation
        (:mod:`baseline.tiger.multi_label_cls.baseline`), which needs
        per-label sigmoid + masked BCE instead of the softmax CE that the
        binary / multiclass task heads use. This method is PURELY ADDITIVE:
        it duplicates the encoder-through-``fc2`` composition of
        :meth:`forward` without altering that method, so the existing
        binary / multiclass softmax behaviour is unchanged.

        Returns ``(logits, aux_loss)`` where ``logits`` are the raw ``fc2``
        outputs and ``aux_loss`` is ``sub_coeff * MI(drug, mol-atom)
        [+ mi_coeff * MI(drug, sub-node)]`` (the same MI terms
        :meth:`forward` adds to its label loss). The label-side BCE is
        computed by the caller from these logits.
        """
        # ── 1. Mol channel ─────────────────────────────────────────
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        mol1_graph_emb, mol1_atom_emb, _ = self.mol_representation_learning(
            mol1_atom_feature, drug1_mol
        )
        mol2_graph_emb, mol2_atom_emb, _ = self.mol_representation_learning(
            mol2_atom_feature, drug2_mol
        )

        if self.mol_only:
            drug1_emb = self.fc1(torch.cat([mol1_graph_emb, mol1_graph_emb], dim=-1))
            drug2_emb = self.fc1(torch.cat([mol2_graph_emb, mol2_graph_emb], dim=-1))
            logits = self.fc2(torch.cat([drug1_emb, drug2_emb], dim=-1))
            loss_s_m = self.loss_MI(self.MI(drug1_emb, mol1_atom_emb)) + self.loss_MI(
                self.MI(drug2_emb, mol2_atom_emb)
            )
            aux_loss = self.mol_coeff * loss_s_m
            return logits, aux_loss

        # ── 2. KG/BKG channel ──────────────────────────────────────
        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        if (
            unseen_ids is not None
            and len(unseen_ids) > 0
            and batch_idx1 is not None
            and batch_idx2 is not None
        ):
            batch_idx1 = batch_idx1.to(drug1_node_feature.device)
            batch_idx2 = batch_idx2.to(drug2_node_feature.device)
            self._patch_unseen_center_nodes(
                drug1_node_feature,
                drug1_subgraph,
                mol1_graph_emb,
                batch_idx1,
                unseen_ids,
            )
            self._patch_unseen_center_nodes(
                drug2_node_feature,
                drug2_subgraph,
                mol2_graph_emb,
                batch_idx2,
                unseen_ids,
            )

        drug1_node_emb, drug1_sub_emb, _ = self.node_representation_learning(
            drug1_node_feature, drug1_subgraph
        )
        drug2_node_emb, drug2_sub_emb, _ = self.node_representation_learning(
            drug2_node_feature, drug2_subgraph
        )

        # ── 3. Fuse + score (raw logits, no softmax) ───────────────
        drug1_emb = self.fc1(torch.cat([drug1_node_emb, mol1_graph_emb], dim=-1))
        drug2_emb = self.fc1(torch.cat([drug2_node_emb, mol2_graph_emb], dim=-1))
        logits = self.fc2(torch.cat([drug1_emb, drug2_emb], dim=-1))

        # ── 4. MI auxiliary losses (same as forward()) ─────────────
        loss_s_m = self.loss_MI(self.MI(drug1_emb, mol1_atom_emb)) + self.loss_MI(
            self.MI(drug2_emb, mol2_atom_emb)
        )
        loss_s_d = self.loss_MI(self.MI(drug1_emb, drug1_sub_emb)) + self.loss_MI(
            self.MI(drug2_emb, drug2_sub_emb)
        )
        aux_loss = self.mol_coeff * loss_s_m + self.mi_coeff * loss_s_d
        return logits, aux_loss
