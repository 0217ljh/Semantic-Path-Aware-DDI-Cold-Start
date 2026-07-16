"""MRCGNN model core — multi-relational RGCN + DGI-style contrastive + molecular skip.

Ported + adapted from upstream
``Paper/Reference/Original-Code/MRCGNN/codes for MRCGNN/layer.py:83-190``
(AAAI-2023 MRCGNN). File-independence (CLAUDE.md §Baseline): COPY+adapt, no import of
``Paper/Reference/Original-Code/`` or ``reproductions/``.

Architecture (faithful to upstream):
  * two stacked ``RGCNConv`` over the DDI-event graph, ``num_relations = K_global``
    (upstream hard-codes 65, ``layer.py:88-89``; parameterized, correction #5).
  * layer-attention fusion of the two RGCN layer outputs via a learnable 2-scalar
    ``self.attt`` (upstream ``layer.py:91-94,176``).
  * TWO contrastive discriminations against the true-graph summary
    (upstream ``layer.py:161-168``):
      - view A = feature-shuffled nodes on the SAME edges/edge_type
        (upstream ``data_s`` -> ``x_a`` at ``layer.py:148-152``);
      - view B = the ORIGINAL adjacency with only the RELATION LABELS shuffled
        (upstream ``data_a`` -> reuses ``adj`` with shuffled ``e_type1`` at
        ``layer.py:154-158``). NOT a shuffled-edge graph (correction #2).
  * TrimNet molecular features used in BOTH branches (correction #1):
      - as the RGCN node input ``x`` (upstream ``data_o.x`` = TrimNet feats,
        ``data_preprocess.py:142-186``), and
      - as the residual/skip concatenated at pair scoring
        (upstream ``self.features1[idx]``, ``layer.py:114,182-185``).

Key deviations from upstream (all mandated by the corrections / the benchmark contract):
  * All hard-codes parameterized (correction #4): ``num_relations``/head width -> ``K``,
    ``n_drugs`` -> ``n_drugs`` arg, Discriminator width -> ``hidden2``.
  * Molecular features, ``K``, ``n_drugs``, dims injected via the constructor. The model
    does NOT re-load ``data/drug_listxiao.csv`` or ``drug_emb_trimnet*.npy`` internally
    (upstream ``layer.py:109-123`` deleted).
  * Device-agnostic: no unconditional ``.cuda()`` (upstream ``layer.py:123,182-183``);
    the skip feature tensor is a registered buffer that follows ``.to(device)``.
  * ``forward`` takes explicit tensors (feats/adj/edge_type for the true graph, the
    two corrupted-view inputs, and the pair index tensors) instead of the upstream
    PyG ``Data`` bundles, but the computation is line-for-line equivalent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from baseline.mrcgnn.layers import AvgReadout, Discriminator, MLPHead, mlp_input_width


class MRCGNN(nn.Module):
    """MRCGNN encoder + contrastive heads + pair classifier.

    Parameters
    ----------
    feature:
        Input molecular-feature dim (upstream ``args.dimensions`` = 128).
    n_classes:
        Global number of DDI-event classes ``K_global`` — drives BOTH the RGCN
        ``num_relations`` and the classifier head width (correction #5).
    n_drugs:
        Number of drug nodes in the DDI graph (upstream fixed 572; correction #4).
        Kept for shape validation / documentation of the node axis.
    mol_features:
        ``(n_drugs, feature)`` float tensor of TrimNet molecular embeddings, used as
        the skip/residual at pair scoring (correction #1). Registered as a buffer so
        it moves with ``.to(device)``. The SAME tensor is expected to be passed as the
        RGCN node input ``x`` at ``forward`` time (the caller owns that plumbing).
    hidden1, hidden2:
        RGCN layer widths (upstream 64 / 32).
    dropout:
        Dropout prob applied after the first RGCN layer (upstream 0.5).
    """

    def __init__(self, *, feature: int, n_classes: int, n_drugs: int,
                 mol_features: torch.Tensor, hidden1: int = 64, hidden2: int = 32,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.feature = int(feature)
        self.n_classes = int(n_classes)
        self.n_drugs = int(n_drugs)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.dropout = float(dropout)

        # two multi-relational GCN layers over the DDI-event graph
        # (upstream layer.py:88-89; num_relations parameterized -> K_global).
        self.encoder_o1 = RGCNConv(self.feature, self.hidden1, num_relations=self.n_classes)
        self.encoder_o2 = RGCNConv(self.hidden1, self.hidden2, num_relations=self.n_classes)

        # learnable 2-scalar layer attention (upstream layer.py:91-94).
        attt = torch.zeros(2)
        attt[0] = 0.5
        attt[1] = 0.5
        self.attt = nn.Parameter(attt)

        # DGI-style contrastive discriminator over the hidden2-dim node embeddings
        # (upstream layer.py:95, Discriminator(hidden2*2) but the Bilinear is (32,32,1)).
        self.disc = Discriminator(self.hidden2)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

        # pair classifier: input width recomputed from the fusion arithmetic
        # (layers.mlp_input_width), output = K_global (corrections #4, #5).
        skip_dim = self.feature
        self.mlp = MLPHead(mlp_input_width(self.hidden1, self.hidden2, skip_dim),
                           self.n_classes)

        # molecular skip/residual features (upstream self.features1, layer.py:114).
        # Registered as a buffer -> device-agnostic (correction #4).
        feats = torch.as_tensor(mol_features, dtype=torch.float)
        if feats.shape != (self.n_drugs, self.feature):
            raise ValueError(
                f"mol_features shape {tuple(feats.shape)} != (n_drugs={self.n_drugs}, "
                f"feature={self.feature})")
        self.register_buffer("mol_skip", feats)

    def forward(self, x_o: torch.Tensor, adj: torch.Tensor, edge_type: torch.Tensor,
                x_a: torch.Tensor, edge_type_shuf: torch.Tensor,
                idx_a: torch.Tensor, idx_b: torch.Tensor):
        """Faithful re-implementation of upstream ``forward`` (layer.py:130-190).

        Inputs
        ------
        x_o:            (n_drugs, feature) true-graph node features (= TrimNet feats).
        adj:            (2, E) edge_index of the true DDI-event graph (bidirectional).
        edge_type:      (E,) global DDI-event id per edge (view-A also uses these).
        x_a:            (n_drugs, feature) feature-shuffled node features (view A / data_s).
        edge_type_shuf: (E,) shuffled relation labels on the SAME ``adj`` (view B / data_a).
        idx_a, idx_b:   (B,) long node indices of the two drugs in each pair.

        Returns
        -------
        log:      (B, K) classification logits.
        ret_os:   (n_drugs, 2) view-A contrastive discriminator output.
        ret_os_a: (n_drugs, 2) view-B contrastive discriminator output.
        x2_os:    (n_drugs, hidden2) true-graph 2nd-layer node embeddings (for downstream/eval).
        """
        edge_type = edge_type.to(torch.int64)
        edge_type_shuf = edge_type_shuf.to(torch.int64)

        # ---- true DDI-event graph (upstream layer.py:141-145) ----
        x1_o = F.relu(self.encoder_o1(x_o, adj, edge_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o, adj, edge_type)

        # ---- view A: feature-shuffled nodes, SAME edges/edge_type (upstream layer.py:148-152) ----
        x1_o_a = F.relu(self.encoder_o1(x_a, adj, edge_type))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
        x2_os_a = self.encoder_o2(x1_o_a, adj, edge_type)

        # ---- view B: ORIGINAL adjacency, shuffled RELATION LABELS (upstream layer.py:154-158) ----
        # NOTE (correction #2): reuses ``adj`` and ``x_o``; only ``edge_type_shuf`` differs.
        x1_o_a_a = F.relu(self.encoder_o1(x_o, adj, edge_type_shuf))
        x1_o_a_a = F.dropout(x1_o_a_a, self.dropout, training=self.training)
        x2_os_a_a = self.encoder_o2(x1_o_a_a, adj, edge_type_shuf)

        # ---- graph summary + two contrastive discriminations (upstream layer.py:161-168) ----
        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)
        ret_os = self.disc(h_os, x2_o, x2_os_a)
        ret_os_a = self.disc(h_os, x2_o, x2_os_a_a)

        # ---- layer-attention fusion (upstream layer.py:176) ----
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)

        aa = idx_a.to(torch.long)
        bb = idx_b.to(torch.long)
        entity1 = final[aa]
        entity2 = final[bb]

        # ---- molecular skip/residual (correction #1; upstream layer.py:182-185) ----
        entity1 = torch.cat((entity1, self.mol_skip[aa]), dim=1)
        entity2 = torch.cat((entity2, self.mol_skip[bb]), dim=1)

        concatenate = torch.cat((entity1, entity2), dim=1)
        log = self.mlp(concatenate)

        return log, ret_os, ret_os_a, x2_o


__all__ = ["MRCGNN"]
