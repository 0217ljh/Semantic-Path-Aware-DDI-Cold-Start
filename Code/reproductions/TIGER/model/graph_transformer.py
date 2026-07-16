"""GraphTransformer — relation-aware heterogeneous graph transformer (paper Sec
"Relation-aware Self-Attention" + "Relation-aware Heterogeneous Graph
Transformer", Eqs. 1-5).

Direct port of upstream ``model/GraphTransformer.py`` (Blair1213/TIGER), with
only cosmetic additions (``from __future__ import annotations``, type hints
on public APIs, module/class docstrings). Algorithm and tensor math are kept
byte-identical to source so behaviour matches the paper.

NOTE — paper-vs-impl quirks preserved verbatim (flagged in code review):
  * ``GraphTransformer.__init__`` loops ``range(layer_num - 1)`` — i.e. when
    paper says L=2, the actual stack has only 1 encoder layer. This is the
    behaviour reported in Table 2; we keep it as-is.
  * ``MultiheadAttention.softmax_kernel_transformation`` is dead code (defined
    but never called); kept for fidelity.
  * Attention numerator uses ``q[col]`` and ``k[row]`` (source convention),
    NOT the more common ``q[row], k[col]``. Match paper code exactly.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)


class GraphTransformerEncode(torch.nn.Module):
    """One transformer encoder block: pre-LN → MHA → residual → pre-LN → FFN
    → residual (paper Eqs. 4-5)."""

    def __init__(
        self,
        num_heads: int,
        in_dim: int,
        dim_forward: int,
        rel_encoder: torch.nn.Module,
        spatial_encoder: torch.nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim),
        )

        self.multiHeadAttention = MultiheadAttention(
            dim_model=self.in_dim,
            num_heads=self.num_heads,
            rel_encoder=rel_encoder,
            spatial_encoder=spatial_encoder,
        )

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()
        self.multiHeadAttention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(
        self,
        feature: torch.Tensor,
        sp_edge_index: torch.Tensor,
        sp_value: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multiHeadAttention(
            x_norm, sp_edge_index, sp_value, edge_rel
        )
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + feature

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, attn_weight


class SpatialEncoding(torch.nn.Module):
    """Spatial encoder — maps shortest-path-length (scalar) → 1-d bias added
    to the pre-softmax attention logits.
    """

    def __init__(self, dim_model: int) -> None:
        super().__init__()
        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU(),
        )

    def reset_parameters(self) -> None:
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap: torch.Tensor) -> torch.Tensor:
        lap_ = torch.unsqueeze(lap, dim=-1)  # [n_edges, 1]
        out = self.fnn(lap_)
        return out


class MultiheadAttention(MessagePassing):
    """Relation-aware multi-head attention (paper Eq. 1-2).

    For each edge (row, col):
      q[col] += rel_embedding(edge_rel)        # paper term (II)+(IV)
      k[row] += rel_embedding(edge_rel)        # paper term (III)+(IV)
      score  = (q[col] · k[row]) / sqrt(depth) + spatial_bias(sp_value)
    Aggregation via :class:`MessagePassing` ``propagate`` (sum aggregator).
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        rel_encoder: torch.nn.Module,
        spatial_encoder: torch.nn.Module,
        **kwargs,
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.d_model = dim_model
        self.num_heads = num_heads

        self.rel_embedding = rel_encoder
        self.rel_encoding = Sequential(Linear(dim_model, 1), ReLU())

        self.spatial_encoding = spatial_encoder

        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads

        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)

        self.dense = Linear(dim_model, dim_model)

    def reset_parameters(self) -> None:
        self.rel_embedding.reset_parameters()
        self.rel_encoding[0].reset_parameters()
        self.spatial_encoding.reset_parameters()
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.dense.reset_parameters()

    def softmax_kernel_transformation(
        self,
        data: torch.Tensor,
        is_query: bool,
        projection_matrix: Optional[torch.nn.Module] = None,
        numerical_stabilizer: float = 0.000001,
    ) -> torch.Tensor:
        """Performer-style kernel approximation. Defined but NOT called in
        the current code path; preserved verbatim from upstream for fidelity."""
        data_normalizer = 1.0 / torch.sqrt(
            torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32))
        )
        data = data_normalizer * data
        ratio = data_normalizer
        data_dash = projection_matrix(data)
        diag_data = torch.square(data)
        diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
        diag_data = diag_data / 2.0
        diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
        last_dims_t = len(data_dash.shape) - 1
        attention_dims_t = len(data_dash.shape) - 3
        if is_query:
            data_dash = ratio * (
                torch.exp(
                    data_dash
                    - diag_data
                    - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]
                )
                + numerical_stabilizer
            )
        else:
            data_dash = ratio * (
                torch.exp(
                    data_dash
                    - diag_data
                    - torch.max(
                        torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                        dim=attention_dims_t,
                        keepdim=True,
                    )[0]
                )
                + numerical_stabilizer
            )
        return data_dash

    def denominator(self, qs: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
        """Normaliser used in original code: sum_j K_j then dot with Q.
        qs: [num_nodes, num_heads, depth]; ks: [num_nodes, num_heads, depth].
        """
        all_ones = torch.ones([ks.shape[0]]).to(qs.device)
        ks_sum = torch.einsum("nhm,n->hm", ks, all_ones)
        return torch.einsum("nhm,hm->nh", qs, ks_sum)

    def forward(
        self,
        x: torch.Tensor,
        sp_edge_index: torch.Tensor,
        sp_value: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rel_embedding = self.rel_embedding(edge_rel)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x).view(x.shape[0], self.num_heads, self.depth)

        row, col = sp_edge_index
        # NOTE upstream uses q[col], k[row] (NOT q[row], k[col]) — kept for
        # bit-faithfulness with original code & reported metrics.
        query_end, key_start = q[col], k[row]
        query_end += rel_embedding
        key_start += rel_embedding

        query_end = query_end.view(sp_edge_index.shape[1], self.num_heads, self.depth)
        key_start = key_start.view(sp_edge_index.shape[1], self.num_heads, self.depth)

        edge_attn_num = torch.einsum("ehd,ehd->eh", query_end, key_start)
        data_normalizer = 1.0 / torch.sqrt(
            torch.sqrt(torch.tensor(edge_attn_num.shape[-1], dtype=torch.float32))
        )
        edge_attn_num *= data_normalizer
        edge_attn_bias = self.spatial_encoding(sp_value)
        edge_attn_num += edge_attn_bias

        attn_normalizer = self.denominator(
            q.view(x.shape[0], self.num_heads, self.depth),
            k.view(x.shape[0], self.num_heads, self.depth),
        )
        edge_attn_dem = attn_normalizer[col]
        attention_weight = edge_attn_num / edge_attn_dem  # [edge_nums, num_heads]

        outputs = []
        for i in range(self.num_heads):
            output_per_head = self.propagate(
                edge_index=sp_edge_index,
                x=v[:, i, :],
                edge_weight=attention_weight[:, i],
                size=None,
            )
            outputs.append(output_per_head)

        out = torch.cat(outputs, dim=-1)
        return self.dense(out), attention_weight


class GraphTransformer(torch.nn.Module):
    """Stack of :class:`GraphTransformerEncode` blocks with shared relation +
    spatial encoders.

    NOTE — preserved upstream quirk: the encoder ModuleList has
    ``layer_num - 1`` blocks (not ``layer_num``). When the paper specifies
    L=2, this code instantiates 1 block. Behaviour matches reported numbers.

    type=='graph': READOUT via mean-pool → graph-level representation
    type=='node' : pick centre-drug node embedding via ``data.id``
    """

    def __init__(
        self,
        layer_num: int = 3,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_rel: int = 10,
        dropout: float = 0.2,
        type: str = "graph",
    ) -> None:
        super().__init__()

        self.type = type
        self.rel_encoder = torch.nn.Embedding(num_rel, embedding_dim)
        self.spatial_encoder = SpatialEncoding(embedding_dim)

        self.encoder = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.encoder.append(
                GraphTransformerEncode(
                    num_heads=num_heads,
                    in_dim=embedding_dim,
                    dim_forward=embedding_dim * 2,
                    rel_encoder=self.rel_encoder,
                    spatial_encoder=self.spatial_encoder,
                    dropout=dropout,
                )
            )

    def reset_parameters(self) -> None:
        for e in self.encoder:
            e.reset_parameters()

    def forward(
        self, feature: torch.Tensor, data
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        x = feature
        graph_embedding_layer: list[torch.Tensor] = []
        attn_layer: list[torch.Tensor] = []
        for graphEncoder in self.encoder:
            x, attn = graphEncoder(x, data.sp_edge_index, data.sp_value, data.sp_edge_rel)
            graph_embedding_layer.append(x)
            attn_layer.append(attn)

        if self.type == "graph":
            sub_representation: list[torch.Tensor] = []
            for index, _drug_mol_graph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = global_mean_pool(x, batch=data.batch)
        else:
            sub_representation = []
            for index, _drug_subgraph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = x[data.id.nonzero().flatten()]

        return representation, sub_representation, attn_layer
