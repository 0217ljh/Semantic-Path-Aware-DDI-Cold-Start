"""Paper-faithful NBFNet model module for cold-start DDI prediction.

Implements the generalized Bellman-Ford recurrence from Zhu et al.
(NeurIPS 2021, arxiv 2106.06935). Design passed 3-round codex faithfulness
review (thread 019e95e4).

Mathematical specification (paper Eq. 4-6).
    boundary:   h_v^(0) = INDICATOR(u, v, q) = 1[v=u] * q
    message:    m_(x,r,v)^(t) = MESSAGE(h_x^(t-1), w_q(x,r,v))
    aggregate:  h_v^(t) = AGGREGATE({m_(x,r,v)^(t) for (x,r,v) in E(v)} U {h_v^(0)})

Specific choices.
    MESSAGE = DistMult.   m = h_x * w_q(r)
    w_q(r)  = W_r^(t) @ q + b_r^(t)   (per-(layer, relation) transform)
    AGGREGATE = PNA-style (mean, max, min, std) x 3 scalers (identity, amp, atten)
    nonlinearity = ReLU after each layer aggregate

Pair score (paper Eq. 7).
    h_q(u, v) = h_v^(L) after L iterations rooted at source u
    score = sigmoid(MLP(concat([h_q_sym, q])))
    where h_q_sym = h_q(a, b) + h_q(b, a) (representation-level symmetrization)

Implementation notes.
- Drug a, b are NOT learnable embeddings. drug_a serves as INDICATOR anchor;
  drug_b's representation comes from message passing.
- KG augmented with inverse edges before propagation (paper KG convention).
- Query-edge masking removes only the direct queried pair edge during training,
  not all DDI edges (relation-aware mask on (a, r_ddi, b) and inverse).
- Vectorized message passing via torch_scatter (NOT Python loops over nodes).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_add, scatter_max, scatter_mean
    _HAS_TORCH_SCATTER = True
except ImportError:
    _HAS_TORCH_SCATTER = False


# ----------------------------------------------------------------------------
# PNA aggregator (Corso et al. 2020, used as NBFNet's recommended advanced agg)
# ----------------------------------------------------------------------------


class PNAAggregator(nn.Module):
    """Principal Neighborhood Aggregation.

    Aggregators. mean, max, min, std (4 aggs)
    Scalers.     identity, amplification (log(deg+1)), attenuation (1/log(deg+1))
    Output proj. (4 * 3 * d) -> d linear

    Per Corso et al. 2020. Used as NBFNet's strongest aggregation choice
    (paper Section 4.2). Codex round-2 review confirmed this is PNA-style
    faithful but not bit-identical to NBFNet official implementation, which
    is acceptable for our purpose.
    """

    def __init__(self, d: int, scalers: tuple[str, ...] = ("identity", "amp", "atten")) -> None:
        super().__init__()
        self.d = int(d)
        self.scalers = tuple(scalers)
        self.n_agg = 4  # mean, max, min, std
        self.n_sca = len(self.scalers)
        self.proj = nn.Linear(self.d * self.n_agg * self.n_sca, self.d)

    def forward(
        self,
        messages: torch.Tensor,    # (E, d) message per edge
        target: torch.Tensor,      # (E,) target node index per edge
        n_nodes: int,
    ) -> torch.Tensor:
        """Vectorized scatter-based PNA aggregation.

        Returns (n_nodes, d) per-node aggregated representation.
        """
        if not _HAS_TORCH_SCATTER:
            raise RuntimeError("torch_scatter required for PNA aggregation")

        # 4 base aggregators (vectorized scatter)
        agg_mean = scatter_mean(messages, target, dim=0, dim_size=n_nodes)
        agg_max = scatter_max(messages, target, dim=0, dim_size=n_nodes)[0]
        agg_min = -scatter_max(-messages, target, dim=0, dim_size=n_nodes)[0]
        # std via E[x^2] - E[x]^2
        agg_sq_mean = scatter_mean(messages ** 2, target, dim=0, dim_size=n_nodes)
        agg_var = (agg_sq_mean - agg_mean ** 2).clamp(min=0.0)
        agg_std = torch.sqrt(agg_var + 1e-12)

        # Concat 4 aggregators
        aggs = torch.cat([agg_mean, agg_max, agg_min, agg_std], dim=-1)  # (n_nodes, 4d)

        # Degree per target node (for scalers)
        ones = torch.ones(target.size(0), device=messages.device)
        deg = scatter_add(ones, target, dim=0, dim_size=n_nodes)         # (n_nodes,)
        log_deg = torch.log(deg.clamp(min=1.0) + 1.0).unsqueeze(-1)      # (n_nodes, 1)

        # Apply scalers
        scaled = []
        for s in self.scalers:
            if s == "identity":
                scaled.append(aggs)
            elif s == "amp":
                scaled.append(aggs * log_deg)
            elif s == "atten":
                scaled.append(aggs / log_deg.clamp(min=1e-12))
            else:
                raise ValueError(f"unknown PNA scaler {s!r}")

        scaled_concat = torch.cat(scaled, dim=-1)  # (n_nodes, 4d * n_sca)
        return self.proj(scaled_concat)            # (n_nodes, d)

    def forward_batched(
        self,
        messages: torch.Tensor,    # (S, M, d) message per edge, batched over S sources
        target: torch.Tensor,      # (M,) target node index per edge (SHARED across S)
        n_nodes: int,
    ) -> torch.Tensor:
        """Batched PNA aggregation (ADDITIVE). Scatters along dim=1 over the
        edge axis for all S sources at once. Mathematically identical to
        calling :meth:`forward` on each source slice and stacking.

        Returns (S, n_nodes, d).
        """
        if not _HAS_TORCH_SCATTER:
            raise RuntimeError("torch_scatter required for PNA aggregation")

        # 4 base aggregators (scatter along edge axis dim=1).
        agg_mean = scatter_mean(messages, target, dim=1, dim_size=n_nodes)
        agg_max = scatter_max(messages, target, dim=1, dim_size=n_nodes)[0]
        agg_min = -scatter_max(-messages, target, dim=1, dim_size=n_nodes)[0]
        agg_sq_mean = scatter_mean(messages ** 2, target, dim=1, dim_size=n_nodes)
        agg_var = (agg_sq_mean - agg_mean ** 2).clamp(min=0.0)
        agg_std = torch.sqrt(agg_var + 1e-12)

        aggs = torch.cat([agg_mean, agg_max, agg_min, agg_std], dim=-1)  # (S, n_nodes, 4d)

        # Degree per target node (shared across S since `target` is shared).
        ones = torch.ones(target.size(0), device=messages.device)
        deg = scatter_add(ones, target, dim=0, dim_size=n_nodes)         # (n_nodes,)
        log_deg = torch.log(deg.clamp(min=1.0) + 1.0).view(1, n_nodes, 1)  # (1, n_nodes, 1)

        scaled = []
        for s in self.scalers:
            if s == "identity":
                scaled.append(aggs)
            elif s == "amp":
                scaled.append(aggs * log_deg)
            elif s == "atten":
                scaled.append(aggs / log_deg.clamp(min=1e-12))
            else:
                raise ValueError(f"unknown PNA scaler {s!r}")

        scaled_concat = torch.cat(scaled, dim=-1)  # (S, n_nodes, 4d * n_sca)
        return self.proj(scaled_concat)            # (S, n_nodes, d)


# ----------------------------------------------------------------------------
# NBF layer (single iteration of generalized Bellman-Ford)
# ----------------------------------------------------------------------------


class NBFLayer(nn.Module):
    """One iteration of generalized Bellman-Ford with per-relation transform.

    Per-(layer, relation) parameters. each NBFLayer owns its own
    W_r (n_rel, d, d) and b_r (n_rel, d) tensors. Different layers
    have different W_r / b_r (paper Section 4.2 noncommutative paths).

    Forward computation.
        m_(x,r,v) = h_x * w_q(r)              # DistMult MESSAGE
        w_q(r)    = W_r @ q + b_r             # relation-query transform
        h_v^(t)   = AGGREGATE_PNA(            # AGGREGATE with PNA
                      {m_(x,r,v) | (x,r,v) in E(v)} U {h_v^(0)}
                    )

    Boundary reinjection. h_v^(0) is included in the aggregation set at
    every layer (paper Eq. 6, codex round-1 verified). Implemented by
    appending self-edges with identity message in the message tensor.
    """

    def __init__(self, d: int, n_rel: int, aggregator: Optional[PNAAggregator] = None) -> None:
        super().__init__()
        self.d = int(d)
        self.n_rel = int(n_rel)

        # Per-(layer, relation) transform. W_r and b_r are layer-local.
        # Total params per layer. n_rel * (d*d + d) = n_rel * (d^2 + d).
        # For n_rel=50, d=32. 50 * (1024 + 32) = 52800 params per layer.
        self.W_r = nn.Parameter(torch.empty(self.n_rel, self.d, self.d))
        self.b_r = nn.Parameter(torch.zeros(self.n_rel, self.d))
        # Xavier init per-relation slice
        for r in range(self.n_rel):
            nn.init.xavier_uniform_(self.W_r[r])

        self.aggregator = aggregator if aggregator is not None else PNAAggregator(self.d)

    def compute_w_q(self, q: torch.Tensor) -> torch.Tensor:
        """Compute relation-query transformed edge embedding.

        Args.
            q. (d,) query embedding for single binary "interact" relation
        Returns.
            w_q. (n_rel, d) per-relation transformed embedding
        """
        # Per-relation. w_q[r] = W_r @ q + b_r
        # W_r is (n_rel, d, d), q is (d,). bmm-style. (n_rel, d, d) @ (d, 1) -> (n_rel, d)
        # Implement with einsum for clarity.
        w_q = torch.einsum("rij,j->ri", self.W_r, q) + self.b_r  # (n_rel, d)
        return w_q

    def forward(
        self,
        h: torch.Tensor,                       # (n_nodes, d) prev-layer node states
        h0: torch.Tensor,                      # (n_nodes, d) boundary state (reinjected)
        edge_src: torch.Tensor,                # (E,) source node index per edge
        edge_dst: torch.Tensor,                # (E,) target node index per edge
        edge_rel: torch.Tensor,                # (E,) relation type per edge
        q: torch.Tensor,                       # (d,) query embedding
        n_nodes: int,
    ) -> torch.Tensor:
        """Vectorized single-layer Bellman-Ford forward.

        Returns (n_nodes, d) updated node states.
        """
        # Compute per-relation edge embedding (relation-query transform)
        w_q = self.compute_w_q(q)              # (n_rel, d)

        # Per-edge transformed embedding. lookup w_q by edge_rel
        w_e = w_q[edge_rel]                    # (E, d)

        # MESSAGE. DistMult-style element-wise product.
        # h[edge_src] gives (E, d) source-node state per edge.
        msg = h[edge_src] * w_e                # (E, d)

        # AGGREGATE. PNA-style scatter aggregation with boundary reinjection.
        # Strategy. concatenate boundary self-messages (h0) with real messages,
        # and use scatter with target indices including self-loops.
        self_idx = torch.arange(n_nodes, device=h.device)
        msg_with_boundary = torch.cat([msg, h0], dim=0)                # (E + n_nodes, d)
        target_with_boundary = torch.cat([edge_dst, self_idx], dim=0)  # (E + n_nodes,)

        agg = self.aggregator(msg_with_boundary, target_with_boundary, n_nodes)  # (n_nodes, d)

        # ReLU nonlinearity (paper Section 4.2 specifies ReLU in hidden layers)
        return F.relu(agg)

    def forward_batched(
        self,
        h: torch.Tensor,                       # (S, n_nodes, d) prev-layer states
        h0: torch.Tensor,                      # (S, n_nodes, d) boundary states
        edge_src: torch.Tensor,                # (E,) shared across S
        edge_dst: torch.Tensor,                # (E,) shared across S
        edge_rel: torch.Tensor,                # (E,) shared across S
        q: torch.Tensor,                       # (d,) query embedding
        n_nodes: int,
    ) -> torch.Tensor:
        """Batched single-layer Bellman-Ford (ADDITIVE). Identical math to
        :meth:`forward` but vectorized over S sources sharing one edge set.

        Returns (S, n_nodes, d).
        """
        w_q = self.compute_w_q(q)              # (n_rel, d)
        w_e = w_q[edge_rel]                    # (E, d)

        # MESSAGE: DistMult element-wise product, broadcast over the S axis.
        msg = h[:, edge_src] * w_e             # (S, E, d) * (E, d) -> (S, E, d)

        # AGGREGATE with boundary reinjection (h0 appended as self-messages).
        self_idx = torch.arange(n_nodes, device=h.device)
        msg_with_boundary = torch.cat([msg, h0], dim=1)               # (S, E+n_nodes, d)
        target_with_boundary = torch.cat([edge_dst, self_idx], dim=0)  # (E+n_nodes,)

        agg = self.aggregator.forward_batched(
            msg_with_boundary, target_with_boundary, n_nodes
        )                                                              # (S, n_nodes, d)
        return F.relu(agg)


# ----------------------------------------------------------------------------
# Full NBFNet model for binary DDI
# ----------------------------------------------------------------------------


class NBFNetDDI(nn.Module):
    """Paper-faithful NBFNet for binary DDI prediction.

    Architecture (Zhu et al. NeurIPS 2021).
    - Single binary query "interact" with learned embedding w_interact in R^d.
    - L NBF layers (default 6), each with per-relation W_r^(t), b_r^(t).
    - Inverse-edge augmentation. each (h, r, t) -> (t, r + n_base_rel, h).
    - Bellman-Ford rooted at source drug, propagate L layers.
    - Symmetric DDI. forward (a -> b) + (b -> a), combine representations.
    - Score. sigmoid(MLP(concat([h_q_sym, q])))

    Endpoint stripping (weak version, codex round-1 accepted).
    - Drug nodes have NO learnable embedding.
    - drug_a is INDICATOR anchor (h_a^(0) = q).
    - drug_b's score-side representation = h_b^(L) from propagation.
    - Score MLP input does NOT contain raw drug feature vectors.

    Inductive guarantee (paper Section 4.1).
    - No node-id embeddings. only relation/query parameters and learnable layers.
    - Unseen drugs participate via their KG-typed edges (which are precomputed
      KG annotation, not learned from DDI labels).
    """

    def __init__(
        self,
        n_nodes: int,
        n_base_rel: int,                       # number of base relation types (forward only)
        d: int = 32,                           # paper default hidden dim
        n_layers: int = 6,                     # paper default L
        mlp_hidden: int = 64,                  # paper default 2-layer MLP head with hidden=64
        ddi_rel_id: Optional[int] = None,      # relation index for "interact" (binary DDI) edge
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_base_rel = int(n_base_rel)
        # Add inverse relations. n_total_rel = 2 * n_base_rel (forward + inverse).
        # Self-loop relation handled via boundary reinjection, not a separate relation.
        self.n_rel = 2 * self.n_base_rel
        self.d = int(d)
        self.L = int(n_layers)
        self.ddi_rel_id = ddi_rel_id           # for query-edge masking

        # Single binary query embedding w_interact (learned)
        self.query = nn.Parameter(torch.empty(self.d))
        nn.init.normal_(self.query, mean=0.0, std=1.0 / math.sqrt(self.d))

        # NBF layers (L layers, each with per-relation transform AND its own PNA aggregator).
        # Codex round-4 fix. shared aggregator across layers ties learnable proj params
        # which is an architectural tie not in NBFNet paper. Each layer gets a fresh
        # PNAAggregator (default-constructed inside NBFLayer when aggregator=None).
        self.layers = nn.ModuleList(
            [NBFLayer(self.d, self.n_rel, aggregator=None) for _ in range(self.L)]
        )

        # MLP score head. input = concat([h_q_sym, q]) = 2*d -> mlp_hidden -> 1
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * self.d, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    # ------------------------------------------------------------------
    # Edge augmentation utilities
    # ------------------------------------------------------------------

    def augment_inverse_edges(
        self,
        edge_src: torch.Tensor,    # (E,)
        edge_dst: torch.Tensor,    # (E,)
        edge_rel: torch.Tensor,    # (E,) base relation IDs in [0, n_base_rel)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Augment KG with inverse-relation edges (paper KG convention).

        Each (h, r, t) -> appends (t, r + n_base_rel, h).
        Returns expanded (edge_src_aug, edge_dst_aug, edge_rel_aug).
        """
        # Forward edges
        fwd_src, fwd_dst, fwd_rel = edge_src, edge_dst, edge_rel
        # Inverse edges (swap src/dst, shift relation by n_base_rel)
        inv_src = edge_dst
        inv_dst = edge_src
        inv_rel = edge_rel + self.n_base_rel

        aug_src = torch.cat([fwd_src, inv_src], dim=0)
        aug_dst = torch.cat([fwd_dst, inv_dst], dim=0)
        aug_rel = torch.cat([fwd_rel, inv_rel], dim=0)
        return aug_src, aug_dst, aug_rel

    def build_query_edge_mask(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        drug_a: int,
        drug_b: int,
    ) -> torch.Tensor:
        """Build boolean mask to EXCLUDE direct queried pair edge during training.

        Returns (E,) bool mask. True = keep edge, False = remove.

        Relation-aware mask. removes only edges of relation = ddi_rel_id (binary DDI)
        between drug_a and drug_b in either direction (and inverse-shifted).
        Other relation edges between (a, b) are kept (e.g., shared target, shared CYP).
        """
        if self.ddi_rel_id is None:
            return torch.ones(edge_src.size(0), dtype=torch.bool, device=edge_src.device)

        ddi_fwd_id = self.ddi_rel_id
        ddi_inv_id = self.ddi_rel_id + self.n_base_rel

        # Build mask. (src=a, dst=b, rel=ddi_fwd) OR (src=b, dst=a, rel=ddi_fwd)
        # OR (src=a, dst=b, rel=ddi_inv) OR (src=b, dst=a, rel=ddi_inv)
        is_ddi_rel = (edge_rel == ddi_fwd_id) | (edge_rel == ddi_inv_id)
        is_pair_ab = (edge_src == drug_a) & (edge_dst == drug_b)
        is_pair_ba = (edge_src == drug_b) & (edge_dst == drug_a)

        # Remove ddi-relation edges between (a, b) in either direction
        should_remove = is_ddi_rel & (is_pair_ab | is_pair_ba)
        return ~should_remove

    def build_union_query_edge_mask(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        source: int,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized union query-edge mask for one source vs many targets.

        Additive helper (codex Round 6 Fix 2). Same semantics as
        :meth:`build_query_edge_mask` but vectorized over a set of targets:
        returns the intersection of the per-target keep masks, i.e. removes
        every DDI-relation edge connecting ``source`` to ANY target in
        ``targets`` (either direction, forward + inverse-shifted relation).

        For a single target ``t`` this is identical to
        ``build_query_edge_mask(..., source, t)``; for a set ``{t_i}`` it
        equals ``& over t_i of build_query_edge_mask(..., source, t_i)``
        (de Morgan: keep = NOT(union of per-target removals)).

        Args.
            edge_src/dst/rel. (E,) augmented-KG edge tensors.
            source. int, the BF root drug for this batch group.
            targets. (T,) tensor (or list) of target drug node indices.

        Returns.
            keep_mask. (E,) bool. True = keep edge, False = remove.
        """
        if self.ddi_rel_id is None:
            return torch.ones(edge_src.size(0), dtype=torch.bool, device=edge_src.device)

        targets_t = torch.as_tensor(targets, device=edge_src.device, dtype=edge_src.dtype)

        ddi_fwd_id = self.ddi_rel_id
        ddi_inv_id = self.ddi_rel_id + self.n_base_rel
        is_ddi_rel = (edge_rel == ddi_fwd_id) | (edge_rel == ddi_inv_id)

        src_is_source = edge_src == source
        dst_is_source = edge_dst == source
        dst_in_targets = torch.isin(edge_dst, targets_t)
        src_in_targets = torch.isin(edge_src, targets_t)

        # (source -> target_i) or (target_i -> source)
        pair_fwd = src_is_source & dst_in_targets
        pair_rev = dst_is_source & src_in_targets

        should_remove = is_ddi_rel & (pair_fwd | pair_rev)
        return ~should_remove

    # ------------------------------------------------------------------
    # Bellman-Ford propagation
    # ------------------------------------------------------------------

    def bellman_ford(
        self,
        source: int,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Run L-layer Bellman-Ford rooted at source node.

        Args.
            source. int, source node index (drug_a)
            edge_src/dst/rel. (E,) edges of the augmented KG
                              (with inverse edges, query-pair masked if training)

        Returns.
            h. (n_nodes, d) final node representations h_v^(L)
        """
        # Boundary INDICATOR. h_v^(0) = q if v=source else 0
        h0 = torch.zeros(self.n_nodes, self.d, device=self.query.device)
        h0[source] = self.query

        h = h0

        # L iterations of Bellman-Ford
        for layer in self.layers:
            h = layer(h, h0, edge_src, edge_dst, edge_rel, self.query, self.n_nodes)

        return h

    # ------------------------------------------------------------------
    # Pair forward (symmetric DDI)
    # ------------------------------------------------------------------

    def score_pair(
        self,
        drug_a: int,
        drug_b: int,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Score a single DDI pair (drug_a, drug_b) with symmetric BF.

        Args.
            edge_src/dst/rel. base KG edges (will be augmented with inverse,
                              and query-edge masked if training)

        Returns.
            logit. scalar tensor. raw pre-sigmoid logit. Apply
                   torch.sigmoid externally for probability, or use
                   torch.nn.functional.binary_cross_entropy_with_logits
                   for loss (preferred for numerical stability).
        """
        # Augment with inverse edges
        aug_src, aug_dst, aug_rel = self.augment_inverse_edges(edge_src, edge_dst, edge_rel)

        # Query-edge masking during training (remove only direct queried pair edge)
        if training:
            keep_mask = self.build_query_edge_mask(aug_src, aug_dst, aug_rel, drug_a, drug_b)
            aug_src = aug_src[keep_mask]
            aug_dst = aug_dst[keep_mask]
            aug_rel = aug_rel[keep_mask]

        # Forward direction. BF rooted at drug_a, readout at drug_b
        h_from_a = self.bellman_ford(drug_a, aug_src, aug_dst, aug_rel)
        h_q_ab = h_from_a[drug_b]                  # (d,)

        # Reverse direction. BF rooted at drug_b, readout at drug_a
        h_from_b = self.bellman_ford(drug_b, aug_src, aug_dst, aug_rel)
        h_q_ba = h_from_b[drug_a]                  # (d,)

        # Symmetric. add representations (paper Section 4.1 undirected convention)
        h_q_sym = h_q_ab + h_q_ba                  # (d,)

        # Concat query embedding (official-repo convention)
        z = torch.cat([h_q_sym, self.query], dim=-1)  # (2d,)

        # MLP score (sigmoid applied externally in loss computation for numerical stability)
        logit = self.mlp_head(z).squeeze(-1)       # ()
        return logit

    # ------------------------------------------------------------------
    # Per-source encoder (for trainer-controlled batch processing)
    # ------------------------------------------------------------------

    def encode_from_source(
        self,
        source: int,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        already_augmented: bool = False,
        edge_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One Bellman-Ford pass from source, return per-node representations.

        Codex round-4 fix. renamed from score_from_source to encode_from_source.
        This method ONLY runs BF and returns h_q(source, *) for all nodes.
        The trainer is responsible for. (1) query-edge masking, (2) reverse-pass
        BF for symmetric DDI, (3) symmetrization, (4) MLP score head.

        Args.
            source. int, source node index
            edge_src/dst/rel. raw or augmented KG edge tensors
            already_augmented. if True, skip inverse-edge augmentation (caller did it)
            edge_keep_mask. (E,) bool mask to filter edges (for query-edge masking).
                            if None, keep all edges.

        Returns.
            h_from_source. (n_nodes, d) per-node representation after L BF iterations.
                           h_from_source[v] is h_q(source, v) per paper notation.

        Usage in trainer (per-source amortization with symmetric DDI).
            aug_src, aug_dst, aug_rel = model.augment_inverse_edges(es, ed, er)
            mask = model.build_query_edge_mask(aug_src, aug_dst, aug_rel, src, tgt)
            h_src = model.encode_from_source(src, aug_src, aug_dst, aug_rel,
                                              already_augmented=True, edge_keep_mask=mask)
            h_tgt = model.encode_from_source(tgt, aug_src, aug_dst, aug_rel,
                                              already_augmented=True, edge_keep_mask=mask)
            h_q_sym = h_src[tgt] + h_tgt[src]
            logit = model.mlp_head(torch.cat([h_q_sym, model.query], dim=-1)).squeeze(-1)
        """
        if not already_augmented:
            edge_src, edge_dst, edge_rel = self.augment_inverse_edges(
                edge_src, edge_dst, edge_rel
            )

        if edge_keep_mask is not None:
            edge_src = edge_src[edge_keep_mask]
            edge_dst = edge_dst[edge_keep_mask]
            edge_rel = edge_rel[edge_keep_mask]

        return self.bellman_ford(source, edge_src, edge_dst, edge_rel)

    # ------------------------------------------------------------------
    # Batched multi-source encoder (ADDITIVE — parallel acceleration)
    # ------------------------------------------------------------------
    # The methods below run the SAME generalized Bellman-Ford as the
    # single-source path, but vectorized over a batch of S sources at once
    # (one batched propagation instead of S sequential Python-loop passes).
    # They are mathematically equivalent to running bellman_ford() per source
    # and stacking, when the SAME (shared) edge set is used for all sources
    # (i.e. no per-source query-edge masking). The single-source functions
    # above are untouched (Round-7 GO preserved); these only add a fast path.

    def bellman_ford_batched(
        self,
        sources: torch.Tensor,                 # (S,) source node indices
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized L-layer Bellman-Ford rooted at S sources simultaneously.

        Returns (S, n_nodes, d) where row i is h_q(sources[i], *) — identical
        (within float reduction order) to stacking bellman_ford(sources[i], ...).
        """
        S = int(sources.size(0))
        # Boundary INDICATOR per source: h0[i, sources[i]] = q, else 0.
        h0 = torch.zeros(S, self.n_nodes, self.d, device=self.query.device)
        h0[torch.arange(S, device=sources.device), sources] = self.query
        h = h0
        for layer in self.layers:
            h = layer.forward_batched(
                h, h0, edge_src, edge_dst, edge_rel, self.query, self.n_nodes
            )
        return h

    def encode_from_sources(
        self,
        sources: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        already_augmented: bool = False,
        edge_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched counterpart of :meth:`encode_from_source`.

        Args.
            sources. (S,) source node indices (long tensor or list).
            edge_keep_mask. OPTIONAL single SHARED (E,) mask applied to ALL
                sources. Per-source masking is NOT supported here by design —
                the trainer only uses this fast path when query-edge masking
                is a no-op for the whole batch (see _query_edges_present), so a
                shared mask suffices.

        Returns.
            (S, n_nodes, d) per-source per-node representations.
        """
        if not already_augmented:
            edge_src, edge_dst, edge_rel = self.augment_inverse_edges(
                edge_src, edge_dst, edge_rel
            )
        if edge_keep_mask is not None:
            edge_src = edge_src[edge_keep_mask]
            edge_dst = edge_dst[edge_keep_mask]
            edge_rel = edge_rel[edge_keep_mask]
        sources = torch.as_tensor(sources, device=self.query.device, dtype=torch.long)
        return self.bellman_ford_batched(sources, edge_src, edge_dst, edge_rel)
