"""NBFNet v3.0 — Joint binary + multi-class DDI model.

See Notes/Log/nbfnet_v3_0_design.md for full design spec. Quick summary.

Forward (per pair (a, b), per mechanism m).
    boundary:    h^(m,0)_v = INDICATOR(a, v, q_m) = 1[v=a] * q_m
    message:     m^(m,t)_(x,r,v) = h^(m,t-1)_x * (W_r^(t) @ q_m + b_r^(t))
    aggregate:   h^(m,t)_v = PNA({m^(m,t)_(x,r,v) for (x,r,v) in E(v)} U {h^(m,0)_v})

After L iters from a, readout h_q^(m)(a, b) = h^(m,L)_b.
Then symmetric: another L iters from b, readout h_q^(m)(b, a) = h^(m,L)_a.
Symmetric: h_q^(m)_sym = h_q^(m)(a, b) + h_q^(m)(b, a)

Stack H = [h_q^(m)_sym]_{m=1..N_m} in R^(N_m, d).

Dual head.
    multi-cls: s_m = MLP_multi(concat([H[m], q_m]))  -> softmax over m
    binary:    z = LogSumExp_m(H)                    -> MLP_binary(concat([z, q_bar])) -> sigmoid

Implementation notes.
- Reuses v1.7 NBFLayer + PNAAggregator UNCHANGED (per CLAUDE.md no-modification rule).
- Multi-query implementation: initial version uses Python loop over mechanisms
  (clean, slow ~N_m x v1.7 cost). Vectorization is left for v3.0.1+.
- Drug nodes have NO learnable embeddings (inductive, cold-start safe).
- Per-source amortization. trainer-controlled via encode_all_mechanisms_from_source.

Status. Initial scaffold (2026-06-05). Pending codex review (Task #5).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_code.models.nbfnet_v1_7.nbfnet_model import NBFLayer, PNAAggregator


# ----------------------------------------------------------------------------
# Multi-query NBFNet joint DDI model
# ----------------------------------------------------------------------------


class NBFNetJointDDI(nn.Module):
    """NBFNet with multiple mechanism queries + dual readout (binary + multi-cls).

    Architecture (see Notes/Log/nbfnet_v3_0_design.md):
    - N_m learnable mechanism queries q_m (one per DDI mechanism type)
    - Shared NBFLayer params across queries (per-(layer, relation) W_r, b_r, PNA)
    - For each pair (a, b), run BF from both directions, for each query q_m
    - Symmetric: h_q^(m)_sym = h_q^(m)(a,b) + h_q^(m)(b,a)  in R^d
    - Stack H = [h_q^(m)_sym]_m  in R^(N_m, d)
    - Multi-cls: per-mechanism MLP score -> softmax
    - Binary: soft-OR aggregator (default LogSumExp) -> MLP -> sigmoid

    Endpoint stripping (inherited from v1.7).
    - Drug nodes have NO learnable embedding
    - drug_a is INDICATOR anchor (h^(m,0)_a = q_m)
    - Score-side rep comes from message-passing evidence, not raw drug features

    Inductive guarantee (inherited from v1.7).
    - No node-id embeddings
    - Unseen drugs participate via their KG-typed edges
    """

    SUPPORTED_BINARY_AGGREGATORS = ("logsumexp", "max", "mean", "attention")

    def __init__(
        self,
        n_nodes: int,
        n_base_rel: int,                      # base relations (forward only, BEFORE inverse augmentation)
        n_mechanisms: int,                    # K, frozen at 80 for v3.0 (see Notes/Log/nbfnet_v3_0_design.md §1.4)
        d: int = 32,                          # paper default hidden dim
        n_layers: int = 6,                    # paper default L
        mlp_hidden: int = 64,                 # paper default head hidden
        ddi_rel_id: Optional[int] = None,     # for query-edge masking (any DDI-bundle relation id)
        binary_aggregator: str = "logsumexp",
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_base_rel = int(n_base_rel)
        self.n_rel = 2 * self.n_base_rel  # forward + inverse
        self.n_mech = int(n_mechanisms)
        self.d = int(d)
        self.L = int(n_layers)
        self.ddi_rel_id = ddi_rel_id

        if binary_aggregator not in self.SUPPORTED_BINARY_AGGREGATORS:
            raise ValueError(
                f"binary_aggregator must be one of {self.SUPPORTED_BINARY_AGGREGATORS}, "
                f"got {binary_aggregator!r}"
            )
        self.binary_aggregator = binary_aggregator

        # Per-mechanism learnable query embeddings q_m
        # Shape. (N_m, d). Each row q_m parameterizes one mechanism's propagation
        self.mechanism_queries = nn.Parameter(torch.empty(self.n_mech, self.d))
        nn.init.normal_(self.mechanism_queries, mean=0.0, std=1.0 / math.sqrt(self.d))

        # Shared NBFNet layers (reused from v1.7, per CLAUDE.md no-modification)
        # Each NBFLayer has its own per-(layer, relation) W_r^(t), b_r^(t)
        # and its own PNAAggregator (codex Round 4 fix from v1.7)
        self.layers = nn.ModuleList(
            [NBFLayer(self.d, self.n_rel, aggregator=None) for _ in range(self.L)]
        )

        # Dual heads
        # Multi-cls head: shared across mechanisms, conditions on query embedding
        #   input = concat([h_q^(m)_sym, q_m])  shape (2d,) per mechanism
        #   output = scalar per mechanism (logit before softmax)
        self.mlp_multi = nn.Sequential(
            nn.Linear(2 * self.d, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

        # Binary head:
        #   input = concat([soft-OR(H), q_bar])  shape (2d,)
        #   output = scalar binary logit
        self.mlp_binary = nn.Sequential(
            nn.Linear(2 * self.d, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

        # Optional attention weights for binary aggregator (only if binary_aggregator='attention')
        if self.binary_aggregator == "attention":
            self.attn_query = nn.Parameter(torch.empty(self.d))
            nn.init.normal_(self.attn_query, mean=0.0, std=1.0 / math.sqrt(self.d))

    # ------------------------------------------------------------------
    # Edge augmentation utilities (delegated to NBFLayer's parent contract)
    # ------------------------------------------------------------------

    def augment_inverse_edges(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as v1.7. (h, r, t) -> + (t, r + n_base_rel, h). Returns augmented edges."""
        inv_src = edge_dst
        inv_dst = edge_src
        inv_rel = edge_rel + self.n_base_rel
        aug_src = torch.cat([edge_src, inv_src], dim=0)
        aug_dst = torch.cat([edge_dst, inv_dst], dim=0)
        aug_rel = torch.cat([edge_rel, inv_rel], dim=0)
        return aug_src, aug_dst, aug_rel

    def build_query_edge_mask(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        drug_a: int,
        drug_b: int,
    ) -> torch.Tensor:
        """Build boolean mask to EXCLUDE direct queried DDI edge during training.

        Note. masks ALL DDI-relation edges between (a, b) (mechanism-agnostic on the
        DDI bundle), per design doc open question #2. If finer-grain mechanism-specific
        masking is needed later, build a separate masking helper.
        """
        if self.ddi_rel_id is None:
            return torch.ones(edge_src.size(0), dtype=torch.bool, device=edge_src.device)

        ddi_fwd_id = self.ddi_rel_id
        ddi_inv_id = self.ddi_rel_id + self.n_base_rel

        is_ddi_rel = (edge_rel == ddi_fwd_id) | (edge_rel == ddi_inv_id)
        is_pair_ab = (edge_src == drug_a) & (edge_dst == drug_b)
        is_pair_ba = (edge_src == drug_b) & (edge_dst == drug_a)

        should_remove = is_ddi_rel & (is_pair_ab | is_pair_ba)
        return ~should_remove

    # ------------------------------------------------------------------
    # Multi-query Bellman-Ford propagation
    # ------------------------------------------------------------------

    def bellman_ford_for_mechanism(
        self,
        source: int,
        mechanism_idx: int,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Run L-layer Bellman-Ford for a single (source, mechanism).

        Args.
            source. int, root node (drug_a or drug_b)
            mechanism_idx. int in [0, N_m), which query embedding to use
            edge_src/dst/rel. (E,) augmented + (optionally) query-edge-masked edges

        Returns.
            h. (n_nodes, d) per-node final representation h^(m, L)_v
        """
        q_m = self.mechanism_queries[mechanism_idx]      # (d,)

        # Boundary INDICATOR. h^(m,0)_v = q_m if v=source else 0
        h0 = torch.zeros(self.n_nodes, self.d, device=q_m.device)
        h0[source] = q_m

        h = h0
        # L iterations of Bellman-Ford with shared NBFLayer params, query=q_m
        for layer in self.layers:
            h = layer(h, h0, edge_src, edge_dst, edge_rel, q_m, self.n_nodes)
        return h

    def encode_all_mechanisms_from_source(
        self,
        source: int,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        already_augmented: bool = False,
        edge_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run BF for source under EACH mechanism query, stack results.

        Args.
            source. int, root node
            edge_src/dst/rel. base or augmented edge tensors
            already_augmented. if False, augment_inverse_edges first
            edge_keep_mask. (E,) bool mask (None = keep all)

        Returns.
            h_per_mech. (N_m, n_nodes, d) per-(mechanism, node) representation
                        h_per_mech[m, v] = h_q^(m)(source, v)

        Cost.
            N_m BF passes. Initial Python-loop impl is clean but slow (~v1.7 x N_m).
            v3.0.1+ should vectorize via stacked queries + batched scatter.

        Usage in trainer (per-source amortization for symmetric DDI).
            aug_src, aug_dst, aug_rel = model.augment_inverse_edges(es, ed, er)
            mask = model.build_query_edge_mask(aug_src, aug_dst, aug_rel, src, tgt)
            h_src_per_mech = model.encode_all_mechanisms_from_source(
                src, aug_src, aug_dst, aug_rel, already_augmented=True,
                edge_keep_mask=mask)
            h_tgt_per_mech = model.encode_all_mechanisms_from_source(
                tgt, aug_src, aug_dst, aug_rel, already_augmented=True,
                edge_keep_mask=mask)
            # h_q^(m)_sym at (src, tgt) = h_src_per_mech[:, tgt] + h_tgt_per_mech[:, src]
            #                       shape (N_m, d)
        """
        if not already_augmented:
            edge_src, edge_dst, edge_rel = self.augment_inverse_edges(
                edge_src, edge_dst, edge_rel
            )

        if edge_keep_mask is not None:
            edge_src = edge_src[edge_keep_mask]
            edge_dst = edge_dst[edge_keep_mask]
            edge_rel = edge_rel[edge_keep_mask]

        # Loop over mechanisms (v3.0 initial impl; vectorize later)
        outs = []
        for m in range(self.n_mech):
            h_m = self.bellman_ford_for_mechanism(
                source, m, edge_src, edge_dst, edge_rel
            )                                            # (n_nodes, d)
            outs.append(h_m)
        return torch.stack(outs, dim=0)                  # (N_m, n_nodes, d)

    # ------------------------------------------------------------------
    # Dual-head readout
    # ------------------------------------------------------------------

    def _binary_soft_or(self, H: torch.Tensor) -> torch.Tensor:
        """Aggregate per-mechanism representations H (N_m, d) -> (d,) via soft OR.

        Implements the binary "evidence aggregation" step from design doc 1.2.
        """
        if self.binary_aggregator == "logsumexp":
            return torch.logsumexp(H, dim=0)                       # (d,)
        if self.binary_aggregator == "max":
            return H.max(dim=0).values                              # (d,)
        if self.binary_aggregator == "mean":
            return H.mean(dim=0)                                    # (d,)
        if self.binary_aggregator == "attention":
            # alpha_m = softmax(<H_m, attn_query>) ; output = sum_m alpha_m * H_m
            scores = H @ self.attn_query                            # (N_m,)
            alpha = F.softmax(scores, dim=0).unsqueeze(-1)          # (N_m, 1)
            return (alpha * H).sum(dim=0)                           # (d,)
        raise ValueError(f"unknown binary_aggregator {self.binary_aggregator!r}")

    def dual_head_readout(self, H_sym: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dual head to symmetric per-mechanism stack H_sym (N_m, d).

        Returns.
            multi_logits. (N_m,) raw per-mechanism logits (apply softmax + CE externally)
            binary_logit. scalar raw binary logit (apply sigmoid + BCE externally)
        """
        # Multi-cls. per-mechanism input = concat([H_sym[m], q_m])  shape (N_m, 2d)
        Q = self.mechanism_queries                                  # (N_m, d)
        multi_in = torch.cat([H_sym, Q], dim=-1)                    # (N_m, 2d)
        multi_logits = self.mlp_multi(multi_in).squeeze(-1)         # (N_m,)

        # Binary. soft-OR aggregator over mechanism axis, then concat global q_bar
        z_bin = self._binary_soft_or(H_sym)                         # (d,)
        q_bar = self.mechanism_queries.mean(dim=0)                  # (d,)
        binary_in = torch.cat([z_bin, q_bar], dim=-1)               # (2d,)
        binary_logit = self.mlp_binary(binary_in).squeeze(-1)       # ()

        return multi_logits, binary_logit

    # ------------------------------------------------------------------
    # Single-pair forward (symmetric, dual-head)
    # ------------------------------------------------------------------

    def score_pair_dual(
        self,
        drug_a: int,
        drug_b: int,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a single DDI pair with dual-head output.

        Args.
            drug_a, drug_b. int node indices
            edge_src/dst/rel. base KG edges (will be augmented + masked internally)
            training. if True, apply query-edge masking (remove direct DDI edges
                      between a and b)

        Returns.
            multi_logits. (N_m,) per-mechanism logits (apply softmax + CE externally)
            binary_logit. scalar binary logit (apply sigmoid + BCE externally)

        Cost.
            2 x N_m BF passes per pair. For per-batch efficiency, trainer should
            use per-source amortization via encode_all_mechanisms_from_source.
        """
        # Augment + (optionally) mask
        aug_src, aug_dst, aug_rel = self.augment_inverse_edges(edge_src, edge_dst, edge_rel)
        if training:
            keep_mask = self.build_query_edge_mask(aug_src, aug_dst, aug_rel, drug_a, drug_b)
            aug_src = aug_src[keep_mask]
            aug_dst = aug_dst[keep_mask]
            aug_rel = aug_rel[keep_mask]

        # Forward BF from drug_a, all mechanisms
        h_from_a = self.encode_all_mechanisms_from_source(
            drug_a, aug_src, aug_dst, aug_rel, already_augmented=True
        )                                                          # (N_m, n_nodes, d)
        h_q_ab = h_from_a[:, drug_b, :]                            # (N_m, d)

        # Reverse BF from drug_b, all mechanisms
        h_from_b = self.encode_all_mechanisms_from_source(
            drug_b, aug_src, aug_dst, aug_rel, already_augmented=True
        )                                                          # (N_m, n_nodes, d)
        h_q_ba = h_from_b[:, drug_a, :]                            # (N_m, d)

        # Representation-level symmetrize per mechanism (paper convention, also v1.7)
        H_sym = h_q_ab + h_q_ba                                    # (N_m, d)

        # Dual readout
        return self.dual_head_readout(H_sym)

    # ------------------------------------------------------------------
    # Joint loss helper (convenience; trainer can also implement inline)
    # ------------------------------------------------------------------

    @staticmethod
    def joint_loss(
        multi_logits: torch.Tensor,        # (B, N_m) batched
        binary_logits: torch.Tensor,       # (B,) batched
        binary_labels: torch.Tensor,       # (B,) 0/1
        multi_labels: torch.Tensor,        # (B,) mechanism index, -1 for negatives
        lambda_binary: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute joint loss = CE(multi, on pos only) + lambda * BCE(binary, on all).

        Args.
            multi_logits. (B, N_m) per-pair per-mechanism logits
            binary_logits. (B,) per-pair binary logit
            binary_labels. (B,) 0/1 float (or long)
            multi_labels. (B,) long mechanism index; use -1 to mark "no mechanism"
                          (i.e. negative pairs); these contribute only to binary loss
            lambda_binary. weight on binary loss term (default 1.0)

        Returns.
            (loss, components_dict). loss is scalar; components_dict has
            'multi_loss', 'binary_loss' (both scalars) for logging.

        Math.
            L = mean_{i where multi_labels[i] >= 0} CE(multi_logits[i], multi_labels[i])
              + lambda * mean_{all i} BCE(sigmoid(binary_logits[i]), binary_labels[i])

            If batch has no positives, multi loss = 0 (return scalar 0 tensor).
        """
        # Binary loss (always over all pairs)
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logits, binary_labels.float()
        )

        # Multi-cls loss (only over positives, i.e. multi_labels >= 0)
        pos_mask = multi_labels >= 0
        if pos_mask.any():
            multi_loss = F.cross_entropy(
                multi_logits[pos_mask], multi_labels[pos_mask]
            )
        else:
            # No positives in this batch; multi loss not defined. Return zero
            # tensor that still preserves gradient graph to mlp_multi (zeroed).
            multi_loss = torch.zeros((), device=binary_logits.device)

        loss = multi_loss + lambda_binary * binary_loss
        components = {
            "multi_loss": multi_loss.detach(),
            "binary_loss": binary_loss.detach(),
            "joint_loss": loss.detach(),
            "lambda_binary": float(lambda_binary),
        }
        return loss, components
