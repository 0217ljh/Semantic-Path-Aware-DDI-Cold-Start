"""Flattened per-batch support tensors shared by both adapter heads.

A ``SupportBatch`` carries the variable-length mediator lists of a batch of
drug pairs in flattened (CSR-like) form, exactly the layout that
``run_spmn_v1_phase2c._gather`` already produces, plus the per-mediator
``d_a`` / ``d_b`` distances (needed for asymmetric stratification) and optional
K3 cross-level edges (reserved; unused until the K3 step).

All index tensors are 1-D ``long`` over the concatenated mediators of the batch
(length ``M = sum of per-pair support sizes``). ``pair_idx`` maps each mediator
to its local pair index in ``[0, n_pairs)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SupportBatch:
    med_id: torch.Tensor        # (M,) long — mediator global entity id
    pair_idx: torch.Tensor      # (M,) long — local pair index in [0, n_pairs)
    type_idx: torch.Tensor      # (M,) long — canonical type id in [0, N_TYPES)
    rel_a: torch.Tensor         # (M,) long — rel(a -> mediator) bucket
    rel_b: torch.Tensor         # (M,) long — rel(mediator -> b) bucket
    d_a: torch.Tensor           # (M,) long — d_sym(a, mediator)
    d_b: torch.Tensor           # (M,) long — d_sym(b, mediator)
    struct_feats: torch.Tensor  # (n_pairs, struct_dim) float
    n_pairs: int
    # Reserved for the K3 step (protein -> higher-level membership edges, in
    # batch-global mediator-index space). Unused while use_k3 is False.
    k3_src: torch.Tensor | None = None
    k3_dst: torch.Tensor | None = None
    # Per-pair gate-conditioning regime features (n_pairs, n_q) for the Step-3
    # pair-conditioned hub gate. Optional/reserved like k3_*; stays None unless the
    # aware runner sets it, so existing call sites are unaffected.
    pair_q: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.struct_feats.device
