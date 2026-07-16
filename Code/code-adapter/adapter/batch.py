"""Stage 4 / DDI-LoRA adapter - the per-mediator flattened batch (I/O contract).

A batch of B drug pairs, each with a variable-size shared-mediator set M_uv, is
represented PyG-style: all (pair, mediator) rows are concatenated into one long
axis of length M = sum_b |M_uv(b)|, with a `pair_idx` telling which pair each row
belongs to. A pair with M_uv=empty contributes zero rows (its z_uv is all-zeros).

All tensors are on the same device. r_a / r_b are the arm relation features
r_tilde_a(m) / r_tilde_b(m) (paper eq. for relation_uvm); during step-1/2 they may
be STUBBED (zeros) until the real path-relation aggregation (step 4) is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AdapterBatch:
    n_pairs: int                 # B
    pair_idx: torch.Tensor       # [M] long, in 0..B-1 (which pair each mediator is in)
    med_type: torch.Tensor       # [M] long, in 0..T-1 (mediator type t(m))
    z_m: torch.Tensor            # [M, d_z] frozen mediator semantics
    r_a: torch.Tensor            # [M, d_z] arm-a relation feature r_tilde_a(m) (u->m)
    r_b: torch.Tensor            # [M, d_z] arm-b relation feature r_tilde_b(m) (v->m)
    h_base: torch.Tensor         # [M, d]   frozen base rep h^base_m of each mediator
    aa_weight: Optional[torch.Tensor] = None  # [M] additive prior on the pooling logit
    #                                          (inverse-log-degree Adamic-Adar warm-start,
    #                                           step 5; None -> learned attention only)
    # Path Design 1 (factorized) route tokens: top-2 sparse mixture over route-signature
    # tokens per arm. None unless arm_mode="pd1".
    tok_a_idx: Optional[torch.Tensor] = None  # [M, 2] long   arm-a (u->m) top-2 token ids
    tok_a_wt: Optional[torch.Tensor] = None   # [M, 2] float  arm-a token weights
    tok_b_idx: Optional[torch.Tensor] = None  # [M, 2] long   arm-b (v->m) top-2 token ids
    tok_b_wt: Optional[torch.Tensor] = None   # [M, 2] float  arm-b token weights
    med_idx: Optional[torch.Tensor] = None    # [M] long  KG node idx per mediator row (analysis)

    @property
    def n_mediators(self) -> int:
        return int(self.pair_idx.shape[0])

    @property
    def device(self) -> torch.device:
        return self.pair_idx.device

    def validate(self) -> None:
        m = self.n_mediators
        for name, t, dim in (("med_type", self.med_type, 1), ("z_m", self.z_m, 2),
                             ("r_a", self.r_a, 2), ("r_b", self.r_b, 2),
                             ("h_base", self.h_base, 2)):
            if t.shape[0] != m:
                raise ValueError(f"{name}: axis-0 {t.shape[0]} != n_mediators {m}")
            if t.dim() != dim:
                raise ValueError(f"{name}: expected {dim}-D, got {t.dim()}-D")
        if self.z_m.shape[1] != self.r_a.shape[1] or self.r_a.shape[1] != self.r_b.shape[1]:
            raise ValueError("z_m / r_a / r_b must share d_z")
        # grouping keys index into params / scatter groups: must be 1-D long, nonnegative
        for name, t in (("pair_idx", self.pair_idx), ("med_type", self.med_type)):
            if t.dim() != 1:
                raise ValueError(f"{name}: expected 1-D, got {t.dim()}-D")
            if t.dtype != torch.long:
                raise ValueError(f"{name}: expected torch.long, got {t.dtype}")
            if m and int(t.min()) < 0:
                raise ValueError(f"{name}: negative index")
        if m and int(self.pair_idx.max()) >= self.n_pairs:
            raise ValueError("pair_idx out of range [0, n_pairs)")
        if self.aa_weight is not None:
            if self.aa_weight.shape != (m,):
                raise ValueError(f"aa_weight: expected [{m}], got {tuple(self.aa_weight.shape)}")
            if not torch.is_floating_point(self.aa_weight):
                raise ValueError(f"aa_weight: expected floating dtype, got {self.aa_weight.dtype}")
            if self.aa_weight.device != self.pair_idx.device:
                raise ValueError("aa_weight: device must match the batch")


__all__ = ["AdapterBatch"]
