"""E-frag — shared/residual molecular<->KG alignment on top of the KG backbone (codex 019e6747).

Fixes the molecular-no-lift problem WITHOUT dropping the required molecular<->KG alignment:
  m_shared = proj_shared(mol_src)  -- aligned to a RELATION-TYPED KG target (InfoNCE)
  m_resid  = proj_resid(mol_src)   -- KG-ORTHOGONAL residual (orthogonality + decorrelation)
The molecular pair logit uses BOTH; combined = emergnn + beta * mol_logit (fused on top of KG).
Aux losses (joint-trained with DDI): lambda_align*InfoNCE(shared, k) + lambda_orth*cos^2(resid,
sg k) + lambda_dec*decorr(resid). mol_src = BRICS fragments (default) or any per-drug vector.

This is the clean single-lever test: does richer molecular, residual-aligned to typed-KG, lift
test_s2 AUROC over the 0.772 KG backbone? Success >0.775 (real), >0.778 (strong); noise ~+-0.3pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v3_multimodal.v2_trainer import _PerModeEmerGNN_V2

FRAG_NPZ = PROJECT_ROOT / "Code/data/_cache/molecular_fragments_brics.npz"
KTYPED_NPZ = PROJECT_ROOT / "Code/data/_cache/kg_typed_target_pubmedbert.npz"


def _offdiag_decorr(z: torch.Tensor) -> torch.Tensor:
    """Barlow-style off-diagonal feature-covariance penalty on batch-normalized z."""
    if z.size(0) < 2:
        return z.new_zeros(())
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    cov = (zc.t() @ zc) / (z.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / z.size(1)


class ResHead(nn.Module):
    """Shared/residual molecular head over a per-drug molecular source vector."""

    def __init__(self, *, src_dim: int, ktyped_dim: int, d: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj_shared = nn.Sequential(nn.Linear(src_dim, 256), nn.LayerNorm(256), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(256, d))
        self.proj_resid = nn.Sequential(nn.Linear(src_dim, 256), nn.LayerNorm(256), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(256, d))
        self.proj_k = nn.Linear(ktyped_dim, d)  # linear, stable target
        self.mol_mlp = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(d, 1))

    def mol_logit(self, src_a, src_b):
        zs_a, zs_b = self.proj_shared(src_a), self.proj_shared(src_b)
        zr_a, zr_b = self.proj_resid(src_a), self.proj_resid(src_b)
        feat = torch.cat([zs_a * zs_b, (zs_a - zs_b).abs(),
                          zr_a * zr_b, (zr_a - zr_b).abs()], dim=1)
        return self.mol_mlp(feat).squeeze(-1)


class _PerModeEmerGNN_V2RES(_PerModeEmerGNN_V2):
    def __init__(self, *, res_d: int = 128, lambda_align: float = 0.05, lambda_orth: float = 0.02,
                 lambda_dec: float = 0.002, lambda_std: float = 0.02,
                 mol_src: str = "fragments", **kwargs) -> None:
        kwargs.setdefault("v2_effect_mode", "count")  # effect path unused here
        super().__init__(**kwargs)
        self.res_d = int(res_d)
        # codex 019e674f: lowered lambdas under BCE sum-reduction + VICReg std hinge.
        self.lambda_align = float(lambda_align)
        self.lambda_orth = float(lambda_orth)
        self.lambda_dec = float(lambda_dec)
        self.lambda_std = float(lambda_std)
        self.mol_src = mol_src
        self._res_loaded = False

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        src_dim = 768  # BRICS fragment vocab dim
        return ResHead(src_dim=src_dim, ktyped_dim=8 * 768, d=self.res_d, dropout=self.mnah_dropout)

    def _load_res_assets(self) -> None:
        if self._res_loaded:
            return
        dev = self.device
        fr = np.load(FRAG_NPZ, allow_pickle=True)
        self._frag_row = {str(x): i for i, x in enumerate(fr["drug_ids"])}
        self._frag = torch.from_numpy(fr["frag"].astype(np.float32)).to(dev)
        kt = np.load(KTYPED_NPZ, allow_pickle=True)
        self._kt_row = {str(x): i for i, x in enumerate(kt["drug_ids"])}
        self._kt = torch.from_numpy(kt["k_typed"].astype(np.float32)).to(dev)
        self._frag_dim = self._frag.shape[1]
        self._kt_dim = self._kt.shape[1]
        self._res_loaded = True
        print(f"[v2res] assets: frag {tuple(self._frag.shape)} k_typed {tuple(self._kt.shape)}",
              flush=True)

    def _src_rows(self, ids, row_map, mat):
        idx = torch.tensor([row_map.get(str(x), -1) for x in ids], dtype=torch.long, device=self.device)
        ok = idx >= 0
        idx_clamped = idx.clamp(min=0)
        out = mat[idx_clamped]
        out = out * ok.float().unsqueeze(1)  # missing -> zeros
        return out

    def _breakdown(self, head, tail, edge_src, edge_dst, edge_rel, batch_df):
        self._load_res_assets()
        emergnn_logit = self._model(head, tail, edge_src, edge_dst, edge_rel)
        a = batch_df["drug_a_id"].astype(str).tolist()
        b = batch_df["drug_b_id"].astype(str).tolist()
        src_a = self._src_rows(a, self._frag_row, self._frag)
        src_b = self._src_rows(b, self._frag_row, self._frag)
        h: ResHead = self._aux_mlp
        mol_logit = h.mol_logit(src_a, src_b)
        combined = emergnn_logit + self._beta() * mol_logit
        # stash for aux losses (incl. drug ids for dedup)
        self._last_src = torch.cat([src_a, src_b], dim=0)
        self._last_kt = torch.cat([
            self._src_rows(a, self._kt_row, self._kt),
            self._src_rows(b, self._kt_row, self._kt)], dim=0)
        self._last_ids = list(a) + list(b)
        return {"combined": combined, "emergnn": emergnn_logit, "head": mol_logit,
                "mol": mol_logit, "eff": torch.zeros_like(mol_logit), "g": torch.zeros_like(mol_logit)}

    def _aux_losses(self, d, batch_df, y):
        h: ResHead = self._aux_mlp
        # codex 019e674f: dedup by DRUG ID (avoid InfoNCE false negatives from repeated drugs)
        seen = {}
        keep_rows = []
        for i, did in enumerate(self._last_ids):
            if did not in seen:
                seen[did] = i
                keep_rows.append(i)
        keep = torch.tensor(keep_rows, dtype=torch.long, device=self.device)
        src = self._last_src[keep]
        kt = self._last_kt[keep]
        has_k = (kt.abs().sum(dim=1) > 0)
        if int(has_k.sum()) < 2:
            return torch.zeros((), device=self.device)
        src = src[has_k]; kt = kt[has_k]
        rs = h.proj_resid(src)
        zs = F.normalize(h.proj_shared(src), dim=1)
        zk = F.normalize(h.proj_k(kt), dim=1)
        logits = zs @ zk.t() / 0.1
        tgt = torch.arange(zs.size(0), device=zs.device)
        nce = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
        zr = F.normalize(rs, dim=1)
        orth = (F.cosine_similarity(zr, zk.detach(), dim=1) ** 2).mean()
        dec = _offdiag_decorr(rs)
        # VICReg std hinge: keep per-dim std of raw residual >= 1 (prevents collapse)
        std = torch.sqrt(rs.var(dim=0) + 1e-4)
        std_hinge = F.relu(1.0 - std).mean()
        bs = max(y.numel(), 1)
        return bs * (self.lambda_align * nce + self.lambda_orth * orth
                     + self.lambda_dec * dec + self.lambda_std * std_hinge)


__all__ = ["_PerModeEmerGNN_V2RES", "ResHead"]
