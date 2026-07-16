"""Stage 3 (i1): PK/PD dual-channel meeting-node aux head.

Per codex round 17: split the 22 shared-mediator COUNT features into a
PK/molecular channel and a PD/effect channel, each with its own MLP head and a
global softplus gate. NO PK/PD label supervision in training (architectural
prior). PK/PD labels used ONLY for post-hoc per-class difference-in-differences
(DiD) channel-ablation analysis.

Success = PK/PD specialization WITHOUT performance loss (expect combined ~ Stage 1),
NOT a metric bump. This is i1 MECHANISM evidence, not the headline.

Channel assignment over the 11 kind-groups (×{1hop,2hop}):
  PK / molecular : protein_gene, pathway, biological_process, molecular_function,
                   cellular_component
  PD / effect    : side_effect, disease, anatomy
  neutral        : compound, pharmacologic_class, exposure  (fed to BOTH by default)

KIND_ORDER (from precompute_meet_features) index map:
  0 protein_gene 1 pathway 2 side_effect 3 disease 4 anatomy 5 compound
  6 biological_process 7 molecular_function 8 cellular_component
  9 pharmacologic_class 10 exposure   (+11 for the 2hop block)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_FILE = Path(__file__).resolve()
sys.path.insert(0, str(_FILE.parents[4] / "Code"))

from my_code.models.screen_s2_v2_meetnode.mnah_trainer import _PerModeEmerGNN_MNAH, AuxMLP

# index groups within the 11-kind block (0..10); 2hop = +11
PK_KINDS = [0, 1, 6, 7, 8]      # molecular
PD_KINDS = [2, 3, 4]            # effect-system
NEUTRAL_KINDS = [5, 9, 10]      # compound / pharm-class / exposure


def _expand(idx_list: list[int]) -> list[int]:
    return idx_list + [i + 11 for i in idx_list]


PK_IDX = _expand(PK_KINDS)        # 10 dims
PD_IDX = _expand(PD_KINDS)        # 6 dims
NEUTRAL_IDX = _expand(NEUTRAL_KINDS)  # 6 dims


class DualChannelAux(nn.Module):
    """PK head + PD head over disjoint count-feature subsets, two global gates.

    forward(x, channel) where channel in {None,'pk_only','pd_only'}:
      None     -> g_pk*pk + g_pd*pd   (full)
      'pk_only'-> g_pk*pk             (ablate PD)
      'pd_only'-> g_pd*pd             (ablate PK)
    """

    def __init__(self, hidden: int = 32, dropout: float = 0.2,
                 neutral_to: str = "both"):
        super().__init__()
        pk_extra = NEUTRAL_IDX if neutral_to in ("both", "pk") else []
        pd_extra = NEUTRAL_IDX if neutral_to in ("both", "pd") else []
        self.pk_cols = PK_IDX + pk_extra
        self.pd_cols = PD_IDX + pd_extra
        self.pk_head = AuxMLP(in_dim=len(self.pk_cols), hidden=hidden, dropout=dropout)
        self.pd_head = AuxMLP(in_dim=len(self.pd_cols), hidden=hidden, dropout=dropout)
        # softplus(0.5413) ~= 1.0 ; init both gates ~1
        self.b_pk = nn.Parameter(torch.tensor(0.5413))
        self.b_pd = nn.Parameter(torch.tensor(0.5413))
        self._pk_cols_t = torch.tensor(self.pk_cols, dtype=torch.long)
        self._pd_cols_t = torch.tensor(self.pd_cols, dtype=torch.long)

    def gates(self) -> tuple[float, float]:
        return float(F.softplus(self.b_pk)), float(F.softplus(self.b_pd))

    def forward(self, x: torch.Tensor, channel: str | None = None) -> torch.Tensor:
        pk_cols = self._pk_cols_t.to(x.device)
        pd_cols = self._pd_cols_t.to(x.device)
        gpk = F.softplus(self.b_pk)
        gpd = F.softplus(self.b_pd)
        pk = gpk * self.pk_head(x.index_select(1, pk_cols))
        pd = gpd * self.pd_head(x.index_select(1, pd_cols))
        if channel == "pk_only":
            return pk
        if channel == "pd_only":
            return pd
        return pk + pd


class _PerModeEmerGNN_PKPD(_PerModeEmerGNN_MNAH):
    """Stage 3: MNAH with PK/PD dual-channel aux head over the 22 counts."""

    def __init__(self, *, pkpd_neutral_to: str = "both", **kwargs) -> None:
        # Stage 3 is counts-only (no text); force text cache off.
        kwargs["mnah_text_cache"] = None
        super().__init__(**kwargs)
        self.pkpd_neutral_to = pkpd_neutral_to

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        if in_dim != 22:
            raise ValueError(f"Stage 3 expects 22 count dims, got {in_dim}")
        return DualChannelAux(hidden=self.mnah_hidden, dropout=self.mnah_dropout,
                              neutral_to=self.pkpd_neutral_to)

    @torch.no_grad()
    def predict_proba_channel(self, pairs, channel: str | None = None) -> np.ndarray:
        """combined prob with aux restricted to a channel (for DiD ablation).
        channel: None=full, 'pk_only'=ablate PD, 'pd_only'=ablate PK,
                 'no_aux'=emergnn only.
        """
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            edge_src, edge_dst, edge_rel = self._eval_edges
        else:
            edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval(); self._aux_mlp.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        beta = self._beta()
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head = head.to(self.device); tail = tail.to(self.device)
            emer = self._model(head, tail, edge_src, edge_dst, edge_rel)
            if channel == "no_aux":
                logit = emer
            else:
                feats = self._lookup_features(batch)
                aux = self._aux_mlp(feats, channel=channel)
                logit = emer + beta * aux
            out[start:start + len(batch)] = torch.sigmoid(logit).cpu().numpy()
        return out


__all__ = ["_PerModeEmerGNN_PKPD", "DualChannelAux", "PK_IDX", "PD_IDX", "NEUTRAL_IDX"]
