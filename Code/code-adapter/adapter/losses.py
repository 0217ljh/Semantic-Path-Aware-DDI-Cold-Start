"""Stage 4 / DDI-LoRA adapter - loss module (kept separate from the model so loss
terms can be changed / added without touching the forward pass).

Total loss = task_loss(s_uv, y) - eta_H * entropy(beta)   (paper: task loss minus an
entropy term on the assignment beta; subtracting entropy REWARDS spread assignments,
i.e. discourages collapse onto a single prototype). eta_H = cfg.eta_entropy (0 = off).

`forward` returns (total, components) where components is a dict of scalar tensors for
logging (task / entropy / total). To add a new regularizer: give it a `_term_*` method
and add one line in `forward`; the components dict makes it show up in the logs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdapterConfig

TASK_KINDS = ("binary", "multiclass", "multilabel")


class AdapterLoss(nn.Module):
    def __init__(self, cfg: AdapterConfig, task_kind: str):
        super().__init__()
        if task_kind not in TASK_KINDS:
            raise ValueError(f"task_kind must be one of {TASK_KINDS}, got {task_kind!r}")
        if task_kind == "binary" and cfg.n_out != 1:
            raise ValueError(f"binary task expects n_out=1, got n_out={cfg.n_out}")
        self.cfg = cfg
        self.task_kind = task_kind

    def _term_task(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: [B, n_out]. binary -> BCEWithLogits ([B,1] vs float y);
        multilabel -> BCEWithLogits ([B,K] vs float y); multiclass -> CE ([B,K] vs long y)."""
        if self.task_kind == "multiclass":
            return F.cross_entropy(logits, target.long())
        y = target.float()
        if self.task_kind == "binary":
            logits = logits.squeeze(-1)                       # [B,1] -> [B]
            y = y.squeeze(-1) if y.dim() > 1 else y
        return F.binary_cross_entropy_with_logits(logits, y)

    def _term_entropy(self, beta: torch.Tensor) -> torch.Tensor:
        """Mean assignment entropy H(beta) = -sum_k beta_k log beta_k over mediators."""
        if beta.numel() == 0:
            return beta.new_zeros(())
        return -(beta.clamp_min(1e-12).log() * beta).sum(-1).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                beta: torch.Tensor | None = None):
        """-> (total: scalar, components: dict[str, scalar tensor])."""
        task = self._term_task(logits, target)
        total = task
        comps = {"task": task.detach()}
        if beta is not None:
            ent = self._term_entropy(beta)                    # always logged (diagnostic)
            comps["entropy"] = ent.detach()
            if self.cfg.eta_entropy != 0.0:
                total = total - self.cfg.eta_entropy * ent    # subtract -> reward spread
        comps["total"] = total.detach()
        return total, comps


__all__ = ["AdapterLoss", "TASK_KINDS"]
