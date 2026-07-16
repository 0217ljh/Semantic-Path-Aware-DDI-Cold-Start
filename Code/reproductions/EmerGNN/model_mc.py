"""EmerGNN multi-class variant model (reproduction-side copy).

Lifts only the output-head from the EmerGNN core: instead of `Wr` producing
a single scalar logit per (head, tail) pair, it produces `n_classes` logits
for multi-class softmax (DrugBank 86 DDI types).

All message-passing layers reused from the binary `EmerGNN` model verbatim.

Reference: LARS-research/EmerGNN/DrugBank/base_model.py uses a softmax margin
loss over relation classes. We replicate the multi-class output dim here;
the actual loss (cross-entropy vs softmax-margin) is chosen in the runner.

Note: this file is an INDEPENDENT COPY of
``baseline/emergnn/multi_cls/model.py`` per CLAUDE.md §"Baseline 规范"
line 447-448 "严禁 import 互调". The two sides are byte-equivalent on
import 2026-05-18; future drift requires re-sync via a documented diff.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from model import EmerGNN  # local copy (was: baseline.emergnn.model)


class EmerGNN_MC(EmerGNN):
    """EmerGNN multi-class output. Inherits binary model verbatim, swaps
    final `Wr` Linear from out_dim=1 to out_dim=n_classes.
    """

    def __init__(
        self,
        n_ent: int,
        n_base_rel: int,
        n_classes: int,
        n_dim: int = 64,
        length: int = 3,
        feat: str = "M",
        morgan_features: Optional[np.ndarray] = None,
        morgan_feat_dim: int = 1024,
    ) -> None:
        super().__init__(
            n_ent=n_ent,
            n_base_rel=n_base_rel,
            n_dim=n_dim,
            length=length,
            feat=feat,
            morgan_features=morgan_features,
            morgan_feat_dim=morgan_feat_dim,
        )
        self.n_classes = n_classes
        # Replace single-logit head with n_classes-logit head
        if feat == "E":
            self.Wr = nn.Linear(4 * n_dim, n_classes)
        else:
            self.Wr = nn.Linear(2 * n_dim, n_classes)
        # Re-init the replaced layer with xavier
        nn.init.xavier_uniform_(self.Wr.weight)
        if self.Wr.bias is not None:
            nn.init.zeros_(self.Wr.bias)
