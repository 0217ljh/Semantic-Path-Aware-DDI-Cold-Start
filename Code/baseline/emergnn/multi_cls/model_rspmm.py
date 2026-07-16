"""EmerGNN multi-class model, rspmm backend.

The rspmm twin of :class:`baseline.emergnn.multi_cls.model.EmerGNN_MC`. Inherits
the paper-faithful ``generalized_rspmm`` forward from
:class:`baseline.emergnn.model_rspmm.EmerGNN_RSPMM` UNCHANGED, and swaps only the
final ``Wr`` head from a single logit to ``n_classes`` logits (identical to how
``EmerGNN_MC`` extends the chunk ``EmerGNN``).

``forward`` is inherited: ``EmerGNN_RSPMM.forward`` ends with ``Wr(embed).squeeze(-1)``;
with ``Wr`` out_dim = ``n_classes`` > 1, ``squeeze(-1)`` is a no-op, so it returns
``(B, n_classes)`` logits — exactly matching ``EmerGNN_MC`` (which relies on the same
squeeze-is-a-no-op behavior inherited from ``EmerGNN``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from torch import nn

from baseline.emergnn.model_rspmm import EmerGNN_RSPMM


class EmerGNN_MC_RSPMM(EmerGNN_RSPMM):
    """EmerGNN multi-class output (rspmm backend). Inherits the rspmm forward;
    replaces the final ``Wr`` Linear out_dim 1 -> n_classes."""

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
        if feat == "E":
            self.Wr = nn.Linear(4 * n_dim, n_classes)
        else:
            self.Wr = nn.Linear(2 * n_dim, n_classes)
        nn.init.xavier_uniform_(self.Wr.weight)
        if self.Wr.bias is not None:
            nn.init.zeros_(self.Wr.bias)


__all__ = ["EmerGNN_MC_RSPMM"]
