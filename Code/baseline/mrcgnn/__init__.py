"""MRCGNN baseline (AAAI-2023) ported for the unified cold-start DDI benchmark.

Step 2 exposes the model core only (Discriminator / AvgReadout / MLP head + the
``MRCGNN`` module). The multiclass training core and the unified wrapper land in
later steps under ``baseline/mrcgnn/multi_cls/``.
"""
from __future__ import annotations

from baseline.mrcgnn.layers import (
    AvgReadout,
    Discriminator,
    MLPHead,
    mlp_input_width,
)
from baseline.mrcgnn.models import MRCGNN

__all__ = ["MRCGNN", "Discriminator", "AvgReadout", "MLPHead", "mlp_input_width"]
