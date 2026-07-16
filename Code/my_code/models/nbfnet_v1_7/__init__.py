"""NBFNet v1.7 — Paper-faithful Neural Bellman-Ford Networks for cold-start DDI.

Independent branch (not integrated with PMP). Faithful to NBFNet
(Zhu et al., NeurIPS 2021, arxiv 2106.06935). Design passed 3-round
codex faithfulness review (thread 019e95e4).

Paper-faithful design choices.
- L=6 layers, d=32 hidden (paper defaults)
- INDICATOR. h_v^(0) = q if v=source else 0
- MESSAGE. DistMult-style m = h_x * w_r^(t)
- AGGREGATE. PNA-style (mean/max/min/std x 3 scalers) + boundary reinjection
- Per-(layer, relation) W_r^(t), b_r^(t) (relation-specific, not shared)
- Inverse-edge augmentation for KG
- Query-edge masking (relation-aware on direct queried pair)
- Representation-level symmetrization (h_q(a,b) + h_q(b,a)) before MLP
- MLP input. concat([h_q_sym, q]) per official-repo convention
- ReLU nonlinearity after each AGGREGATE
- Per-source amortized BF in training (sample multiple targets per source)
- AUROC + AP for validation/early stopping

Goal. Verify vanilla NBFNet on our DrugBank 800-drug S2 cold-start
binary DDI vs v1.5A (0.7764) and v2i4 (0.7804) baselines.

Reference. github.com/DeepGraphLearning/NBFNet (algorithmic reference
only, NOT forked — old TorchDrug/PyTorch 1.8 stack).
"""
from __future__ import annotations

from my_code.models.nbfnet_v1_7.nbfnet_model import (
    NBFNetDDI,
    NBFLayer,
    PNAAggregator,
)

__all__ = ["NBFNetDDI", "NBFLayer", "PNAAggregator"]
