"""NBFNet v3.0 — Joint binary + multi-class DDI prediction.

Series 3 = joint dual-task models. v3.0 is the MVP without ULTRA.

Architectural pillars (per Notes/Log/nbfnet_v3_0_design.md).
- Multi-query NBFNet: N_m learnable mechanism queries q_m, shared propagation params
- Dual readout: per-mechanism softmax multi-cls + soft-OR (LogSumExp) binary
- Joint loss: CE(multi, on positives) + lambda * BCE(binary, on all)
- Theoretical basis: "Proposition B" -- binary is OR-aggregation of mechanism evidence,
  not gate-before-multi (which has single-direction error propagation risk)

Reuses v1.7 NBFLayer + PNAAggregator UNCHANGED (per CLAUDE.md no-modification rule).
v3.0 model wraps them in an outer multi-query loop; vectorization is left for v3.0.1+.

Scope.
- v3.0  : MVP, fixed mechanism set, learnable q_m, Step 1-3 of user's plan
- v3.1+ : add ULTRA-style relation-conditional q_m via mechanism meta-graph (Step 4)

Status. Initial scaffold (2026-06-05). Pending codex review (Task #5).
"""
from __future__ import annotations

from my_code.models.nbfnet_v3_0.nbfnet_v3_model import (
    NBFNetJointDDI,
)
from my_code.models.nbfnet_v3_0.multicls_vocab import MultiClsVocab

__all__ = ["NBFNetJointDDI", "MultiClsVocab"]
