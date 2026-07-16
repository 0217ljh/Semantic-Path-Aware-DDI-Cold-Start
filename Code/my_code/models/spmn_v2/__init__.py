"""SPMN v2 — structural-variable adapter (asymmetric common-reachability).

New module. Migrates the KG-only hyper-edge structural variable from
``spmn_v1.rel_head`` into a pluggable adapter with two additions:

  * asymmetric stratification: per-type pooling split into symmetric
    (d_a == d_b) vs asymmetric (d_a != d_b) channels;
  * (reserved) K3 cross-level aggregation: protein -> pathway/GO/disease.

The core (``StructuralVariableCore``) produces a pair-level vector ``z_adapter``;
two thin output stages share it:

  * ``StandaloneHead``  = scorer(z_adapter)            -> logit  (solo A/B test)
  * ``FusionHead``      = mlp(cat[z_backbone, z_adapter]) -> logit (adapter on a backbone)

With ``use_asym=False`` (and no absdiff embedding / K3) the core reproduces
``spmn_v1.rel_head.SPMNRelHead`` exactly up to the final scoring MLP — see
``Code/scripts/verify_spmn_v2_core_equiv.py``. ``spmn_v1`` is never modified.
"""
from __future__ import annotations

from .batch import SupportBatch
from .core import StructuralVariableCore
from .heads import FusionHead, StandaloneHead

__all__ = [
    "SupportBatch",
    "StructuralVariableCore",
    "StandaloneHead",
    "FusionHead",
]
