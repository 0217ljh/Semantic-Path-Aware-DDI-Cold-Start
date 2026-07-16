"""KnowDDI multiclass task package (unified benchmark).

Re-exports the multiclass unified wrapper so callers can
``from baseline.knowddi.multi_cls import KnowDDIUnifiedMulticlass``.
"""
from __future__ import annotations

from .baseline_unified import KnowDDIUnifiedMulticlass

__all__ = ["KnowDDIUnifiedMulticlass"]
