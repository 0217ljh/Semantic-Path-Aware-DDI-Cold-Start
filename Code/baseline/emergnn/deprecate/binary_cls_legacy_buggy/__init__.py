"""DEPRECATED 2026-05-20 — original single-S2-mode binary baseline.

Kept as historical record. See ``baseline.py`` docstring for deprecation note.
The class was renamed to ``EmerGNNBaselineLegacyBuggy`` and the ``"emergnn"``
registry decorator was removed so importing this file does not collide with
the new official baseline at ``baseline/emergnn/binary_cls/baseline.py``.
"""
from __future__ import annotations

from baseline.emergnn.deprecate.binary_cls_legacy_buggy.baseline import (
    EmerGNNBaselineLegacyBuggy,
)

__all__ = ["EmerGNNBaselineLegacyBuggy"]
