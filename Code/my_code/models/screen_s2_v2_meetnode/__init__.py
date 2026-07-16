"""v2 Stage 1: Meeting-Node Auxiliary Head (MNAH) on EmerGNN backbone.

See README.md for design rationale.
"""
from __future__ import annotations

from my_code.models.screen_s2_v2_meetnode.mnah_trainer import _PerModeEmerGNN_MNAH, AuxMLP

__all__ = ["_PerModeEmerGNN_MNAH", "AuxMLP"]
