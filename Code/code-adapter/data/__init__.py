"""Stage 1 - dataset load. Re-exports the loader entry points."""
from __future__ import annotations

from data.loader import (RankData, SUPPORTED_DATASETS, SUPPORTED_FOLDS,
                         load_rank_data)

__all__ = ["RankData", "load_rank_data", "SUPPORTED_DATASETS", "SUPPORTED_FOLDS"]
