"""Re-export data loading from data module. See docs/7, 8."""
from __future__ import annotations

from my_code.data import (
    compute_data_key,
    get_or_load_data,
    get_or_load_features,
)

__all__ = ["compute_data_key", "get_or_load_data", "get_or_load_features"]
