# Generic types for pipeline (Sample/Batch, etc.)
from __future__ import annotations

from typing import Any

# Batch is dict-like: at least "x"; optional "y", "meta"
# Project-defined; no strict type here.
Batch = dict[str, Any]
