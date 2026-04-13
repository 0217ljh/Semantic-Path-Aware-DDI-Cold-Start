"""Cora collate: full-batch single graph -> return batch[0]."""
from __future__ import annotations

from typing import List


def collate_cora(batch: List[dict]) -> dict:
    return batch[0]
