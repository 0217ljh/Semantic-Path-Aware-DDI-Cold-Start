"""Message-passing backend selector for the EmerGNN baseline.

Two backends produce the SAME EmerGNN model family but differ only in the
sparse relational message-passing kernel:

  * ``"chunk"`` (default) — the pure-PyTorch chunked scatter loop in
    :mod:`baseline.emergnn.model` (no third-party CUDA kernel; portable but
    slow / memory-heavy on the full merged KG).
  * ``"rspmm"`` — torchdrug's fused ``generalized_rspmm`` CUDA kernel (the
    ORIGINAL EmerGNN kernel; ~20x faster + far less memory on the full KG,
    which is what makes the 7.1M-edge merged KG feasible).

Selected via the ``EMERGNN_BACKEND`` environment variable so no existing
function signature has to change and the default (chunk) path stays byte-for-byte
untouched. The chosen backend is recorded by the caller in run manifests/logs
for provenance.
"""
from __future__ import annotations

import os

VALID_BACKENDS = ("chunk", "rspmm")
DEFAULT_BACKEND = "chunk"


def get_backend() -> str:
    """Return the selected backend, validated. Defaults to ``"chunk"``."""
    b = os.environ.get("EMERGNN_BACKEND", DEFAULT_BACKEND).strip().lower()
    if b not in VALID_BACKENDS:
        raise ValueError(
            f"EMERGNN_BACKEND must be one of {VALID_BACKENDS}; got {b!r}"
        )
    return b
