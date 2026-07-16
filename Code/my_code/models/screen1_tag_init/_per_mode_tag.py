"""Per-mode wrapper of EmerGNN with external init injection.

Reuses `_PerModeEmerGNN.fit()` verbatim — we only swap the model class via
a temporary monkey-patch of `baseline.emergnn._per_mode.EmerGNN` so that
training uses our `EmerGNN_TAG` (feat='X') instead of the upstream `EmerGNN`
(feat='M' / 'E').

Why this approach?
  - CLAUDE.md forbids modifying baseline code
  - Copy-pasting fit() would diverge over time
  - Monkey-patch is local (context-manager-scoped) and reversible
  - Trainer's `feat='E'` keeps save/load shape-compatible (X mode uses the
    same `ent_kg` Embedding name and same Wr Linear(4*n_dim, 1) head)
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from baseline.emergnn._per_mode import _PerModeEmerGNN  # noqa: E402
import baseline.emergnn._per_mode as _upstream_pm       # noqa: E402

from my_code.models.screen1_tag_init.emergnn_with_init import EmerGNN_TAG  # noqa: E402


class _EmerGNN_TAG_Factory:
    """Adapter callable matching upstream EmerGNN(...) signature.

    Upstream calls `EmerGNN(n_ent, n_base_rel, n_dim, length, feat, morgan_features)`.
    We discard `feat` and `morgan_features`; force feat='X' + external_init.
    """

    def __init__(self, external_init_aligned: torch.Tensor, freeze_init: bool):
        self._init = external_init_aligned
        self._freeze = bool(freeze_init)

    def __call__(self, n_ent, n_base_rel, n_dim, length, feat, morgan_features=None):
        if self._init.shape != (n_ent, n_dim):
            raise ValueError(
                f"factory got external_init shape {tuple(self._init.shape)} "
                f"but upstream wants (n_ent={n_ent}, n_dim={n_dim})"
            )
        return EmerGNN_TAG(
            n_ent=n_ent,
            n_base_rel=n_base_rel,
            n_dim=n_dim,
            length=length,
            external_init=self._init,
            freeze_init=self._freeze,
            feat="X",
        )


@contextmanager
def _patch_emergnn_class(factory):
    old = _upstream_pm.EmerGNN
    _upstream_pm.EmerGNN = factory
    try:
        yield
    finally:
        _upstream_pm.EmerGNN = old


class _PerModeEmerGNN_TAG(_PerModeEmerGNN):
    """Per-mode EmerGNN trainer that uses EmerGNN_TAG via temporary patch.

    Init arguments
      external_init : Tensor[n_ent, n_dim] aligned to the merged-KG entity vocab.
                      Caller is responsible for ensuring the row order matches
                      `_PerModeEmerGNN._entity2id`. Alignment is verified at
                      first fit() call.
      external_init_node_ids : list of node IDs corresponding to rows of
                      `external_init` (used to remap to entity-id order).
      freeze_init : if True, ent_kg.weight is frozen during training.

    All other kwargs are forwarded to `_PerModeEmerGNN.__init__`. We force
    `feat='E'` on the trainer so save/load doesn't try to access Morgan
    feature buffers (X mode's nn.Embedding + Wr(4*n_dim, 1) is identical
    shape to 'E' mode, so state-dict round-trip works.)
    """

    def __init__(
        self,
        *,
        external_init: np.ndarray | torch.Tensor,
        external_init_node_ids: list[str],
        freeze_init: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("feat", "E")
        if kwargs.get("feat") != "E":
            raise ValueError("_PerModeEmerGNN_TAG forces feat='E' (X-shape isomorphic)")
        super().__init__(**kwargs)

        if isinstance(external_init, np.ndarray):
            external_init = torch.from_numpy(external_init.astype(np.float32))
        self._external_init_raw = external_init.float()
        self._external_init_node_ids = list(external_init_node_ids)
        self._freeze_init = bool(freeze_init)

        if self._external_init_raw.shape[0] != len(self._external_init_node_ids):
            raise ValueError(
                f"external_init has {self._external_init_raw.shape[0]} rows but "
                f"external_init_node_ids has {len(self._external_init_node_ids)} entries"
            )

    def _align_init_to_entity_vocab(self) -> torch.Tensor:
        """Reorder external_init rows to match `self._entity2id` order.

        Drugbank-KG trainer vocab uses RAW entity IDs (e.g. `BE0001234`) but
        the merged-KG external init uses PREFIXED IDs (e.g. `db:enzyme:BE0001234`).
        Try direct match first; if miss and not a drug ID, try each drugbank
        role-prefix in order: target → enzyme → transporter → carrier → pathway.
        First hit wins (same biological entity tends to have identical text
        across roles since the encoder uses the entity's name).
        """
        if self._entity2id is None:
            raise RuntimeError("Trainer must run _setup_graph before alignment")
        n_dim = self._external_init_raw.shape[1]
        n_ent = self._n_ent
        aligned = torch.zeros(n_ent, n_dim)
        id2row = {nid: r for r, nid in enumerate(self._external_init_node_ids)}
        drugbank_prefixes = ("db:target:", "db:enzyme:", "db:transporter:",
                             "db:carrier:", "db:pathway:")
        n_direct = 0
        n_via_prefix = 0
        n_miss = 0
        for ent, idx in self._entity2id.items():
            row = id2row.get(ent)
            if row is not None:
                aligned[idx] = self._external_init_raw[row]
                n_direct += 1
                continue
            # Try drugbank role prefixes if entity isn't already prefixed/DB
            if not ent.startswith(("DB", "db:", "het:", "prime:")):
                for prefix in drugbank_prefixes:
                    row = id2row.get(f"{prefix}{ent}")
                    if row is not None:
                        aligned[idx] = self._external_init_raw[row]
                        n_via_prefix += 1
                        break
            if row is None:
                n_miss += 1
        n_hit = n_direct + n_via_prefix
        print(f"[per_mode_tag] external_init alignment: hit={n_hit:,} / "
              f"trainer_vocab={n_ent:,} "
              f"(direct={n_direct:,}, via_prefix={n_via_prefix:,}, "
              f"miss={n_miss:,} -> zero vec)")
        return aligned

    def fit(self, train, val=None, *, kg=None):
        # Build entity vocab first (so we can align init)
        if kg is None:
            kg = train.kg
        # Run upstream _setup_graph to populate _entity2id et al.
        self._setup_graph(train, kg)
        aligned_init = self._align_init_to_entity_vocab()
        factory = _EmerGNN_TAG_Factory(aligned_init, freeze_init=self._freeze_init)
        with _patch_emergnn_class(factory):
            super().fit(train, val, kg=kg)


__all__ = ["_PerModeEmerGNN_TAG"]
