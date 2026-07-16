"""Per-mode trainer for Screen 5 (PK/PD dual subgraph).

Two parallel EmerGNN_TAG networks (one per subgraph) sharing the external
init but with their own flow + attention parameters. Late merge at the
score head: Linear(8*n_dim, 1) over [pk_h_emb, pk_t_emb, pk_h_hid, pk_t_hid,
pd_h_emb, pd_t_emb, pd_h_hid, pd_t_hid].

Per first_step_plan.md §4.7 variants:
  S0 : full KG, no split (=  _PerModeEmerGNN_TAG)
  S3 : PK ∥ PD dual flow + late merge (PRIMARY)
  S4 : S3 + VME injection at gap positions
  S6 : random 2-way split (control)
  S7 : strict no-overlap split
  S8 : S3 + scrambled VME (semantic control)

This trainer mirrors _PerModeEmerGNN.fit() shuffle_train logic with:
  - Two separate edge tensors (PK, PD) instead of one
  - Two model instances trained jointly via summed loss + joint merge head
  - VME injection placeholder (S4/S8 — not yet wired)

NOTE: This is a FUNCTIONAL skeleton suitable for code review; running
training requires verifying junction relation IDs map correctly into the
subgraph-restricted relation set, which I have not fully audited because
the trainer is not on the critical path during this 10h autonomous run.
A NotImplementedError is left in `fit()` until that audit is complete.
"""
from __future__ import annotations

import copy
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from baseline.emergnn._per_mode import _PerModeEmerGNN
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train

from my_code.models.screen1_tag_init.emergnn_with_init import EmerGNN_TAG
from my_code.models.screen5_pkpd_subgraph.subgraph_builder import split_edges_pk_pd

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


class _DualFlowHead(nn.Module):
    """Score head that merges PK and PD branch outputs.

    Input: pk_head_emb, pk_tail_emb, pk_head_hid, pk_tail_hid (4 * n_dim)
           pd_head_emb, pd_tail_emb, pd_head_hid, pd_tail_hid (4 * n_dim)
    Output: scalar logit
    """

    def __init__(self, n_dim: int):
        super().__init__()
        self.merge = nn.Linear(8 * n_dim, 1)
        nn.init.xavier_uniform_(self.merge.weight)


class _PerModeEmerGNN_PKPD(_PerModeEmerGNN):
    """Per-mode dual-subgraph EmerGNN trainer.

    Args specific to dual-flow:
      external_init             : Tensor[N, n_dim] node init
      external_init_node_ids    : node-ID list
      variant                   : S0 | S3 | S4 | S6 | S7 | S8
      merged_edges_for_split    : path to merged-KG edges parquet (for subgraph extraction)
      merged_nodes_for_split    : path to merged-KG nodes parquet (for canonical kind lookup)
      vme_table                 : dict (gap_key -> embedding) for S4/S8, None for others
    """

    def __init__(
        self,
        *,
        external_init: np.ndarray | torch.Tensor,
        external_init_node_ids: list[str],
        variant: str,
        merged_edges_for_split: str | Path,
        merged_nodes_for_split: str | Path,
        vme_table: dict | None = None,
        freeze_init: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("feat", "E")
        super().__init__(**kwargs)
        if variant not in ("S0", "S3", "S4", "S6", "S7", "S8"):
            raise ValueError(f"unknown variant {variant!r}")
        self._variant = variant
        if isinstance(external_init, np.ndarray):
            external_init = torch.from_numpy(external_init.astype(np.float32))
        self._external_init = external_init.float()
        self._external_init_node_ids = list(external_init_node_ids)
        self._merged_edges = Path(merged_edges_for_split)
        self._merged_nodes = Path(merged_nodes_for_split)
        self._vme_table = vme_table
        self._freeze_init = bool(freeze_init)

    def _align_init(self) -> torch.Tensor:
        n_ent = self._n_ent
        aligned = torch.zeros(n_ent, self._external_init.shape[1])
        id2row = {nid: r for r, nid in enumerate(self._external_init_node_ids)}
        prefixes = ("db:target:", "db:enzyme:", "db:transporter:", "db:carrier:", "db:pathway:")
        for ent, idx in self._entity2id.items():
            row = id2row.get(ent)
            if row is None and not ent.startswith(("DB", "db:", "het:", "prime:")):
                for p in prefixes:
                    row = id2row.get(f"{p}{ent}")
                    if row is not None:
                        break
            if row is not None:
                aligned[idx] = self._external_init[row]
        return aligned

    def _build_subgraph_edges(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load merged KG, split into PK / PD subgraphs per variant.

        Returns (pk_edges, pd_edges) DataFrames. Schema matches input parquet.
        For S0 (full KG): both = all edges.
        For S6 (random 2-way): random partition matching PK/PD sizes.
        For S7 (strict): drug-only edges go to PK only (arbitrary canonical choice).
        """
        edges_df = pd.read_parquet(self._merged_edges)
        nodes_df = pd.read_parquet(self._merged_nodes)

        if self._variant == "S0":
            return edges_df.copy(), edges_df.copy()

        if self._variant in ("S3", "S4", "S8"):
            return split_edges_pk_pd(edges_df, nodes_df, overlap_mode="include_both")

        if self._variant == "S7":
            return split_edges_pk_pd(edges_df, nodes_df, overlap_mode="strict_pk")

        if self._variant == "S6":
            pk0, pd0 = split_edges_pk_pd(edges_df, nodes_df, overlap_mode="include_both")
            return ift_random_split_by_size(edges_df, (len(pk0), len(pd0)), seed=42)

        raise ValueError(f"unknown variant {self._variant!r}")

    def fit(self, train, val=None, *, kg=None):
        """Production dual-flow training.

        Status: SKELETON. The flow propagation needs to use the SUBGRAPH-
        restricted edge_src/edge_dst/edge_rel tensors (built from PK/PD
        subgraphs), but the relation index space depends on how we
        re-encode primekg/hetionet relation types when restricting to
        subgraph. Need to audit before relying on numerical results.

        For now, raise NotImplementedError to prevent silent garbage runs.
        Production version will:
          1. Build PK/PD subgraph edge tensors with their own relation vocab
          2. Two EmerGNN_TAG instances (pk_model, pd_model) sharing init
          3. Joint optimizer over both + _DualFlowHead
          4. Per-batch forward: get hidden states from both branches, merge via head
          5. (S4 only) Inject VME at gap positions during flow

        See first_step_plan.md §4.7 for full design.
        """
        raise NotImplementedError(
            "Screen 5 _PerModeEmerGNN_PKPD.fit() is FUNCTIONAL skeleton only. "
            "Production version requires relation-vocab audit between PK/PD subgraphs. "
            "Deferred per first_step_plan.md §4.7 execution priority."
        )

    def predict_proba(self, pairs, *, kg=None):
        raise NotImplementedError("see fit()")


def ift_random_split_by_size(edges_df, sizes, seed=0):
    from my_code.models.screen5_pkpd_subgraph.subgraph_builder import split_edges_random_2way
    return split_edges_random_2way(edges_df, sizes, seed=seed)


__all__ = ["_PerModeEmerGNN_PKPD", "_DualFlowHead"]
