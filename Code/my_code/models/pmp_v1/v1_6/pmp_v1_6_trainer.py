"""PMP v1.6 trainer (created 2026-06-04).

PyG MessagePassing-based re-implementation of v1.5 A. Inherits the v1.5
trainer (which inherits v1.2) so all cache construction is reused.

Overrides:
  - `_build_aux_head` to build `PMPv1_6_Module`.
  - `_combined_logit` to call the GNN module's 5-tuple forward.
  - `_predict_branches` / `_predict_branches_v1_6` for the
    single-branch diagnostic convention.
  - `_validate_branches` for single-branch val dict.
  - `_format_epoch_log` for the `[pmp-v1.6]` prefix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.pmp_v1.v1_5.pmp_v1_5_trainer import (  # noqa: E402
    _PerModeEmerGNN_PMP_v1_5, DEFAULT_CLUSTER_CACHE,
)
from my_code.models.pmp_v1.v1_6.cluster_head import (  # noqa: E402
    PMPv1_6_Module,
)


class _PerModeEmerGNN_PMP_v1_6(_PerModeEmerGNN_PMP_v1_5):
    """v1.6 trainer: PyG MessagePassing equivalent of v1.5 A.

    Inherits all args from v1.5 trainer. Drops the `content_hidden` arg from
    v1.5 (g_mlp is not present here — it was dead code in v1.5 A anyway).
    """

    def __init__(self, **kwargs) -> None:
        # Strip content_hidden if present (v1.6 has no g_mlp).
        kwargs.pop("content_hidden", None)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # _build_aux_head: PMPv1_6_Module
    # ------------------------------------------------------------------

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        if (
            self._cluster_n_clusters is None
            or self._cluster_member_pmp_ids is None
            or self._pmp_n_mediators is None
            or self._pmp_n_types is None
            or self._pmp_n_rels_total is None
        ):
            raise RuntimeError(
                "Caches must be loaded before _build_aux_head; call "
                "_load_feature_cache first."
            )

        module = PMPv1_6_Module(
            n_mediators=self._pmp_n_mediators,
            n_types=self._pmp_n_types,
            n_rels_plus_special=self._pmp_n_rels_total,
            n_clusters=self._cluster_n_clusters,
            d=self.n_dim,
            type_dim=None,
            rel_dim=None,
            hidden=self.cluster_hidden,
            dropout=self.cluster_dropout,
        )
        return module

    # ------------------------------------------------------------------
    # _combined_logit: same input construction as v1.5, but the GNN module
    # only uses the intersection mediator tensors (per-drug per-cluster
    # are accepted but ignored, mirroring v1.5 A's dead-code semantics).
    # ------------------------------------------------------------------

    def _combined_logit(
        self,
        head: torch.Tensor,
        tail: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        batch_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        self._ensure_caches_on_device()

        a_list = batch_df["drug_a_id"].astype(str).tolist()
        b_list = batch_df["drug_b_id"].astype(str).tolist()
        try:
            a_indices = [self._cluster_drug2id[d] for d in a_list]  # type: ignore[index]
            b_indices = [self._cluster_drug2id[d] for d in b_list]  # type: ignore[index]
        except KeyError as exc:
            raise KeyError(
                f"Drug id {exc.args[0]!r} not in cluster drug2id. Rebuild "
                f"cluster cache against a KG that contains all eval/train drugs."
            ) from exc
        a_idx = torch.tensor(a_indices, dtype=torch.long, device=device)
        b_idx = torch.tensor(b_indices, dtype=torch.long, device=device)

        affinity_a = self._cluster_affinity_t[a_idx]
        affinity_b = self._cluster_affinity_t[b_idx]

        # Per-drug per-cluster tensors are built (inherited code path) but
        # NOT passed to the GNN module — it doesn't accept them as Step 1
        # is purely analytical (affinity-driven).
        # We just skip the construction entirely to save time.

        # Intersection mediator tensors for Step 2.
        (
            med_idx_per_cluster,
            rel_a_per_cluster,
            rel_b_per_cluster,
            mask_int_per_cluster,
        ) = self._build_per_cluster_mediator_tensors(batch_df)

        score, _z_evidence, _alpha, _attn_a, _attn_b = self._aux_mlp(
            affinity_a, affinity_b,
            med_idx_per_cluster=med_idx_per_cluster,
            rel_a_per_cluster=rel_a_per_cluster,
            rel_b_per_cluster=rel_b_per_cluster,
            mask_per_cluster=mask_int_per_cluster,
            type_id_per_cluster=self._type_id_per_cluster,  # type: ignore[arg-type]
        )

        zero = torch.zeros_like(score)
        return score, zero, zero

    # ------------------------------------------------------------------
    # _predict_branches override — single-branch convention.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_branches(self, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_branches_v1_6(pairs)

    @torch.no_grad()
    def _predict_branches_v1_6(
        self, pairs: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_probs, combined_probs_copy, zeros)."""
        if self._model is None or self._aux_mlp is None:
            raise RuntimeError("PMP v1.6 trainer not yet fitted")
        self._model.eval()
        self._aux_mlp.eval()
        self._ensure_caches_on_device()

        c_out = np.empty(len(pairs), dtype=np.float32)

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            a_list = batch["drug_a_id"].astype(str).tolist()
            b_list = batch["drug_b_id"].astype(str).tolist()
            a_idx = torch.tensor(
                [self._cluster_drug2id[d] for d in a_list],  # type: ignore[index]
                dtype=torch.long, device=self.device,
            )
            b_idx = torch.tensor(
                [self._cluster_drug2id[d] for d in b_list],  # type: ignore[index]
                dtype=torch.long, device=self.device,
            )
            affinity_a = self._cluster_affinity_t[a_idx]
            affinity_b = self._cluster_affinity_t[b_idx]

            (
                med_idx_per_cluster, rel_a_per_cluster,
                rel_b_per_cluster, mask_int_per_cluster,
            ) = self._build_per_cluster_mediator_tensors(batch)

            score, _z, _a, _aa, _ab = self._aux_mlp(
                affinity_a, affinity_b,
                med_idx_per_cluster=med_idx_per_cluster,
                rel_a_per_cluster=rel_a_per_cluster,
                rel_b_per_cluster=rel_b_per_cluster,
                mask_per_cluster=mask_int_per_cluster,
                type_id_per_cluster=self._type_id_per_cluster,  # type: ignore[arg-type]
            )
            n = len(batch)
            c_out[start:start + n] = torch.sigmoid(score).detach().cpu().numpy()

        zeros = np.zeros_like(c_out)
        return c_out, c_out.copy(), zeros

    # ------------------------------------------------------------------
    # _validate_branches override — single-branch dict.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_branches(self, val) -> dict:
        from sklearn.metrics import roc_auc_score

        pos = val.splits.val_s2[["drug_a_id", "drug_b_id"]]
        neg = val.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]
        if len(pos) == 0 or len(neg) == 0:
            return {
                "combined": float("nan"),
                "emergnn": float("nan"),
                "aux": float("nan"),
            }
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, _, _ = self._predict_branches_v1_6(all_pairs)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        v_combined = float(roc_auc_score(y, c))
        return {
            "combined": v_combined,
            "emergnn": v_combined,
            "aux": float("nan"),
        }

    # ------------------------------------------------------------------
    # _format_epoch_log override — [pmp-v1.6] prefix.
    # ------------------------------------------------------------------

    def _format_epoch_log(
        self,
        *,
        epoch: int,
        losses: list,
        ep_time: float,
        v_auc: float,
        br: dict,
        beta_val: float,
    ) -> str:
        return (
            f"[pmp-v1.6] [ep {epoch + 1}/{self.n_epochs}] "
            f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
            f"val_combined={v_auc:.4f} "
            f"(PyG MessagePassing equivalent of v1.5 A) "
            f"beta={beta_val:.3f}"
        )


__all__ = [
    "_PerModeEmerGNN_PMP_v1_6",
    "DEFAULT_CLUSTER_CACHE",
]
