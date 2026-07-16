"""PMP v1.3 trainer (created 2026-06-03).

Layer-1-only ablation of v1.2. Inherits the v1.2 trainer for cache loading
(PMP cache + v1.1 cluster cache v2 + per-drug per-cluster mediator map), then
overrides `_build_aux_head`, `_combined_logit`, `_predict_branches`,
`_validate_branches`, and `_format_epoch_log` to score from the cluster path
ONLY, without the within-cluster pool.
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

from my_code.models.pmp_v1.v1_2.pmp_v1_2_trainer import (  # noqa: E402
    _PerModeEmerGNN_PMP_v1_2, DEFAULT_CLUSTER_CACHE,
)
from my_code.models.pmp_v1.v1_3.cluster_head import PMPv1_3Module  # noqa: E402


class _PerModeEmerGNN_PMP_v1_3(_PerModeEmerGNN_PMP_v1_2):
    """v1.3 trainer: Layer 1 (pair-conditional cluster path) only.

    Inherits all args from v1.2, including `cluster_max_mediators_per_drug`.
    Does NOT use intersection mediator tensors at training time, since the
    within-cluster pool is dropped entirely.
    """

    # ------------------------------------------------------------------
    # _build_aux_head: PMPv1_3Module (cluster path only)
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

        module = PMPv1_3Module(
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
    # _combined_logit: cluster-path-only forward
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
        rel_dist_a = self._cluster_rel_dist_t[a_idx]
        rel_dist_b = self._cluster_rel_dist_t[b_idx]

        med_a_per_cluster, mask_a_per_cluster = (
            self._build_per_drug_per_cluster_tensors(a_list)
        )
        med_b_per_cluster, mask_b_per_cluster = (
            self._build_per_drug_per_cluster_tensors(b_list)
        )

        # v1.3 forward does NOT take intersection tensors.
        score, _z_cluster_pair = self._aux_mlp(
            affinity_a, affinity_b, rel_dist_a, rel_dist_b,
            med_a_per_cluster, mask_a_per_cluster,
            med_b_per_cluster, mask_b_per_cluster,
        )

        zero = torch.zeros_like(score)
        return score, zero, zero

    # ------------------------------------------------------------------
    # _predict_branches override — v1.3 has only one branch (cluster path).
    # Return (combined, combined, zeros) so callers expecting 3-tuple work,
    # but the "cluster_only" slot is just a duplicate of combined and the
    # "within_only" slot is zeros (no within branch in v1.3).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_branches(self, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_branches_v1_3(pairs)

    @torch.no_grad()
    def _predict_branches_v1_3(
        self, pairs: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_probs, combined_probs_copy, zeros).

        v1.3 has no within branch to decompose, so cluster_only_probs is a
        duplicate of combined for diagnostic-slot compatibility.
        """
        if self._model is None or self._aux_mlp is None:
            raise RuntimeError("PMP v1.3 trainer not yet fitted")
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
            rel_dist_a = self._cluster_rel_dist_t[a_idx]
            rel_dist_b = self._cluster_rel_dist_t[b_idx]

            med_a_per_cluster, mask_a_per_cluster = (
                self._build_per_drug_per_cluster_tensors(a_list)
            )
            med_b_per_cluster, mask_b_per_cluster = (
                self._build_per_drug_per_cluster_tensors(b_list)
            )

            score, _z = self._aux_mlp(
                affinity_a, affinity_b, rel_dist_a, rel_dist_b,
                med_a_per_cluster, mask_a_per_cluster,
                med_b_per_cluster, mask_b_per_cluster,
            )
            n = len(batch)
            c_out[start:start + n] = torch.sigmoid(score).detach().cpu().numpy()

        zeros = np.zeros_like(c_out)
        return c_out, c_out.copy(), zeros

    # ------------------------------------------------------------------
    # _validate_branches override — single-branch dict
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
                "cluster_only": float("nan"),
                "within_only": float("nan"),
            }
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, _, _ = self._predict_branches_v1_3(all_pairs)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        v_combined = float(roc_auc_score(y, c))
        # v1.3 has no within branch — slots are placeholders.
        return {
            "combined": v_combined,
            # Legacy keys still expected by parent fit() print-line path
            # (only used as fallback if subclass doesn't override
            # _format_epoch_log). Stuffed with v_combined for consistency.
            "emergnn": v_combined,
            "aux": float("nan"),
            "cluster_only": v_combined,
            "within_only": float("nan"),
        }

    # ------------------------------------------------------------------
    # _format_epoch_log override — single-branch log line
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
            f"[pmp-v1.3] [ep {epoch + 1}/{self.n_epochs}] "
            f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
            f"val_combined={v_auc:.4f} "
            f"(layer1-only; no within branch) "
            f"beta={beta_val:.3f}"
        )


__all__ = [
    "_PerModeEmerGNN_PMP_v1_3",
    "DEFAULT_CLUSTER_CACHE",
]
