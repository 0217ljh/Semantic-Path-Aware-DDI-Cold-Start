"""PMP v1.4 trainer (created 2026-06-03).

Layer-2-only ablation of v1.2. Inherits the v1.1 trainer directly (NOT v1.2)
because v1.4 does not need the per-drug per-cluster mediator map; the
intersection-based within-pool is sufficient.

Overrides `_build_aux_head`, `_combined_logit`, `_predict_branches`,
`_validate_branches`, and `_format_epoch_log` to score from the within-pool
ONLY, without the cluster path.
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

from my_code.models.pmp_v1.v1_1.pmp_v1_1_trainer import (  # noqa: E402
    _PerModeEmerGNN_PMP_v1_1, DEFAULT_CLUSTER_CACHE,
)
from my_code.models.pmp_v1.v1_4.cluster_head import PMPv1_4Module  # noqa: E402


class _PerModeEmerGNN_PMP_v1_4(_PerModeEmerGNN_PMP_v1_1):
    """v1.4 trainer: Layer 2 (within-cluster pool) only.

    Inherits all args from v1.1. Does NOT use pair-conditional per-drug
    per-cluster mediator tensors at training time, since the cluster path is
    dropped entirely.
    """

    # ------------------------------------------------------------------
    # _build_aux_head: PMPv1_4Module (within-pool only)
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

        module = PMPv1_4Module(
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
        # v1.4 has no cluster_embed; nothing to init from members.
        return module

    # ------------------------------------------------------------------
    # _combined_logit: within-pool-only forward
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

        affinity_a = self._cluster_affinity_t[a_idx]  # (B, K)
        affinity_b = self._cluster_affinity_t[b_idx]

        # Intersection mediators for within-pool (uses v1.1's helper).
        (
            med_idx_per_cluster,
            rel_a_per_cluster,
            rel_b_per_cluster,
            mask_int_per_cluster,
        ) = self._build_per_cluster_mediator_tensors(batch_df)

        # v1.4 forward takes affinity (for p_a, p_b) + intersection tensors only.
        score, _z_within = self._aux_mlp(
            affinity_a, affinity_b,
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_int_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
        )

        zero = torch.zeros_like(score)
        return score, zero, zero

    # ------------------------------------------------------------------
    # _predict_branches override — v1.4 has only the within-pool branch.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_branches(self, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_branches_v1_4(pairs)

    @torch.no_grad()
    def _predict_branches_v1_4(
        self, pairs: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_probs, zeros, combined_probs_copy).

        v1.4 has no cluster path to decompose, so within_only_probs is a
        duplicate of combined for diagnostic-slot compatibility, and the
        cluster_only slot is zeros.
        """
        if self._model is None or self._aux_mlp is None:
            raise RuntimeError("PMP v1.4 trainer not yet fitted")
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

            score, _z_within = self._aux_mlp(
                affinity_a, affinity_b,
                med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
                mask_int_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
            )
            n = len(batch)
            c_out[start:start + n] = torch.sigmoid(score).detach().cpu().numpy()

        zeros = np.zeros_like(c_out)
        return c_out, zeros, c_out.copy()

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
        c, _, _ = self._predict_branches_v1_4(all_pairs)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        v_combined = float(roc_auc_score(y, c))
        return {
            "combined": v_combined,
            "emergnn": float("nan"),
            "aux": v_combined,
            "cluster_only": float("nan"),
            "within_only": v_combined,
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
            f"[pmp-v1.4] [ep {epoch + 1}/{self.n_epochs}] "
            f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
            f"val_combined={v_auc:.4f} "
            f"(layer2-only; no cluster path) "
            f"beta={beta_val:.3f}"
        )


__all__ = [
    "_PerModeEmerGNN_PMP_v1_4",
    "DEFAULT_CLUSTER_CACHE",
]
