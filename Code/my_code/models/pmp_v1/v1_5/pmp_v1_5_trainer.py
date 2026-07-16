"""PMP v1.5 trainer (created 2026-06-03).

Sequential cluster-attention -> within-pool variant with prior + content
residual attention (Codex Option B). Inherits the v1.2 trainer because v1.5
needs the per-drug per-cluster mediator map (for H_a^k / H_b^k pair-conditional
pool used inside Step 1's content residual).

Overrides:
  - `_build_aux_head` to build PMPv1_5Module.
  - `_combined_logit` to build BOTH per-drug per-cluster tensors (Step 1
    content residual) AND intersection mediator tensors (Step 2 within-pool),
    then call the v1.5 forward.
  - `_predict_branches` / `_predict_branches_v1_5` for diagnostic 3-tuple.
  - `_validate_branches` to return single-branch dict (combined only) with
    legacy keys preserved for backward compatibility.
  - `_format_epoch_log` for correctly-labeled `[pmp-v1.5]` log line.
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
from my_code.models.pmp_v1.v1_5.cluster_head import PMPv1_5Module  # noqa: E402


class _PerModeEmerGNN_PMP_v1_5(_PerModeEmerGNN_PMP_v1_2):
    """v1.5 trainer: sequential cluster-attention -> within-pool with content
    residual attention.

    Inherits all args from v1.2, including `cluster_max_mediators_per_drug`
    (controls per-drug per-cluster mediator cap used for H_a / H_b
    AttnPool).

    Additional args:
      content_hidden: hidden width of the content residual MLP `g` (default 16).
    """

    def __init__(
        self,
        *,
        content_hidden: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.content_hidden = int(content_hidden)

    # ------------------------------------------------------------------
    # _build_aux_head: PMPv1_5Module
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

        module = PMPv1_5Module(
            n_mediators=self._pmp_n_mediators,
            n_types=self._pmp_n_types,
            n_rels_plus_special=self._pmp_n_rels_total,
            n_clusters=self._cluster_n_clusters,
            d=self.n_dim,
            type_dim=None,
            rel_dim=None,
            hidden=self.cluster_hidden,
            content_hidden=self.content_hidden,
            dropout=self.cluster_dropout,
        )
        return module

    # ------------------------------------------------------------------
    # _combined_logit: full v1.5 forward (Step 1 content-residual attn + Step 2
    # attention-marginal weighted within-pool)
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

        # Step 1 inputs: per-drug per-cluster mediators (for H_a / H_b).
        med_a_per_cluster, mask_a_per_cluster = (
            self._build_per_drug_per_cluster_tensors(a_list)
        )
        med_b_per_cluster, mask_b_per_cluster = (
            self._build_per_drug_per_cluster_tensors(b_list)
        )

        # Step 2 inputs: intersection mediators (for within-pool).
        (
            med_idx_per_cluster,
            rel_a_per_cluster,
            rel_b_per_cluster,
            mask_int_per_cluster,
        ) = self._build_per_cluster_mediator_tensors(batch_df)

        score, _z_evidence, _alpha, _attn_a, _attn_b = self._aux_mlp(
            affinity_a, affinity_b,
            med_a_per_cluster, mask_a_per_cluster,
            med_b_per_cluster, mask_b_per_cluster,
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_int_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
        )

        zero = torch.zeros_like(score)
        return score, zero, zero

    # ------------------------------------------------------------------
    # _predict_branches override — v1.5 has a single fused branch.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_branches(self, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_branches_v1_5(pairs)

    @torch.no_grad()
    def _predict_branches_v1_5(
        self, pairs: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_probs, combined_probs_copy, zeros).

        v1.5 has no separate parallel branch to decompose; the cluster
        attention modulates a single forward path through within-pool.
        """
        if self._model is None or self._aux_mlp is None:
            raise RuntimeError("PMP v1.5 trainer not yet fitted")
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

            med_a_per_cluster, mask_a_per_cluster = (
                self._build_per_drug_per_cluster_tensors(a_list)
            )
            med_b_per_cluster, mask_b_per_cluster = (
                self._build_per_drug_per_cluster_tensors(b_list)
            )
            (
                med_idx_per_cluster, rel_a_per_cluster,
                rel_b_per_cluster, mask_int_per_cluster,
            ) = self._build_per_cluster_mediator_tensors(batch)

            score, _z_e, _a, _aa, _ab = self._aux_mlp(
                affinity_a, affinity_b,
                med_a_per_cluster, mask_a_per_cluster,
                med_b_per_cluster, mask_b_per_cluster,
                med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
                mask_int_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
            )
            n = len(batch)
            c_out[start:start + n] = torch.sigmoid(score).detach().cpu().numpy()

        zeros = np.zeros_like(c_out)
        return c_out, c_out.copy(), zeros

    # ------------------------------------------------------------------
    # _validate_branches override — single-branch dict (combined only)
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
                "lambda_content": float("nan"),
            }
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, _, _ = self._predict_branches_v1_5(all_pairs)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        v_combined = float(roc_auc_score(y, c))
        # Read current lambda_content value for diagnostic.
        try:
            lam = float(
                self._aux_mlp.lambda_content.detach().cpu().item()  # type: ignore[attr-defined]
            )
        except Exception:
            lam = float("nan")
        return {
            "combined": v_combined,
            # Legacy keys (kept for any parent code that may still read them).
            "emergnn": v_combined,
            "aux": float("nan"),
            # v1.5-specific diagnostic.
            "lambda_content": lam,
        }

    # ------------------------------------------------------------------
    # _format_epoch_log override — single-branch log line + lambda diagnostic
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
        lam = br.get("lambda_content", float("nan"))
        return (
            f"[pmp-v1.5] [ep {epoch + 1}/{self.n_epochs}] "
            f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
            f"val_combined={v_auc:.4f} "
            f"lambda_content={lam:.4f} "
            f"(sequential cluster-attn -> within-pool; "
            f"prior + content residual) "
            f"beta={beta_val:.3f}"
        )


__all__ = [
    "_PerModeEmerGNN_PMP_v1_5",
    "DEFAULT_CLUSTER_CACHE",
]
