"""PMP v1.2 trainer (created 2026-06-03).

Inherits from v1.1's `_PerModeEmerGNN_PMP_v1_1`. The only architectural change
is in the cluster path:

  v1.1:  e_i = cluster_embed.weight[i]                       (fixed, shared)
  v1.2:  e_i = AttnPool({mediator_embed[m] : m ∈ members(i) ∩ N(a)})  (pair-conditional)

To support pair-conditional e_i, v1.2 builds a per-drug per-cluster mediator
index map at cache load time (`self._drug_per_cluster_mediators`), and at every
forward gathers (B, M_max_k) tensors of drug-a's and drug-b's cluster-k
mediators (separately, NOT the intersection).

The within-cluster pool, EmerGNN backbone bypass, and cluster cache schema are
unchanged from v1.1.

Optional cap `cluster_max_mediators_per_drug` (default 64) bounds the per-drug
per-cluster mediator list size at load time. Deterministic truncation by PMP
mediator id ascending (reproducibility). Set to None to disable.

Diagnostic API change:
  - `_predict_branches` overridden to return (combined, cluster_only, within_only)
    via `_predict_branches_v1_2` (parallels v1.1's `_predict_branches_v1_1`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_FILE = Path(__file__).resolve()
# v1_2/ lives under Code/my_code/models/pmp_v1/, so parents[5] = project root.
PROJECT_ROOT = _FILE.parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.pmp_v1.v1_1.pmp_v1_1_trainer import (  # noqa: E402
    _PerModeEmerGNN_PMP_v1_1, DEFAULT_CLUSTER_CACHE,
)
from my_code.models.pmp_v1.v1_2.cluster_head import PMPv1_2Module  # noqa: E402


class _PerModeEmerGNN_PMP_v1_2(_PerModeEmerGNN_PMP_v1_1):
    """v1.2 trainer with pair-conditional cluster representations.

    Beyond v1.1 args:
      cluster_max_mediators_per_drug: int | None
        Maximum number of mediators kept per (drug, cluster) for the
        pair-conditional cluster pool. Deterministic truncation by ascending
        PMP mediator id. Default 64. Set to None to disable cap.

    Inherits all other args (and cluster cache loader) from v1.1 trainer.
    """

    def __init__(
        self,
        *,
        cluster_max_mediators_per_drug: int | None = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if cluster_max_mediators_per_drug is not None:
            cap_int = int(cluster_max_mediators_per_drug)
            if cap_int <= 0:
                raise ValueError(
                    f"cluster_max_mediators_per_drug must be positive or None, "
                    f"got {cluster_max_mediators_per_drug!r}"
                )
            self.cluster_max_mediators_per_drug: int | None = cap_int
        else:
            self.cluster_max_mediators_per_drug = None

        # Built in _load_feature_cache (after v1.1's loader populates PMP n1/n2
        # and cluster_n_clusters): drug_id -> list of K lists, each holding
        # PMP indices of cluster-k mediators that are 1-hop or 2-hop neighbors
        # of the drug. Sorted ascending by PMP id; capped at
        # cluster_max_mediators_per_drug if set.
        self._drug_per_cluster_mediators: dict[str, list[list[int]]] | None = None

    # ------------------------------------------------------------------
    # _load_feature_cache: extend v1.1 to precompute per-drug per-cluster
    # mediator lists.
    # ------------------------------------------------------------------

    def _load_feature_cache(
        self,
        train_pos_pairs: pd.DataFrame,
        train_neg_pairs: pd.DataFrame | None = None,
    ) -> None:
        super()._load_feature_cache(train_pos_pairs, train_neg_pairs)

        if (
            self._cluster_n_clusters is None
            or self._pmp_n1 is None
            or self._pmp_n2 is None
            or self._pmp_mediator2id is None
        ):
            raise RuntimeError(
                "[pmp-v1.2] super()._load_feature_cache did not populate the "
                "expected fields; cannot build per-drug per-cluster index map."
            )

        K = int(self._cluster_n_clusters)
        cap = self.cluster_max_mediators_per_drug

        per_drug: dict[str, list[list[int]]] = {}
        # Codex r1 fix: include every drug from cluster_drug2id, NOT just drugs
        # present in PMP n1/n2. A cluster-cache-resident drug with zero 1-hop
        # and zero 2-hop mediators (e.g. extremely isolated node) would
        # otherwise miss from `per_drug` and trigger KeyError at batch time;
        # the spec says empty M_k(a) -> H_a[:, k, :] = 0, which means an
        # all-empty per-cluster list is the correct, well-defined fallback.
        # cluster_drug2id is populated by v1.1 loader (super() ran above), so
        # union with PMP n1/n2 keys covers any drug we might evaluate.
        all_drugs: set[str] = (
            set(self._pmp_n1.keys())
            | set(self._pmp_n2.keys())
            | set((self._cluster_drug2id or {}).keys())
        )
        n_capped_cells = 0
        n_total_refs = 0
        n_empty_drugs = 0
        max_count_per_cluster = [0] * K
        for d in all_drugs:
            per_cluster: list[list[int]] = [[] for _ in range(K)]
            seen_med_ids: set[str] = set()
            # 1-hop: (m_id, type_id, rel_id). 1-hop takes precedence over 2-hop
            # so seen_med_ids guards against double counting.
            for m_id, type_id, _rel_id in self._pmp_n1.get(d, []):
                if m_id in seen_med_ids:
                    continue
                seen_med_ids.add(m_id)
                pmp_idx = self._pmp_mediator2id.get(m_id)
                if pmp_idx is None:
                    continue
                k = int(type_id)
                if 0 <= k < K:
                    per_cluster[k].append(int(pmp_idx))
            # 2-hop: (m_id, type_id)
            for m_id, type_id in self._pmp_n2.get(d, []):
                if m_id in seen_med_ids:
                    continue
                seen_med_ids.add(m_id)
                pmp_idx = self._pmp_mediator2id.get(m_id)
                if pmp_idx is None:
                    continue
                k = int(type_id)
                if 0 <= k < K:
                    per_cluster[k].append(int(pmp_idx))

            # Deterministic truncation: sort ascending by PMP id, then cap.
            drug_total = 0
            for k in range(K):
                per_cluster[k].sort()
                if cap is not None and len(per_cluster[k]) > cap:
                    n_capped_cells += 1
                    per_cluster[k] = per_cluster[k][:cap]
                cnt = len(per_cluster[k])
                drug_total += cnt
                n_total_refs += cnt
                if cnt > max_count_per_cluster[k]:
                    max_count_per_cluster[k] = cnt
            if drug_total == 0:
                # Drug has no PMP mediators at all -> H_a for this drug will be
                # zero across all clusters, per spec. Track for diagnostics.
                n_empty_drugs += 1
            per_drug[d] = per_cluster
        self._drug_per_cluster_mediators = per_drug

        print(
            f"[pmp-v1.2] per-drug per-cluster cache built: "
            f"n_drugs={len(per_drug)} "
            f"n_empty_drugs={n_empty_drugs} "
            f"cap={cap} n_capped_cells={n_capped_cells} "
            f"total_mediator_refs={n_total_refs} "
            f"max_count_per_cluster={max_count_per_cluster}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # _build_aux_head: PMPv1_2Module (no cluster_embed init step; v1.2 has none)
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

        module = PMPv1_2Module(
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
        # v1.2 has no cluster_embed parameter; cluster representation is
        # computed pair-conditionally at forward. Nothing to init here.
        return module

    # ------------------------------------------------------------------
    # Helper: build per-drug per-cluster mediator tensors for a list of drugs
    # ------------------------------------------------------------------

    def _build_per_drug_per_cluster_tensors(
        self, drugs: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """For each drug in the list, look up the per-cluster mediator lists
        and pad to (B, M_max_k) per cluster.

        Args:
          drugs: length-B list of drug id strings.

        Returns:
          med_per_cluster:  list of K (B, M_max_k) long tensors on self.device.
          mask_per_cluster: list of K (B, M_max_k) bool tensors on self.device.

        Missing drug ids fail loudly (KeyError) — v1.1's cluster cache loader
        already verifies all train + train-neg drugs are in cluster_drug2id,
        and v1.2's `_load_feature_cache` seeds `_drug_per_cluster_mediators`
        from the UNION of PMP n1/n2 keys AND cluster_drug2id keys (codex r1
        fix). Drugs with zero PMP neighborhood get K empty lists. A
        cluster-cache-resident drug not present in `_drug_per_cluster_mediators`
        therefore signals a cache build mismatch worth catching loudly.
        """
        if self._drug_per_cluster_mediators is None:
            raise RuntimeError(
                "Per-drug per-cluster cache not loaded; call "
                "_load_feature_cache first."
            )
        device = self.device
        B = len(drugs)
        K = int(self._cluster_n_clusters)  # type: ignore[arg-type]

        # Resolve all per-drug entries up front so KeyErrors surface before any
        # tensor allocation.
        per_drug_entries: list[list[list[int]]] = []
        for drug in drugs:
            entry = self._drug_per_cluster_mediators.get(drug)
            if entry is None:
                raise KeyError(
                    f"Drug id {drug!r} not in per-drug per-cluster cache. "
                    f"Rebuild PMP cache (it should cover all eval drugs)."
                )
            per_drug_entries.append(entry)

        med_per_cluster: list[torch.Tensor] = []
        mask_per_cluster: list[torch.Tensor] = []
        for k in range(K):
            # Find M_max_k across the batch (entries are length-K, list[k] is
            # the cluster-k mediator id list for drug i).
            M_max_k = 0
            for entry in per_drug_entries:
                if len(entry[k]) > M_max_k:
                    M_max_k = len(entry[k])
            M_max_k = max(M_max_k, 1)  # keep tensor well-shaped

            med_t = torch.zeros((B, M_max_k), dtype=torch.long, device=device)
            mask_t = torch.zeros((B, M_max_k), dtype=torch.bool, device=device)
            for i, entry in enumerate(per_drug_entries):
                ids = entry[k]
                n = len(ids)
                if n == 0:
                    continue
                med_t[i, :n] = torch.tensor(ids, dtype=torch.long, device=device)
                mask_t[i, :n] = True
            med_per_cluster.append(med_t)
            mask_per_cluster.append(mask_t)
        return med_per_cluster, mask_per_cluster

    # ------------------------------------------------------------------
    # _combined_logit: forward through v1.2 module
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
        rel_dist_a = self._cluster_rel_dist_t[a_idx]  # (B, K, n_rels+1)
        rel_dist_b = self._cluster_rel_dist_t[b_idx]

        # Per-drug per-cluster tensors for pair-conditional repr (NEW vs v1.1).
        med_a_per_cluster, mask_a_per_cluster = (
            self._build_per_drug_per_cluster_tensors(a_list)
        )
        med_b_per_cluster, mask_b_per_cluster = (
            self._build_per_drug_per_cluster_tensors(b_list)
        )

        # Intersection mediators for within-pool (UNCHANGED from v1.1).
        (
            med_idx_per_cluster,
            rel_a_per_cluster,
            rel_b_per_cluster,
            mask_int_per_cluster,
        ) = self._build_per_cluster_mediator_tensors(batch_df)

        score, _z_cluster_pair, _z_within = self._aux_mlp(
            affinity_a, affinity_b, rel_dist_a, rel_dist_b,
            med_a_per_cluster, mask_a_per_cluster,
            med_b_per_cluster, mask_b_per_cluster,
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_int_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
        )

        zero = torch.zeros_like(score)
        return score, zero, zero

    # ------------------------------------------------------------------
    # _predict_branches override (parallels v1.1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_branches(self, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_branches_v1_2(pairs)

    # ------------------------------------------------------------------
    # _validate_branches override — return dict with both legacy keys
    # (combined/emergnn/aux) AND v1.2-semantic keys (cluster_only/within_only).
    # Legacy keys are kept so any code reading br['emergnn'] / br['aux'] sees
    # the v1.2-correct values; the log label fix is done via _format_epoch_log
    # below, not by extra print here.
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
        c, cluster_only, within_only = self._predict_branches_v1_2(all_pairs)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        v_combined = float(roc_auc_score(y, c))
        v_cluster_only = float(roc_auc_score(y, cluster_only))
        v_within_only = float(roc_auc_score(y, within_only))
        return {
            "combined": v_combined,
            # Legacy keys still expected by parent code (now hold v1.2 values).
            "emergnn": v_cluster_only,
            "aux": v_within_only,
            # v1.2-semantic keys — `_format_epoch_log` reads these.
            "cluster_only": v_cluster_only,
            "within_only": v_within_only,
        }

    # ------------------------------------------------------------------
    # _format_epoch_log override — produce a v1.2-correctly-labeled line.
    # Uses cluster_only / within_only keys we stuffed into br above. No more
    # stale val_emer / val_aux strings in the log.
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
            f"[pmp-v1.2] [ep {epoch + 1}/{self.n_epochs}] "
            f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
            f"val_combined={v_auc:.4f} "
            f"val_cluster_only={br['cluster_only']:.4f} "
            f"val_within_only={br['within_only']:.4f} "
            f"beta={beta_val:.3f}"
        )

    @torch.no_grad()
    def _predict_branches_v1_2(
        self, pairs: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_probs, cluster_only_probs, within_only_probs).

        Diagnostic decomposition: zero out one branch's contribution to
        MLP_score input, take sigmoid. Since MLP_score is not linear the two
        do NOT sum to combined; use for relative ranking only.
        """
        if self._model is None or self._aux_mlp is None:
            raise RuntimeError("PMP v1.2 trainer not yet fitted")
        self._model.eval()
        self._aux_mlp.eval()
        self._ensure_caches_on_device()

        c_out = np.empty(len(pairs), dtype=np.float32)
        cluster_only_out = np.empty(len(pairs), dtype=np.float32)
        within_only_out = np.empty(len(pairs), dtype=np.float32)

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
            (
                med_idx_per_cluster, rel_a_per_cluster,
                rel_b_per_cluster, mask_int_per_cluster,
            ) = self._build_per_cluster_mediator_tensors(batch)

            score, z_cluster_pair, z_within = self._aux_mlp(
                affinity_a, affinity_b, rel_dist_a, rel_dist_b,
                med_a_per_cluster, mask_a_per_cluster,
                med_b_per_cluster, mask_b_per_cluster,
                med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
                mask_int_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
            )

            z_within_zero = torch.zeros_like(z_within)
            z_pair_cluster_only = torch.cat(
                [z_cluster_pair, z_within_zero], dim=-1
            )
            cluster_only_logit = self._aux_mlp.mlp_score(
                z_pair_cluster_only
            ).squeeze(-1)

            z_cluster_zero = torch.zeros_like(z_cluster_pair)
            z_pair_within_only = torch.cat([z_cluster_zero, z_within], dim=-1)
            within_only_logit = self._aux_mlp.mlp_score(
                z_pair_within_only
            ).squeeze(-1)

            n = len(batch)
            c_out[start:start + n] = torch.sigmoid(score).detach().cpu().numpy()
            cluster_only_out[start:start + n] = torch.sigmoid(
                cluster_only_logit
            ).detach().cpu().numpy()
            within_only_out[start:start + n] = torch.sigmoid(
                within_only_logit
            ).detach().cpu().numpy()
        return c_out, cluster_only_out, within_only_out


__all__ = [
    "_PerModeEmerGNN_PMP_v1_2",
    "DEFAULT_CLUSTER_CACHE",
]
