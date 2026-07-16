"""PMP v1.1 trainer (rewrite 2026-06-02).

Architectural departure from v1.0 PMP. The score function is REPLACED entirely
by the v1.1 cluster path + per-cluster within-pool + final MLP. EmerGNN
backbone scoring and v1 PMP residual fusion are bypassed.

Score function:

    z_cluster_pair = MLP_pair_cluster(h_path)
        with h_path = sum_{i,j} alpha_ij * psi(i, j, a, b)
        and  alpha_ij = p_a[i] * T[i,j] * p_b[j] / sum_{u,v} p_a[u] * T[u,v] * p_b[v]
        T = softmax(W_T)
        psi includes cluster_embed[i], cluster_embed[j], elementwise, plus
        r_a_to_i and r_j_to_b expected relation embeddings.

    z_within = sum_k (p_a[k] * p_b[k]) * z_within^k
        with z_within^k = AttnPool over members(cluster_k) ∩ N_<=2(a) ∩ N_<=2(b)

    z_pair = concat([z_cluster_pair, z_within])
    score  = MLP_score(z_pair)

Trainer inheritance: keeps the v1 inheritance chain (_PerModeEmerGNN_MNAH) so
the parent fit() loop, shuffle_train(mode='S2') protocol, and optimizer setup
all stay intact. The override:

  - _load_feature_cache: loads BOTH the PMP mediator cache (for mediator2id +
    n1/n2 lookup) and the v1.1 cluster cache v2 (drug2id, drug_cluster_affinity,
    drug_cluster_rel_dist, cluster_members).
  - _build_aux_head: returns the PMPv1_1Module and initializes cluster_embed
    from member mediator means before returning.
  - _combined_logit: computes the v1.1 score and returns (score, zeros, zeros)
    so the parent fit() signature is satisfied. EmerGNN backbone and v1 PMP
    fusion are not summed into the score; their parameters still exist and
    remain in the optimizer but will receive zero gradient.
  - _predict_branches: returns (score_probs, zeros, zeros) since v1's branch
    semantics no longer apply. A new method _predict_branches_v1_1 returns
    (combined_probs, cluster_only_probs, within_only_probs) approximations
    for diagnostic logging.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

_FILE = Path(__file__).resolve()
# v1_1/ lives under Code/my_code/models/pmp_v1/, so parents[5] = project root.
PROJECT_ROOT = _FILE.parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.pmp_v1.pmp_trainer import (  # noqa: E402
    _PerModeEmerGNN_PMP, DEFAULT_PMP_CACHE,
)
from my_code.models.pmp_v1.v1_1.cluster_head import PMPv1_1Module  # noqa: E402


DEFAULT_CLUSTER_CACHE = (
    PROJECT_ROOT
    / "Code/data/_cache/pmp_v1_1_cluster_cache_drugbank_seed42_kgonly_v2.pkl"
)

_ACCEPTED_CLUSTER_SCHEMAS = {"pmp_v1_1_cluster_v2"}
_ACCEPTED_PMP_SCHEMAS = {"pmp_v1_codex_r2"}


class _PerModeEmerGNN_PMP_v1_1(_PerModeEmerGNN_PMP):
    """v1.1 trainer with full architectural replacement of v1 scoring.

    Args (beyond v1):
      cluster_cache_path        : path to v1.1 cluster cache v2 pickle.
      cluster_hidden            : MLP hidden dim inside PMPv1_1Module
                                  (default 128, same as v1).
      cluster_dropout           : dropout inside PMPv1_1Module (default 0.2).

    Inherits all other args from v1 _PerModeEmerGNN_PMP (which itself accepts
    pmp_cache_path, pmp_hidden, pmp_dropout, pmp_max_mediators). v1.1 still
    uses pmp_cache_path to load mediator vocab and per-drug 1-hop / 2-hop
    mediator lists. The v1 PMP head's separate residual fusion is NOT used,
    but the cache contents are reused.
    """

    def __init__(
        self,
        *,
        cluster_cache_path: str | None = None,
        cluster_hidden: int = 128,
        cluster_dropout: float = 0.2,
        **kwargs,
    ) -> None:
        # Force v1's PMP residual beta to effectively zero. Init beta = 0
        # routes to raw_init = -5.0 via the v1 guard, so softplus(beta) ≈ 7e-3.
        # We will further bypass v1's PMP contribution in _combined_logit, so
        # this is belt-and-suspenders.
        kwargs.setdefault("mnah_init_beta", 0.0)
        kwargs.setdefault("mnah_text_cache", None)
        super().__init__(**kwargs)

        self.cluster_cache_path = cluster_cache_path or str(DEFAULT_CLUSTER_CACHE)
        self.cluster_hidden = int(cluster_hidden)
        self.cluster_dropout = float(cluster_dropout)

        # Cluster cache state populated in _load_feature_cache.
        self._cluster_n_clusters: int | None = None
        self._cluster_n_drugs: int | None = None
        self._cluster_drug2id: dict[str, int] | None = None
        self._cluster_member_pmp_ids: dict[int, list[int]] | None = None
        # Tensors holding (n_drugs, K) and (n_drugs, K, n_rels+1) inputs.
        # Moved to model device lazily on first forward.
        self._cluster_affinity_t: torch.Tensor | None = None
        self._cluster_rel_dist_t: torch.Tensor | None = None
        # type_id list for the 12 clusters (range(n_clusters))
        self._type_id_per_cluster: list[int] | None = None

    # ------------------------------------------------------------------
    # _load_feature_cache: load both PMP cache (mediator2id + n1/n2) and
    # v1.1 cluster cache v2
    # ------------------------------------------------------------------

    def _load_feature_cache(
        self,
        train_pos_pairs: pd.DataFrame,
        train_neg_pairs: pd.DataFrame | None = None,
    ) -> None:
        # Reuse v1's loader for the PMP mediator cache so we get
        # self._pmp_n1, self._pmp_n2, self._pmp_mediator2id, etc.
        super()._load_feature_cache(train_pos_pairs, train_neg_pairs)

        # Codex r1 fix: re-open the PMP cache to extract type2id and rel2id
        # so we can validate consistency with the v1.1 cluster cache below.
        # v1's loader does not stash these as attributes.
        with open(self.pmp_cache_path, "rb") as f:
            pmp_payload = pickle.load(f)
        pmp_type2id = dict(pmp_payload.get("type2id", {}))
        pmp_kind_order = list(pmp_payload.get("kind_order", []))
        pmp_rel2id = dict(pmp_payload.get("rel2id", {}))

        # Load v1.1 cluster cache v2.
        cache_path = Path(self.cluster_cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"PMP v1.1 cluster cache (schema v2) not found at {cache_path}. "
                f"Run Code/scripts/precompute_pmp_v1_1_cluster_cache.py first."
            )
        print(f"[pmp-v1.1] loading cluster cache: {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        schema = payload.get("schema_version", "")
        if schema not in _ACCEPTED_CLUSTER_SCHEMAS:
            raise RuntimeError(
                f"v1.1 cluster cache at {cache_path} has unsupported "
                f"schema_version={schema!r}. Accepted: "
                f"{sorted(_ACCEPTED_CLUSTER_SCHEMAS)}. Rebuild via "
                f"`python Code/scripts/precompute_pmp_v1_1_cluster_cache.py --force`."
            )

        # Codex r1 fix: validate type2id, kind_order, rel2id agree between PMP
        # and v1.1 cluster caches. If PMP cache lacks "kind_order" (older PMP
        # cache versions did not store it), fall back to just type2id +
        # rel2id checks. Mismatch means cluster k != type id k under one of
        # the two caches, which would silently misread cluster IDs and
        # relation channels. Refuse to proceed.
        cluster_type2id = dict(payload.get("type2id", {}))
        cluster_kind_order = list(payload.get("kind_order", []))
        cluster_rel2id = dict(payload.get("rel2id", {}))
        if pmp_type2id and cluster_type2id and pmp_type2id != cluster_type2id:
            raise RuntimeError(
                f"[pmp-v1.1] type2id mismatch between PMP cache and v1.1 cluster cache. "
                f"PMP type2id keys: {sorted(pmp_type2id.keys())[:5]}... "
                f"Cluster type2id keys: {sorted(cluster_type2id.keys())[:5]}... "
                f"Rebuild BOTH caches against the same KG parquet."
            )
        if pmp_kind_order and cluster_kind_order and pmp_kind_order != cluster_kind_order:
            raise RuntimeError(
                f"[pmp-v1.1] kind_order mismatch between PMP cache and v1.1 cluster cache. "
                f"Rebuild BOTH caches against the same KG parquet."
            )
        if pmp_rel2id and cluster_rel2id and pmp_rel2id != cluster_rel2id:
            raise RuntimeError(
                f"[pmp-v1.1] rel2id mismatch between PMP cache and v1.1 cluster cache. "
                f"Rebuild BOTH caches against the same KG parquet."
            )

        self._cluster_n_clusters = int(payload["n_clusters"])
        self._cluster_n_drugs = int(payload["n_drugs"])
        self._cluster_drug2id = dict(payload["drug2id"])

        affinity = np.asarray(payload["drug_cluster_affinity"], dtype=np.float32)
        rel_dist = np.asarray(payload["drug_cluster_rel_dist"], dtype=np.float32)
        # Shape validation
        if affinity.shape != (
            self._cluster_n_drugs, self._cluster_n_clusters
        ):
            raise ValueError(
                f"cluster affinity shape {affinity.shape} != "
                f"({self._cluster_n_drugs}, {self._cluster_n_clusters})"
            )
        # rel_dist shape (n_drugs, n_clusters, n_rels_plus_special) — verify against
        # PMP cache's n_rels_plus_special (must match for rel_embed lookups).
        if rel_dist.shape != (
            self._cluster_n_drugs,
            self._cluster_n_clusters,
            self._pmp_n_rels_total,
        ):
            raise ValueError(
                f"cluster rel_dist shape {rel_dist.shape} != "
                f"({self._cluster_n_drugs}, {self._cluster_n_clusters}, "
                f"{self._pmp_n_rels_total}). PMP cache and v1.1 cluster cache "
                f"are inconsistent — rebuild both."
            )
        self._cluster_affinity_t = torch.from_numpy(affinity)
        self._cluster_rel_dist_t = torch.from_numpy(rel_dist)

        # Map cluster_members[k] (KG node id strings) to PMP mediator IDs.
        # Members not in the PMP mediator vocab are silently dropped (they are
        # not reachable via PMP's mediator2id and would never appear in M_k
        # anyway).
        cluster_members_str = payload["cluster_members"]
        self._cluster_member_pmp_ids = {}
        n_dropped = 0
        for k_str, mediator_ids in cluster_members_str.items():
            k = int(k_str)
            pmp_ids: list[int] = []
            for m_id in mediator_ids:
                pmp_idx = self._pmp_mediator2id.get(m_id)
                if pmp_idx is None:
                    n_dropped += 1
                    continue
                pmp_ids.append(int(pmp_idx))
            self._cluster_member_pmp_ids[k] = pmp_ids

        self._type_id_per_cluster = list(range(self._cluster_n_clusters))

        # Coverage diagnostic. Include train_neg_pairs (else KeyError mid-train).
        train_drugs = set(
            train_pos_pairs["drug_a_id"].astype(str).tolist()
            + train_pos_pairs["drug_b_id"].astype(str).tolist()
        )
        if train_neg_pairs is not None:
            train_drugs |= set(
                train_neg_pairs["drug_a_id"].astype(str).tolist()
                + train_neg_pairs["drug_b_id"].astype(str).tolist()
            )
        missing = train_drugs - set(self._cluster_drug2id.keys())
        if missing:
            raise RuntimeError(
                f"[pmp-v1.1] FATAL: {len(missing)} train drugs missing from "
                f"cluster drug2id. Rebuild cluster cache. Examples: "
                f"{sorted(missing)[:5]}"
            )

        member_counts = {k: len(v) for k, v in self._cluster_member_pmp_ids.items()}
        print(
            f"[pmp-v1.1] cluster cache stats: n_clusters={self._cluster_n_clusters} "
            f"n_drugs={self._cluster_n_drugs} "
            f"member_pmp_count={member_counts} "
            f"dropped_member_ids_not_in_pmp_vocab={n_dropped}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # _build_aux_head: construct PMPv1_1Module and init cluster_embed
    # ------------------------------------------------------------------

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        if (self._cluster_n_clusters is None or self._cluster_member_pmp_ids is None
                or self._pmp_n_mediators is None or self._pmp_n_types is None
                or self._pmp_n_rels_total is None):
            raise RuntimeError(
                "Caches must be loaded before _build_aux_head; call "
                "_load_feature_cache first."
            )

        module = PMPv1_1Module(
            n_mediators=self._pmp_n_mediators,
            n_types=self._pmp_n_types,
            n_rels_plus_special=self._pmp_n_rels_total,
            n_clusters=self._cluster_n_clusters,
            d=self.n_dim,
            type_dim=None,  # default = max(d // 4, 8)
            rel_dim=None,
            hidden=self.cluster_hidden,
            dropout=self.cluster_dropout,
        )
        # Initialize cluster_embed[k] = mean(mediator_embed[m] for m in members(k)).
        # Mediator embed has just been Xavier-init'd in PMPv1_1Module.__init__.
        # Both are on CPU at this point; parent moves the whole module to device
        # afterward.
        module.init_cluster_embed_from_members(self._cluster_member_pmp_ids)
        return module

    # ------------------------------------------------------------------
    # Helper: move cluster caches to device (once, lazily)
    # ------------------------------------------------------------------

    def _ensure_caches_on_device(self) -> None:
        if (self._cluster_affinity_t is not None
                and self._cluster_affinity_t.device != self.device):
            self._cluster_affinity_t = self._cluster_affinity_t.to(self.device)
        if (self._cluster_rel_dist_t is not None
                and self._cluster_rel_dist_t.device != self.device):
            self._cluster_rel_dist_t = self._cluster_rel_dist_t.to(self.device)

    # ------------------------------------------------------------------
    # Helper: build per-cluster padded mediator tensors for a batch
    # ------------------------------------------------------------------

    def _build_per_cluster_mediator_tensors(
        self, batch_df: pd.DataFrame
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Gather per-cluster mediator tensors for the batch.

        For each pair, calls v1's `_gather_mediators_for_pair` which already
        returns the merged 1-hop + 2-hop mediator list with type IDs and
        relation IDs. Then groups mediators by cluster (type_id) and pads.

        Returns: (med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
                  mask_per_cluster)
        each a list of length n_clusters, where the k-th entry is a (B, M_max_k)
        tensor on self.device.
        """
        device = self.device
        B = len(batch_df)
        K = self._cluster_n_clusters  # type: ignore[arg-type]
        a_list = batch_df["drug_a_id"].astype(str).tolist()
        b_list = batch_df["drug_b_id"].astype(str).tolist()

        # First, gather all mediators per pair via v1's helper.
        per_pair: list[tuple[list[int], list[int], list[int], list[int]]] = []
        for a, b in zip(a_list, b_list):
            ent_ids, type_ids, rel_a_ids, rel_b_ids = (
                self._gather_mediators_for_pair(a, b)
            )
            per_pair.append((ent_ids, type_ids, rel_a_ids, rel_b_ids))

        # For each cluster k, collect per-pair mediators of that type.
        med_idx_per_cluster: list[torch.Tensor] = []
        rel_a_per_cluster: list[torch.Tensor] = []
        rel_b_per_cluster: list[torch.Tensor] = []
        mask_per_cluster: list[torch.Tensor] = []

        for k in range(K):
            per_pair_med: list[list[int]] = []
            per_pair_rel_a: list[list[int]] = []
            per_pair_rel_b: list[list[int]] = []
            M_max_k = 0
            for (ent_ids, type_ids, rel_a_ids, rel_b_ids) in per_pair:
                indices = [i for i in range(len(ent_ids)) if type_ids[i] == k]
                med_k = [ent_ids[i] for i in indices]
                ra_k = [rel_a_ids[i] for i in indices]
                rb_k = [rel_b_ids[i] for i in indices]
                per_pair_med.append(med_k)
                per_pair_rel_a.append(ra_k)
                per_pair_rel_b.append(rb_k)
                M_max_k = max(M_max_k, len(med_k))
            M_max_k = max(M_max_k, 1)  # ensure tensor shape is well-defined

            med_t = torch.zeros((B, M_max_k), dtype=torch.long, device=device)
            ra_t = torch.zeros((B, M_max_k), dtype=torch.long, device=device)
            rb_t = torch.zeros((B, M_max_k), dtype=torch.long, device=device)
            mask_t = torch.zeros((B, M_max_k), dtype=torch.bool, device=device)
            for i in range(B):
                n = len(per_pair_med[i])
                if n == 0:
                    continue
                med_t[i, :n] = torch.tensor(
                    per_pair_med[i], dtype=torch.long, device=device,
                )
                ra_t[i, :n] = torch.tensor(
                    per_pair_rel_a[i], dtype=torch.long, device=device,
                )
                rb_t[i, :n] = torch.tensor(
                    per_pair_rel_b[i], dtype=torch.long, device=device,
                )
                mask_t[i, :n] = True
            med_idx_per_cluster.append(med_t)
            rel_a_per_cluster.append(ra_t)
            rel_b_per_cluster.append(rb_t)
            mask_per_cluster.append(mask_t)

        return (
            med_idx_per_cluster,
            rel_a_per_cluster,
            rel_b_per_cluster,
            mask_per_cluster,
        )

    # ------------------------------------------------------------------
    # _combined_logit: replace EmerGNN scoring entirely with v1.1 score
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
        """Compute v1.1 score directly; ignore EmerGNN backbone and v1 PMP fusion.

        Returns (score, zero, zero) so the parent fit() loop and
        _predict_branches consumers (which expect 3 tensors) still work.
        EmerGNN backbone and v1 PMP residual params still exist but their
        gradient contribution to the score is zero — they will not update.
        """
        device = self.device
        self._ensure_caches_on_device()

        # Drug indices into cluster cache.
        a_list = batch_df["drug_a_id"].astype(str).tolist()
        b_list = batch_df["drug_b_id"].astype(str).tolist()
        try:
            a_indices = [self._cluster_drug2id[d] for d in a_list]
            b_indices = [self._cluster_drug2id[d] for d in b_list]
        except KeyError as exc:
            raise KeyError(
                f"Drug id {exc.args[0]!r} not in cluster drug2id. Rebuild "
                f"cluster cache against a KG that contains all eval/train drugs."
            ) from exc
        a_idx = torch.tensor(a_indices, dtype=torch.long, device=device)
        b_idx = torch.tensor(b_indices, dtype=torch.long, device=device)

        # Cluster-path inputs.
        affinity_a = self._cluster_affinity_t[a_idx]  # (B, K)
        affinity_b = self._cluster_affinity_t[b_idx]
        rel_dist_a = self._cluster_rel_dist_t[a_idx]  # (B, K, n_rels+1)
        rel_dist_b = self._cluster_rel_dist_t[b_idx]

        # Per-cluster within-pool inputs.
        (
            med_idx_per_cluster,
            rel_a_per_cluster,
            rel_b_per_cluster,
            mask_per_cluster,
        ) = self._build_per_cluster_mediator_tensors(batch_df)

        # Forward.
        score, _z_cluster_pair, _z_within = self._aux_mlp(
            affinity_a, affinity_b, rel_dist_a, rel_dist_b,
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
        )

        # Return v1.1 score in the "combined" slot and zeros for the unused
        # emergnn_logit / pmp_logit slots so parent infra continues to work.
        zero = torch.zeros_like(score)
        return score, zero, zero

    # ------------------------------------------------------------------
    # _predict_branches override (codex r1 fix). MNAH parent reports three
    # branches as (combined, emergnn, aux); v1.1 reinterprets them as
    # (combined, cluster_only, within_only) so the training-time val log
    # surfaces v1.1-specific signal instead of meaningless 0.5 sigmoid-of-zero
    # values for the unused emergnn / pmp residual branches.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_branches(self, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_branches_v1_1(pairs)

    # ------------------------------------------------------------------
    # Extended diagnostic: per-component probabilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_branches_v1_1(
        self, pairs: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_probs, cluster_only_probs, within_only_probs).

        cluster_only_probs and within_only_probs are computed by zeroing out the
        complementary component going into MLP_score, then taking sigmoid. This
        is a rough decomposition for diagnostic purposes (since MLP_score is
        not linear, the two parts do not sum to combined).
        """
        if self._model is None or self._aux_mlp is None:
            raise RuntimeError("PMP v1.1 trainer not yet fitted")
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            edge_src, edge_dst, edge_rel = self._eval_edges
        else:
            edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval()
        self._aux_mlp.eval()
        self._ensure_caches_on_device()

        c_out = np.empty(len(pairs), dtype=np.float32)
        cluster_only_out = np.empty(len(pairs), dtype=np.float32)
        within_only_out = np.empty(len(pairs), dtype=np.float32)

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head = head.to(self.device); tail = tail.to(self.device)

            # Build inputs ourselves so we can compute the per-component
            # diagnostics. Mirrors _combined_logit setup.
            a_list = batch["drug_a_id"].astype(str).tolist()
            b_list = batch["drug_b_id"].astype(str).tolist()
            a_idx = torch.tensor(
                [self._cluster_drug2id[d] for d in a_list],
                dtype=torch.long, device=self.device,
            )
            b_idx = torch.tensor(
                [self._cluster_drug2id[d] for d in b_list],
                dtype=torch.long, device=self.device,
            )
            affinity_a = self._cluster_affinity_t[a_idx]
            affinity_b = self._cluster_affinity_t[b_idx]
            rel_dist_a = self._cluster_rel_dist_t[a_idx]
            rel_dist_b = self._cluster_rel_dist_t[b_idx]
            (
                med_idx_per_cluster, rel_a_per_cluster,
                rel_b_per_cluster, mask_per_cluster,
            ) = self._build_per_cluster_mediator_tensors(batch)

            score, z_cluster_pair, z_within = self._aux_mlp(
                affinity_a, affinity_b, rel_dist_a, rel_dist_b,
                med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
                mask_per_cluster, self._type_id_per_cluster,  # type: ignore[arg-type]
            )
            # Cluster-only: replace z_within with zeros
            z_within_zero = torch.zeros_like(z_within)
            z_pair_cluster_only = torch.cat(
                [z_cluster_pair, z_within_zero], dim=-1
            )
            cluster_only_logit = self._aux_mlp.mlp_score(z_pair_cluster_only).squeeze(-1)
            # Within-only: replace z_cluster_pair with zeros
            z_cluster_zero = torch.zeros_like(z_cluster_pair)
            z_pair_within_only = torch.cat([z_cluster_zero, z_within], dim=-1)
            within_only_logit = self._aux_mlp.mlp_score(z_pair_within_only).squeeze(-1)

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
    "_PerModeEmerGNN_PMP_v1_1",
    "DEFAULT_CLUSTER_CACHE",
]
