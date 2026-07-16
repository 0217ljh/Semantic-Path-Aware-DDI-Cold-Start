"""PMP v1 — Pair-Mediator Pooling trainer (locked Algorithm 1).

Inherits _PerModeEmerGNN_MNAH to reuse:
  - EmerGNN backbone construction
  - shuffle_train(mode='S2') training protocol
  - fit() loop scaffolding
  - per-branch logging utilities

Overrides:
  - __init__:           accept pmp_cache_path + Layer 2 hyperparams
  - _load_feature_cache: load PMP per-drug mediator dict cache
  - _build_aux_head:    return PMPHead (Layer 2 attention pool)
  - _combined_logit:    use mediator gather + attention pool instead of 22-d count

Locked Algorithm 1 (do not change without re-review):
  1. M = N_<=2(a) intersect N_<=2(b)   (typed common neighbors, non-drug)
  2. if M empty: z = 0
  3. else: phi(m) = MLP([h_m, e_type, e_rel(a,m), e_rel(m,b)])
          alpha = softmax(w^T tanh(W phi(m)))
          z = sum_m alpha_m * phi_m
  4. l_base = EmerGNN_logit(a, b)
  5. l_pmp  = MLP_score(z)
  6. y_hat  = sigmoid(l_base + beta * l_pmp)
"""
from __future__ import annotations

import copy
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
# pmp_v1 lives directly under Code/my_code/models/, so parents[4] = project root.
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v2_meetnode.mnah_trainer import _PerModeEmerGNN_MNAH  # noqa
from baseline.emergnn.shuffle_utils import (  # noqa
    build_edge_lists_from_triplets, shuffle_train,
)
from baseline.emergnn.model import EmerGNN  # noqa


DEFAULT_PMP_CACHE = (
    PROJECT_ROOT
    / "Code/data/_cache/pmp_mediator_cache_drugbank_seed42_kgonly_v1.pkl"
)


# ----------------------------------------------------------------------------
# PMP Head module — Algorithm 1 lines 3-9 (phi + attention pool) + line 11 (MLP_score)
# ----------------------------------------------------------------------------

class PMPHead(nn.Module):
    """Per-mediator typed embedding + attention pool + MLP_score.

    Forward signature:
        forward(h_m, type_ids, rel_am, rel_mb, mask) -> logit (B,)

    where:
        h_m       : (B, M_max, n_dim)   per-mediator KG-embedding (from EmerGNN._entity_embed)
        type_ids  : (B, M_max) long     mediator type index
        rel_am    : (B, M_max) long     relation a->m (REL_2HOP_ID for 2-hop)
        rel_mb    : (B, M_max) long     relation m->b (REL_2HOP_ID for 2-hop)
        mask      : (B, M_max) bool     True for real mediator, False for padding

    Empty mediator pairs (mask all False) produce z = 0 (Algorithm 1 line 2).
    """

    def __init__(
        self,
        n_dim: int,
        n_mediators: int,            # PMP-internal mediator vocab size (codex r2 fix)
        n_types: int,
        n_rels: int,
        hidden: int = 128,
        dropout: float = 0.2,
        type_dim: int | None = None,
        rel_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_dim = int(n_dim)
        self.n_mediators = int(n_mediators)
        if type_dim is None:
            type_dim = max(n_dim // 4, 8)
        if rel_dim is None:
            rel_dim = max(n_dim // 4, 8)
        self.type_dim = int(type_dim)
        self.rel_dim = int(rel_dim)

        # Codex r2 fix #1: PMP owns its mediator embedding table, decoupled from
        # EmerGNN backbone. For feat='M', backbone's _entity_embed returns
        # Went(0) = constant for ALL non-drug mediators (since Morgan features
        # only exist for drug entities, non-drug rows are zero). That makes
        # Algorithm 1 line 3's h_m a constant, breaking the entire pool.
        # We add a separate trainable nn.Embedding(n_mediators, n_dim) that
        # gives each merged-KG mediator a distinct learnable representation.
        # Trained end-to-end via DDI BCE.
        self.mediator_embed = nn.Embedding(self.n_mediators, self.n_dim)
        nn.init.xavier_uniform_(self.mediator_embed.weight)

        self.type_embed = nn.Embedding(n_types, self.type_dim)
        self.rel_embed = nn.Embedding(n_rels, self.rel_dim)

        # phi(m) = MLP([h_m ; e_type ; e_rel(a,m) ; e_rel(m,b)])
        phi_in_dim = self.n_dim + self.type_dim + 2 * self.rel_dim
        self.phi_mlp = nn.Sequential(
            nn.Linear(phi_in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_dim),
        )

        # Attention: s_m = w^T tanh(W phi(m))
        self.attn_W = nn.Linear(self.n_dim, self.n_dim)
        self.attn_w = nn.Linear(self.n_dim, 1, bias=False)

        # MLP_score: z -> scalar logit
        self.score_mlp = nn.Sequential(
            nn.Linear(self.n_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        mediator_idx: torch.Tensor,  # (B, M_max) long — PMP-internal mediator vocab indices
        type_ids: torch.Tensor,    # (B, M_max) long
        rel_am: torch.Tensor,      # (B, M_max) long
        rel_mb: torch.Tensor,      # (B, M_max) long
        mask: torch.Tensor,        # (B, M_max) bool
    ) -> torch.Tensor:
        # Algorithm 1 line 3-5: build phi(m). h_m comes from PMP's own
        # mediator_embed table (codex r2 fix), NOT from EmerGNN backbone.
        h_m = self.mediator_embed(mediator_idx)  # (B, M_max, n_dim)
        e_type = self.type_embed(type_ids)      # (B, M_max, type_dim)
        e_ram = self.rel_embed(rel_am)          # (B, M_max, rel_dim)
        e_rmb = self.rel_embed(rel_mb)          # (B, M_max, rel_dim)
        phi_in = torch.cat([h_m, e_type, e_ram, e_rmb], dim=-1)
        phi = self.phi_mlp(phi_in)              # (B, M_max, n_dim)

        # Algorithm 1 line 6-7: attention score + softmax
        s = self.attn_w(torch.tanh(self.attn_W(phi))).squeeze(-1)  # (B, M_max)
        # Mask out padded positions before softmax: set to -inf
        neg_inf = torch.finfo(s.dtype).min
        s = s.masked_fill(~mask, neg_inf)

        # Detect all-padded rows (empty mediator set): mask sum == 0
        any_valid = mask.any(dim=1)  # (B,)
        # For all-padded rows, replace s with 0 so softmax doesn't NaN
        # (we'll zero out the contribution via alpha * mask below anyway)
        s_safe = torch.where(any_valid.unsqueeze(1), s, torch.zeros_like(s))
        alpha = F.softmax(s_safe, dim=1)        # (B, M_max)
        alpha = alpha * mask.float()            # zero out padding contribution

        # Algorithm 1 line 8: z = sum_m alpha_m * phi_m   (z=0 for empty rows)
        z = (alpha.unsqueeze(-1) * phi).sum(dim=1)  # (B, n_dim)
        # All-padded rows: z is already 0 (mask zeros out everything)

        # Algorithm 1 line 11: l_pmp = MLP_score(z)
        logit = self.score_mlp(z).squeeze(-1)  # (B,)
        return logit


# ----------------------------------------------------------------------------
# PMP Trainer
# ----------------------------------------------------------------------------

class _PerModeEmerGNN_PMP(_PerModeEmerGNN_MNAH):
    """EmerGNN + PMP Layer 2 attention pool aux head (replaces MNAH 22-d count head).

    Args extending MNAH parent:
      pmp_cache_path : path to per-drug mediator dict pickle
      pmp_hidden     : MLP hidden dim inside PMPHead (default 128)
      pmp_dropout    : dropout inside PMPHead (default 0.2)
      pmp_init_beta  : initial softplus(raw_beta) for residual fusion (default 1.0,
                       same as MNAH; reused via parent's mnah_init_beta)
      pmp_max_mediators : optional cap on mediators per pair (None = no cap)
    """

    def __init__(
        self,
        *,
        pmp_cache_path: str | None = None,
        pmp_hidden: int = 128,
        pmp_dropout: float = 0.2,
        pmp_max_mediators: int | None = None,
        **kwargs,
    ) -> None:
        # Disable MNAH's 22-d feature cache loading; PMP has its own cache.
        # We set mnah_feat_cache to a placeholder, but override _load_feature_cache.
        # Force mnah_text_cache=None (PMP doesn't use Stage 2 text cache).
        kwargs.setdefault("mnah_text_cache", None)
        super().__init__(**kwargs)

        self.pmp_cache_path = pmp_cache_path or str(DEFAULT_PMP_CACHE)
        self.pmp_hidden = int(pmp_hidden)
        self.pmp_dropout = float(pmp_dropout)
        self.pmp_max_mediators = (
            int(pmp_max_mediators) if pmp_max_mediators is not None else None
        )

        # PMP-specific cache state (populated in _load_feature_cache)
        self._pmp_n1: dict[str, list[tuple[str, int, int]]] | None = None
        self._pmp_n2: dict[str, list[tuple[str, int]]] | None = None
        self._pmp_n_types: int | None = None
        self._pmp_n_rels_total: int | None = None
        self._pmp_rel_2hop_id: int | None = None
        # Codex r2 fix #2: PMP-internal mediator2id (decoupled from backbone entity2id)
        self._pmp_mediator2id: dict[str, int] | None = None
        self._pmp_n_mediators: int | None = None

    # ------------------------------------------------------------------
    # Override _load_feature_cache: load PMP per-drug mediator dicts
    # ------------------------------------------------------------------

    def _load_feature_cache(
        self,
        train_pos_pairs: pd.DataFrame,
        train_neg_pairs: pd.DataFrame | None = None,
    ) -> None:
        """Load the PMP per-drug mediator dictionary cache.

        Unlike MNAH (which loads a per-pair 22-d feature parquet and fits a
        train-only normalizer), PMP stores per-drug mediator lists and computes
        the per-pair mediator set at runtime via set intersection. No normalizer
        is needed because attention pool / softmax is scale-invariant.
        """
        cache_path = Path(self.pmp_cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"PMP mediator cache not found at {cache_path}. "
                f"Run Code/scripts/precompute_pmp_cache.py first."
            )
        print(f"[pmp] loading mediator cache: {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        # codex r2: required v2+ schema (with PMP-internal mediator2id).
        # codex r2 follow-up: exact-match accepted versions list, not lexical
        # string compare (which would silently reject "pmp_v2" etc.).
        ACCEPTED_PMP_SCHEMAS = {"pmp_v1_codex_r2"}
        schema = payload.get("schema_version", "")
        if "pmp_mediator2id" not in payload or schema not in ACCEPTED_PMP_SCHEMAS:
            raise RuntimeError(
                f"PMP cache at {cache_path} has unsupported schema_version={schema!r}. "
                f"Accepted: {sorted(ACCEPTED_PMP_SCHEMAS)}. Rebuild via "
                f"`python Code/scripts/precompute_pmp_cache.py --force`."
            )
        self._pmp_n1 = payload["n1"]
        self._pmp_n2 = payload["n2"]
        self._pmp_n_types = int(payload["n_types"])
        self._pmp_n_rels_total = int(payload["n_rels_plus_special"])
        self._pmp_rel_2hop_id = int(payload["rel_2hop_id"])
        # codex r2 fix #2: PMP-internal mediator vocab decoupled from backbone entity2id
        self._pmp_mediator2id = payload["pmp_mediator2id"]
        self._pmp_n_mediators = int(payload["n_mediators"])
        print(
            f"[pmp] cache stats: drugs_with_n1={len(self._pmp_n1)} "
            f"drugs_with_n2={len(self._pmp_n2)} n_types={self._pmp_n_types} "
            f"n_rels_total={self._pmp_n_rels_total} rel_2hop_id={self._pmp_rel_2hop_id} "
            f"n_mediators={self._pmp_n_mediators}",
            flush=True,
        )

        # Coverage diagnostic. Verify PMP cache drug vocabulary covers training pairs.
        cache_drugs = set(self._pmp_n1.keys()) | set(self._pmp_n2.keys())
        train_drugs = set(train_pos_pairs["drug_a_id"].astype(str).tolist()
                          + train_pos_pairs["drug_b_id"].astype(str).tolist())
        missing_drugs = train_drugs - cache_drugs
        if missing_drugs:
            print(f"[pmp] WARNING: {len(missing_drugs)} train drugs lack any "
                  f"mediator entry in PMP cache; examples={list(missing_drugs)[:5]}",
                  flush=True)
        else:
            print(f"[pmp] coverage OK: all {len(train_drugs)} train drugs have "
                  f"mediator entries", flush=True)

        # Mark the aux input dim with n_dim so _build_aux_head receives a usable hint;
        # PMPHead doesn't actually consume a single-int in_dim, but we set it for
        # consistency with parent's _build_aux_head signature.
        self._aux_in_dim = self.n_dim

    # ------------------------------------------------------------------
    # Override _build_aux_head: construct PMPHead instead of MNAH AuxMLP
    # ------------------------------------------------------------------

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        if (self._pmp_n_types is None or self._pmp_n_rels_total is None
                or self._pmp_n_mediators is None):
            raise RuntimeError(
                "PMP cache must be loaded before _build_aux_head (call "
                "_load_feature_cache first)."
            )
        return PMPHead(
            n_dim=self.n_dim,
            n_mediators=self._pmp_n_mediators,   # codex r2 fix #1
            n_types=self._pmp_n_types,
            n_rels=self._pmp_n_rels_total,
            hidden=self.pmp_hidden,
            dropout=self.pmp_dropout,
        )

    # ------------------------------------------------------------------
    # Mediator gather (Algorithm 1 line 1)
    # ------------------------------------------------------------------

    def _gather_mediators_for_pair(
        self, a: str, b: str
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Return (mediator_pmp_indices, type_ids, rel_a_ids, rel_b_ids) for pair (a, b).

        Codex r2 fix #2: mediator indices are PMP-internal (built from merged KG
        in precompute), NOT backbone entity2id. This decouples PMP mediator set
        from the EmerGNN backbone's entity vocabulary, so mediators from
        Hetionet / PrimeKG won't be silently dropped when backbone uses
        backbone_kg_source='drugbank' (smaller vocab).

        Combines 1-hop and 2-hop common mediators (typed common neighbors).
        For 1-hop mediators, rel_a_id and rel_b_id are real KG edge relation IDs.
        For 2-hop mediators, both rel_a_id and rel_b_id are REL_2HOP_ID (special).

        If a mediator appears in both 1-hop sets, it is treated as 1-hop only
        (1-hop's relation info is strictly more informative than 2-hop's pseudo-rel).
        """
        if self._pmp_n1 is None or self._pmp_n2 is None or self._pmp_mediator2id is None:
            raise RuntimeError("PMP cache not loaded")
        pmp_id_of = self._pmp_mediator2id  # local alias

        # 1-hop intersection: index a's n1 by mediator_id for fast lookup
        n1_a = {m_id: (type_id, rel_id) for (m_id, type_id, rel_id) in self._pmp_n1.get(a, [])}
        n1_b = {m_id: (type_id, rel_id) for (m_id, type_id, rel_id) in self._pmp_n1.get(b, [])}
        common_1hop = set(n1_a.keys()) & set(n1_b.keys())

        mediator_node_ids: list[int] = []
        type_ids: list[int] = []
        rel_a_ids: list[int] = []
        rel_b_ids: list[int] = []
        seen_mediators: set[str] = set()

        for m_id in common_1hop:
            t_a, r_a = n1_a[m_id]
            t_b, r_b = n1_b[m_id]
            # Type agrees by construction (function of mediator's KG kind). Use a's view.
            pmp_idx = pmp_id_of.get(m_id)
            if pmp_idx is None:
                # Shouldn't happen — PMP cache built mediator2id from same set —
                # raise loudly instead of silent drop (codex r2 #2 lesson).
                raise KeyError(
                    f"Mediator {m_id!r} from n1[{a}] ∩ n1[{b}] missing from "
                    f"pmp_mediator2id — cache schema bug?"
                )
            mediator_node_ids.append(int(pmp_idx))
            type_ids.append(int(t_a))
            rel_a_ids.append(int(r_a))
            rel_b_ids.append(int(r_b))
            seen_mediators.add(m_id)

        # 2-hop intersection: skip mediators already in 1-hop
        n2_a = {m_id: type_id for (m_id, type_id) in self._pmp_n2.get(a, [])}
        n2_b_ids = set(m_id for (m_id, _) in self._pmp_n2.get(b, []))
        common_2hop = (set(n2_a.keys()) & n2_b_ids) - seen_mediators
        rel2hop = int(self._pmp_rel_2hop_id)  # type: ignore[arg-type]

        for m_id in common_2hop:
            type_id = n2_a[m_id]
            pmp_idx = pmp_id_of.get(m_id)
            if pmp_idx is None:
                raise KeyError(
                    f"Mediator {m_id!r} from n2[{a}] ∩ n2[{b}] missing from "
                    f"pmp_mediator2id — cache schema bug?"
                )
            mediator_node_ids.append(int(pmp_idx))
            type_ids.append(int(type_id))
            rel_a_ids.append(rel2hop)
            rel_b_ids.append(rel2hop)

        # Optional cap on mediators per pair (deterministic: head order from sets is
        # nondeterministic; sort by mediator_node_id for reproducibility before cap)
        if self.pmp_max_mediators is not None and len(mediator_node_ids) > self.pmp_max_mediators:
            order = sorted(
                range(len(mediator_node_ids)),
                key=lambda i: mediator_node_ids[i],
            )[: self.pmp_max_mediators]
            mediator_node_ids = [mediator_node_ids[i] for i in order]
            type_ids = [type_ids[i] for i in order]
            rel_a_ids = [rel_a_ids[i] for i in order]
            rel_b_ids = [rel_b_ids[i] for i in order]

        return mediator_node_ids, type_ids, rel_a_ids, rel_b_ids

    def _batch_mediator_tensors(
        self, batch: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build padded mediator tensors for a batch of pairs.

        Codex r2: mediator indices are PMP-internal (in PMPHead.mediator_embed's
        vocab), NOT backbone entity2id.

        Returns:
            mediator_idx_batch : (B, M_max) long  PMP-internal mediator indices (0 for padding)
            type_ids_batch     : (B, M_max) long  (0 for padding)
            rel_am_batch       : (B, M_max) long  (0 for padding)
            rel_mb_batch       : (B, M_max) long  (0 for padding)
            mask_batch         : (B, M_max) bool  True for real mediator
        """
        B = len(batch)
        a_list = batch["drug_a_id"].astype(str).tolist()
        b_list = batch["drug_b_id"].astype(str).tolist()

        all_ent_ids: list[list[int]] = []
        all_type_ids: list[list[int]] = []
        all_rel_am: list[list[int]] = []
        all_rel_mb: list[list[int]] = []
        m_max = 0
        for a, b in zip(a_list, b_list):
            ent_ids, t_ids, r_a, r_b = self._gather_mediators_for_pair(a, b)
            all_ent_ids.append(ent_ids)
            all_type_ids.append(t_ids)
            all_rel_am.append(r_a)
            all_rel_mb.append(r_b)
            m_max = max(m_max, len(ent_ids))
        # ensure m_max >= 1 to keep tensor shape well-defined; empty pairs will have all-False mask
        m_max = max(m_max, 1)

        ent_idx_batch = torch.zeros((B, m_max), dtype=torch.long, device=self.device)
        type_ids_batch = torch.zeros((B, m_max), dtype=torch.long, device=self.device)
        rel_am_batch = torch.zeros((B, m_max), dtype=torch.long, device=self.device)
        rel_mb_batch = torch.zeros((B, m_max), dtype=torch.long, device=self.device)
        mask_batch = torch.zeros((B, m_max), dtype=torch.bool, device=self.device)

        for i in range(B):
            k = len(all_ent_ids[i])
            if k == 0:
                continue
            ent_idx_batch[i, :k] = torch.tensor(all_ent_ids[i], dtype=torch.long, device=self.device)
            type_ids_batch[i, :k] = torch.tensor(all_type_ids[i], dtype=torch.long, device=self.device)
            rel_am_batch[i, :k] = torch.tensor(all_rel_am[i], dtype=torch.long, device=self.device)
            rel_mb_batch[i, :k] = torch.tensor(all_rel_mb[i], dtype=torch.long, device=self.device)
            mask_batch[i, :k] = True
        return ent_idx_batch, type_ids_batch, rel_am_batch, rel_mb_batch, mask_batch

    # ------------------------------------------------------------------
    # Override _combined_logit: residual fusion of EmerGNN logit + PMP head
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
        """Return (combined_logit, emergnn_logit, pmp_logit) per batch.

        Algorithm 1 lines 4-6:
          l_base = EmerGNN_logit(a, b)
          l_pmp  = PMPHead(h_m, ...)   # which already includes MLP_score
          combined = l_base + softplus(raw_beta) * l_pmp
        """
        # EmerGNN backbone logit (l_base). Same call as MNAH parent.
        emergnn_logit = self._model(head, tail, edge_src, edge_dst, edge_rel)

        # Build mediator tensors for this batch (PMP-internal mediator indices).
        med_idx, type_ids, rel_am, rel_mb, mask = self._batch_mediator_tensors(batch_df)

        # Codex r2 fix #1: PMPHead owns its own mediator embedding table; we no
        # longer call self._model._entity_embed (which would give Went(0) ~ constant
        # for all non-drug mediators under feat='M'). PMPHead.forward takes
        # mediator_idx directly (PMP-internal vocab) and looks up via its own
        # trainable nn.Embedding(n_mediators, n_dim).
        pmp_logit = self._aux_mlp(med_idx, type_ids, rel_am, rel_mb, mask)  # (B,)

        # Residual fusion (MNAH-style); _beta() = softplus(raw_beta)
        combined = emergnn_logit + self._beta() * pmp_logit
        return combined, emergnn_logit, pmp_logit


__all__ = ["_PerModeEmerGNN_PMP", "PMPHead", "DEFAULT_PMP_CACHE"]
