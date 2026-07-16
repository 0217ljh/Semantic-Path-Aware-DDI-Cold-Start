"""SPMN v1 — multi-modal head (Phase 3): KG relation-pool + molecular FragAlign.

Combines the locked KG backbone (relation-enriched per-(pair,type) attention
pool + explicit struct features) with the molecule↔KG bridge:

  F2 (dense)  fragment-aligned entity evidence:  e_d = Σ_v w_d(v) · h(v)
              (per-type pooled; w from :class:`SPMNBridge`)
  F3 (sparse) fragment-weighted cross-type channel:
              q_{τ,τ'} = Σ_{(u,v) co-path} w_a(u) · w_b(v)
  κ           KG-availability gate → fragment-only fallback when support empty

Correctness-first: the molecular part is computed per pair in a clear loop
(small GPU ops); vectorise later once molecular gain is confirmed. No drug-id
embeddings; entity embeddings shared (inductive).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

from my_code.models.spmn_v1.bridge import SPMNBridge


class SPMNMolHead(nn.Module):
    def __init__(
        self,
        n_entities: int,
        n_types: int,
        n_rel_buckets: int,
        struct_dim: int,
        d_frag: int,
        d: int = 32,
        type_dim: int = 8,
        rel_dim: int = 8,
        r_bridge: int = 32,
        hidden: int = 128,
        dropout: float = 0.2,
        bridge_temp: float = 0.5,
        use_frag_fallback: bool = True,
        use_bridge: bool = True,
    ) -> None:
        super().__init__()
        self.n_types = int(n_types)
        self.d = int(d)
        self.d_frag = int(d_frag)
        # Ablation switches for gain ATTRIBUTION (codex): isolate whether any
        # molecular gain comes from the raw fragment fallback (f) or the
        # grounded bridge (F2/F3). Blocks stay in the head (zeroed) so dims hold.
        self.use_frag_fallback = bool(use_frag_fallback)
        self.use_bridge = bool(use_bridge)

        # --- KG backbone (mirrors rel_head) ---
        self.entity_embed = nn.Embedding(int(n_entities), self.d)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        self.type_embed = nn.Embedding(self.n_types, int(type_dim))
        nn.init.xavier_uniform_(self.type_embed.weight)
        self.rel_embed = nn.Embedding(int(n_rel_buckets), int(rel_dim))
        nn.init.xavier_uniform_(self.rel_embed.weight)
        self.phi_mlp = nn.Sequential(
            nn.Linear(self.d + int(type_dim) + 2 * int(rel_dim), hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, self.d),
        )
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)
        self.W_T = nn.Parameter(torch.empty(self.n_types, self.n_types))
        nn.init.xavier_uniform_(self.W_T)

        # --- molecular bridge ---
        self.bridge = SPMNBridge(d_frag=d_frag, d_kg=self.d, n_types=self.n_types,
                                 r=r_bridge, temp=bridge_temp)

        # --- head: [routed KG channels (K*d) | struct
        #            | f_a (d_frag) | f_b (d_frag)   <- fragment-only fallback, UNGATED
        #            | e_a (d) | e_b (d) | q (K*K) | kappa(1)  <- KG-grounded, kappa-gated] ---
        head_in = self.n_types * self.d + int(struct_dim) + 2 * self.d_frag \
            + 2 * self.d + self.n_types * self.n_types + 1
        self.head_mlp = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    # ---- per-pair pieces -------------------------------------------------

    def _kg_pool(self, glob, typ, rel_a, rel_b):
        """relation-enriched per-type attention pool -> routed channels (K*d,)
        and entity states h_ent (E,d) for the bridge."""
        K, d = self.n_types, self.d
        device = self.W_T.device
        h_ent = self.entity_embed(glob)                       # (E, d)
        if glob.numel() == 0:
            return torch.zeros(K * d, device=device), h_ent
        phi = self.phi_mlp(torch.cat([
            h_ent, self.type_embed(typ), self.rel_embed(rel_a), self.rel_embed(rel_b),
        ], dim=-1))                                           # (E, d)
        scores = self.within_attn_w(torch.tanh(self.within_attn_W(phi))).squeeze(-1)
        c = torch.zeros(K, d, device=device)
        # per-type attention pool (softmax within each type group)
        for t in range(K):
            m = typ == t
            if m.any():
                a = torch.softmax(scores[m], dim=0)
                c[t] = (a.unsqueeze(-1) * phi[m]).sum(0)
        transition = F.softmax(self.W_T, dim=-1)
        c_tilde = transition @ c                              # (K, d)
        return c_tilde.reshape(K * d), h_ent

    def forward_pair(self, glob, typ, rel_a, rel_b, struct,
                     z_a, z_b, iu, iv, kappa):
        """Score one pair. glob/typ/rel_a/rel_b:(E,); struct:(struct_dim,);
        z_a:(Fa,d_frag) z_b:(Fb,d_frag); iu,iv:(P,) co-path local idx; kappa: scalar."""
        K, d = self.n_types, self.d
        device = self.W_T.device
        kg_chan, h_ent = self._kg_pool(glob, typ, rel_a, rel_b)

        # F-fallback: raw mean-pooled fragment reps (ALWAYS available, UNGATED
        # → carries molecular signal even when the KG corridor is empty).
        if self.use_frag_fallback:
            f_a = z_a.mean(0) if z_a.size(0) > 0 else torch.zeros(self.d_frag, device=device)
            f_b = z_b.mean(0) if z_b.size(0) > 0 else torch.zeros(self.d_frag, device=device)
        else:
            f_a = torch.zeros(self.d_frag, device=device)
            f_b = torch.zeros(self.d_frag, device=device)

        # molecular alignment weights over the support entities
        if self.use_bridge:
            w_a, _ = self.bridge.entity_weights(z_a, h_ent, typ)   # (E,)
            w_b, _ = self.bridge.entity_weights(z_b, h_ent, typ)
        else:
            w_a = torch.zeros(h_ent.size(0), device=device)
            w_b = torch.zeros(h_ent.size(0), device=device)

        # F2 dense: fragment-aligned entity evidence
        if h_ent.numel() > 0:
            e_a = (w_a.unsqueeze(-1) * h_ent).sum(0)           # (d,)
            e_b = (w_b.unsqueeze(-1) * h_ent).sum(0)
        else:
            e_a = torch.zeros(d, device=device)
            e_b = torch.zeros(d, device=device)

        # F3 sparse: fragment-weighted cross-type co-path channel q[τ,τ']
        q = torch.zeros(K, K, device=device)
        if iu.numel() > 0:
            contrib = w_a[iu] * w_b[iv]                        # (P,)
            flat = typ[iu] * K + typ[iv]                       # (P,)
            q.view(-1).scatter_add_(0, flat, contrib)

        z = torch.cat([kg_chan, struct,
                       f_a, f_b,                      # fragment-only fallback (ungated)
                       e_a * kappa, e_b * kappa,      # KG-grounded F2 (gated)
                       (q * kappa).reshape(K * K),    # KG-grounded F3 (gated)
                       kappa.reshape(1)], dim=-1)
        return self.head_mlp(z).squeeze(-1)


    # ------------------------------------------------------------------
    # Vectorised batched forward (GPU-efficient; replaces the per-pair loop).
    # Numerically equals forward_pair (verified). Flattened-segment layout.
    # ------------------------------------------------------------------

    def _seg_mean(self, fz, fpair, B):
        d = fz.size(-1)
        s = torch.zeros(B, d, device=fz.device)
        if fz.size(0) > 0:
            s.scatter_add_(0, fpair.unsqueeze(-1).expand(-1, d), fz)
        cnt = torch.zeros(B, device=fz.device)
        if fz.size(0) > 0:
            cnt.scatter_add_(0, fpair, torch.ones(fz.size(0), device=fz.device))
        return s / cnt.clamp(min=1.0).unsqueeze(-1)

    def _bridge_weights_batch(self, fz, fpair, he_t, typ, group, B):
        """Batched bridge weights over the flattened support (M,). he_t already
        = B_{φ(v)} h̃(v). Returns w (M,) — per-(pair,type) softmax of the
        multi-instance logsumexp over the pair's own fragments; 0 for pairs with
        no fragments (matches forward_pair)."""
        M = he_t.size(0)
        device = he_t.device
        K = self.n_types
        fa_count = torch.zeros(B, dtype=torch.long, device=device)
        if fz.size(0) > 0:
            fa_count.scatter_add_(0, fpair, torch.ones(fz.size(0), dtype=torch.long, device=device))
        pair_of = group // K                                   # pair id per med
        counts_med = fa_count[pair_of]                         # (M,) frags per med's pair
        total = int(counts_med.sum().item())
        w = torch.zeros(M, device=device)
        if total == 0 or fz.size(0) == 0:
            return w
        fa_start = torch.cumsum(fa_count, 0) - fa_count        # (B,)
        zf = self.bridge.W_f(fz)                               # (F, r)
        # same-pair (f, m) enumeration
        fm_m = torch.repeat_interleave(torch.arange(M, device=device), counts_med)
        blk = torch.repeat_interleave(torch.cumsum(counts_med, 0) - counts_med, counts_med)
        within = torch.arange(total, device=device) - blk
        fm_f = torch.repeat_interleave(fa_start[pair_of], counts_med) + within
        dots = (zf[fm_f] * he_t[fm_m]).sum(-1) / math.sqrt(self.bridge.r)   # (total,)
        # segment logsumexp over fm_m -> aggscore (M,)
        maxv = torch.full((M,), float("-inf"), device=device)
        maxv = maxv.scatter_reduce(0, fm_m, dots, reduce="amax", include_self=False)
        se = torch.exp(dots - maxv[fm_m])
        sum_se = torch.zeros(M, device=device).scatter_add_(0, fm_m, se)
        has = counts_med > 0
        agg = torch.where(has, torch.log(sum_se.clamp(min=1e-20)) + maxv, torch.zeros(M, device=device))
        w = pyg_softmax(agg / self.bridge.temp, group, num_nodes=B * K)
        return w * has.float()

    def forward_batch(self, *, med, pair_idx, typ, rel_a, rel_b, n_pairs,
                      fa_z, fa_pair, fb_z, fb_pair, iu_g, iv_g, struct, kappa):
        B, K, d = int(n_pairs), self.n_types, self.d
        device = self.W_T.device
        M = med.size(0)
        group = pair_idx * K + typ

        # KG pool (batched)
        h_ent = self.entity_embed(med)
        phi = self.phi_mlp(torch.cat([h_ent, self.type_embed(typ),
              self.rel_embed(rel_a), self.rel_embed(rel_b)], dim=-1))
        scores = self.within_attn_w(torch.tanh(self.within_attn_W(phi))).squeeze(-1)
        alpha = pyg_softmax(scores, group, num_nodes=B * K)
        c = torch.zeros(B * K, d, device=device)
        c.scatter_add_(0, group.unsqueeze(-1).expand(-1, d), alpha.unsqueeze(-1) * phi)
        c = c.view(B, K, d)
        transition = F.softmax(self.W_T, dim=-1)
        c_tilde = torch.einsum("kj,bjd->bkd", transition, c).reshape(B, K * d)

        # fragment-only fallback (ungated)
        if self.use_frag_fallback:
            f_a = self._seg_mean(fa_z, fa_pair, B)
            f_b = self._seg_mean(fb_z, fb_pair, B)
        else:
            f_a = torch.zeros(B, self.d_frag, device=device)
            f_b = torch.zeros(B, self.d_frag, device=device)

        # bridge weights over support
        if self.use_bridge:
            he = self.bridge.W_h(h_ent)
            he_t = torch.einsum("erc,ec->er", self.bridge.B[typ], he)
            w_a = self._bridge_weights_batch(fa_z, fa_pair, he_t, typ, group, B)
            w_b = self._bridge_weights_batch(fb_z, fb_pair, he_t, typ, group, B)
        else:
            w_a = torch.zeros(M, device=device)
            w_b = torch.zeros(M, device=device)

        # F2 dense
        e_a = torch.zeros(B, d, device=device)
        e_b = torch.zeros(B, d, device=device)
        if M > 0:
            e_a.scatter_add_(0, pair_idx.unsqueeze(-1).expand(-1, d), w_a.unsqueeze(-1) * h_ent)
            e_b.scatter_add_(0, pair_idx.unsqueeze(-1).expand(-1, d), w_b.unsqueeze(-1) * h_ent)

        # F3 sparse co-path channel
        q = torch.zeros(B * K * K, device=device)
        if iu_g.numel() > 0:
            contrib = w_a[iu_g] * w_b[iv_g]
            cell = pair_idx[iu_g] * (K * K) + typ[iu_g] * K + typ[iv_g]
            q.scatter_add_(0, cell, contrib)
        q = q.view(B, K * K)

        kap = kappa.view(B, 1)
        z = torch.cat([c_tilde, struct, f_a, f_b, e_a * kap, e_b * kap, q * kap, kap], dim=-1)
        return self.head_mlp(z).squeeze(-1)


__all__ = ["SPMNMolHead"]
