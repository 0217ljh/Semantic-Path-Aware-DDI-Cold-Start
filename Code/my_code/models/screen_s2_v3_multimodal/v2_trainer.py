"""v2 learned-routing trainer — i1 routing + multimodal alignment on the MNAH backbone.

Implements README_i1_routing_v2_learned.md (codex PASS r2). Subclasses the MNAH trainer and
REPLACES its single 22-dim aux head with a two-channel gated head, while keeping the EmerGNN
path-flow backbone (i2) and the logit-level fusion `emergnn + beta * head` untouched:

    head_logit = g * mol_logit + (1 - g) * eff_logit         (learned soft gate)
    combined   = emergnn_logit + beta * head_logit

Channels (i1):
- MOLECULAR channel  = molecular-layer shared-mediator counts + InfoNCE-aligned molecular
  embedding z_m of both drugs (proj_m(m_u), cache molecular_aligned_infonce.npz). PK evidence.
- EFFECT channel     = pair-conditional cross-attention "select-compose" over the top-K=32
  effect-layer neighbors of each drug (PD-composable kinds {side_effect, phenotype, symptom},
  ranked by inverse drug-frequency), frequency-debiased. PD evidence.
- GATE               = low-capacity, layer-composition-features ONLY (mol vs eff mediator
  counts) -> sigmoid. No drug IDs / no learned drug embeddings / no DDI-type labels (codex R4).

The whole head is returned by `_build_aux_head` so its params join the MNAH optimizer; the
per-drug inputs (z_m, effect neighbors) are gathered inside the overridden `_combined_logit`.

Sanity-check flags (codex mandatory, run-time):
- effect_shuffle_control : permute the per-drug effect-neighbor sets (preserve degree) -> the
  effect channel must collapse to ~0 contribution.
- effect_mode='bagpool'  : replace cross-attention with mean-pool -> PD subgroup must drop.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v2_meetnode.mnah_trainer import _PerModeEmerGNN_MNAH
from baseline.emergnn._per_mode import _PerModeEmerGNN  # noqa
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train
from baseline.emergnn.model import EmerGNN

# 22 meet-feature columns: 11 mediator types x 2 hops. Index within each hop block:
#  0 protein_gene 1 pathway 2 side_effect 3 disease 4 anatomy 5 compound
#  6 biological_process 7 molecular_function 8 cellular_component 9 pharmacologic_class 10 exposure
_MOL_TYPES = [0, 1, 5, 6, 7, 8]   # molecular layer (PK)
_EFF_TYPES = [2, 3, 4]            # effect layer (PD)  (disease/anatomy used by gate only)
_MOL_COLS = _MOL_TYPES + [11 + i for i in _MOL_TYPES]   # 1hop + 2hop
_EFF_COLS = _EFF_TYPES + [11 + i for i in _EFF_TYPES]

ALIGN_NPZ = PROJECT_ROOT / "Code/data/_cache/molecular_aligned_infonce.npz"
EFFNBR_NPZ = PROJECT_ROOT / "Code/data/_cache/effect_neighbors_pubmedbert.npz"
PUBMEDBERT_PT = PROJECT_ROOT / "Code/data/KG/_merged_kg/_cache/screen1_tag_init/d_name_only__pubmedbert.pt"


class V2Head(nn.Module):
    """Two-channel gated head. Holds all params; called by _combined_logit with per-pair data."""

    def __init__(self, *, align_dim: int, eff_in: int = 768, d: int = 64,
                 n_mol_cols: int = 12, n_eff_cols: int = 6, dropout: float = 0.2,
                 effect_mode: str = "xattn", freq_debias: float = 1.0,
                 eff_topk_pair: int = 0, eff_attn_temp: float = 1.0,
                 eff_pair_dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.effect_mode = effect_mode
        self.freq_debias = freq_debias
        # codex 019e6251 patch knobs: sparse top-k pair selection over the K*K candidates,
        # attention temperature (<1 sharpens), and effect-pair dropout (train-time).
        self.eff_topk_pair = int(eff_topk_pair)   # 0 = keep all candidates (no-op)
        self.eff_attn_temp = float(eff_attn_temp)
        self.eff_pair_dropout = float(eff_pair_dropout)
        self._last_attn_entropy = None
        # molecular channel: [z_a*z_b, |z_a-z_b|, mol_counts] -> mol_logit
        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * align_dim + n_mol_cols, d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, 1),
        )
        # effect channel: project neighbor emb, compose selected pairs -> eff_logit
        self.eff_proj = nn.Sequential(nn.Linear(eff_in, d), nn.LayerNorm(d), nn.GELU())
        self.eff_out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(d, 1))
        # effect-COUNT channel (codex 019e6295 pivot): pair-conditional MLP over effect-layer
        # count features (the demonstrably-alive PD signal), used when effect_mode=='count'.
        self.eff_count_mlp = nn.Sequential(
            nn.Linear(n_eff_cols, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))
        # gate: low-capacity, composition features only (mol vs eff counts), -> [0,1]
        self.gate = nn.Sequential(nn.Linear(n_mol_cols + n_eff_cols, 8), nn.ReLU(),
                                   nn.Linear(8, 1))

    def molecular_logit(self, z_a, z_b, mol_counts):
        feat = torch.cat([z_a * z_b, (z_a - z_b).abs(), mol_counts], dim=1)
        return self.mol_mlp(feat).squeeze(-1)

    def effect_logit_count(self, eff_counts):
        """Pivot path: effect channel over effect-layer count features (no neighbor emb)."""
        self._last_attn_entropy = None
        return self.eff_count_mlp(eff_counts).squeeze(-1)

    def effect_logit(self, Ea, Eb, ma, mb, deg_a, deg_b):
        """Ea,Eb: (B,K,768) neighbor emb; ma,mb: (B,K) bool masks; deg: (B,K) drug-degree."""
        Pa = self.eff_proj(Ea)  # (B,K,d)
        Pb = self.eff_proj(Eb)
        if self.effect_mode == "bagpool":
            # ablation: independent mean-pool then compose (no cross-selection)
            va = _masked_mean(Pa, ma); vb = _masked_mean(Pb, mb)
            comp = va * vb
            return self.eff_out(comp).squeeze(-1)
        # cross-attention select-compose over K x K candidate effect pairs
        scores = torch.einsum("bkd,bld->bkl", Pa, Pb) / (self.d ** 0.5)  # (B,K,K)
        # frequency debias: down-weight pairs involving generic (high-degree) effects
        bias = -self.freq_debias * (
            torch.log1p(deg_a.clamp(min=0)).unsqueeze(2)
            + torch.log1p(deg_b.clamp(min=0)).unsqueeze(1)
        )
        scores = scores + bias
        pair_mask = ma.unsqueeze(2) & mb.unsqueeze(1)  # (B,K,K)
        # effect-pair dropout (train-time): randomly drop valid candidate pairs
        if self.training and self.eff_pair_dropout > 0.0:
            keep = torch.rand_like(scores) >= self.eff_pair_dropout
            pair_mask = pair_mask & keep
        scores = scores.masked_fill(~pair_mask, float("-inf"))
        flat = scores.flatten(1)  # (B, K*K)
        # sparse top-k pair selection: keep only the top-k candidate pairs per drug-pair
        if self.eff_topk_pair > 0 and self.eff_topk_pair < flat.size(1):
            kth = torch.topk(flat, self.eff_topk_pair, dim=1).values[:, -1:]  # (B,1)
            flat = flat.masked_fill(flat < kth, float("-inf"))
        # all-masked rows -> zero contribution (no effect neighbors / all dropped)
        allmask = torch.isinf(flat).all(dim=1)
        logits = flat.masked_fill(torch.isinf(flat), -1e9) / max(self.eff_attn_temp, 1e-6)
        attn = torch.softmax(logits, dim=1)
        attn = attn.masked_fill(allmask.unsqueeze(1), 0.0)
        # attention entropy (diagnostic; potential reg target)
        with torch.no_grad():
            p = attn.clamp(min=1e-12)
            self._last_attn_entropy = float((-(p * p.log()).sum(dim=1)).mean().item())
        attn = attn.view_as(scores)  # (B,K,K)
        comp = torch.einsum("bkl,bkd,bld->bd", attn, Pa, Pb)  # (B,d)
        self._last_attn = attn
        return self.eff_out(comp).squeeze(-1)

    def gate_value(self, mol_counts, eff_counts):
        g_in = torch.cat([mol_counts, eff_counts], dim=1)
        return torch.sigmoid(self.gate(g_in)).squeeze(-1)


def _masked_mean(x, mask):
    m = mask.float().unsqueeze(-1)
    s = (x * m).sum(dim=1)
    n = m.sum(dim=1).clamp(min=1.0)
    return s / n


class _PerModeEmerGNN_V2(_PerModeEmerGNN_MNAH):
    def __init__(self, *, v2_align_dim: int = 128, v2_d: int = 64, v2_topk: int = 32,
                 v2_effect_mode: str = "xattn", v2_freq_debias: float = 1.0,
                 v2_effect_shuffle_control: bool = False, v2_shuffle_seed: int = 777,
                 v2_eff_topk_pair: int = 0, v2_eff_attn_temp: float = 1.0,
                 v2_eff_pair_dropout: float = 0.0, v2_eff_aux_bce: float = 0.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.v2_align_dim = int(v2_align_dim)
        self.v2_d = int(v2_d)
        self.v2_topk = int(v2_topk)
        self.v2_effect_mode = str(v2_effect_mode)
        self.v2_freq_debias = float(v2_freq_debias)
        self.v2_effect_shuffle_control = bool(v2_effect_shuffle_control)
        self.v2_shuffle_seed = int(v2_shuffle_seed)
        self.v2_eff_topk_pair = int(v2_eff_topk_pair)
        self.v2_eff_attn_temp = float(v2_eff_attn_temp)
        self.v2_eff_pair_dropout = float(v2_eff_pair_dropout)
        # codex 019e6272: aux BCE deep-supervision weight on eff_logit (0 = off => MNAH-equiv)
        self.v2_eff_aux_bce = float(v2_eff_aux_bce)
        self._v2_loaded = False

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        return V2Head(align_dim=self.v2_align_dim, d=self.v2_d,
                      n_mol_cols=len(_MOL_COLS), n_eff_cols=len(_EFF_COLS),
                      dropout=self.mnah_dropout, effect_mode=self.v2_effect_mode,
                      freq_debias=self.v2_freq_debias,
                      eff_topk_pair=self.v2_eff_topk_pair,
                      eff_attn_temp=self.v2_eff_attn_temp,
                      eff_pair_dropout=self.v2_eff_pair_dropout)

    def _load_v2_assets(self) -> None:
        if self._v2_loaded:
            return
        dev = self.device
        # aligned molecular embedding z_m per drug
        al = np.load(ALIGN_NPZ, allow_pickle=True)
        self._zm_row = {str(d): i for i, d in enumerate(al["drug_ids"])}
        self._zm = torch.from_numpy(al["z_m"].astype(np.float32)).to(dev)
        # PubMedBERT node embeddings for effect neighbors
        obj = torch.load(PUBMEDBERT_PT, map_location="cpu", weights_only=False)
        self._node_emb = obj["embeddings"].float().to(dev)  # (178029, 768)
        # effect-neighbor CSR -> per-drug top-K (PD-composable kinds, rarest by drugdeg)
        eff = np.load(EFFNBR_NPZ, allow_pickle=True)
        eids = [str(d) for d in eff["drug_ids"]]
        indptr = eff["indptr"]; nbr_rows = eff["nbr_rows"]
        nbr_kind = eff["nbr_kind"]; nbr_deg = eff["nbr_drugdeg"]
        K = self.v2_topk
        N = len(eids)
        idx_mat = np.zeros((N, K), dtype=np.int64)
        deg_mat = np.zeros((N, K), dtype=np.float32)
        mask_mat = np.zeros((N, K), dtype=bool)
        for i in range(N):
            s, e = indptr[i], indptr[i + 1]
            rows = nbr_rows[s:e]; kinds = nbr_kind[s:e]; degs = nbr_deg[s:e]
            keep = kinds <= 2  # PD-composable: side_effect/phenotype/symptom
            rows = rows[keep]; degs = degs[keep]
            if len(rows) == 0:
                continue
            order = np.argsort(degs)[:K]  # rarest (smallest drug-degree) first
            rows = rows[order]; degs = degs[order]
            idx_mat[i, :len(rows)] = rows
            deg_mat[i, :len(rows)] = degs
            mask_mat[i, :len(rows)] = True
        if self.v2_effect_shuffle_control:
            perm = np.random.default_rng(self.v2_shuffle_seed).permutation(N)
            idx_mat = idx_mat[perm]; deg_mat = deg_mat[perm]; mask_mat = mask_mat[perm]
            print(f"[v2] *** EFFECT SHUFFLE CONTROL *** permuted {N} drug->effect-set bindings",
                  flush=True)
        self._eff_row = {d: i for i, d in enumerate(eids)}
        self._eff_idx = torch.from_numpy(idx_mat).to(dev)
        self._eff_deg = torch.from_numpy(deg_mat).to(dev)
        self._eff_mask = torch.from_numpy(mask_mat).to(dev)
        self._zm_dim = self._zm.shape[1]
        self._v2_loaded = True
        cov_zm = sum(1 for d in self._eff_row if d in self._zm_row)
        print(f"[v2] assets loaded: z_m {tuple(self._zm.shape)}, effect topK={K}, "
              f"drugs with eff set={int(mask_mat.any(1).sum())}/{N}", flush=True)

    def _drug_rows(self, ids, row_map, default=0):
        return torch.tensor([row_map.get(str(x), default) for x in ids],
                            dtype=torch.long, device=self.device)

    def _gather_pair_inputs(self, batch_df: pd.DataFrame):
        a = batch_df["drug_a_id"].astype(str).tolist()
        b = batch_df["drug_b_id"].astype(str).tolist()
        # z_m (missing -> zeros)
        za = self._zm[self._drug_rows(a, self._zm_row)]
        zb = self._zm[self._drug_rows(b, self._zm_row)]
        a_has = torch.tensor([str(x) in self._zm_row for x in a], device=self.device).float().unsqueeze(1)
        b_has = torch.tensor([str(x) in self._zm_row for x in b], device=self.device).float().unsqueeze(1)
        za = za * a_has; zb = zb * b_has
        # effect neighbor sets
        ra = self._drug_rows(a, self._eff_row); rb = self._drug_rows(b, self._eff_row)
        Ea = self._node_emb[self._eff_idx[ra]]  # (B,K,768)
        Eb = self._node_emb[self._eff_idx[rb]]
        ma = self._eff_mask[ra]; mb = self._eff_mask[rb]
        dega = self._eff_deg[ra]; degb = self._eff_deg[rb]
        # zero out effect emb for drugs missing from eff cache (row default 0 but mask handles)
        a_in = torch.tensor([str(x) in self._eff_row for x in a], device=self.device)
        b_in = torch.tensor([str(x) in self._eff_row for x in b], device=self.device)
        ma = ma & a_in.unsqueeze(1); mb = mb & b_in.unsqueeze(1)
        return za, zb, Ea, Eb, ma, mb, dega, degb

    def _breakdown(self, head, tail, edge_src, edge_dst, edge_rel, batch_df):
        """Full per-pair breakdown. Returns dict of tensors (all length B)."""
        self._load_v2_assets()
        emergnn_logit = self._model(head, tail, edge_src, edge_dst, edge_rel)
        feats = self._lookup_features(batch_df)  # (B,22) normalized counts
        mol_counts = feats[:, _MOL_COLS]
        eff_counts = feats[:, _EFF_COLS]
        za, zb, Ea, Eb, ma, mb, dega, degb = self._gather_pair_inputs(batch_df)
        h: V2Head = self._aux_mlp
        mol_logit = h.molecular_logit(za, zb, mol_counts)
        if self.v2_effect_mode == "count":
            eff_logit = h.effect_logit_count(eff_counts)
        else:
            eff_logit = h.effect_logit(Ea, Eb, ma, mb, dega, degb)
        g = h.gate_value(mol_counts, eff_counts)
        head_logit = g * mol_logit + (1.0 - g) * eff_logit
        combined = emergnn_logit + self._beta() * head_logit
        return {
            "combined": combined, "emergnn": emergnn_logit, "head": head_logit,
            "mol": mol_logit, "eff": eff_logit, "g": g,
        }

    def _combined_logit(self, head, tail, edge_src, edge_dst, edge_rel, batch_df):
        d = self._breakdown(head, tail, edge_src, edge_dst, edge_rel, batch_df)
        return d["combined"], d["emergnn"], d["head"]

    def _aux_losses(self, d: dict, batch_df, y) -> "torch.Tensor":
        """Hook for subclasses to add aux losses (default 0). v2res adds align+orth."""
        return torch.zeros((), device=self.device)

    def fit(self, train, val=None, *, kg=None):
        """Copy of MNAH.fit + optional aux-BCE deep-supervision on eff_logit.

        Identical to _PerModeEmerGNN_MNAH.fit when v2_eff_aux_bce == 0. The ONLY change is
        the per-batch loss: total = BCE(combined) + v2_eff_aux_bce * BCE(eff_logit) (codex
        019e6272 falsification test). Uses self._breakdown to obtain eff_logit per batch.
        """
        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)
        try:
            train_neg = train.get_train_negatives(0, regenerate=True)[["drug_a_id", "drug_b_id"]]
        except Exception as exc:
            print(f"[v2] could not load train negatives for normalizer: {exc}", flush=True)
            train_neg = None
        self._load_feature_cache(train.splits.train, train_neg)

        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi
        self._model = EmerGNN(
            n_ent=self._n_ent, n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim, length=self.length, feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)

        self._aux_mlp = self._build_aux_head(self._aux_in_dim).to(self.device)
        if self.mnah_init_beta > 0:
            raw_init = float(np.log(np.exp(self.mnah_init_beta) - 1.0))
        else:
            raw_init = -5.0
        self._raw_beta = nn.Parameter(torch.tensor(raw_init, device=self.device))
        print(f"[v2] aux head built; raw_beta init={raw_init:.3f} -> beta={self.mnah_init_beta:.3f}; "
              f"eff_aux_bce={self.v2_eff_aux_bce}", flush=True)

        params = list(self._model.parameters()) + list(self._aux_mlp.parameters()) + [self._raw_beta]
        opt = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        pos_df = train.splits.train.copy()
        a_ids = pos_df["drug_a_id"].astype(str).map(self._entity2id)
        b_ids = pos_df["drug_b_id"].astype(str).map(self._entity2id)
        valid_mask = a_ids.notna() & b_ids.notna()
        train_ddi_int = np.stack([
            a_ids[valid_mask].astype(np.int64).to_numpy(),
            b_ids[valid_mask].astype(np.int64).to_numpy(),
            np.full(int(valid_mask.sum()), n_kg_rel, dtype=np.int64),
        ], axis=1)
        eval_kg_triplets = np.concatenate([train_ddi_int, self._kg_triplets], axis=0)
        esrc, edst, erel = build_edge_lists_from_triplets(
            eval_kg_triplets, self._n_ent, n_base_rel_with_ddi)
        self._eval_edges = (
            torch.from_numpy(esrc).long().to(self.device),
            torch.from_numpy(edst).long().to(self.device),
            torch.from_numpy(erel).long().to(self.device),
        )

        rng = np.random.default_rng(0)
        best_val_auc = -1.0
        best_state: dict | None = None
        lam = self.v2_eff_aux_bce

        for epoch in range(self.n_epochs):
            epoch_kg, train_pos_targets = shuffle_train(
                train_ddi_int, self._kg_triplets,
                self.shuffle_train_mode, ratio=self.shuffle_ratio, rng=rng,
                extra_kg_ent=self._kg_entity_set,
            )
            if len(train_pos_targets) == 0:
                continue
            esrc, edst, erel = build_edge_lists_from_triplets(
                epoch_kg, self._n_ent, n_base_rel_with_ddi)
            edge_src = torch.from_numpy(esrc).long().to(self.device)
            edge_dst = torch.from_numpy(edst).long().to(self.device)
            edge_rel = torch.from_numpy(erel).long().to(self.device)

            pre_neg = train.get_train_negatives(epoch, regenerate=True)
            n_target_pos = len(train_pos_targets)
            if len(pre_neg) >= n_target_pos:
                neg_idx = rng.choice(len(pre_neg), size=n_target_pos, replace=False)
                neg = pre_neg.iloc[neg_idx].reset_index(drop=True)
            else:
                neg = pre_neg

            id2entity = {v: k for k, v in self._entity2id.items()}
            pos_target_df = pd.DataFrame({
                "drug_a_id": [id2entity[int(h)] for h in train_pos_targets[:, 0]],
                "drug_b_id": [id2entity[int(t)] for t in train_pos_targets[:, 1]],
            })
            pairs_df = pd.concat([
                pos_target_df.assign(label=1),
                neg[["drug_a_id", "drug_b_id"]].assign(label=0),
            ], ignore_index=True).sample(frac=1, random_state=epoch).reset_index(drop=True)

            self._model.train(); self._aux_mlp.train()
            t_epoch = time.time()
            losses = []
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_d = head.to(self.device); tail_d = tail.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                d = self._breakdown(head_d, tail_d, edge_src, edge_dst, edge_rel, batch)
                loss = F.binary_cross_entropy_with_logits(d["combined"], y, reduction="sum")
                if lam > 0:
                    loss = loss + lam * F.binary_cross_entropy_with_logits(
                        d["eff"], y, reduction="sum")
                loss = loss + self._aux_losses(d, batch, y)  # hook (default 0; v2res adds align+orth)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                opt.step()
                losses.append(loss.item() / max(len(batch), 1))

            if val is not None:
                self._model.eval(); self._aux_mlp.eval()
                br = self._validate_branches(val)
                v_auc = br["combined"]
                self._model.train(); self._aux_mlp.train()
                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    best_state = {
                        "emergnn": copy.deepcopy(self._model.state_dict()),
                        "aux": copy.deepcopy(self._aux_mlp.state_dict()),
                        "raw_beta": float(self._raw_beta.detach().cpu().item()),
                    }
                scheduler.step(v_auc)
                ep_time = time.time() - t_epoch
                beta_val = float(self._beta().detach().cpu().item())
                ent = getattr(self._aux_mlp, "_last_attn_entropy", None)
                print(f"[v2] [ep {epoch+1}/{self.n_epochs}] loss={np.mean(losses):.4f} "
                      f"time={ep_time:.0f}s val_combined={v_auc:.4f} val_emer={br['emergnn']:.4f} "
                      f"val_aux={br['aux']:.4f} beta={beta_val:.3f} "
                      f"eff_attn_ent={ent if ent is None else round(ent,3)}", flush=True)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state["emergnn"])
            self._aux_mlp.load_state_dict(best_state["aux"])
            with torch.no_grad():
                self._raw_beta.copy_(torch.tensor(best_state["raw_beta"], device=self.device))
            print(f"[v2] loaded best val_combined={best_val_auc:.4f}", flush=True)

    @torch.no_grad()
    def predict_channels(self, pairs: pd.DataFrame) -> dict:
        """Eval-only per-pair channel dump for diagnostics (codex 019e6251)."""
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            edge_src, edge_dst, edge_rel = self._eval_edges
        else:
            edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval(); self._aux_mlp.eval()
        keys = ["combined", "emergnn", "head", "mol", "eff", "g"]
        out = {k: np.empty(len(pairs), dtype=np.float32) for k in keys}
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            hd, tl = self._pair_indices(batch)
            hd = hd.to(self.device); tl = tl.to(self.device)
            d = self._breakdown(hd, tl, edge_src, edge_dst, edge_rel, batch)
            for k in keys:
                out[k][start:start + len(batch)] = d[k].detach().cpu().numpy()
        return out


__all__ = ["_PerModeEmerGNN_V2", "V2Head"]
