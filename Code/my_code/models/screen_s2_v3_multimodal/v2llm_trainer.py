"""E-llm — MNAH (EmerGNN + meeting-node counts) + LLM-pharmacology RESIDUAL channel.

codex 019e6770: molecular is largely redundant with KG in this cold-start; the better bet for
NEW signal is LLM-distilled pharmacology. This keeps the FULL MNAH backbone (count head = i2)
and ADDS an LLM residual head, so the comparison is the clean "molecular/LLM ON TOP OF KG":

    combined = emergnn_logit + beta * count_logit + beta_llm * llm_logit

count head identical to MNAH (so MNAH 0.772 is recoverable at beta_llm->0). LLM head: per-drug
sanitized-pharmacology PubMedBERT embedding (leakage-redacted), symmetric pair feature
[z_a*z_b, |z_a-z_b|] -> MLP. Per-drug & leakage-safe (no pairwise LLM calls).

No fit() override needed: the combined head's params (count + llm + beta_llm) all live in
self._aux_mlp and join the MNAH optimizer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v2_meetnode.mnah_trainer import _PerModeEmerGNN_MNAH, AuxMLP

LLM_NPZ = PROJECT_ROOT / "Code/data/_cache/llm_pharma/llm_text_pubmedbert.npz"


class CountPlusLLMHead(nn.Module):
    def __init__(self, *, count_in: int = 22, count_hidden: int = 32, llm_in: int = 768,
                 d: int = 128, dropout: float = 0.2, beta_llm_init: float = 0.5):
        super().__init__()
        self.count_mlp = AuxMLP(in_dim=count_in, hidden=count_hidden, dropout=dropout)
        self.llm_mlp = nn.Sequential(nn.Linear(2 * llm_in, d), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(d, 1))
        # learnable LLM scale via softplus(raw) so beta_llm >= 0
        self.raw_beta_llm = nn.Parameter(torch.tensor(float(np.log(np.exp(beta_llm_init) - 1.0))))

    def count_logit(self, feats22):
        return self.count_mlp(feats22)

    def llm_logit(self, z_a, z_b):
        feat = torch.cat([z_a * z_b, (z_a - z_b).abs()], dim=1)
        return self.llm_mlp(feat).squeeze(-1)

    def beta_llm(self):
        return F.softplus(self.raw_beta_llm)


class _PerModeEmerGNN_V2LLM(_PerModeEmerGNN_MNAH):
    def __init__(self, *, v2llm_d: int = 128, v2llm_beta_init: float = 0.5,
                 v2llm_shuffle_control: bool = False, v2llm_shuffle_seed: int = 999,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.v2llm_d = int(v2llm_d)
        self.v2llm_beta_init = float(v2llm_beta_init)
        self.v2llm_shuffle_control = bool(v2llm_shuffle_control)
        self.v2llm_shuffle_seed = int(v2llm_shuffle_seed)
        self._llm_loaded = False

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        return CountPlusLLMHead(count_in=22, count_hidden=self.mnah_hidden, llm_in=768,
                                d=self.v2llm_d, dropout=self.mnah_dropout,
                                beta_llm_init=self.v2llm_beta_init)

    def _load_llm_assets(self) -> None:
        if self._llm_loaded:
            return
        ll = np.load(LLM_NPZ, allow_pickle=True)
        ids = [str(x) for x in ll["drug_ids"]]
        emb = ll["emb"].astype(np.float32)
        if self.v2llm_shuffle_control:
            perm = np.random.default_rng(self.v2llm_shuffle_seed).permutation(len(ids))
            emb = emb[perm]
            print(f"[v2llm] *** LLM SHUFFLE CONTROL *** permuted {len(ids)} drug->emb", flush=True)
        self._llm_row = {d: i for i, d in enumerate(ids)}
        self._llm = torch.from_numpy(emb).to(self.device)
        self._llm_loaded = True
        print(f"[v2llm] LLM emb loaded: {emb.shape}, leak_flagged="
              f"{int(ll['leakage_flag'].sum()) if 'leakage_flag' in ll.files else 'NA'}", flush=True)

    def _llm_vecs(self, ids):
        idx = torch.tensor([self._llm_row.get(str(x), -1) for x in ids], dtype=torch.long,
                           device=self.device)
        ok = (idx >= 0).float().unsqueeze(1)
        return self._llm[idx.clamp(min=0)] * ok

    def _combined_logit(self, head, tail, edge_src, edge_dst, edge_rel, batch_df):
        self._load_llm_assets()
        emergnn_logit = self._model(head, tail, edge_src, edge_dst, edge_rel)
        feats = self._lookup_features(batch_df)
        h: CountPlusLLMHead = self._aux_mlp
        count_logit = h.count_logit(feats)
        za = self._llm_vecs(batch_df["drug_a_id"].astype(str).tolist())
        zb = self._llm_vecs(batch_df["drug_b_id"].astype(str).tolist())
        llm_logit = h.llm_logit(za, zb)
        combined = emergnn_logit + self._beta() * count_logit + h.beta_llm() * llm_logit
        aux = self._beta() * count_logit + h.beta_llm() * llm_logit
        return combined, emergnn_logit, aux

    @torch.no_grad()
    def predict_channels(self, pairs: pd.DataFrame) -> dict:
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            es, ed, er = self._eval_edges
        else:
            es, ed, er = self._edges_on_device()
        self._model.eval(); self._aux_mlp.eval()
        self._load_llm_assets()
        h: CountPlusLLMHead = self._aux_mlp
        keys = ["combined", "emergnn", "count", "llm"]
        out = {k: np.empty(len(pairs), dtype=np.float32) for k in keys}
        for s in range(0, len(pairs), self.batch_size):
            b = pairs.iloc[s:s + self.batch_size]
            hd, tl = self._pair_indices(b); hd = hd.to(self.device); tl = tl.to(self.device)
            emer = self._model(hd, tl, es, ed, er)
            feats = self._lookup_features(b)
            cl = h.count_logit(feats)
            za = self._llm_vecs(b["drug_a_id"].astype(str).tolist())
            zb = self._llm_vecs(b["drug_b_id"].astype(str).tolist())
            ll = h.llm_logit(za, zb)
            out["combined"][s:s+len(b)] = (emer + self._beta()*cl + h.beta_llm()*ll).cpu().numpy()
            out["emergnn"][s:s+len(b)] = emer.cpu().numpy()
            out["count"][s:s+len(b)] = (self._beta()*cl).cpu().numpy()
            out["llm"][s:s+len(b)] = (h.beta_llm()*ll).cpu().numpy()
        return out


__all__ = ["_PerModeEmerGNN_V2LLM", "CountPlusLLMHead"]
