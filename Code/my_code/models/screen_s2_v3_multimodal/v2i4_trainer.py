"""E-i4 — MNAH + structured-LLM MECHANISTIC pair features (codex 019e67af final feature test).

Replaces the inert free-text LLM CLS approach (E-llm) with EXPLICIT pairwise mechanistic
features derived from the leakage-sanitized structured fields. Keeps the full MNAH (count
head = i2) and adds an I4 residual head:

    combined = emergnn_logit + beta * count_logit + beta_i4 * i4_logit

13 pair features per drug pair (interpretable, mechanistic):
  PK:  shared CYP substrate, shared CYP inhibitor, shared CYP inducer,
       CYP inhibitor->substrate cross (asym, both directions summed),
       CYP inducer->substrate cross,
       shared transporter substrate, transporter inhib->substrate cross
  PD:  shared therapeutic class, shared primary target, shared pd_effects,
       shared toxicity mechanisms, shared clearance pathway
  Other: jaccard of full-token union/intersection

Hard-stop rule (codex 019e67af): lift < +1.5pt AND shuffle close to main -> pivot.
"""
from __future__ import annotations

import json
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

I4_JSON = PROJECT_ROOT / "Code/data/_cache/llm_pharma/i4_typed_sets.json"

PAIR_FEAT_NAMES = [
    "shared_cyp_substrate", "shared_cyp_inhibitor", "shared_cyp_inducer",
    "cyp_inhib_subst", "cyp_induc_subst",
    "shared_transporter_substrate", "transporter_inhib_subst",
    "shared_target", "shared_class", "shared_pd_effect",
    "shared_toxicity", "shared_clearance", "jaccard_overall",
]
N_PAIR = len(PAIR_FEAT_NAMES)


def _pair_feat(sa: dict, sb: dict) -> np.ndarray:
    """13 mechanistic overlap features for a drug pair (both sets of typed fields)."""
    if sa is None or sb is None:
        return np.zeros(N_PAIR, dtype=np.float32)

    def S(d, k):
        return set(d.get(k, []) or [])

    cs_a, cs_b = S(sa, "cyp_substrate"), S(sb, "cyp_substrate")
    ci_a, ci_b = S(sa, "cyp_inhibitor"), S(sb, "cyp_inhibitor")
    cu_a, cu_b = S(sa, "cyp_inducer"), S(sb, "cyp_inducer")
    ts_a, ts_b = S(sa, "transporter_substrate"), S(sb, "transporter_substrate")
    ti_a, ti_b = S(sa, "transporter_inhibitor"), S(sb, "transporter_inhibitor")
    tc_a, tc_b = S(sa, "therapeutic_class"), S(sb, "therapeutic_class")
    tg_a, tg_b = S(sa, "primary_targets"), S(sb, "primary_targets")
    pd_a, pd_b = S(sa, "pd_effects"), S(sb, "pd_effects")
    tx_a, tx_b = S(sa, "toxicity_mechanisms"), S(sb, "toxicity_mechanisms")
    cl_a, cl_b = S(sa, "clearance"), S(sb, "clearance")

    f = np.zeros(N_PAIR, dtype=np.float32)
    f[0] = len(cs_a & cs_b); f[1] = len(ci_a & ci_b); f[2] = len(cu_a & cu_b)
    f[3] = len(ci_a & cs_b) + len(ci_b & cs_a)
    f[4] = len(cu_a & cs_b) + len(cu_b & cs_a)
    f[5] = len(ts_a & ts_b); f[6] = len(ti_a & ts_b) + len(ti_b & ts_a)
    f[7] = len(tg_a & tg_b); f[8] = len(tc_a & tc_b); f[9] = len(pd_a & pd_b)
    f[10] = len(tx_a & tx_b); f[11] = len(cl_a & cl_b)
    # jaccard across all token universes
    all_a = cs_a | ci_a | cu_a | ts_a | ti_a | tc_a | tg_a | pd_a | tx_a | cl_a
    all_b = cs_b | ci_b | cu_b | ts_b | ti_b | tc_b | tg_b | pd_b | tx_b | cl_b
    u = len(all_a | all_b)
    f[12] = (len(all_a & all_b) / u) if u > 0 else 0.0
    return f


class CountPlusI4Head(nn.Module):
    def __init__(self, *, count_in: int = 22, count_hidden: int = 32, dropout: float = 0.2,
                 i4_hidden: int = 32, beta_i4_init: float = 0.5):
        super().__init__()
        self.count_mlp = AuxMLP(in_dim=count_in, hidden=count_hidden, dropout=dropout)
        self.i4_mlp = nn.Sequential(nn.Linear(N_PAIR, i4_hidden), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(i4_hidden, 1))
        self.raw_beta_i4 = nn.Parameter(torch.tensor(float(np.log(np.exp(beta_i4_init) - 1.0))))

    def count_logit(self, feats22):
        return self.count_mlp(feats22)

    def i4_logit(self, pair_feat):
        return self.i4_mlp(pair_feat).squeeze(-1)

    def beta_i4(self):
        return F.softplus(self.raw_beta_i4)


class _PerModeEmerGNN_V2I4(_PerModeEmerGNN_MNAH):
    def __init__(self, *, v2i4_hidden: int = 32, v2i4_beta_init: float = 0.5,
                 v2i4_shuffle_control: bool = False, v2i4_random_control: bool = False,
                 v2i4_shuffle_seed: int = 1234,
                 v2i4_pair_normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.v2i4_hidden = int(v2i4_hidden)
        self.v2i4_beta_init = float(v2i4_beta_init)
        self.v2i4_shuffle_control = bool(v2i4_shuffle_control)
        self.v2i4_random_control = bool(v2i4_random_control)
        self.v2i4_shuffle_seed = int(v2i4_shuffle_seed)
        self.v2i4_pair_normalize = bool(v2i4_pair_normalize)
        self._i4_loaded = False
        self._i4_sets: dict[str, dict] = {}
        self._i4_mean = None
        self._i4_std = None

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        return CountPlusI4Head(count_in=22, count_hidden=self.mnah_hidden,
                               dropout=self.mnah_dropout,
                               i4_hidden=self.v2i4_hidden, beta_i4_init=self.v2i4_beta_init)

    def _load_i4_assets(self) -> None:
        if self._i4_loaded:
            return
        sets = json.loads(I4_JSON.read_text(encoding="utf-8"))
        # codex 019e769b sanitizer audit: drop tokens containing DDI/interaction language
        # from free-text fields, in case residual interaction phrasing slipped through.
        BAD = ("interact", "coadminist", "co-administ", "combined with", "combination with",
               "concomitant", "avoid with", "contraindicated", "with inhibitor", "with inducer",
               "increase levels", "decrease levels", "co-medic", "co-prescri")
        FREE = ("therapeutic_class", "primary_targets", "pd_effects",
                "toxicity_mechanisms", "clearance")
        n_drop = 0
        for d, fields in sets.items():
            for k in FREE:
                kept = [t for t in fields.get(k, []) if not any(b in t for b in BAD)]
                n_drop += len(fields.get(k, [])) - len(kept)
                fields[k] = kept
        print(f"[v2i4] sanitizer-2 dropped {n_drop} free-text tokens with interaction language",
              flush=True)
        if self.v2i4_shuffle_control:
            # permute only AMONG drugs that have at least one non-empty field, so
            # missingness/coverage structure is preserved (codex 019e769b control fix).
            keys_nonempty = [d for d, f in sets.items()
                             if any(len(v) > 0 for v in f.values())]
            keys_empty = [d for d in sets if d not in set(keys_nonempty)]
            rng = np.random.default_rng(self.v2i4_shuffle_seed)
            perm = rng.permutation(len(keys_nonempty))
            new = {keys_nonempty[i]: sets[keys_nonempty[perm[i]]]
                   for i in range(len(keys_nonempty))}
            for d in keys_empty:
                new[d] = sets[d]  # keep empty drugs as empty
            sets = new
            print(f"[v2i4] *** SHUFFLE CONTROL *** permuted {len(keys_nonempty)} non-empty "
                  f"drug->set bindings (empty drugs preserved)", flush=True)
        if self.v2i4_random_control:
            # replace each drug's typed sets with random tokens (preserve cardinalities)
            rng = np.random.default_rng(self.v2i4_shuffle_seed)
            vocab_pool = [f"rand_{i}" for i in range(2000)]
            for d, fields in sets.items():
                for k, v in fields.items():
                    n = len(v)
                    fields[k] = list(rng.choice(vocab_pool, size=n, replace=False)) if n else []
            print(f"[v2i4] *** RANDOM CONTROL *** replaced typed tokens with random",
                  flush=True)
        self._i4_sets = sets
        self._i4_loaded = True
        print(f"[v2i4] typed sets loaded: {len(sets)} drugs", flush=True)

    def _build_pair_feats(self, batch_df: pd.DataFrame) -> torch.Tensor:
        self._load_i4_assets()
        a = batch_df["drug_a_id"].astype(str).tolist()
        b = batch_df["drug_b_id"].astype(str).tolist()
        feats = np.stack([_pair_feat(self._i4_sets.get(ai), self._i4_sets.get(bi))
                          for ai, bi in zip(a, b)], axis=0)
        if self.v2i4_pair_normalize:
            # train-stats normalization: lazy fit on first call from this batch's stats?
            # Better: fit once on train pairs, then frozen. Fit in fit() before training.
            if self._i4_mean is not None and self._i4_std is not None:
                feats = (feats - self._i4_mean) / self._i4_std
        return torch.from_numpy(feats.astype(np.float32)).to(self.device)

    def _fit_pair_norm(self, train_pos: pd.DataFrame, train_neg: pd.DataFrame | None = None) -> None:
        """Fit normalizer on train POSITIVES + train NEGATIVES (codex 019e769b major fix).
        Pos-only stats would distort because overlap features are systematically higher on
        positives. std floored at 1e-3 to prevent sparse-feature blowup."""
        self._load_i4_assets()
        rows = []
        for src in (train_pos, train_neg):
            if src is None:
                continue
            a = src["drug_a_id"].astype(str).tolist()
            b = src["drug_b_id"].astype(str).tolist()
            rows.extend([_pair_feat(self._i4_sets.get(ai), self._i4_sets.get(bi))
                         for ai, bi in zip(a, b)])
        feats = np.stack(rows, axis=0)
        self._i4_mean = feats.mean(axis=0).astype(np.float32)
        std = feats.std(axis=0).astype(np.float32)
        self._i4_std = np.maximum(std, 1e-3).astype(np.float32)
        n_floored = int((std < 1e-3).sum())
        print(f"[v2i4] pair-feat normalizer fit on {len(feats)} pairs "
              f"(pos+neg); means range [{self._i4_mean.min():.2f},{self._i4_mean.max():.2f}], "
              f"std floored: {n_floored}/{len(std)}", flush=True)

    def _combined_logit(self, head, tail, edge_src, edge_dst, edge_rel, batch_df):
        emergnn_logit = self._model(head, tail, edge_src, edge_dst, edge_rel)
        feats = self._lookup_features(batch_df)
        h: CountPlusI4Head = self._aux_mlp
        count_logit = h.count_logit(feats)
        pair_feat = self._build_pair_feats(batch_df)
        i4_logit = h.i4_logit(pair_feat)
        combined = emergnn_logit + self._beta() * count_logit + h.beta_i4() * i4_logit
        aux = self._beta() * count_logit + h.beta_i4() * i4_logit
        return combined, emergnn_logit, aux

    def fit(self, train, val=None, *, kg=None):
        # codex 019e769b: fit normalizer on TRAIN POS + TRAIN NEG (not pos-only).
        try:
            tr_pos = train.splits.train[["drug_a_id", "drug_b_id"]]
            try:
                tr_neg = train.get_train_negatives(0, regenerate=True)[
                    ["drug_a_id", "drug_b_id"]]
            except Exception as exc:
                print(f"[v2i4] couldn't load train negatives for norm: {exc}", flush=True)
                tr_neg = None
            self._fit_pair_norm(tr_pos, tr_neg)
        except Exception as exc:
            print(f"[v2i4] pair-norm fit failed: {exc}", flush=True)
        return super().fit(train, val=val, kg=kg)

    @torch.no_grad()
    def predict_channels(self, pairs: pd.DataFrame) -> dict:
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            es, ed, er = self._eval_edges
        else:
            es, ed, er = self._edges_on_device()
        self._model.eval(); self._aux_mlp.eval()
        h: CountPlusI4Head = self._aux_mlp
        keys = ["combined", "emergnn", "count", "i4"]
        out = {k: np.empty(len(pairs), dtype=np.float32) for k in keys}
        for s in range(0, len(pairs), self.batch_size):
            b = pairs.iloc[s:s + self.batch_size]
            hd, tl = self._pair_indices(b); hd = hd.to(self.device); tl = tl.to(self.device)
            emer = self._model(hd, tl, es, ed, er)
            feats = self._lookup_features(b)
            cl = h.count_logit(feats)
            pf = self._build_pair_feats(b)
            il = h.i4_logit(pf)
            out["combined"][s:s+len(b)] = (emer + self._beta()*cl + h.beta_i4()*il).cpu().numpy()
            out["emergnn"][s:s+len(b)] = emer.cpu().numpy()
            out["count"][s:s+len(b)] = (self._beta()*cl).cpu().numpy()
            out["i4"][s:s+len(b)] = (h.beta_i4()*il).cpu().numpy()
        return out


__all__ = ["_PerModeEmerGNN_V2I4", "CountPlusI4Head", "PAIR_FEAT_NAMES"]
