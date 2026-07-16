"""RGCNRankWrapper — R-GCN (node paradigm) plugged into the rank-analysis harness.

Wraps the existing RGCNPairModel (binary, 1-logit head) / RGCNMultiClass (K-way
head) + the merged KG, implementing setup / train_epoch / encode_pairs. Encodes the
whole KG once per graph then reads out pairs (node paradigm). DRUG-ID space; the
harness-provided ScoringContext declares which DDI facts are present (leak-free).

NOTE: "wrapper" = model-specific glue to the shared harness, NOT the method's
semantic M_A·M_B adapter. The word "adapter" is reserved for that method module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "Code"))
sys.path.insert(0, str(_ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
from train_rgcn_rank_pilot import build_relation_edges, RGCNPairModel  # noqa: E402
from train_rgcn_multicls_s2 import RGCNMultiClass  # noqa: E402

from ..framework import RankModel, EpochData
from ..specs import PairEncoding, ScoringContext, TaskSpec, TrainEpochOutput

_DEFAULTS = dict(hidden=128, num_layers=2, num_bases=16, bottleneck=256,
                 lr=1e-3, weight_decay=5e-4, dropout=0.2, amp=True, seed=42)


class RGCNRankWrapper(RankModel):
    def setup(self, task: TaskSpec, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        kg = MergedKG.from_parquet(DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
        self.id2i = kg.id_to_idx; n = kg.n_nodes
        logdeg = np.log1p(kg.degree.astype(np.float64))
        x = np.zeros((n, N_TYPES + 1), dtype=np.float32)
        x[np.arange(n), kg.type_id.astype(np.int64)] = 1.0
        x[:, N_TYPES] = ((logdeg - logdeg.mean()) / (logdeg.std() + 1e-8)).astype(np.float32)
        self.x = torch.from_numpy(x).to(self.device)
        bio_ei, bio_et, self.num_base_rel, _, _ = build_relation_edges(DEFAULT_EDGES_PATH, self.id2i)
        self.bio_ei = bio_ei.to(self.device); self.bio_et = bio_et.to(self.device)

        # DDI relations: binary -> 1 slot; multiclass -> K typed slots
        self.n_ddi_rel = 1 if task.is_binary else task.n_classes
        aug_nr = self.num_base_rel + self.n_ddi_rel
        common = dict(in_dim=x.shape[1], hidden=self.hp["hidden"], bottleneck=self.hp["bottleneck"],
                      dropout=self.hp["dropout"], num_relations=aug_nr,
                      num_bases=self.hp["num_bases"], num_layers=self.hp["num_layers"])
        if task.is_binary:
            self.model = RGCNPairModel(**common).to(self.device)
        else:
            self.model = RGCNMultiClass(**common, n_classes=task.n_classes).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.use_amp = bool(self.hp["amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # -- graph + pair index helpers ------------------------------------------
    def _idx(self, ids) -> np.ndarray:
        miss = [str(x) for x in ids if str(x) not in self.id2i]
        if miss:
            raise ValueError(f"{len(miss)} drug ids not in merged KG, e.g. {miss[:5]}")
        return np.array([self.id2i[str(x)] for x in ids], dtype=np.int64)

    def _aug_graph(self, ctx: ScoringContext):
        ddi = ctx.ddi_edges
        if len(ddi) == 0:
            return self.bio_ei, self.bio_et
        a = self._idx(ddi[:, 0]); b = self._idx(ddi[:, 1])
        rel = self.num_base_rel + (np.zeros(len(a), np.int64) if self.task.is_binary
                                   else ddi[:, 2].astype(np.int64))
        di = torch.from_numpy(np.vstack([np.concatenate([a, b]),
                                         np.concatenate([b, a])])).to(self.device)
        dt = torch.from_numpy(np.concatenate([rel, rel])).to(self.device)
        return torch.cat([self.bio_ei, di], dim=1), torch.cat([self.bio_et, dt])

    def _pair_idx(self, pairs):
        pairs = np.asarray(pairs)
        ia = self._idx(pairs[:, 0]); ib = self._idx(pairs[:, 1])
        return (torch.as_tensor(ia, device=self.device), torch.as_tensor(ib, device=self.device))

    # -- RankModel contract --------------------------------------------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        aug_ei, aug_et = self._aug_graph(epoch.fact_context)
        ia, ib = self._pair_idx(epoch.target_pairs)
        y = torch.as_tensor(epoch.target_labels, device=self.device)
        self.model.train(); self.opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits, _ = self.model(self.x, aug_ei, aug_et, ia, ib)
            if self.task.is_binary:
                loss = F.binary_cross_entropy_with_logits(logits, y.float())
            else:
                loss = F.cross_entropy(logits, y.long())
        self.scaler.scale(loss).backward(); self.scaler.step(self.opt); self.scaler.update()
        return TrainEpochOutput(mean_loss=float(loss.item()), n_targets=len(epoch.target_labels))

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        aug_ei, aug_et = self._aug_graph(context)
        ia, ib = self._pair_idx(pairs)
        self.model.eval()
        h = self.model.encode(self.x, aug_ei, aug_et)
        z = self.model.pair_z(h, ia, ib)
        logits = self.model.out(z)                       # (n,1) binary | (n,K) multiclass
        return PairEncoding(pair_ids=np.asarray(pairs),
                            pair_repr=z.cpu().numpy().astype(np.float32),
                            logits=logits.cpu().numpy().astype(np.float32),
                            repr_kind="node_readout", repr_stage="pair_mlp_out")

    @torch.no_grad()
    def head_from_repr(self, pair_repr) -> np.ndarray:
        z = torch.as_tensor(np.asarray(pair_repr), dtype=torch.float32, device=self.device)
        self.model.eval()
        return self.model.out(z).cpu().numpy().astype(np.float32)   # (n,1) binary | (n,K) multi

    def known_drugs(self) -> set:
        return set(self.id2i.keys())

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)
