"""KnowDDIRankWrapper — KnowDDI (subgraph paradigm) plugged into the rank-analysis harness.

Reuses the codex-reviewed KnowDDI pilot machinery (build_graph_tensors /
extract_enclosing_subgraph / prepare_subgraph / KnowDDIDataset / collate /
KnowDDIClassifier / make_params) and drives training/encoding through the SHARED
protocol-2 harness so KnowDDI sees the SAME per-epoch S2 emerging shuffle as R-GCN
and EmerGNN.

Integration design (codex-discussed 2026-07-07, user-sanctioned "Option B" for
protocol-2 fit + acceleration; KnowDDI fills the subgraph cell of the intro figure):
  * BIO-ONLY node sets (enclosing-subgraph BFS on the DDI-free incidence) are
    epoch-invariant -> extracted once per pair, cached, deterministic per (pair,seed).
  * Each epoch/context the GLOBAL graph's DDI edges are REBUILT from the harness's
    fact_context / ScoringContext (bio edges fixed + this call's DDI edges), so the
    induced pair subgraph (prepare_subgraph) picks up exactly the current fact-KG
    DDI edges. Leak-free: prepare_subgraph removes the query pair's own edge (both
    directions), and the context already excludes it.
KnowDDI is transductive (learnable per-node embedding table) => cold-start is
expected WEAK, a property to measure, not a defect. Binary uses the pilot's 1-logit
head; multi-cls swaps W_final for a K-way head (BCE -> CE), the DDI event type being
the LABEL space (baseline num_rels=K), NOT a graph edge relation. The subgraph/GNN
(GraphSAGE + GSL) is identical across tasks.

NOTE: "wrapper" = model glue to the shared harness, NOT the method's M_A·M_B adapter.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "Code"))
sys.path.insert(0, str(_ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
from train_knowddi_rank_pilot import (  # noqa: E402
    build_graph_tensors, extract_enclosing_subgraph, KnowDDIDataset, collate,
    KnowDDIClassifier, make_params)

from ..framework import RankModel, EpochData
from ..specs import PairEncoding, ScoringContext, TaskSpec, TrainEpochOutput

_DEFAULTS = dict(
    emb_dim=32, num_gcn_layers=2, gcn_aggregator_type="mean", gcn_dropout=0.2,
    num_infer_layers=3, num_dig_layers=3, MLP_hidden_dim=16, MLP_num_layers=2,
    MLP_dropout=0.2, func_num=1, sparsify=1, threshold=0.05, edge_softmax=1,
    gsl_rel_emb_dim=32, lamda=0.7, gsl_has_edge_emb=1,
    hop=2, max_nodes_per_hop=200, lr=5e-3, weight_decay=1e-5, batch_size=128,
    grad_clip=10.0, seed=42)


class KnowDDIRankWrapper(RankModel):
    def setup(self, task: TaskSpec, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        kg = MergedKG.from_parquet(DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
        self.id2i = kg.id_to_idx
        self.n_nodes = kg.n_nodes

        # BIO-only base: build_graph_tensors with EMPTY DDI -> bio edge tensors +
        # bio-only incidence (epoch-invariant node-set scaffold, Option B).
        empty = np.zeros((0, 2), dtype=np.int64)
        g_bio, self.inc_bio, self.layout = build_graph_tensors(kg, str(DEFAULT_EDGES_PATH), empty)
        self.bio_src, self.bio_dst = g_bio.edges()
        self.bio_typ = g_bio.edata["type"]
        self.ddi_id = int(self.layout["ddi_id"])

        args = SimpleNamespace(**self.hp)
        gg0 = self._global_graph(empty)                       # bio-only for model init
        params = make_params(args, kg, self.layout, gg0.to(self.device), self.device)
        self.model = KnowDDIClassifier(params).to(self.device)
        if not task.is_binary:
            # KnowDDIClassifier is binary (W_final out=1); swap for a K-way head. The
            # subgraph/GNN is unchanged; the DDI event type is the LABEL space (num_rels=K
            # in the baseline), NOT a graph edge relation (aug_num_rels = BKG + self-loop).
            # forward's .squeeze(-1) is a no-op for out=K, so logits become (n, K).
            self.model.W_final = nn.Linear(3 * self.model.score_dim, task.n_classes).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["lr"],
                                    weight_decay=self.hp["weight_decay"])
        self._nodeset: dict = {}                              # (u,v) -> (nodes, labels)

    # -- graph + index helpers -----------------------------------------------
    def _global_graph(self, ddi_idx: np.ndarray):
        """CPU DGL global graph = fixed bio edges + these DDI edges (both dirs, ddi_id).
        The model's GraphSAGE runs on the .to(device) copy; prepare_subgraph induces
        each pair subgraph from the CPU copy so it carries THIS call's DDI edges."""
        if len(ddi_idx):
            a = torch.as_tensor(ddi_idx[:, 0], dtype=torch.long)
            b = torch.as_tensor(ddi_idx[:, 1], dtype=torch.long)
            ds = torch.cat([a, b]); dd = torch.cat([b, a])
            dt = torch.full((ds.numel(),), self.ddi_id, dtype=self.bio_typ.dtype)
            src = torch.cat([self.bio_src, ds]); dst = torch.cat([self.bio_dst, dd])
            typ = torch.cat([self.bio_typ, dt])
        else:
            src, dst, typ = self.bio_src, self.bio_dst, self.bio_typ
        g = dgl.graph((src, dst), num_nodes=self.n_nodes)
        g.edata["type"] = typ
        g.ndata["idx"] = torch.arange(self.n_nodes, dtype=torch.long)
        return g

    def _to_idx(self, cols: np.ndarray) -> np.ndarray:
        miss = [str(x) for x in cols if str(x) not in self.id2i]
        if miss:
            raise ValueError(f"{len(miss)} drug ids not in merged KG, e.g. {miss[:5]}")
        return np.array([self.id2i[str(x)] for x in cols], dtype=np.int64)

    def _ddi_idx(self, ddi_edges: np.ndarray) -> np.ndarray:
        ddi_edges = np.asarray(ddi_edges)
        if len(ddi_edges) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([self._to_idx(ddi_edges[:, 0]), self._to_idx(ddi_edges[:, 1])], axis=1)

    def _pair_idx(self, pairs: np.ndarray) -> np.ndarray:
        pairs = np.asarray(pairs)
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([self._to_idx(pairs[:, 0]), self._to_idx(pairs[:, 1])], axis=1)

    def _node_set(self, u: int, v: int):
        """Enclosing-subgraph node set from the BIO-only incidence (epoch-invariant),
        cached + deterministic per (u, v, seed)."""
        key = (int(u), int(v))
        if key not in self._nodeset:
            rng = np.random.default_rng(int(self.hp["seed"]) * 1_000_003 + int(u) * 31 + int(v))
            self._nodeset[key] = extract_enclosing_subgraph(
                u, v, self.inc_bio, self.hp["hop"], self.hp["max_nodes_per_hop"], rng)
        return self._nodeset[key]

    def _dataset(self, pairs_idx: np.ndarray, labels: np.ndarray, gg_cpu):
        # Enclosing-subgraph extraction is the slow one-time cost (~40-78ms/pair); log
        # progress+timing so it never looks stuck. Only counts genuine cache misses.
        t0 = time.perf_counter(); total = len(pairs_idx); n_new = 0; cache = []
        for j, (u, v) in enumerate(pairs_idx):
            was_new = (int(u), int(v)) not in self._nodeset
            cache.append(self._node_set(u, v))
            if was_new:
                n_new += 1
                if n_new % 2000 == 0:
                    print(f"[knowddi] subgraph extract {n_new} new ({j + 1}/{total} pairs, "
                          f"{time.perf_counter() - t0:.0f}s, cache={len(self._nodeset)})", flush=True)
        if n_new >= 2000:
            print(f"[knowddi] subgraph extract done: {n_new} new / {total} pairs "
                  f"({time.perf_counter() - t0:.0f}s, cache={len(self._nodeset)})", flush=True)
        return KnowDDIDataset(gg_cpu, cache, labels, self.hp["num_dig_layers"])

    # -- RankModel contract --------------------------------------------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        gg = self._global_graph(self._ddi_idx(epoch.fact_context.ddi_edges))
        self.model.global_graph = gg.to(self.device)         # GraphSAGE runs on this
        y = np.asarray(epoch.target_labels, dtype=np.float32)
        n = len(y)
        if n == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)
        ds = self._dataset(self._pair_idx(epoch.target_pairs), y, gg)
        bs = int(self.hp["batch_size"]); order = rng.permutation(n)
        self.model.train(); total = 0.0
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            g, yb = collate([ds[int(i)] for i in idx])
            g = g.to(self.device); yb = yb.to(self.device)
            self.opt.zero_grad()
            logits = self.model(g)                           # (b,) binary | (b,K) multiclass
            if self.task.is_binary:
                loss = F.binary_cross_entropy_with_logits(logits, yb.float())
            else:
                loss = F.cross_entropy(logits, yb.long())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.hp["grad_clip"])
            self.opt.step()
            total += float(loss.item()) * len(idx)          # sample-weighted (last batch shorter)
        return TrainEpochOutput(mean_loss=total / max(n, 1), n_targets=n)

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        if len(pairs) == 0:                                  # defensive; harness guards this
            return PairEncoding(pair_ids=pairs,
                                pair_repr=np.zeros((0, 3 * self.model.score_dim), np.float32),
                                logits=np.zeros((0,), np.float32),
                                repr_kind="subgraph_gsl_readout", repr_stage="pre_scorer_pred")
        gg = self._global_graph(self._ddi_idx(context.ddi_edges))
        self.model.global_graph = gg.to(self.device)
        ds = self._dataset(self._pair_idx(pairs), np.zeros(len(pairs), np.float32), gg)
        self.model.eval()
        bs = int(self.hp["batch_size"]); reprs, logits = [], []
        for s in range(0, len(pairs), bs):
            g, _ = collate([ds[i] for i in range(s, min(s + bs, len(pairs)))])
            g = g.to(self.device)
            score, pred = self.model(g, return_pred=True)
            reprs.append(pred.float().cpu().numpy()); logits.append(score.float().cpu().numpy())
        return PairEncoding(pair_ids=pairs,
                            pair_repr=np.concatenate(reprs).astype(np.float32),
                            logits=np.concatenate(logits).astype(np.float32),
                            repr_kind="subgraph_gsl_readout", repr_stage="pre_scorer_pred")

    @torch.no_grad()
    def head_from_repr(self, pair_repr) -> np.ndarray:
        p = torch.as_tensor(np.asarray(pair_repr), dtype=torch.float32, device=self.device)
        self.model.eval()
        return self.model.W_final(p).squeeze(-1).cpu().numpy().astype(np.float32)

    def known_drugs(self) -> set:
        return set(self.id2i.keys())

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)
