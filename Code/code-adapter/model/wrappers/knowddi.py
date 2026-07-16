"""KnowDDIRankWrapper - KnowDDI (subgraph paradigm) backbone plugged into the
code-adapter harness. Ported from `my_code/rank_analysis/wrappers/knowddi.py`,
re-targeted to the code-adapter RankModel contract. Reuses the codex-reviewed
KnowDDI pilot machinery (build_graph_tensors / extract_enclosing_subgraph /
KnowDDIDataset / collate / KnowDDIClassifier / make_params); not re-implemented.

BIO-only enclosing-subgraph node sets are epoch-invariant -> extracted once per
pair, cached, deterministic per (pair, seed). Each epoch/context the global graph's
DDI edges are rebuilt from the harness ScoringContext, so the induced pair subgraph
picks up exactly the current fact-KG DDI edges (leak-free: prepare_subgraph removes
the query pair's own edge; the context already excludes it). KnowDDI is transductive
=> cold-start is expected WEAK (a property to measure). Native protocol is P1.

"wrapper" = harness glue, NOT the M_A·M_B adapter.
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

from ..contracts import (EpochData, PairEncoding, RankModel, ScoringContext,  # noqa: E402
                         TrainEpochOutput)

#: tunable hyperparameters (exposed via hp)
_DEFAULTS = dict(
    emb_dim=32, num_gcn_layers=2, gcn_aggregator_type="mean", gcn_dropout=0.2,
    num_infer_layers=3, num_dig_layers=3, MLP_hidden_dim=16, MLP_num_layers=2,
    MLP_dropout=0.2, func_num=1, sparsify=1, threshold=0.05, edge_softmax=1,
    gsl_rel_emb_dim=32, lamda=0.7, gsl_has_edge_emb=1,
    hop=2, max_nodes_per_hop=100, lr=5e-3, weight_decay=1e-5, batch_size=128,   # 200->100: efficiency
    #     (user-approved 2026-07-14; smaller enclosing subgraph = faster extraction + quadratic GSL.
    #      CHANGES KnowDDI results vs the paper's 200. Pass max_nodes_per_hop=200 to restore.)
    grad_clip=10.0, seed=42)


class KnowDDIRankWrapper(RankModel):
    def setup(self, task, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        kg = MergedKG.from_parquet(DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
        self.id2i = kg.id_to_idx
        self.n_nodes = kg.n_nodes

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
            self.model.W_final = nn.Linear(3 * self.model.score_dim, task.n_classes).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["lr"],
                                    weight_decay=self.hp["weight_decay"])
        self._nodeset: dict = {}                              # (u,v) -> (nodes, labels)
        self._load_subgraph_cache()                           # prebuilt disk cache (if present)

    def _load_subgraph_cache(self) -> None:
        """Load a prebuilt per-pair enclosing-subgraph cache into self._nodeset (byte-identical;
        misses fall back to on-the-fly sequential extraction). Build it with
        Code/scripts/prepare_knowddi_subgraphs.py (parallel, offline). The cache is keyed by
        node idx + a signature (KG identity/hop/max_nodes/seed/code), so a stale cache is ignored."""
        try:
            from kg import knowddi_subgraph_cache as C
            sig = C.signature(hop=int(self.hp["hop"]), max_nodes_per_hop=int(self.hp["max_nodes_per_hop"]),
                              seed=int(self.hp["seed"]), n_nodes=int(self.n_nodes),
                              inc=self.inc_bio)
            path = C.cache_path(sig)
            cached = C.load(path, sig)
            if cached:
                self._nodeset.update(cached)
                print(f"[knowddi] subgraph cache HIT: {len(cached)} pairs <- {path.name}", flush=True)
            else:
                print(f"[knowddi] subgraph cache MISS ({path.name}); extracting on the fly. "
                      f"Prebuild with prepare_knowddi_subgraphs.py for speed.", flush=True)
        except Exception as e:                                # noqa: BLE001 — cache is best-effort
            print(f"[knowddi] subgraph cache load skipped ({type(e).__name__}: {e})", flush=True)

    # -- graph + index helpers -----------------------------------------------
    def _global_graph(self, ddi_idx: np.ndarray):
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
        key = (int(u), int(v))
        if key not in self._nodeset:
            rng = np.random.default_rng(int(self.hp["seed"]) * 1_000_003 + int(u) * 31 + int(v))
            self._nodeset[key] = extract_enclosing_subgraph(
                u, v, self.inc_bio, self.hp["hop"], self.hp["max_nodes_per_hop"], rng)
        return self._nodeset[key]

    def _dataset(self, pairs_idx: np.ndarray, labels: np.ndarray, gg_cpu):
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
        if self.hp.get("graphsage_hoist"):                   # opt-in full-batch-grad path (below)
            return self._train_epoch_graphsage_hoist(epoch, rng)
        gg = self._global_graph(self._ddi_idx(epoch.fact_context.ddi_edges))
        self.model.global_graph = gg.to(self.device)
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
            total += float(loss.item()) * len(idx)
        return TrainEpochOutput(mean_loss=total / max(n, 1), n_targets=n)

    def _train_epoch_graphsage_hoist(self, epoch: EpochData, rng) -> TrainEpochOutput:
        """Full-batch-gradient training path (user-chosen scheme, hp ``graphsage_hoist``).

        The full-KG GraphSAGE over the 178k-node ``global_graph`` is the dominant cost of the
        default per-minibatch ``train_epoch`` (its BACKWARD scatters over ~14M edges EVERY
        minibatch). Here it is computed and back-propagated ONCE per epoch instead: the encoder's
        node embeddings (``h`` AND its side-effect ``repr``) are detached into leaf proxies, so
        each minibatch's backward flows only through the cheap GSL/head and ACCUMULATES into the
        proxy ``.grad``. The GraphSAGE grad (proxy grads -> weights) is applied ONCE at epoch end,
        then one ``opt.step()``. Summing per-minibatch grads is exactly the full-batch gradient, so
        this is mathematically full-batch GD (one optimizer step/epoch), a deliberate deviation from
        KnowDDI's minibatch SGD to make one epoch tractable on the merged KG. Loss is summed and
        divided by ``n`` so ``mean_loss`` is the epoch mean.

        MEMORY (manual gradient checkpointing): retaining the full-KG GraphSAGE forward graph for a
        whole epoch costs ~19GB of activations (14M edges) and OOMs a 32GB card once the epoch-end
        backward allocates grads on top. So the proxy VALUES are computed under ``no_grad`` (nothing
        retained), and the forward is RECOMPUTED with grad at epoch end (weights are unchanged until
        ``opt.step``) to back-propagate the accumulated proxy grads. The torch RNG is snapshotted
        and restored around the two forwards so the recompute reproduces the SAME dropout masks (and
        therefore the same node embeddings) the minibatches actually consumed -- otherwise the grad
        would be evaluated at a different point than the loss.
        Note: ``repr`` MUST be proxied too (the forward reads global ``repr`` at line ~519); leaving
        it attached would re-enter the shared GraphSAGE graph each minibatch -> double-backward."""
        gg = self._global_graph(self._ddi_idx(epoch.fact_context.ddi_edges))
        self.model.global_graph = gg.to(self.device)
        y = np.asarray(epoch.target_labels, dtype=np.float32)
        n = len(y)
        if n == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)
        ds = self._dataset(self._pair_idx(epoch.target_pairs), y, gg)
        bs = int(self.hp["batch_size"]); order = rng.permutation(n)
        self.model.train(); self.opt.zero_grad()
        # proxy VALUES under no_grad (no retained graph). Snapshot RNG so the epoch-end recompute
        # reproduces the SAME dropout masks -> identical embeddings.
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state(self.device)
                    if torch.cuda.is_available() and self.device.type == "cuda" else None)
        with torch.no_grad():
            h_val = self.model.embedding_model(self.model.global_graph)
            repr_val = self.model.global_graph.ndata["repr"]     # side effect of the encoder
        h_proxy = h_val.detach().requires_grad_(True)
        repr_proxy = repr_val.detach().requires_grad_(True)
        self.model.global_graph.ndata["h"] = h_proxy
        self.model.global_graph.ndata["repr"] = repr_proxy
        total = 0.0; n_batches = (n + bs - 1) // bs
        for bi, s in enumerate(range(0, n, bs), start=1):
            idx = order[s:s + bs]
            g, yb = collate([ds[int(i)] for i in idx])
            g = g.to(self.device); yb = yb.to(self.device)
            logits = self.model(g, skip_global_embed=True)      # reads the proxy leaves
            if self.task.is_binary:
                loss = F.binary_cross_entropy_with_logits(logits, yb.float(), reduction="sum") / n
            else:
                loss = F.cross_entropy(logits, yb.long(), reduction="sum") / n
            loss.backward()                                     # accumulate into GSL/head + proxies
            total += float(loss.item())
            if bi % 200 == 0:
                print(f"[knowddi-hoist] ep step {bi}/{n_batches} accum_mean_loss={total:.4f}",
                      flush=True)
        # recompute the full-KG GraphSAGE WITH grad (same weights + restored RNG => same masks =>
        # same values as the proxies), then back-propagate the accumulated proxy grads ONCE.
        # Snapshot the post-loop RNG first and restore it after the backward, so rewinding for the
        # recompute does NOT perturb the global RNG stream that later epochs consume (seeded
        # reproducibility -- codex 019f6347).
        post_cpu_rng = torch.get_rng_state()
        post_cuda_rng = (torch.cuda.get_rng_state(self.device) if cuda_rng is not None else None)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, self.device)
        h_re = self.model.embedding_model(self.model.global_graph)
        repr_re = self.model.global_graph.ndata["repr"]
        outs, grads = [], []
        if h_proxy.grad is not None:
            outs.append(h_re); grads.append(h_proxy.grad)
        if repr_proxy.grad is not None:
            outs.append(repr_re); grads.append(repr_proxy.grad)
        if outs:
            torch.autograd.backward(outs, grads)
        torch.set_rng_state(post_cpu_rng)                       # continue the stream as if no rewind
        if post_cuda_rng is not None:
            torch.cuda.set_rng_state(post_cuda_rng, self.device)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.hp["grad_clip"])
        self.opt.step()
        return TrainEpochOutput(mean_loss=total, n_targets=n)

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        if len(pairs) == 0:
            return PairEncoding(pair_ids=pairs,
                                pair_repr=np.zeros((0, 3 * self.model.score_dim), np.float32),
                                logits=np.zeros((0,), np.float32),
                                repr_kind="subgraph_gsl_readout", repr_stage="pre_scorer_pred")
        gg = self._global_graph(self._ddi_idx(context.ddi_edges))
        self.model.global_graph = gg.to(self.device)
        ds = self._dataset(self._pair_idx(pairs), np.zeros(len(pairs), np.float32), gg)
        self.model.eval()
        # PERF: the full-KG GraphSAGE over global_graph is invariant here (fixed context, eval),
        # so compute it ONCE instead of per batch (was ~#batches recomputes). Identical outputs;
        # batches pass skip_global_embed=True. (codex 019f5ece; do NOT do this in train_epoch.)
        self.model.global_graph.ndata["h"] = self.model.embedding_model(self.model.global_graph)
        bs = int(self.hp["batch_size"]); reprs, logits = [], []
        for s in range(0, len(pairs), bs):
            g, _ = collate([ds[i] for i in range(s, min(s + bs, len(pairs)))])
            g = g.to(self.device)
            score, pred = self.model(g, return_pred=True, skip_global_embed=True)
            reprs.append(pred.float().cpu().numpy()); logits.append(score.float().cpu().numpy())
        return PairEncoding(pair_ids=pairs,
                            pair_repr=np.concatenate(reprs).astype(np.float32),
                            logits=np.concatenate(logits).astype(np.float32),
                            repr_kind="subgraph_gsl_readout", repr_stage="pre_scorer_pred")

    def pair_forward(self, pairs, context: ScoringContext) -> "torch.Tensor":
        """Grad-enabled pre-scorer pair rep pred [N, 3*score_dim] (= [g_out||head||tail]) as a
        torch tensor, for JOINT adapter composition. Same subgraph + model(return_pred=True) as
        encode_pairs but WITHOUT no_grad and WITHOUT the W_final head; respects caller train/eval
        mode. Global graph rebuilt from the leak-free context (per minibatch in joint)."""
        pairs = np.asarray(pairs)
        gg = self._global_graph(self._ddi_idx(context.ddi_edges))
        self.model.global_graph = gg.to(self.device)
        ds = self._dataset(self._pair_idx(pairs), np.zeros(len(pairs), np.float32), gg)
        bs = int(self.hp["batch_size"]); preds = []
        for s in range(0, len(pairs), bs):
            g, _ = collate([ds[i] for i in range(s, min(s + bs, len(pairs)))])
            g = g.to(self.device)
            _, pred = self.model(g, return_pred=True)        # [b, 3*score_dim] grad-enabled
            preds.append(pred)
        return (torch.cat(preds, dim=0) if preds
                else torch.zeros((0, 3 * self.model.score_dim), device=self.device))

    def pair_streams(self, pairs, context: ScoringContext) -> "list":
        """Per-layer grad-enabled pair reps for design-R 'default'. KnowDDI's pred is
        [g_out || head || tail], each score_dim = (1+num_gcn+num_infer)*emb_dim wide and itself
        an ORDERED concat of L=(1+num_gcn+num_infer) per-layer emb_dim blocks (pre-embed, GCN
        layers, GSL infer layers). Stream s = [g_out_s || head_s || tail_s] (the s-th emb_dim
        block of each), [N, 3*emb_dim]. Derived purely from pred by block gather -- no model
        change. (stream[-1] is the LAST GSL layer, NOT the full pred; 'last' mode uses full pred.)"""
        pred = self.pair_forward(pairs, context)             # [N, 3*score_dim] grad
        emb = int(self.hp["emb_dim"]); sd = int(self.model.score_dim); L = sd // emb
        g_out, head, tail = pred[:, :sd], pred[:, sd:2 * sd], pred[:, 2 * sd:3 * sd]
        streams = []
        for s in range(L):
            sl = slice(s * emb, (s + 1) * emb)
            streams.append(torch.cat([g_out[:, sl], head[:, sl], tail[:, sl]], dim=-1))  # [N,3*emb]
        return streams

    @torch.no_grad()
    def head_from_repr(self, pair_repr) -> np.ndarray:
        p = torch.as_tensor(np.asarray(pair_repr), dtype=torch.float32, device=self.device)
        self.model.eval()
        return self.model.W_final(p).squeeze(-1).cpu().numpy().astype(np.float32)

    def effective_hp(self) -> dict:
        return dict(self.hp)

    def known_drugs(self) -> set:
        return set(self.id2i.keys())

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)


__all__ = ["KnowDDIRankWrapper"]
