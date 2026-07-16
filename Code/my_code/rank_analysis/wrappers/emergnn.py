"""EmerGNNRankWrapper — EmerGNN (path paradigm, rspmm backend) plugged into the
rank-analysis harness.

Reuses the codex-reviewed EmerGNN machinery for SETUP ONLY (``_PerModeEmerGNN_RSPMM.
_setup_graph`` -> entity2id + bio KG triplets + Morgan features), then builds the
``EmerGNN_RSPMM`` (binary) / ``EmerGNN_MC_RSPMM`` (multiclass) model and drives
training/encoding through the SHARED harness. The harness owns the protocol-2 epoch
loop, so EmerGNN sees the EXACT same emerging/fact split as every other model
(EmerGNN's own ``shuffle_train`` is NOT used here).

Training is EmerGNN-faithful: per-epoch KG rebuilt once, minibatch (batch_size) SGD
with sum-reduction BCE/CE (matching the paper optimizer regime), NOT the full-batch
mean used by the node-paradigm R-GCN wrapper. DRUG-ID space; the harness-provided
ScoringContext declares which DDI facts are present (leak-free).

NOTE: "wrapper" = model glue to the shared harness, NOT the method's M_A·M_B adapter.
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

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402
from data_utils.unified import DATASET_DIRS  # noqa: E402
from baseline.emergnn._per_mode_rspmm import _PerModeEmerGNN_RSPMM  # noqa: E402
from baseline.emergnn._rspmm_utils import build_sparse_kg_from_triplets  # noqa: E402
from baseline.emergnn.model_rspmm import EmerGNN_RSPMM  # noqa: E402
from baseline.emergnn.multi_cls.model_rspmm import EmerGNN_MC_RSPMM  # noqa: E402

from ..framework import RankModel, EpochData
from ..specs import PairEncoding, ScoringContext, TaskSpec, TrainEpochOutput

# Merged-KG edge file under the canonical project KG dir Code/data/KG/_merged_kg/
# (dataset-independent; verified extract_emergnn_fact_protocol_rank.py:57 + on disk).
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"

_DEFAULTS = dict(n_dim=64, length=3, feat="M", lr=1e-3, batch_size=32,
                 weight_decay=1e-8, seed=42)


class EmerGNNRankWrapper(RankModel):
    def setup(self, task: TaskSpec, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        dataset = self.hp["dataset"]; fold = self.hp["fold"]
        # harness dataset key is the on-disk DIR name (drugbank_ryu); EmerGNN's
        # load_leaf wants the GROUP key (ryu). Invert DATASET_DIRS (no hardcoding).
        dir2group = {v: k for k, v in DATASET_DIRS.items()}
        if dataset not in dir2group:
            raise ValueError(f"dataset dir {dataset!r} not in DATASET_DIRS values "
                             f"{sorted(DATASET_DIRS.values())}")
        group = dir2group[dataset]
        leaf_task = "binary" if task.is_binary else "multiclass"

        unified_root = _ROOT / "Code" / "data" / "ddi_unified"
        leaf = U.load_leaf(str(unified_root), group, leaf_task, "cold_s2", fold)
        ds = make_dataset(leaf.train, leaf.val, leaf.resources)
        # Merged KG is a project-level artifact (dataset-independent), NOT the
        # per-dataset leaf.resources.kg.source (null for KG-free unified leaves like
        # drugbank_ryu). Canonical path matches EmerGNN's validated runs; hp-overridable.
        merged_kg_path = Path(self.hp.get("merged_kg_path")
                              or _ROOT / "Code" / "data" / "KG" / "_merged_kg" / MERGED_EDGES)
        if not merged_kg_path.is_file():
            raise FileNotFoundError(
                f"merged KG edges not found at {merged_kg_path}; pass hp['merged_kg_path'] "
                f"to the wrapper to point at edges__...mask1.parquet")

        # SETUP ONLY: build entity2id + bio KG triplets + Morgan features via the
        # codex-reviewed _setup_graph (no fit() training loop).
        self.core = _PerModeEmerGNN_RSPMM(
            n_dim=self.hp["n_dim"], length=self.hp["length"], feat=self.hp["feat"],
            learning_rate=self.hp["lr"], batch_size=self.hp["batch_size"],
            weight_decay=self.hp["weight_decay"], device="cuda" if self.device.type == "cuda" else "cpu",
            backbone_kg_source="merged", merged_kg_path=merged_kg_path,
            shuffle_train_mode="S2", shuffle_ratio=0.8)
        morgan_mat, _drug_ids = self.core._setup_graph(ds, ds.kg)

        self.e2i = self.core._entity2id
        self.n_ent = int(self.core._n_ent)
        self.n_base_rel = int(self.core._n_base_rel)
        # DDI relation slots: binary -> 1; multiclass -> K typed slots (paper-faithful).
        self.n_ddi_rel = 1 if task.is_binary else task.n_classes
        self.n_base_rel_with_ddi = self.n_base_rel + self.n_ddi_rel
        self.core._n_base_rel_with_ddi = self.n_base_rel_with_ddi
        self.bio_triplets = np.asarray(self.core._kg_triplets, dtype=np.int64)

        if task.is_binary:
            model = EmerGNN_RSPMM(
                n_ent=self.n_ent, n_base_rel=self.n_base_rel_with_ddi,
                n_dim=self.hp["n_dim"], length=self.hp["length"], feat=self.hp["feat"],
                morgan_features=morgan_mat if self.hp["feat"] == "M" else None)
        else:
            model = EmerGNN_MC_RSPMM(
                n_ent=self.n_ent, n_base_rel=self.n_base_rel_with_ddi,
                n_classes=task.n_classes, n_dim=self.hp["n_dim"], length=self.hp["length"],
                feat=self.hp["feat"], morgan_features=morgan_mat if self.hp["feat"] == "M" else None)
        self.core._model = model.to(self.device)
        self.opt = torch.optim.Adam(self.core._model.parameters(), lr=self.hp["lr"],
                                    weight_decay=self.hp["weight_decay"])

    # -- KG + pair index helpers ---------------------------------------------
    def _map(self, ids) -> np.ndarray:
        miss = [str(x) for x in ids if str(x) not in self.e2i]
        if miss:
            raise ValueError(f"{len(miss)} drug ids not in EmerGNN entity vocab, e.g. {miss[:5]}")
        return np.array([self.e2i[str(x)] for x in ids], dtype=np.int64)

    def _kg_from_context(self, ctx: ScoringContext) -> torch.Tensor:
        """Sparse KG = bio triplets + this context's DDI facts (one direction; the
        rspmm builder adds the reverse + self-loops). Mirrors fit()'s eval-KG build."""
        tris = [self.bio_triplets]
        ddi = ctx.ddi_edges
        if len(ddi):
            a = self._map(ddi[:, 0]); b = self._map(ddi[:, 1])
            rel = self.n_base_rel + (np.zeros(len(a), np.int64) if self.task.is_binary
                                     else ddi[:, 2].astype(np.int64))
            tris.append(np.stack([a, b, rel], axis=1))
        all_tri = np.concatenate(tris, axis=0)
        return build_sparse_kg_from_triplets(all_tri, self.n_ent,
                                             self.n_base_rel_with_ddi, device=self.device)

    @torch.no_grad()
    def _batched_embed_logit(self, ia, ib, kg):
        """Batched enc_ht + score head. Mirrors extract_emergnn_fact_protocol_rank.
        _embed_logit_on_kg: emb=(N,2*n_dim) pair repr, logit=(N,) binary / (N,K) multi."""
        m = self.core._model; m.eval()
        bs = self.hp["batch_size"]
        embs, logits = [], []
        for s in range(0, len(ia), bs):
            h = torch.as_tensor(ia[s:s + bs], device=self.device)
            t = torch.as_tensor(ib[s:s + bs], device=self.device)
            emb = m.enc_ht(h, t, kg)
            logit = m.Wr(emb).squeeze(-1)                 # (B,) binary | (B,K) multi
            embs.append(emb.float().cpu().numpy()); logits.append(logit.float().cpu().numpy())
        return np.concatenate(embs), np.concatenate(logits)

    # -- RankModel contract --------------------------------------------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        kg = self._kg_from_context(epoch.fact_context)           # built once per epoch
        ia = self._map(epoch.target_pairs[:, 0]); ib = self._map(epoch.target_pairs[:, 1])
        y = np.asarray(epoch.target_labels)
        n = len(y)
        if n == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)
        perm = rng.permutation(n)                                # EmerGNN reshuffles targets/epoch
        ia, ib, y = ia[perm], ib[perm], y[perm]
        m = self.core._model; m.train()
        bs = self.hp["batch_size"]; total = 0.0
        for s in range(0, n, bs):
            h = torch.as_tensor(ia[s:s + bs], device=self.device)
            t = torch.as_tensor(ib[s:s + bs], device=self.device)
            yb = torch.as_tensor(y[s:s + bs], device=self.device)
            self.opt.zero_grad(set_to_none=True)
            logits = m(h, t, kg)                                  # (B,) binary | (B,K) multi
            if self.task.is_binary:
                loss = F.binary_cross_entropy_with_logits(logits, yb.float(), reduction="sum")
            else:
                loss = F.cross_entropy(logits, yb.long(), reduction="sum")
            loss.backward(); self.opt.step()
            total += float(loss.item())
        return TrainEpochOutput(mean_loss=total / n, n_targets=n)

    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        kg = self._kg_from_context(context)
        ia = self._map(pairs[:, 0]); ib = self._map(pairs[:, 1])
        emb, logit = self._batched_embed_logit(ia, ib, kg)
        return PairEncoding(pair_ids=pairs,
                            pair_repr=emb.astype(np.float32),
                            logits=logit.astype(np.float32),
                            repr_kind="enc_ht", repr_stage="bidirectional_message_passing")

    @torch.no_grad()
    def head_from_repr(self, pair_repr) -> np.ndarray:
        emb = torch.as_tensor(np.asarray(pair_repr), dtype=torch.float32, device=self.device)
        self.core._model.eval()
        return self.core._model.Wr(emb).squeeze(-1).cpu().numpy().astype(np.float32)

    def known_drugs(self) -> set:
        return set(self.e2i.keys())

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.core._model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.core._model.load_state_dict(state)
