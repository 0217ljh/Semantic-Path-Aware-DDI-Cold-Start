"""EmerGNNRankWrapper - EmerGNN (path paradigm, rspmm backend) backbone plugged
into the code-adapter harness. Ported from `my_code/rank_analysis/wrappers/emergnn.py`,
re-targeted to the code-adapter RankModel contract. Reuses the codex-reviewed EmerGNN
machinery (setup only) + EmerGNN_RSPMM / EmerGNN_MC_RSPMM models (not re-implemented).

EmerGNN's native protocol is protocol-2, and the harness owns the P2 emerging/fact
shuffle, so EmerGNN sees the EXACT same split as every other model (EmerGNN's own
`shuffle_train` is NOT used). Training is EmerGNN-faithful: per-epoch KG rebuilt once,
minibatch SGD with sum-reduction BCE/CE. DRUG-ID space; ScoringContext is leak-free.

Needs hp['dataset'] + hp['fold'] (to load its leaf for the entity vocab + Morgan
features); the entrypoint injects them. "wrapper" = harness glue, not the M_A·M_B adapter.
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

from ..contracts import (EpochData, PairEncoding, RankModel, ScoringContext,  # noqa: E402
                         TrainEpochOutput)

#: merged-KG edge file under Code/data/KG/_merged_kg/ (dataset-independent)
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"

#: tunable hyperparameters (exposed via hp)
_DEFAULTS = dict(n_dim=64, length=3, feat="M", lr=1e-3, batch_size=32,
                 weight_decay=1e-8, seed=42)


class EmerGNNRankWrapper(RankModel):
    def setup(self, task, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        dataset = self.hp["dataset"]; fold = self.hp["fold"]
        dir2group = {v: k for k, v in DATASET_DIRS.items()}
        if dataset not in dir2group:
            raise ValueError(f"dataset dir {dataset!r} not in DATASET_DIRS values "
                             f"{sorted(DATASET_DIRS.values())}")
        group = dir2group[dataset]
        leaf_task = "binary" if task.is_binary else "multiclass"

        unified_root = _ROOT / "Code" / "data" / "ddi_unified"
        leaf = U.load_leaf(str(unified_root), group, leaf_task, "cold_s2", fold)
        ds = make_dataset(leaf.train, leaf.val, leaf.resources)
        merged_kg_path = Path(self.hp.get("merged_kg_path")
                              or _ROOT / "Code" / "data" / "KG" / "_merged_kg" / MERGED_EDGES)
        if not merged_kg_path.is_file():
            raise FileNotFoundError(
                f"merged KG edges not found at {merged_kg_path}; pass hp['merged_kg_path']")

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

    @property
    def model(self):
        """The trainable nn.Module. EmerGNN keeps it at core._model; this alias gives the
        composer the same `.model` handle every other wrapper exposes (parameters / train /
        state_dict / load_state_dict for FROZEN freeze + JOINT training)."""
        return self.core._model

    # -- KG + pair index helpers ---------------------------------------------
    def _map(self, ids) -> np.ndarray:
        miss = [str(x) for x in ids if str(x) not in self.e2i]
        if miss:
            raise ValueError(f"{len(miss)} drug ids not in EmerGNN entity vocab, e.g. {miss[:5]}")
        return np.array([self.e2i[str(x)] for x in ids], dtype=np.int64)

    def _kg_from_context(self, ctx: ScoringContext) -> torch.Tensor:
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
        perm = rng.permutation(n)
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
        return PairEncoding(pair_ids=pairs, pair_repr=emb.astype(np.float32),
                            logits=logit.astype(np.float32),
                            repr_kind="enc_ht", repr_stage="bidirectional_message_passing")

    def pair_forward(self, pairs, context: ScoringContext) -> "torch.Tensor":
        """Grad-enabled pre-scorer pair embedding enc_ht(h,t,kg) [N, d] as a torch tensor,
        for JOINT adapter composition (backbone stays in the autograd graph). Same enc_ht as
        encode_pairs but WITHOUT no_grad and WITHOUT the .Wr() head; respects the caller-set
        train/eval mode. KG rebuilt from the leak-free context (per minibatch in joint)."""
        pairs = np.asarray(pairs)
        kg = self._kg_from_context(context)
        ia = self._map(pairs[:, 0]); ib = self._map(pairs[:, 1])
        m = self.core._model
        bs = int(self.hp["batch_size"]); embs = []
        for s in range(0, len(ia), bs):
            h = torch.as_tensor(ia[s:s + bs], device=self.device)
            t = torch.as_tensor(ib[s:s + bs], device=self.device)
            embs.append(m.enc_ht(h, t, kg))                  # [b, d] grad-enabled
        width = (4 if m.feat == "E" else 2) * int(self.hp["n_dim"])
        return torch.cat(embs, dim=0) if embs else torch.zeros((0, width), device=self.device)

    def pair_streams(self, pairs, context: ScoringContext) -> "list":
        """Per-hop grad-enabled pair reps [enc_ht@depth1, .., enc_ht@depthL], each [N, d], for
        design-R 'default' (one correction per propagation hop). Reuses the model's OWN enc_ht
        at truncated depth (temporarily sets m.L=k, k=1..L, restored in finally) -- NO kernel
        re-implementation, so stream[-1] (k=L) == enc_ht == pair_forward exactly by construction.
        Depth-k readout uses the first k message-passing layers (nested, faithful to enc_ht).
        NOTE: mutating m.L is shared model state, so this is single-threaded only (fine for the
        sequential train/eval loop; do not call concurrently on the same wrapper)."""
        pairs = np.asarray(pairs)
        kg = self._kg_from_context(context)
        ia = self._map(pairs[:, 0]); ib = self._map(pairs[:, 1])
        m = self.core._model; L_full = int(m.L)
        bs = int(self.hp["batch_size"])
        chunks = [[] for _ in range(L_full)]                 # per-hop accumulators
        try:
            for s in range(0, len(ia), bs):
                h = torch.as_tensor(ia[s:s + bs], device=self.device)
                t = torch.as_tensor(ib[s:s + bs], device=self.device)
                for k in range(1, L_full + 1):
                    m.L = k
                    chunks[k - 1].append(m.enc_ht(h, t, kg))  # depth-k readout [b, d]
        finally:
            m.L = L_full                                     # always restore the real depth
        width = (4 if m.feat == "E" else 2) * int(self.hp["n_dim"])
        return [torch.cat(c, dim=0) if c else torch.zeros((0, width), device=self.device)
                for c in chunks]

    @torch.no_grad()
    def head_from_repr(self, pair_repr) -> np.ndarray:
        emb = torch.as_tensor(np.asarray(pair_repr), dtype=torch.float32, device=self.device)
        self.core._model.eval()
        return self.core._model.Wr(emb).squeeze(-1).cpu().numpy().astype(np.float32)

    def effective_hp(self) -> dict:
        return dict(self.hp)

    def known_drugs(self) -> set:
        return set(self.e2i.keys())

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.core._model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.core._model.load_state_dict(state)


__all__ = ["EmerGNNRankWrapper"]
