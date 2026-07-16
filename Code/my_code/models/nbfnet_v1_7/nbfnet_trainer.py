"""NBFNet v1.7 trainer — paper-faithful, fresh implementation.

Codex round-5 verdict. fresh trainer (NOT inheriting _PerModeEmerGNN). NBFNet
is not "EmerGNN backbone + aux head"; forcing it into EmerGNN's hooks distorts
the design. We selectively reuse protocol utilities only.

Reused from EmerGNN scaffolding (utilities, not class inheritance).
- shuffle_train(mode='S2') drug-level cold simulation
- KG construction utilities (kg_builder_merged)
- Standard logger / training progress helpers

Trainer responsibilities (codex round 4 + 5 spec).
- Per-source amortized training loop. group batch by source drug
- Union edge mask per source. mask {(src, ddi_rel, tgt_i) for tgt_i in batch}
- Symmetric pair scoring. encode_from_source(a) + encode_from_source(b)
  then h_q_sym = h_a[b] + h_b[a], score = mlp(concat([h_q_sym, q]))
- Negative sampling per source (1:1 ratio default, configurable)
- BCE-with-logits loss (numerical stability)
- Validation. AUROC + AP, early stop on best val AUROC
- Checkpoint save/load
- Log to Code/runs/<run_id>/ per project log convention

Anti-leakage.
- shuffle_train(mode='S2') masks emerging-drug DDI edges (drug-level)
- NBFNet query-edge masking removes direct queried pair edge (relation-aware)
- Both applied together (codex Q3 clarified they target different leakage)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.nbfnet_v1_7.nbfnet_model import NBFNetDDI  # noqa: E402
# NOTE: deliberately NOT importing build_edge_lists_from_triplets — it adds
# self-loops + EmerGNN relation convention that conflict with NBFNet's own
# inverse-edge augmentation (handoff Round 6 Fix 1). NBFNet builds edge
# tensors directly from shuffle_train's [h, t, r] triplets.
from baseline.emergnn.shuffle_utils import shuffle_train  # noqa: E402
from collections import defaultdict  # noqa: E402


class NBFNetTrainer:
    """Paper-faithful NBFNet trainer for cold-start S2 binary DDI.

    Per-source amortized training with symmetric pair scoring.
    Two BF encodings per pair (a-source + b-source), symmetrize at
    representation level before MLP head.
    """

    def __init__(
        self,
        # Model hyperparams (paper defaults)
        d: int = 32,
        n_layers: int = 6,
        mlp_hidden: int = 64,
        # Training hyperparams
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-8,
        neg_ratio: int = 1,                     # negatives per positive
        # Optimization
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        early_stop_patience: int = 10,
        # KG handling
        ddi_rel_id: Optional[int] = None,        # relation index for "interact"
        shuffle_train_mode: str = "S2",          # EmerGNN cold simulation mode
        shuffle_ratio: float = 0.8,              # fraction of in-KG drugs kept "old"
        use_batched_bf: bool = True,             # batched multi-source BF (parallel accel)
        # Logging
        log_step_every: int = 50,
        eval_strategy: str = "epoch",
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.d = int(d)
        self.n_layers = int(n_layers)
        self.mlp_hidden = int(mlp_hidden)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.neg_ratio = int(neg_ratio)
        self.scheduler_patience = int(scheduler_patience)
        self.scheduler_factor = float(scheduler_factor)
        self.early_stop_patience = int(early_stop_patience)
        self.ddi_rel_id = ddi_rel_id
        self.shuffle_train_mode = str(shuffle_train_mode)
        self.shuffle_ratio = float(shuffle_ratio)
        self.use_batched_bf = bool(use_batched_bf)
        self.log_step_every = int(log_step_every)
        self.eval_strategy = str(eval_strategy)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)

        # Will be set during fit()
        self.model: Optional[NBFNetDDI] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None
        self.n_nodes: Optional[int] = None
        self.n_base_rel: Optional[int] = None
        self.best_val_auc: float = -float("inf")
        self.best_epoch: int = 0
        self.best_state: Optional[dict] = None

        # KG state set during setup_graph()
        self.base_kg_triplets: Optional[np.ndarray] = None    # (m, 3) [h, t, r] non-DDI facts
        self.train_ddi_triplets: Optional[np.ndarray] = None  # (n, 3) [h, t, r=ddi_rel_id]
        self._kg_entity_set: set = set()                      # base-KG entity ids (paper extra_kg_ent)
        self._eval_edges: Optional[tuple] = None              # static eval KG (train_ddi + base_kg)

    # ------------------------------------------------------------------
    # KG setup
    # ------------------------------------------------------------------

    def setup_graph(
        self,
        base_kg_triplets: np.ndarray,
        train_ddi_triplets: np.ndarray,
        n_nodes: int,
        n_base_rel: int,
        ddi_rel_id: int,
    ) -> None:
        """Store the KG + train-DDI triplets for per-epoch shuffle_train.

        Codex Round 6 Fix 1. Keeps BOTH the raw numpy triplets (consumed by
        :func:`shuffle_train` each epoch) AND the static eval-KG edge tensors
        (= train_ddi + base_kg, used by validation/inference). Per-epoch
        training edges are rebuilt in :meth:`_build_epoch_kg`.

        Triplet column convention is ``[head, tail, rel]`` for BOTH inputs
        (matches ``baseline.emergnn.kg_builder.build_kg_from_kb`` output and
        ``shuffle_train``'s expected layout).

        Args.
            base_kg_triplets. (m, 3) [h, t, r] non-DDI facts (rel in [0, ddi_rel_id)).
            train_ddi_triplets. (n, 3) [h, t, r] train DDI positives, r == ddi_rel_id.
            n_nodes. total entity count in the merged KG (drugs + KG entities).
            n_base_rel. number of base forward relations INCLUDING the DDI slot
                        (e.g. 5 KG buckets + 1 DDI = 6); the model doubles this
                        for inverse edges.
            ddi_rel_id. relation index of the binary "interact" edge (used for
                        relation-aware query-edge masking).
        """
        self.n_nodes = int(n_nodes)
        self.n_base_rel = int(n_base_rel)
        self.ddi_rel_id = int(ddi_rel_id)

        self.base_kg_triplets = np.asarray(base_kg_triplets, dtype=np.int64)
        self.train_ddi_triplets = np.asarray(train_ddi_triplets, dtype=np.int64)

        # Paper convention (process_files_kg): ddi_in_kg is computed against the
        # union of all KG entities. Mirror EmerGNN by passing the base-KG entity
        # set as shuffle_train's extra_kg_ent.
        if len(self.base_kg_triplets):
            self._kg_entity_set = set(
                np.unique(self.base_kg_triplets[:, :2]).astype(np.int64).tolist()
            )
        else:
            self._kg_entity_set = set()

        # Static eval KG = train_ddi + base_kg (paper vKG = train_ddi + train_kg).
        # Stored as raw [h, t, r] tensors; scoring augments inverse edges itself.
        eval_kg = (
            np.concatenate([self.train_ddi_triplets, self.base_kg_triplets], axis=0)
            if len(self.base_kg_triplets)
            else self.train_ddi_triplets
        )
        eval_src = torch.from_numpy(eval_kg[:, 0]).long().to(self.device)
        eval_dst = torch.from_numpy(eval_kg[:, 1]).long().to(self.device)
        eval_rel = torch.from_numpy(eval_kg[:, 2]).long().to(self.device)
        self._eval_edges = (eval_src, eval_dst, eval_rel)

        print(
            f"[nbfnet] KG setup. n_nodes={self.n_nodes} n_base_rel={self.n_base_rel} "
            f"ddi_rel_id={self.ddi_rel_id} n_base_kg_edges={len(self.base_kg_triplets)} "
            f"n_train_ddi={len(self.train_ddi_triplets)} n_eval_edges={len(eval_src)}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def init_model(self) -> None:
        """Initialize NBFNetDDI model on device.

        Called after setup_graph (needs n_nodes, n_base_rel).
        """
        if self.n_nodes is None or self.n_base_rel is None:
            raise RuntimeError("setup_graph must be called before init_model")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = NBFNetDDI(
            n_nodes=self.n_nodes,
            n_base_rel=self.n_base_rel,
            d=self.d,
            n_layers=self.n_layers,
            mlp_hidden=self.mlp_hidden,
            ddi_rel_id=self.ddi_rel_id,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[nbfnet] model init. d={self.d} L={self.n_layers} "
              f"n_params={n_params:,}", flush=True)

    # ------------------------------------------------------------------
    # Per-source amortized forward (training-time helper)
    # ------------------------------------------------------------------

    def encode_targets_from_source(
        self,
        source: int,
        targets: torch.Tensor,                  # (T,) target node indices
        epoch_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Single-source encoder. ONE Bellman-Ford pass, read out at targets.

        Codex Round 6 Fix 3. Renamed from ``score_pairs_from_source`` and the
        docstring corrected: this method returns FORWARD-ONLY hidden states
        ``h_q(source, target_i)`` of shape ``(T, d)`` — it does NOT return
        logits and does NOT symmetrize. The caller is responsible for the
        reverse pass, representation-level symmetrization, and the MLP head.

        The amortized training/eval path uses
        :meth:`model.build_union_query_edge_mask` +
        :meth:`model.encode_from_source` directly with per-source caching
        (see :meth:`_score_pairs_amortized`); this method is the clean
        single-source convenience encoder used for debugging / smoke tests.

        Args.
            source. source drug node index.
            targets. (T,) target drug node indices.
            epoch_edges. (edge_src, edge_dst, edge_rel) raw (un-augmented)
                         KG edges for this snapshot.
            training. if True, mask the direct (source, ddi, target_i) edges.

        Returns.
            h_q_st. (T, d) forward hidden states h_q(source, target_i).
        """
        edge_src, edge_dst, edge_rel = epoch_edges

        # Augment with inverse edges (once)
        aug_src, aug_dst, aug_rel = self.model.augment_inverse_edges(
            edge_src, edge_dst, edge_rel
        )

        # Vectorized union mask covering all (source, target_i) DDI edges
        if training and self.model.ddi_rel_id is not None:
            keep_mask = self.model.build_union_query_edge_mask(
                aug_src, aug_dst, aug_rel, source, targets
            )
        else:
            keep_mask = None

        # ONE BF pass from source
        h_source = self.model.encode_from_source(
            source, aug_src, aug_dst, aug_rel,
            already_augmented=True,
            edge_keep_mask=keep_mask,
        )

        # Read out forward hidden state at each target
        h_q_st = h_source[targets]                              # (T, d)
        return h_q_st

    def score_pair_symmetric(
        self,
        drug_a: int,
        drug_b: int,
        epoch_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Single-pair symmetric scoring. two BF passes.

        Used for debugging / smoke tests (one pair at a time, simpler API).
        Training and validation use batched per-source amortization via
        _score_pairs_amortized.
        """
        edge_src, edge_dst, edge_rel = epoch_edges
        aug_src, aug_dst, aug_rel = self.model.augment_inverse_edges(
            edge_src, edge_dst, edge_rel
        )
        if training and self.model.ddi_rel_id is not None:
            keep_mask = self.model.build_query_edge_mask(
                aug_src, aug_dst, aug_rel, drug_a, drug_b
            )
        else:
            keep_mask = None

        h_from_a = self.model.encode_from_source(
            drug_a, aug_src, aug_dst, aug_rel,
            already_augmented=True, edge_keep_mask=keep_mask,
        )
        h_from_b = self.model.encode_from_source(
            drug_b, aug_src, aug_dst, aug_rel,
            already_augmented=True, edge_keep_mask=keep_mask,
        )
        h_q_sym = h_from_a[drug_b] + h_from_b[drug_a]            # (d,)
        z = torch.cat([h_q_sym, self.model.query], dim=-1)       # (2d,)
        logit = self.model.mlp_head(z).squeeze(-1)               # scalar
        return logit

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        val_pos_pairs: pd.DataFrame,
        val_neg_pairs: pd.DataFrame,
        drug_id_map: dict[str, int],          # drug string ID -> int node index
        run_dir: Path,
    ) -> dict:
        """Train NBFNet on cold-start S2 DDI.

        Codex Round 6 Fix 1/5. Train positives/facts are produced PER EPOCH by
        :func:`shuffle_train` from the stored ``train_ddi_triplets`` /
        ``base_kg_triplets`` (set in :meth:`setup_graph`), and negatives are
        sampled against each epoch's emerging-drug pool — so the caller only
        passes the (static) validation set.

        Args.
            val_pos_pairs / val_neg_pairs. validation DataFrames with columns
                ``drug_a_id`` / ``drug_b_id`` (string drug ids).
            drug_id_map. mapping drug_id string -> int KG node index.
            run_dir. directory to save best checkpoint + metrics.

        Returns.
            metrics dict with best validation AUC, best epoch, training time.
        """
        if self.model is None:
            raise RuntimeError("init_model must be called before fit")
        if self.train_ddi_triplets is None:
            raise RuntimeError("setup_graph must be called before fit")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Convert val pair DataFrames to node-index arrays once (drop unknowns).
        val_pos_a, val_pos_b = self._pairs_to_node_idx(val_pos_pairs, drug_id_map)
        val_neg_a, val_neg_b = self._pairs_to_node_idx(val_neg_pairs, drug_id_map)
        val_a = np.concatenate([val_pos_a, val_neg_a])
        val_b = np.concatenate([val_pos_b, val_neg_b])
        val_y = np.concatenate([
            np.ones(len(val_pos_a)), np.zeros(len(val_neg_a)),
        ])
        print(
            f"[nbfnet] val set. n_pos={len(val_pos_a)} n_neg={len(val_neg_a)}",
            flush=True,
        )

        fit_start = time.time()

        for epoch in range(1, self.n_epochs + 1):
            epoch_start = time.time()

            # Per-epoch shuffle_train: fact/target resampling (cold simulation).
            (edge_src, edge_dst, edge_rel), epoch_targets = self._build_epoch_kg(epoch)
            if len(epoch_targets) == 0:
                print(
                    f"[nbfnet] [ep {epoch}/{self.n_epochs}] 0 targets after "
                    f"shuffle_train(mode={self.shuffle_train_mode}); skip",
                    flush=True,
                )
                continue

            # Sample negatives against THIS epoch's emerging-drug pool (Fix 5).
            neg_rng = np.random.default_rng(self.seed + 100_000 + epoch)
            n_neg = self.neg_ratio * len(epoch_targets)
            neg_pairs = self._sample_negatives(epoch_targets, n_neg, neg_rng)

            epoch_loss = self._train_epoch(
                epoch_targets, neg_pairs, (edge_src, edge_dst, edge_rel), epoch,
            )

            # Validation on the STATIC eval KG (train_ddi + base_kg).
            if self.eval_strategy == "epoch":
                val_auc, val_ap = self._validate(val_a, val_b, val_y)
                self.scheduler.step(val_auc)
                print(
                    f"[nbfnet] [ep {epoch}/{self.n_epochs}] loss={epoch_loss:.4f} "
                    f"val_auc={val_auc:.4f} val_ap={val_ap:.4f} "
                    f"time={time.time()-epoch_start:.1f}s",
                    flush=True,
                )
                # Best-checkpoint tracking + persistence (Fix 4/5).
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.best_epoch = epoch
                    self.best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self.save_state(run_dir / "best_model.pt")
                if self._should_early_stop(epoch):
                    print(
                        f"[nbfnet] early stop at epoch {epoch} "
                        f"(best ep {self.best_epoch}, val_auc={self.best_val_auc:.4f})",
                        flush=True,
                    )
                    break

        # Load best checkpoint
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        fit_time = time.time() - fit_start
        print(
            f"[nbfnet] fit done. best_val_auc={self.best_val_auc:.4f} "
            f"best_epoch={self.best_epoch} time={fit_time:.1f}s",
            flush=True,
        )

        return {
            "best_val_auc": float(self.best_val_auc),
            "best_epoch": int(self.best_epoch),
            "fit_sec": float(fit_time),
        }

    def _pairs_to_node_idx(
        self, pairs: pd.DataFrame, drug_id_map: dict[str, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a ``[drug_a_id, drug_b_id]`` DataFrame to node-index arrays.

        Rows whose drug ids are absent from ``drug_id_map`` are dropped (with
        a warning) rather than raising, mirroring EmerGNN's lenient handling.
        """
        a_idx = pairs["drug_a_id"].astype(str).map(drug_id_map)
        b_idx = pairs["drug_b_id"].astype(str).map(drug_id_map)
        valid = a_idx.notna() & b_idx.notna()
        if not valid.all():
            print(
                f"[nbfnet] dropping {(~valid).sum()} val pairs with drugs "
                f"not in node vocab",
                flush=True,
            )
        return (
            a_idx[valid].to_numpy(dtype=np.int64),
            b_idx[valid].to_numpy(dtype=np.int64),
        )

    def _build_epoch_kg(
        self, epoch: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]:
        """Per-epoch shuffle_train snapshot (Codex Round 6 Fix 1).

        Calls :func:`shuffle_train` on the stored train-DDI + base-KG triplets
        to split this epoch's positives into KG facts vs prediction targets
        (drug-level cold simulation), then materializes the fact set as direct
        ``[h, t, r]`` -> edge tensors. NBFNet's own inverse-edge augmentation
        and query-edge masking are applied downstream, so we deliberately do
        NOT route through ``build_edge_lists_from_triplets`` (which would add
        self-loops + the EmerGNN relation convention).

        Returns.
            ((edge_src, edge_dst, edge_rel), epoch_targets) where the edge
            tensors are this epoch's KG (fact triplets + base KG) and
            ``epoch_targets`` is the (n, 3) [h, t, r] array of training targets.
        """
        rng = np.random.default_rng(self.seed + epoch)
        epoch_kg, epoch_targets = shuffle_train(
            self.train_ddi_triplets,
            self.base_kg_triplets,
            self.shuffle_train_mode,
            ratio=self.shuffle_ratio,
            rng=rng,
            extra_kg_ent=self._kg_entity_set,
        )
        epoch_kg = np.asarray(epoch_kg, dtype=np.int64)
        edge_src = torch.from_numpy(epoch_kg[:, 0].copy()).long().to(self.device)
        edge_dst = torch.from_numpy(epoch_kg[:, 1].copy()).long().to(self.device)
        edge_rel = torch.from_numpy(epoch_kg[:, 2].copy()).long().to(self.device)
        return (edge_src, edge_dst, edge_rel), np.asarray(epoch_targets, dtype=np.int64)

    def _sample_negatives(
        self, epoch_targets: np.ndarray, n_negatives: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample negative pairs from THIS epoch's emerging-drug pool (Fix 5).

        Uniform random pair sampling restricted to the drugs that appear in
        ``epoch_targets`` (the emerging pool), excluding any positive pair
        (symmetric). Returns an (n, 2) int64 array of (head, tail) node ids.
        """
        pos_set: set[tuple[int, int]] = set()
        for h, t in epoch_targets[:, :2]:
            pos_set.add((int(h), int(t)))
            pos_set.add((int(t), int(h)))
        pool = np.unique(epoch_targets[:, :2].reshape(-1))
        if len(pool) < 2 or n_negatives <= 0:
            return np.zeros((0, 2), dtype=np.int64)

        neg: list[tuple[int, int]] = []
        max_tries = int(n_negatives) * 50 + 100
        tries = 0
        while len(neg) < n_negatives and tries < max_tries:
            a = int(rng.choice(pool))
            b = int(rng.choice(pool))
            tries += 1
            if a != b and (a, b) not in pos_set:
                neg.append((a, b))
        if len(neg) < n_negatives:
            print(
                f"[nbfnet] neg sampling under-filled: got {len(neg)}/{n_negatives} "
                f"after {tries} tries (small emerging pool size={len(pool)})",
                flush=True,
            )
        return np.array(neg, dtype=np.int64) if neg else np.zeros((0, 2), dtype=np.int64)

    def _train_epoch(
        self,
        epoch_targets: np.ndarray,            # (n, 3) [h, t, r] positive targets
        neg_pairs: np.ndarray,                # (m, 2) [h, t] sampled negatives
        epoch_kg_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        epoch: int,
    ) -> float:
        """Train one epoch with per-source amortization (Codex Round 6 Fix 2).

        Groups the batch by source-a (forward BF) and by source-b (reverse BF),
        runs ONE Bellman-Ford pass per unique source with a union query-edge
        mask covering all that source's batch targets, caches the per-node
        encodings, then composes each pair's symmetric logit. Cuts BF passes
        per batch from 2*batch_size to (unique-a + unique-b).

        Returns mean epoch loss.
        """
        pos_a = epoch_targets[:, 0].astype(np.int64)
        pos_b = epoch_targets[:, 1].astype(np.int64)
        if len(neg_pairs):
            neg_a = neg_pairs[:, 0].astype(np.int64)
            neg_b = neg_pairs[:, 1].astype(np.int64)
        else:
            neg_a = np.zeros((0,), dtype=np.int64)
            neg_b = np.zeros((0,), dtype=np.int64)

        all_a = np.concatenate([pos_a, neg_a])
        all_b = np.concatenate([pos_b, neg_b])
        all_y = np.concatenate([
            np.ones(len(pos_a), dtype=np.float32),
            np.zeros(len(neg_a), dtype=np.float32),
        ])

        # Shuffle (deterministic per epoch).
        rng = np.random.default_rng(self.seed + 200_000 + epoch)
        perm = rng.permutation(len(all_a))
        all_a, all_b, all_y = all_a[perm], all_b[perm], all_y[perm]

        # Augment inverse edges ONCE for the whole epoch's KG snapshot.
        edge_src, edge_dst, edge_rel = epoch_kg_edges
        aug = self.model.augment_inverse_edges(edge_src, edge_dst, edge_rel)

        self.model.train()
        losses: list[float] = []
        n = len(all_a)
        n_batches = (n + self.batch_size - 1) // self.batch_size

        for step in range(n_batches):
            s = step * self.batch_size
            e = min(s + self.batch_size, n)
            batch_a = all_a[s:e]
            batch_b = all_b[s:e]
            batch_y = torch.from_numpy(all_y[s:e]).to(self.device)

            self.optimizer.zero_grad()
            logits = self._score_pairs(batch_a, batch_b, aug, training=True)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if (step + 1) % self.log_step_every == 0:
                recent = float(np.mean(losses[-self.log_step_every:]))
                print(
                    f"[nbfnet] [ep {epoch}/{self.n_epochs} step {step+1}/{n_batches}] "
                    f"loss={recent:.4f}",
                    flush=True,
                )

        return float(np.mean(losses)) if losses else float("nan")

    def _score_pairs_amortized(
        self,
        batch_a: np.ndarray,                  # (B,) source-a node ids
        batch_b: np.ndarray,                  # (B,) source-b node ids
        aug_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        training: bool,
    ) -> torch.Tensor:
        """Per-source amortized symmetric pair scoring (Codex Round 6 Fix 2).

        Args.
            batch_a / batch_b. node-id arrays for the pairs in this batch.
            aug_edges. ALREADY inverse-augmented (edge_src, edge_dst, edge_rel).
            training. if True, apply per-source union query-edge masking.

        Returns.
            logits. (B,) tensor of raw pre-sigmoid logits.
        """
        aug_src, aug_dst, aug_rel = aug_edges

        # Group targets by source for both directions.
        targets_by_a: dict[int, list[int]] = defaultdict(list)
        targets_by_b: dict[int, list[int]] = defaultdict(list)
        for a, b in zip(batch_a.tolist(), batch_b.tolist()):
            targets_by_a[a].append(b)
            targets_by_b[b].append(a)

        forward_cache: dict[int, torch.Tensor] = {}
        for src, tgts in targets_by_a.items():
            mask = (
                self.model.build_union_query_edge_mask(
                    aug_src, aug_dst, aug_rel, src, tgts
                )
                if training
                else None
            )
            forward_cache[src] = self.model.encode_from_source(
                src, aug_src, aug_dst, aug_rel,
                already_augmented=True, edge_keep_mask=mask,
            )

        reverse_cache: dict[int, torch.Tensor] = {}
        for src, tgts in targets_by_b.items():
            mask = (
                self.model.build_union_query_edge_mask(
                    aug_src, aug_dst, aug_rel, src, tgts
                )
                if training
                else None
            )
            reverse_cache[src] = self.model.encode_from_source(
                src, aug_src, aug_dst, aug_rel,
                already_augmented=True, edge_keep_mask=mask,
            )

        # Compose symmetric logits: h_sym = h_q(a,b) + h_q(b,a).
        logits = []
        for a, b in zip(batch_a.tolist(), batch_b.tolist()):
            h_sym = forward_cache[a][b] + reverse_cache[b][a]      # (d,)
            z = torch.cat([h_sym, self.model.query], dim=-1)       # (2d,)
            logits.append(self.model.mlp_head(z).squeeze(-1))
        return torch.stack(logits)

    def _score_pairs(
        self,
        batch_a: np.ndarray,
        batch_b: np.ndarray,
        aug_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        training: bool,
    ) -> torch.Tensor:
        """Dispatch to the batched fast path or the per-source amortized path.

        Uses the batched multi-source BF when ``use_batched_bf`` is on AND
        per-source query-edge masking is a no-op for this batch (verified by
        :meth:`_query_edges_present`). Otherwise falls back to the exact
        per-source masked path so correctness is never compromised.
        """
        if self.use_batched_bf:
            needs_mask = training and self._query_edges_present(aug_edges, batch_a, batch_b)
            if not needs_mask:
                return self._score_pairs_batched(batch_a, batch_b, aug_edges)
        return self._score_pairs_amortized(batch_a, batch_b, aug_edges, training=training)

    def _query_edges_present(
        self,
        aug_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_a: np.ndarray,
        batch_b: np.ndarray,
    ) -> bool:
        """True if any DDI-relation edge connects two of this batch's drugs.

        When False, per-source query-edge masking would remove nothing, so the
        unmasked batched path is exactly equivalent. (Holds for S2 training:
        target/negative drugs are emerging, their DDI edges are not in the
        epoch KG facts.) Cheap conservative check over DDI edges only.
        """
        if self.model.ddi_rel_id is None:
            return False
        aug_src, aug_dst, aug_rel = aug_edges
        ddi_fwd = self.model.ddi_rel_id
        ddi_inv = ddi_fwd + self.model.n_base_rel
        is_ddi = (aug_rel == ddi_fwd) | (aug_rel == ddi_inv)
        if not bool(is_ddi.any()):
            return False
        nodes = torch.as_tensor(
            np.unique(np.concatenate([batch_a, batch_b])),
            device=self.device, dtype=aug_src.dtype,
        )
        src_in = torch.isin(aug_src[is_ddi], nodes)
        dst_in = torch.isin(aug_dst[is_ddi], nodes)
        return bool((src_in & dst_in).any())

    def _score_pairs_batched(
        self,
        batch_a: np.ndarray,
        batch_b: np.ndarray,
        aug_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Batched (parallel) symmetric pair scoring — no per-source masking.

        One batched forward BF over unique source-a, one over unique source-b,
        then vectorized symmetric compose. Equivalent to _score_pairs_amortized
        with training=False (verified by the equivalence test).
        """
        aug_src, aug_dst, aug_rel = aug_edges

        ua, inv_a = np.unique(batch_a, return_inverse=True)
        ub, inv_b = np.unique(batch_b, return_inverse=True)
        ua_t = torch.as_tensor(ua, device=self.device, dtype=torch.long)
        ub_t = torch.as_tensor(ub, device=self.device, dtype=torch.long)

        hf = self.model.encode_from_sources(            # (Sa, n_nodes, d)
            ua_t, aug_src, aug_dst, aug_rel, already_augmented=True)
        hr = self.model.encode_from_sources(            # (Sb, n_nodes, d)
            ub_t, aug_src, aug_dst, aug_rel, already_augmented=True)

        rows_a = torch.as_tensor(inv_a, device=self.device, dtype=torch.long)
        rows_b = torch.as_tensor(inv_b, device=self.device, dtype=torch.long)
        bt_a = torch.as_tensor(batch_a, device=self.device, dtype=torch.long)
        bt_b = torch.as_tensor(batch_b, device=self.device, dtype=torch.long)

        fa = hf[rows_a, bt_b]                            # (B, d): h_q(a, b)
        rb = hr[rows_b, bt_a]                            # (B, d): h_q(b, a)
        h_sym = fa + rb                                  # (B, d)
        q = self.model.query.unsqueeze(0).expand(h_sym.size(0), -1)
        z = torch.cat([h_sym, q], dim=-1)               # (B, 2d)
        return self.model.mlp_head(z).squeeze(-1)       # (B,)

    def _validate(
        self, val_a: np.ndarray, val_b: np.ndarray, val_y: np.ndarray
    ) -> tuple[float, float]:
        """Compute AUROC + AP on the validation set (static eval KG).

        Uses the static eval KG (train_ddi + base_kg) stored in setup_graph,
        amortized over sources, with training=False (no query-edge masking —
        S2 val pairs are held out and never present in the eval KG).
        """
        if self._eval_edges is None:
            raise RuntimeError("setup_graph must be called before _validate")
        self.model.eval()
        aug = self.model.augment_inverse_edges(*self._eval_edges)

        logits_all = []
        with torch.no_grad():
            for s in range(0, len(val_a), self.batch_size):
                e = min(s + self.batch_size, len(val_a))
                batch_logits = self._score_pairs(
                    val_a[s:e], val_b[s:e], aug, training=False
                )
                logits_all.append(batch_logits.detach().cpu().numpy())

        logits_arr = np.concatenate(logits_all) if logits_all else np.zeros(0)
        probs = 1.0 / (1.0 + np.exp(-logits_arr))
        auc = float(roc_auc_score(val_y, probs))
        ap = float(average_precision_score(val_y, probs))
        return auc, ap

    def _should_early_stop(self, epoch: int) -> bool:
        """Early stop if val AUC hasn't improved in early_stop_patience epochs.

        Codex Round 6 Fix 4. Compares the current epoch against the best epoch
        tracked in :meth:`fit` (``self.best_epoch``).
        """
        return (epoch - self.best_epoch) >= self.early_stop_patience

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self, pairs: pd.DataFrame, drug_id_map: dict[str, int]
    ) -> np.ndarray:
        """Predict DDI probabilities for ``[drug_a_id, drug_b_id]`` pairs.

        Uses the static eval KG (train_ddi + base_kg) with the amortized
        symmetric scorer (no query-edge masking — test pairs are held out).
        Pairs with unknown drug ids are scored as 0.0 (kept in order so the
        caller's label alignment is preserved).
        """
        if self.model is None or self._eval_edges is None:
            raise RuntimeError("fit()/setup_graph() must run before predict_proba")

        a_idx = pairs["drug_a_id"].astype(str).map(drug_id_map)
        b_idx = pairs["drug_b_id"].astype(str).map(drug_id_map)
        valid = (a_idx.notna() & b_idx.notna()).to_numpy()
        a_full = a_idx.to_numpy()
        b_full = b_idx.to_numpy()

        out = np.zeros(len(pairs), dtype=np.float32)
        val_pos = np.where(valid)[0]
        if len(val_pos) == 0:
            return out

        va = a_full[valid].astype(np.int64)
        vb = b_full[valid].astype(np.int64)

        self.model.eval()
        aug = self.model.augment_inverse_edges(*self._eval_edges)
        probs_valid = np.empty(len(va), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(va), self.batch_size):
                e = min(s + self.batch_size, len(va))
                logits = self._score_pairs(va[s:e], vb[s:e], aug, training=False)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs_valid[s:e] = p
        out[val_pos] = probs_valid
        return out

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def save_state(self, path: Path) -> None:
        """Save model state to disk."""
        torch.save({
            "model_state": self.model.state_dict(),
            "best_val_auc": self.best_val_auc,
            "config": {
                "d": self.d,
                "n_layers": self.n_layers,
                "mlp_hidden": self.mlp_hidden,
                "n_nodes": self.n_nodes,
                "n_base_rel": self.n_base_rel,
                "ddi_rel_id": self.ddi_rel_id,
            },
        }, path)

    def load_state(self, path: Path) -> None:
        """Load model state from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        # Rebuild model if needed
        if self.model is None:
            cfg = checkpoint["config"]
            self.n_nodes = cfg["n_nodes"]
            self.n_base_rel = cfg["n_base_rel"]
            self.ddi_rel_id = cfg["ddi_rel_id"]
            self.init_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.best_val_auc = checkpoint.get("best_val_auc", -float("inf"))
