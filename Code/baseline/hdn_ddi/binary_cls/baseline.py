"""HDN-DDI binary-classification baseline (paper-faithful, BRICS-aware).

Implements the paper's 3 core contributions (Sun & Zheng 2025, BMC
Bioinformatics 2025):

  1. **refined BRICS substructure extraction** — 3-level molecular
     graphs via :func:`baseline.hdn_ddi.mol_features.mol_to_data`
     (atoms y=0, BRICS fragments y=1, super-node y=2).
  2. **66-dim node features** per paper Additional file 1 Table S1.
  3. **Bipartite inter-drug graph restricted to y==1 substructure-level
     nodes** (paper §HDN Encoder: "Unlike existing methods that
     consider all atomic nodes...").

Paper task = per-triplet binary classification "does (drug_a, rel,
drug_b) exist?" with sigmoid + BCE.  Project keeps `rel_total=1` and
trains BCE on `train + train_negatives` (binary view of the multi-
relation DrugBank label set).

Independence: per CLAUDE.md §"文件级独立性" this module does NOT
import anything from :mod:`reproductions.HDN-DDI`; the BRICS
featurizer is a separately-maintained copy in
:mod:`baseline.hdn_ddi.mol_features`.
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Batch

from baseline.base import BaselineModel, register, write_manifest
from baseline.dsn_ddi.models import make_bipartite_data
from baseline.hdn_ddi._shared import (
    bipartite_edge_index_y1 as _bipartite_edge_index_y1,
    drug_smiles_dict as _drug_smiles_dict,
    ensure_mol_graphs_pkl as _ensure_mol_graphs_pkl,
    load_mol_graphs_pkl as _load_mol_graphs_pkl,
)
from baseline.hdn_ddi.mol_features import (
    ATOM_FEATURE_DIM,
    build_drug_graphs,
)
from baseline.hdn_ddi.models import HDN_DDI

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


# Canonical default location of this baseline's 3-level hierarchical
# molecular-graph pkl. Lives in ``baseline/hdn_ddi/_data/necessary/``
# per CLAUDE.md §"Baseline 规范" §1 (necessary files in `_data/necessary/`,
# parallel to reproduction side's `_Original-Dataset/necessary/`).
# Each side owns the pkl files corresponding to ITS dataset; baseline
# pkl is built from THIS PROJECT's DrugBank pool, not from upstream's
# 1706-drug pool.
#
# Filename suffix: ``__mine`` (per CLAUDE.md §"复现规范" §2 + §"Baseline
# 规范" §1). Unlike the reproduction side where both ``__official.pkl``
# (upstream ship) and ``__mine.pkl`` (our builder) can coexist, baseline
# side never has an ``__official`` version — our DrugBank pool has no
# upstream-shipped pkl. The explicit ``__mine`` suffix tells the reader
# "this is OUR builder's output; do not expect an official one".
#
# If the file doesn't exist (or is stale relative to the train dataset),
# ``ensure_mol_graphs_pkl`` will auto-rebuild it by subprocess-invoking
# ``_data/necessary/build_hierarchical_pkl.py`` (baseline-side detection
# DOES auto-build, per §"Baseline 规范" §2 step 3 — contrast with
# reproduction-side which only loads, never builds).
#
# parents math from ``Code/baseline/hdn_ddi/binary_cls/baseline.py``:
#   parents[0] = binary_cls/      parents[2] = baseline/
#   parents[1] = hdn_ddi/         parents[3] = Code/
_HDN_DDI_PKG = Path(__file__).resolve().parents[1]  # baseline/hdn_ddi/
DEFAULT_MOL_PKL_PATH = (
    _HDN_DDI_PKG / "_data" / "necessary" / "hdn_ddi_mol_graphs__mine.pkl"
)


# ---------------------------------------------------------------------
# Baseline class
# ---------------------------------------------------------------------

@register("hdn_ddi")
class HDNDDIBaseline(BaselineModel):
    """HDN-DDI BRICS-aware baseline (paper-faithful)."""

    VERSION = "2.0"  # paper-faithful BRICS-aware (replaces old 1.0 flat variant)

    def __init__(
        self,
        *,
        in_features: int = ATOM_FEATURE_DIM,
        hidd_dim: int = 64,
        kge_dim: int = 128,
        heads_out_feat_params: tuple[int, ...] = (64, 64, 64, 64, 64, 64),
        blocks_params: tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 512,
        n_epochs: int = 5,
        device: str = "auto",
        log_step_every: int = 50,
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        save_strategy: str = "no",
        save_steps: int = 500,
        save_total_limit: int = 3,
        load_best_model_at_end: bool = True,
        run_dir: str | None = None,
        # Path to the 3-level hierarchical mol-graphs pkl (CLAUDE.md
        # §"Baseline 规范" §1: necessary file in `_data/necessary/`).
        # Default = canonical project path (:data:`DEFAULT_MOL_PKL_PATH`).
        # When the file is missing or stale (cached drug set < train.drugs),
        # ``_build_graphs`` auto-rebuilds via subprocess-invoking
        # ``baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py``
        # (each baseline owns its own builder copy; never crosses to
        # reproduction side per §"文件级独立性") — no manual prerequisite
        # step. Pass ``mol_pkl_path=None`` to explicitly opt out and use
        # the legacy on-the-fly fallback (NOT recommended; debug only).
        mol_pkl_path: str | Path | None = DEFAULT_MOL_PKL_PATH,
    ) -> None:
        if len(heads_out_feat_params) != len(blocks_params):
            raise ValueError(
                "heads_out_feat_params and blocks_params must have the same length."
            )
        # Same architectural invariant as the flat variant: intra/inter
        # attention outputs are hard-coded to 32*2=64 dims each, so the
        # concat is 128 and we require head*out_feats == 128 == kge_dim.
        for h, n in zip(heads_out_feat_params, blocks_params):
            if h * n != 128:
                raise ValueError(
                    "HDN-DDI's IntraGraphAttention/InterGraphAttention output "
                    "is hardcoded to 64 dims each; every "
                    "(head_out_feats * n_heads) must equal 128. Got "
                    f"{h} * {n} = {h * n}."
                )
        if kge_dim != 128:
            raise ValueError(
                "HDN-DDI requires kge_dim == 128 to match the per-block "
                f"embedding dim. Got {kge_dim}."
            )
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.heads_out_feat_params = tuple(heads_out_feat_params)
        self.blocks_params = tuple(blocks_params)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = self._resolve_device(device)
        self.log_step_every = int(log_step_every)
        self.eval_strategy = eval_strategy
        self.eval_steps = int(eval_steps)
        self.save_strategy = save_strategy
        self.save_steps = int(save_steps)
        self.save_total_limit = int(save_total_limit)
        self.load_best_model_at_end = bool(load_best_model_at_end)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.mol_pkl_path = Path(mol_pkl_path) if mol_pkl_path is not None else None
        self._model: HDN_DDI | None = None
        self._graphs: dict | None = None
        self._missing: list[str] = []

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _build_graphs(self, train: "PairDataset") -> None:
        # CLAUDE.md §"Baseline 规范" §2 step 3: baseline must auto-detect
        # the necessary pkl, and auto-build if missing/stale/corrupt via
        # the local subprocess builder (no manual prerequisite CLI step).
        # ``ensure_mol_graphs_pkl`` handles HIT / STALE / CORRUPT /
        # MISSING branches + logs which path was taken.
        if self.mol_pkl_path is not None:
            self._graphs = _ensure_mol_graphs_pkl(train, self.mol_pkl_path)
            # ``ensure_mol_graphs_pkl`` guarantees coverage post-build,
            # but a manually-supplied stale pkl could still leave gaps —
            # surface them as a warning.
            train_drug_ids = set(
                map(str, train.drugs["drugbank_id"]) if train.drugs is not None
                else []
            )
            self._missing = sorted(train_drug_ids - set(self._graphs.keys()))
            if self._missing:
                print(
                    f"[hdn_ddi] WARNING: {len(self._missing)} drugs from "
                    f"train.drugs are missing in mol_pkl ({self.mol_pkl_path}). "
                    f"Use ``--force-rebuild`` or delete the pkl to regenerate.",
                    file=sys.stderr,
                )
            return
        # Legacy on-the-fly path: ONLY reachable when caller explicitly
        # sets ``mol_pkl_path=None`` (opt-out). NOT recommended for
        # production — see CLAUDE.md §"Baseline 规范" §2 (auto-build via
        # `_data/necessary/build_hierarchical_pkl.py` is the standard path).
        smiles = _drug_smiles_dict(train)
        self._graphs, self._missing = build_drug_graphs(smiles)
        if not self._graphs:
            raise ValueError(
                "HDN-DDI (BRICS) built zero molecular graphs -- every drug "
                "failed to parse."
            )
        if self._missing:
            print(
                f"[hdn_ddi] {len(self._missing)} drugs skipped "
                f"(no parseable SMILES).",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Batch construction (paper-faithful: bipartite filters y==1 only)
    # ------------------------------------------------------------------

    def _make_pair_batch(self, pairs: pd.DataFrame):
        keep_idx: list[int] = []
        h_list, t_list, b_list = [], [], []
        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            ga = self._graphs.get(a)
            gb = self._graphs.get(b)
            if ga is None or gb is None:
                continue
            ga = ga.clone()
            gb = gb.clone()
            edge_b = _bipartite_edge_index_y1(ga, gb)
            b_data = make_bipartite_data(ga.x, gb.x, edge_b)
            keep_idx.append(i)
            h_list.append(ga)
            t_list.append(gb)
            b_list.append(b_data)
        if not keep_idx:
            return None, None, None, None, np.zeros(len(pairs), dtype=bool)

        h_batch = Batch.from_data_list(h_list)
        t_batch = Batch.from_data_list(t_list)
        b_batch = Batch.from_data_list(b_list)
        rels = torch.zeros(len(keep_idx), dtype=torch.long)
        mask = np.zeros(len(pairs), dtype=bool)
        mask[np.asarray(keep_idx)] = True
        return h_batch, t_batch, rels, b_batch, mask

    # ------------------------------------------------------------------
    # ABC surface (same shape as flat HDNDDIBaseline.fit / predict)
    # ------------------------------------------------------------------

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        self._build_graphs(train)

        self._model = HDN_DDI(
            in_features=self.in_features,
            hidd_dim=self.hidd_dim,
            kge_dim=self.kge_dim,
            rel_total=1,
            heads_out_feat_params=list(self.heads_out_feat_params),
            blocks_params=list(self.blocks_params),
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Paper §Parameters + jcsun-00 inductive_train.py:
        #   scheduler = LambdaLR(0.96 ** epoch)
        # Per-epoch exponential decay, must call scheduler.step() once per epoch
        # (NOT per batch).  Missing this was a P0 review finding.
        scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 0.96 ** epoch)

        pos = train.splits.train.copy()
        best_val_auc = -1.0
        best_state: dict | None = None

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        n_pos_plus_neg = 2 * len(pos)
        steps_per_epoch = (n_pos_plus_neg + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[hdn_ddi] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        ckpt_root = (self.run_dir / "checkpoints") if self.run_dir is not None else None

        def _eval_dict() -> dict:
            auc = self._validate(val) if val is not None else float("nan")
            return {"val_auc": auc}

        def _save_ckpt(tag: str, scope: str) -> None:
            if ckpt_root is None:
                return
            ckpt_path = ckpt_root / f"checkpoint-{tag}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.save(ckpt_path)
            progress.log_save(str(ckpt_path), scope=scope)
            rotate_checkpoints(ckpt_root, self.save_total_limit)

        for epoch in range(self.n_epochs):
            # Paper-faithful: regenerate negatives each epoch (jcsun-00
            # uses per-batch fresh sampling via DrugDataset.__corrupt_*;
            # PairDataset gives us per-epoch granularity which is the
            # closest we can do without rewriting the data loader).
            # Missing this was a P0 review finding (fixed-neg made the
            # model overfit to a small neg set).
            neg = train.get_train_negatives(epoch, regenerate=True)
            pairs_df = (
                pd.concat(
                    [
                        pos[["drug_a_id", "drug_b_id"]].assign(label=1),
                        neg[["drug_a_id", "drug_b_id"]].assign(label=0),
                    ],
                    ignore_index=True,
                )
                .sample(frac=1, random_state=epoch)
                .reset_index(drop=True)
            )

            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start : start + self.batch_size]
                h_b, t_b, rels, b_b, mask = self._make_pair_batch(batch)
                if h_b is None:
                    continue
                h_b = h_b.to(self.device)
                t_b = t_b.to(self.device)
                rels = rels.to(self.device)
                b_b = b_b.to(self.device)
                y = torch.tensor(
                    batch.loc[mask, "label"].to_numpy(),
                    dtype=torch.float32,
                    device=self.device,
                )
                opt.zero_grad(set_to_none=True)
                logits = self._model((h_b, t_b, rels, b_b))
                loss = binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                opt.step()
                progress.step(loss.item())

                if progress.should_eval_step() and val is not None:
                    self._model.eval()
                    metrics = _eval_dict()
                    self._model.train()
                    progress.log_eval(metrics, scope="step")
                    if metrics.get("val_auc", -1) > best_val_auc:
                        best_val_auc = metrics["val_auc"]
                        best_state = copy.deepcopy(self._model.state_dict())
                if progress.should_save_step():
                    _save_ckpt(f"step{progress.step_count}", scope="step")

            extra: dict = {}
            if progress.should_eval_epoch() and val is not None:
                self._model.eval()
                metrics = _eval_dict()
                self._model.train()
                progress.log_eval(metrics, scope="epoch")
                extra.update(metrics)
                if metrics.get("val_auc", -1) > best_val_auc:
                    best_val_auc = metrics["val_auc"]
                    best_state = copy.deepcopy(self._model.state_dict())
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}", scope="epoch")
            progress.epoch_end(extra=extra if extra else None)
            # Per-epoch exponential LR decay (paper §Parameters +
            # jcsun-00 inductive_train.py): step once at epoch end,
            # AFTER eval / save (so the printed val metrics reflect the
            # epoch's actual training-time LR).
            scheduler.step()

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[hdn_ddi] loaded best val_auc={best_val_auc:.4f} state",
                flush=True,
            )

    @torch.no_grad()
    def _validate(self, val: "PairDataset") -> float:
        pos = val.splits.val_s2[["drug_a_id", "drug_b_id"]]
        neg = val.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y_score = np.concatenate(
            [self.predict_proba(pos), self.predict_proba(neg)]
        )
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y_true, y_score))

    @torch.no_grad()
    def predict_proba(
        self,
        pairs: pd.DataFrame,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None or self._graphs is None:
            raise RuntimeError(
                "HDNDDIBaseline must be fitted (or loaded) before prediction."
            )
        self._model.eval()
        out = np.full(len(pairs), 0.5, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            h_b, t_b, rels, b_b, mask = self._make_pair_batch(batch)
            if h_b is None:
                continue
            h_b = h_b.to(self.device)
            t_b = t_b.to(self.device)
            rels = rels.to(self.device)
            b_b = b_b.to(self.device)
            logits = self._model((h_b, t_b, rels, b_b))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            out_slice = np.full(len(batch), 0.5, dtype=np.float32)
            out_slice[mask] = probs
            out[start : start + len(batch)] = out_slice
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump({"graphs": self._graphs, "missing": self._missing}, f)
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "variant": "brics",
                "hyperparameters": {
                    "in_features": self.in_features,
                    "hidd_dim": self.hidd_dim,
                    "kge_dim": self.kge_dim,
                    "heads_out_feat_params": list(self.heads_out_feat_params),
                    "blocks_params": list(self.blocks_params),
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "sklearn_version": sklearn.__version__,
                },
                "graph_metadata": {
                    "n_drugs_with_graph": len(self._graphs),
                    "n_drugs_missing": len(self._missing),
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str", **kwargs) -> "HDNDDIBaseline":
        out = Path(path)
        with (out / "manifest.json").open() as f:
            manifest = json.load(f)
        hp = manifest.get("extra", {}).get("hyperparameters", {})
        # Mix in any caller overrides (e.g. for device).
        init_kwargs = {**hp, **kwargs}
        # heads_out_feat_params / blocks_params must be tuples for __init__.
        init_kwargs["heads_out_feat_params"] = tuple(
            init_kwargs.get("heads_out_feat_params", (64, 64, 64, 64, 64, 64))
        )
        init_kwargs["blocks_params"] = tuple(
            init_kwargs.get("blocks_params", (2, 2, 2, 2, 2, 2))
        )
        obj = cls(**init_kwargs)
        obj._model = HDN_DDI(
            in_features=obj.in_features,
            hidd_dim=obj.hidd_dim,
            kge_dim=obj.kge_dim,
            rel_total=1,
            heads_out_feat_params=list(obj.heads_out_feat_params),
            blocks_params=list(obj.blocks_params),
        ).to(obj.device)
        state = torch.load(out / "model.pt", map_location=obj.device)
        obj._model.load_state_dict(state)
        with (out / "graphs.pkl").open("rb") as f:
            saved = pickle.load(f)
        obj._graphs = saved["graphs"]
        obj._missing = saved.get("missing", [])
        return obj
