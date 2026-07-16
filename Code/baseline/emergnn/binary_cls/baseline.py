"""EmerGNN binary baseline (multimode, paper-faithful 3-sub-model dispatch).

**Official EmerGNN binary baseline for the project** (promoted 2026-05-20
after the multimode-vs-kgonly comparison on seed 42). Old single-S2-mode
binary baseline lives at ``baseline/emergnn/deprecate/binary_cls_legacy_buggy/``;
the alternative kgonly variant lives at
``baseline/emergnn/deprecate/binary_cls_kgonly/``. The per-mode training
helper class ``_PerModeEmerGNN`` used by this wrapper lives at
``baseline/emergnn/_per_mode.py``.

Background — the bug this design fixes:
    The earlier binary baseline trained one EmerGNN with hardcoded
    ``shuffle_train_mode="S2"`` and evaluated it on test_s0/s1/s2. Under
    paper-faithful drugbank KG, this produced degenerate AUC=0.5 on s0/s1
    (warm + half-cold) while s2 worked. Root cause: the S2-mode-trained
    model learned "lots of train_ddi edges adjacent ⇒ this pair is a fact,
    not a target ⇒ predict negative", which collapsed s0/s1 evaluation.
    See ``baseline/emergnn/_results/2026-05-18__binary_cls__seed42.md``
    and ``Code/runs/_diagnose/emergnn_s0s1_collapse.json``.

This baseline reproduces the upstream "one model per setting" pattern
(``DrugBank/evaluate.py:run_model`` dispatch) faithfully:

  * S0 sub-model — shuffle_train_mode='S0' (random 80/20 edge holdout),
                   feat='E' (learned entity embedding), batch_size=128,
                   weight_decay=1e-6
  * S1 sub-model — shuffle_train_mode='S1' (1-cold drug holdout),
                   feat='M' (Morgan fingerprint), batch_size=32,
                   weight_decay=1e-8
  * S2 sub-model — shuffle_train_mode='S2' (2-cold drug holdout),
                   feat='M' (Morgan fingerprint), batch_size=32,
                   weight_decay=1e-8

These four (mode, feat, batch, weight_decay) bundles are exactly what
upstream dispatches per dataset family (``evaluate.py:54-70``).

At ``predict_proba(pairs)`` time, each pair is routed to the sub-model
whose training-time drug distribution matches the pair's composition:

  * both drugs in G1 (warm pool) → s0 sub-model
  * exactly one of (head, tail) in G2 (cold pool) → s1 sub-model
  * both drugs in G2 (cold pool) → s2 sub-model

The 2026-05-20 comparison (seed 42, backbone_kg_source=drugbank) showed this
baseline beats both (a) the original 5/18 buggy single-S2-mode design
on s0/s1 by 49pt/33pt AUC, and (b) the kgonly variant on s0 by 7.5pt
AUC. See ``baseline/emergnn/_results/2026-05-20__binary_cls__seed42__final.md``
for the head-to-head numbers.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from baseline.base import BaselineModel, register, write_manifest
from baseline.emergnn._per_mode import _PerModeEmerGNN

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


# ---------------------------------------------------------------------
# Internal subclass: EmerGNNBaseline with a configurable validation split
# ---------------------------------------------------------------------
# The stock ``EmerGNNBaseline._validate`` is hard-coded to read ``val_s2``,
# which is what biased the original buggy run toward S2 specialization.
# Each sub-model in the multimode wrapper must validate on its OWN cold-
# start split so the best-state selection follows the same distribution
# the sub-model was trained for.

class _ModeSpecificEmerGNN(_PerModeEmerGNN):
    """EmerGNNBaseline with a configurable ``_validate`` split.

    Parameters
    ----------
    val_split : {'val_s0', 'val_s1', 'val_s2'}
        Which PairDataset val split to use for best-checkpoint selection
        and ReduceLROnPlateau step input.
    """

    def __init__(self, *, val_split: str = "val_s2", **kwargs: Any) -> None:
        if val_split not in ("val_s0", "val_s1", "val_s2"):
            raise ValueError(
                f"val_split must be one of val_s0/val_s1/val_s2, got {val_split!r}"
            )
        super().__init__(**kwargs)
        self._val_split = val_split

    @torch.no_grad()
    def _validate(self, val: "PairDataset") -> float:
        # Note: catching all exceptions here mirrors the parent's behavior.
        # In practice the only failure mode is "split not present in PKL"
        # (legacy bundles); other failures are real bugs that the parent
        # also masks. Keep the broad except to match parent semantics.
        pos = getattr(val.splits, self._val_split)[["drug_a_id", "drug_b_id"]]
        try:
            neg = val.get_negatives(self._val_split)[["drug_a_id", "drug_b_id"]]
        except Exception:
            return float("nan")
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y_score = np.concatenate(
            [self.predict_proba(pos), self.predict_proba(neg)]
        )
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y_true, y_score))

    # ------------------------------------------------------------------
    # Save/load: extend parent to preserve mode-specific hyperparams that
    # the parent class's manifest doesn't capture (val_split,
    # shuffle_train_mode, weight_decay, feat, batch_size). Inference-only
    # restore still works without this (the parent's graph.pkl + model.pt
    # are sufficient for predict_proba); we add this so anyone inspecting
    # the saved sub-model can recover the exact training config.
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        super().save(path)
        meta = {
            "val_split": self._val_split,
            "shuffle_train_mode": self.shuffle_train_mode,
            "shuffle_ratio": self.shuffle_ratio,
            "weight_decay": self.weight_decay,
            "feat": self.feat,
            "batch_size": self.batch_size,
            "n_dim": self.n_dim,
            "length": self.length,
            "learning_rate": self.learning_rate,
        }
        with (Path(path) / "mode_meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: "Path | str") -> "_ModeSpecificEmerGNN":
        # Parent.load returns an EmerGNNBaseline instance with restored
        # graph + model state but using parent class. We need to spawn a
        # _ModeSpecificEmerGNN with the right val_split + other
        # mode hyperparams from the sidecar.
        meta_path = Path(path) / "mode_meta.pkl"
        if not meta_path.is_file():
            # Backward compat: no sidecar means we were saved by a pre-fix
            # version. Fall back to parent load + assume val_split='val_s2'.
            base = _PerModeEmerGNN.load(path)
            inst = cls.__new__(cls)
            inst.__dict__.update(base.__dict__)
            inst._val_split = "val_s2"
            return inst
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        # Instantiate this class fresh with the recovered hyperparams, then
        # restore weights via parent's load path.
        init_kwargs = {
            "val_split": meta["val_split"],
            "n_dim": meta["n_dim"],
            "length": meta["length"],
            "feat": meta["feat"],
            "learning_rate": meta["learning_rate"],
            "batch_size": meta["batch_size"],
            "shuffle_train_mode": meta["shuffle_train_mode"],
            "shuffle_ratio": meta["shuffle_ratio"],
            "weight_decay": meta["weight_decay"],
        }
        inst = cls(**init_kwargs)
        base = _PerModeEmerGNN.load(path)
        # Copy restored runtime state from `base` into `inst`. Note we
        # also include ``_n_base_rel_with_ddi`` — parent save uses it to
        # serialize the correct relation count, so an inst that re-saves
        # without it would fall back to raw ``_n_base_rel`` and corrupt
        # the manifest (codex round-2 finding).
        for attr in (
            "_model",
            "_entity2id",
            "_n_ent",
            "_n_base_rel",
            "_n_base_rel_with_ddi",
            "_edge_src",
            "_edge_dst",
            "_edge_rel",
            "_eval_edges",
        ):
            if hasattr(base, attr):
                setattr(inst, attr, getattr(base, attr))
        return inst


# ---------------------------------------------------------------------
# Hardcoded sub-model configs (match upstream evaluate.py:54-70 EXACTLY)
# ---------------------------------------------------------------------
# Upstream ``DrugBank/evaluate.py:run_model`` dispatches per dataset:
#   - S1/S2 setting:  lr=0.001  lamb=0.00000001  feat='M'  n_batch=32   length=3  n_dim=64
#   - S0    setting:  lr=0.001  lamb=0.000001    feat='E'  n_batch=128  length=3  n_dim=64
# We hardcode the mode-specific quartet (shuffle_train_mode, feat,
# batch_size, weight_decay) since deviating from upstream's task-specific
# hyperparams was a contributing factor to the 2026-05-18 collapse bug.
# `lr`, `length`, `n_dim` are shared across all 3 sub-models (per upstream)
# and remain configurable through the wrapper's __init__.
_SUBMODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "s0": {
        "shuffle_train_mode": "S0",
        "feat": "E",
        "batch_size": 128,
        "weight_decay": 1e-6,  # upstream S0 lamb
        "val_split": "val_s0",
    },
    "s1": {
        "shuffle_train_mode": "S1",
        "feat": "M",
        "batch_size": 32,
        "weight_decay": 1e-8,  # upstream S1/S2 lamb
        "val_split": "val_s1",
    },
    "s2": {
        "shuffle_train_mode": "S2",
        "feat": "M",
        "batch_size": 32,
        "weight_decay": 1e-8,  # upstream S1/S2 lamb
        "val_split": "val_s2",
    },
}


@register("emergnn")
class EmerGNNBaseline(BaselineModel):
    """EmerGNN binary baseline — multimode (Method A).

    Trains 3 internal :class:`_ModeSpecificEmerGNN` instances during
    :meth:`fit`, one per S0/S1/S2 mode (matching upstream paper conventions).
    Routes test pairs to the right sub-model based on G1/G2 drug membership.

    All shared hyperparams accept the same name as :class:`EmerGNNBaseline`
    so ``run_baseline.py`` can pass identical ``common_kwargs``. Mode-specific
    hyperparams (``shuffle_train_mode``, ``feat``, ``batch_size``,
    ``weight_decay``, ``val_split``) are HARDCODED per sub-model and NOT
    user-overridable — paper-faithful per CLAUDE.md §"Baseline 规范" §4 case A.
    """

    name = "emergnn"
    VERSION = "1.0"

    def __init__(
        self,
        *,
        # Shared hyperparams (forwarded verbatim to all 3 sub-models)
        n_dim: int = 64,
        length: int = 3,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        device: str = "auto",
        backbone_kg_source: str = "drugbank",
        merged_kg_path: str | Path | None = None,
        merged_kg_blocklist: tuple[str, ...] = (),
        weight_decay: float = 1e-8,
        shuffle_ratio: float = 0.8,
        log_step_every: int = 50,
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        save_strategy: str = "no",
        save_steps: int = 500,
        save_total_limit: int = 3,
        load_best_model_at_end: bool = True,
        run_dir: str | Path | None = None,
        # ── EXPLICITLY REJECTED kwargs (mode-specific, hardcoded per sub-model)
        # If the caller passes any of these we raise to make the error obvious.
        # They live in ``_SUBMODEL_CONFIGS`` and are not user-overridable.
        feat: str | None = None,
        batch_size: int | None = None,
        shuffle_train_mode: str | None = None,
    ) -> None:
        if feat is not None or batch_size is not None or shuffle_train_mode is not None:
            raise TypeError(
                "EmerGNNBaseline takes neither `feat`, `batch_size`, "
                "nor `shuffle_train_mode`. These are HARDCODED per S0/S1/S2 "
                "sub-model to match upstream paper conventions. See "
                "_SUBMODEL_CONFIGS in this file."
            )
        # Pack the kwargs that flow into each sub-model. Note ``weight_decay``
        # is INTENTIONALLY excluded — _SUBMODEL_CONFIGS sets a different
        # weight_decay per sub-model (S0=1e-6, S1/S2=1e-8, matching upstream).
        # The wrapper's ``weight_decay`` kwarg above is accepted only for
        # signature compatibility with EmerGNNBaseline / run_baseline.py
        # common_kwargs and is silently overridden per sub-model.
        del weight_decay  # silence ruff/pyright; intentionally unused
        self._shared_kwargs: dict[str, Any] = dict(
            n_dim=n_dim,
            length=length,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            device=device,
            backbone_kg_source=backbone_kg_source,
            merged_kg_path=merged_kg_path,
            merged_kg_blocklist=merged_kg_blocklist,
            shuffle_ratio=shuffle_ratio,
            log_step_every=log_step_every,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
        )
        # run_dir is per-sub-model: each gets its own subdir so checkpoints
        # don't collide.
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self._models: dict[str, _ModeSpecificEmerGNN] = {}
        # G1/G2 sets are populated in fit() and used by predict_proba routing.
        self._g1_drugs: set[str] = set()
        self._g2_drugs: set[str] = set()

    # ------------------------------------------------------------------
    # ABC surface
    # ------------------------------------------------------------------

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        # Capture drug partition for predict-time routing. PairDataset's
        # SplitFolds protocol exposes g1_drugs / g2_drugs.
        if not hasattr(train.splits, "g1_drugs") or not hasattr(train.splits, "g2_drugs"):
            raise ValueError(
                "EmerGNNBaseline requires PairDataset with explicit "
                "g1_drugs / g2_drugs partition (SplitFolds protocol)."
            )
        self._g1_drugs = set(map(str, train.splits.g1_drugs))
        self._g2_drugs = set(map(str, train.splits.g2_drugs))
        # Defensive: ensure G1 ∩ G2 = ∅. If a drug ends up in both sets due
        # to a corrupted SplitFolds partition, `_classify_pair` would route
        # it by precedence (G1-and-G1 → s0) and silently miscategorize.
        # Refuse to proceed in that case (codex round-2 defensive check).
        overlap = self._g1_drugs & self._g2_drugs
        if overlap:
            raise ValueError(
                f"EmerGNNBaseline.fit: G1 and G2 drug sets overlap "
                f"({len(overlap)} shared drugs; first 5: {sorted(overlap)[:5]}). "
                f"This indicates a corrupted PairDataset partition. Routing "
                f"would silently miscategorize these drugs."
            )

        # Train 3 sub-models sequentially. We could parallelize but that
        # complicates GPU memory; sequential is simpler and adequate.
        for split_key in ("s0", "s1", "s2"):
            cfg = _SUBMODEL_CONFIGS[split_key]
            print(
                f"\n=========== [emergnn] training {split_key} "
                f"sub-model (mode={cfg['shuffle_train_mode']}, "
                f"feat={cfg['feat']}, batch={cfg['batch_size']}) ===========",
                flush=True,
            )
            sub_run_dir = (
                self._run_dir / f"submodel_{split_key}"
                if self._run_dir is not None
                else None
            )
            sub_kwargs = dict(self._shared_kwargs)
            sub_kwargs["shuffle_train_mode"] = cfg["shuffle_train_mode"]
            sub_kwargs["feat"] = cfg["feat"]
            sub_kwargs["batch_size"] = cfg["batch_size"]
            sub_kwargs["weight_decay"] = cfg["weight_decay"]
            sub_kwargs["run_dir"] = sub_run_dir
            sub_model = _ModeSpecificEmerGNN(
                val_split=cfg["val_split"],
                **sub_kwargs,
            )
            sub_model.fit(train, val=val, kg=kg)
            self._models[split_key] = sub_model

    # ------------------------------------------------------------------
    # Predict-time routing
    # ------------------------------------------------------------------

    def _classify_pair(self, drug_a: str, drug_b: str) -> str:
        """Return 's0' | 's1' | 's2' based on G1/G2 membership of (drug_a, drug_b).

        Requires both drugs to be in G1 ∪ G2 (the partition captured at
        ``fit`` time). Raises ``ValueError`` if a drug is in neither set —
        that would indicate either (a) a drug never seen by the model at
        fit time (predict-on-truly-novel), or (b) a corrupted G1/G2 partition.
        Explicit G2 membership is checked rather than treating "not in G1"
        as "in G2", which would silently misclassify drugs the model has
        never encountered.
        """
        a_in_g1 = drug_a in self._g1_drugs
        a_in_g2 = drug_a in self._g2_drugs
        b_in_g1 = drug_b in self._g1_drugs
        b_in_g2 = drug_b in self._g2_drugs
        if not (a_in_g1 or a_in_g2):
            raise ValueError(
                f"drug_a={drug_a!r} is in neither g1_drugs nor g2_drugs; "
                f"model has no training-time exposure to it. Either the "
                f"partition is corrupted or the caller passed an unknown drug."
            )
        if not (b_in_g1 or b_in_g2):
            raise ValueError(
                f"drug_b={drug_b!r} is in neither g1_drugs nor g2_drugs; "
                f"model has no training-time exposure to it. Either the "
                f"partition is corrupted or the caller passed an unknown drug."
            )
        if a_in_g1 and b_in_g1:
            return "s0"
        if a_in_g2 and b_in_g2:
            return "s2"
        return "s1"  # exactly one of a/b is in g1 (and the other in g2)

    @torch.no_grad()
    def predict_proba(
        self,
        pairs: pd.DataFrame,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        """Route each pair to its mode-matched sub-model and concat predictions."""
        if not self._models:
            raise RuntimeError(
                "EmerGNNBaseline must be fitted (or loaded) before prediction."
            )
        if len(pairs) == 0:
            return np.empty(0, dtype=np.float32)

        # Tag each row with its routing split.
        a = pairs["drug_a_id"].astype(str)
        b = pairs["drug_b_id"].astype(str)
        routes = np.empty(len(pairs), dtype=object)
        for i, (da, db) in enumerate(zip(a, b)):
            routes[i] = self._classify_pair(da, db)

        # Predict per route. Preserves original row order by indexing back.
        out = np.empty(len(pairs), dtype=np.float32)
        for split_key in ("s0", "s1", "s2"):
            mask = routes == split_key
            if not mask.any():
                continue
            sub_pairs = pairs.iloc[mask].reset_index(drop=True)
            sub_scores = self._models[split_key].predict_proba(sub_pairs, kg=kg)
            out[mask] = sub_scores
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        """Save all 3 sub-models + routing metadata under one directory."""
        if not self._models:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for split_key, sub in self._models.items():
            sub.save(out / f"submodel_{split_key}")
        with (out / "routing.pkl").open("wb") as f:
            pickle.dump(
                {
                    "g1_drugs": sorted(self._g1_drugs),
                    "g2_drugs": sorted(self._g2_drugs),
                    "shared_kwargs": self._shared_kwargs,
                    "submodel_keys": sorted(self._models.keys()),
                },
                f,
            )
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "submodel_configs": _SUBMODEL_CONFIGS,
                "submodel_keys": sorted(self._models.keys()),
                "n_g1_drugs": len(self._g1_drugs),
                "n_g2_drugs": len(self._g2_drugs),
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "EmerGNNBaseline":
        inp = Path(path)
        with (inp / "routing.pkl").open("rb") as f:
            routing = pickle.load(f)
        shared_kwargs = routing["shared_kwargs"]
        inst = cls(**shared_kwargs)
        inst._g1_drugs = set(routing["g1_drugs"])
        inst._g2_drugs = set(routing["g2_drugs"])
        for split_key in routing["submodel_keys"]:
            sub_dir = inp / f"submodel_{split_key}"
            inst._models[split_key] = _ModeSpecificEmerGNN.load(sub_dir)
        return inst
