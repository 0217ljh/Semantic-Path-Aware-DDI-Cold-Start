"""SSI-DDI multi-class variant — K-way DDI-type prediction.

Restores SSI-DDI's paper multi-relational design for the unified multiclass task:
``rel_total = n_classes`` RESCAL relation matrices, all-relation scoring
(:class:`baseline.ssi_ddi.multi_cls.model.SSI_DDI_MC`), positives-only training
with softmax cross-entropy over the train-observed DDI-type vocab, best checkpoint
by macro-F1 (the multiclass primary metric). Reuses the binary core's molecular
featurization (``_build_graphs`` / ``_make_pair_batch``) UNCHANGED via inheritance.

case-B adaptation (CLAUDE.md §Baseline §4): SSI-DDI's paper trains per-triple
sigmoid/margin over relations; the unified multiclass benchmark is which-of-K +
macro-F1, so the head/loss adapt to all-K softmax CE while the RESCAL
multi-relational scoring core is preserved.
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
import torch
from torch import optim
from torch.nn import functional as F

from baseline.base import register, write_manifest
from baseline.ssi_ddi.baseline import SSIDDIBaseline
from baseline.ssi_ddi.multi_cls.model import SSI_DDI_MC

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


@register("ssi_ddi_mc")
class SSIDDIMulticlassBaseline(SSIDDIBaseline):
    """SSI-DDI multi-class (K DDI types). Inherits binary featurization; K-way head."""

    VERSION = "1.0-mc"

    def __init__(self, *, n_classes: int = 86, **kw) -> None:
        super().__init__(**kw)
        self.n_classes = int(n_classes)
        self._ddi_type_to_idx: dict[str, int] | None = None
        self._idx_to_ddi_type: list[str] | None = None

    def _build_model(self) -> SSI_DDI_MC:
        return SSI_DDI_MC(
            in_features=self.in_features,
            hidd_dim=self.hidd_dim,
            kge_dim=self.kge_dim,
            rel_total=self.n_classes,
            heads_out_feat_params=list(self.heads_out_feat_params),
            blocks_params=list(self.blocks_params),
        ).to(self.device)

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        if "ddi_type" not in train.splits.train.columns:
            raise ValueError(
                "Multi-class SSI-DDI requires `ddi_type` in train.splits.train; got "
                + str(list(train.splits.train.columns))
            )
        types = sorted(train.splits.train["ddi_type"].astype(str).unique())
        self._ddi_type_to_idx = {t: i for i, t in enumerate(types)}
        self._idx_to_ddi_type = list(types)
        if len(types) != self.n_classes:
            print(
                f"[ssi_ddi_mc] observed {len(types)} ddi_types in train, "
                f"n_classes={self.n_classes} -> adjusting to {len(types)}.",
                file=sys.stderr,
            )
            self.n_classes = len(types)

        self._build_graphs(train)
        self._model = self._build_model()
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)

        pos = train.splits.train.copy()
        pos["_y"] = pos["ddi_type"].astype(str).map(self._ddi_type_to_idx)

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.task_eval import eval_multiclass
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.task_eval import eval_multiclass

        steps_per_epoch = max(1, (len(pos) + self.batch_size - 1) // self.batch_size)
        progress = TrainProgress(total_epochs=self.n_epochs, log_step_every=50,
                                 total_steps_per_epoch=steps_per_epoch,
                                 prefix="[ssi_ddi_mc] ")
        best_val_macro_f1 = -1.0
        best_state: dict | None = None

        def _val_metrics() -> dict:
            if val is None:
                return {}
            vp = val.splits.val_s2[["drug_a_id", "drug_b_id", "ddi_type"]]
            if len(vp) == 0:
                return {}
            preds = self.predict_proba(vp[["drug_a_id", "drug_b_id"]])
            mask = vp["ddi_type"].astype(str).isin(self._ddi_type_to_idx)
            preds = preds[mask.values]
            labels = np.array([self._ddi_type_to_idx[str(t)] for t in vp.loc[mask, "ddi_type"]])
            if len(labels) == 0:
                return {}
            m = eval_multiclass(preds, labels, self.n_classes)
            return {"val_top1": m["top1_acc"], "val_macro_f1": m["macro_f1"]}

        for epoch in range(self.n_epochs):
            pairs_df = pos.sample(frac=1, random_state=epoch).reset_index(drop=True)
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start : start + self.batch_size]
                h_batch, t_batch, _rels, mask = self._make_pair_batch(batch)
                if h_batch is None:
                    continue
                h_batch = h_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                y = torch.tensor(batch.loc[mask, "_y"].to_numpy(), dtype=torch.long,
                                 device=self.device)
                opt.zero_grad(set_to_none=True)
                logits = self._model.forward_all((h_batch, t_batch))   # (B, K)
                loss = F.cross_entropy(logits, y, reduction="sum")
                loss.backward()
                opt.step()
                progress.step(loss.item() / max(int(mask.sum()), 1))

            extra: dict = {}
            if val is not None:
                self._model.eval()
                metrics = _val_metrics()
                self._model.train()
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_macro_f1", -1) > best_val_macro_f1:
                        best_val_macro_f1 = metrics["val_macro_f1"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[ssi_ddi_mc] loaded best val_macro_f1={best_val_macro_f1:.4f} state",
                  flush=True)

    @torch.no_grad()
    def predict_proba(
        self,
        pairs: "pd.DataFrame",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None or self._graphs is None:
            raise RuntimeError("SSIDDIMulticlassBaseline must be fitted before predict.")
        self._model.eval()
        # Uniform 1/K for pairs whose drug has no graph (featurizer cold-start),
        # preserving one row per input.
        out = np.full((len(pairs), self.n_classes), 1.0 / self.n_classes, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            h_batch, t_batch, _rels, mask = self._make_pair_batch(batch)
            if h_batch is None:
                continue
            h_batch = h_batch.to(self.device)
            t_batch = t_batch.to(self.device)
            logits = self._model.forward_all((h_batch, t_batch))       # (kept, K)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            block = np.full((len(batch), self.n_classes), 1.0 / self.n_classes, dtype=np.float32)
            block[mask] = probs
            out[start : start + len(batch)] = block
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump({"graphs": self._graphs, "missing": self._missing,
                         "ddi_type_to_idx": self._ddi_type_to_idx,
                         "idx_to_ddi_type": self._idx_to_ddi_type,
                         "n_classes": self.n_classes}, f)
        write_manifest(
            out, baseline_name=self.name,
            extra={"version": self.VERSION, "task": "multiclass", "n_classes": self.n_classes,
                   "hyperparameters": {
                       "in_features": self.in_features, "hidd_dim": self.hidd_dim,
                       "kge_dim": self.kge_dim,
                       "heads_out_feat_params": list(self.heads_out_feat_params),
                       "blocks_params": list(self.blocks_params),
                       "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                       "batch_size": self.batch_size, "n_epochs": self.n_epochs}})

    @classmethod
    def load(cls, path: "Path | str") -> "SSIDDIMulticlassBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        if "heads_out_feat_params" in hparams:
            hparams["heads_out_feat_params"] = tuple(hparams["heads_out_feat_params"])
        if "blocks_params" in hparams:
            hparams["blocks_params"] = tuple(hparams["blocks_params"])
        n_classes = manifest.get("n_classes", 86)
        inst = cls(n_classes=n_classes, **hparams)
        with (p / "graphs.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._graphs = payload["graphs"]
        inst._missing = payload.get("missing", [])
        inst._ddi_type_to_idx = payload["ddi_type_to_idx"]
        inst._idx_to_ddi_type = payload["idx_to_ddi_type"]
        inst.n_classes = int(payload.get("n_classes", n_classes))
        inst._model = inst._build_model()
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        return inst


__all__ = ["SSIDDIMulticlassBaseline"]
