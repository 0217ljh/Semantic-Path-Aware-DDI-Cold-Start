"""MRCGNN BINARY training core — case-B added task (paper is multiclass-only).

MRCGNN (AAAI-2023) is a multiclass DDI-event predictor. This module adds the BINARY
DDI task (does an interaction exist?) for the unified cold-start benchmark, reusing this
package's model core (``baseline.mrcgnn.models.MRCGNN``), featurizer
(``baseline.mrcgnn.mol_features``), and the TrimNet feature builder
(``baseline.mrcgnn._data.necessary.build_trimnet_features`` via ``_shared``).

File-independence (CLAUDE.md §Baseline §6): COPY+adapt of ``multi_cls/baseline.py``; NO
import of ``Paper/Reference/Original-Code/`` or ``reproductions/``. Structurally mirrors
``baseline/ssi_ddi/baseline.py`` (rel_total=1 collapse + BCE + per-epoch negatives +
best-ckpt-by-val-AUROC) for the binary contract.

CASE-B binary deviations (CLAUDE.md §Baseline §4 case-B — allowed, this task is NOT the
paper's task):
  * output head K-way -> 1 logit: ``MRCGNN(n_classes=1)`` collapses BOTH the RGCN
    ``num_relations`` AND the MLP head to 1 (models.py:85-86, 104-105). The single
    "interaction" relation matches SSI-DDI's ``rel_total=1`` collapse (ssi_ddi/baseline.py:179);
  * loss CE -> BCE over train POSITIVES + per-epoch NEGATIVES (ssi_ddi/baseline.py:196-225);
  * primary metric macro-F1 -> AUPRC (run_baseline_unified.py:54); best-ckpt selected on
    val AUROC (ssi_ddi/baseline.py:190,229-236), reporting both AUROC + AUPRC each eval.

THE key case-B decision — VIEW B is DROPPED (coordinator+codex-approved, 2026-07-01).
  MRCGNN trains TWO DGI-style contrastive views (models.py:145-160):
    * View A = feature row-permutation on the SAME edges/edge_type (models.py:145-148);
    * View B = relation-label shuffle on the SAME adjacency (models.py:150-154).
  With ``num_relations=1`` the binary DDI graph's ``edge_type`` is ALL ZEROS, so a
  permutation of View B's relation labels is the IDENTITY — View B's corrupted graph is
  the true graph differing ONLY by dropout noise, and ``loss3 = BCE(ret_b, dgi_target)``
  becomes a mislabeled / contentless term (NOT a faithful relation corruption). We
  therefore KEEP View A + ``0.05 * BCE_A`` and DROP View B + ``0.1 * BCE_B``. View A is
  relation-count-agnostic (pure feature-corruption DGI contrast), so the paper's
  contrastive training structure survives. Binary loss = ``1.0 * BCE_cls + 0.05 * BCE_A``
  (2-loss; ratios from multi_cls/baseline.py:73-74, dropping ``_LOSS_RATIO3``). View B is
  NOT computed at all (its two extra RGCN forwards are skipped).

TrimNet FEATURES sourced from the SIBLING MULTICLASS leaf (coordinator+codex decision Q3).
  The TrimNet builder is a SUPERVISED multiclass DDI-EVENT classifier
  (build_trimnet_features.py:196-215, CrossEntropyLoss over ``ddi_type``); a
  binary-supervised TrimNet would be a DIFFERENT featurizer and break the cache contract
  (the cache key bakes ``n_classes`` + train triples, _shared.py:47-65). So we DECOUPLE:
    * the binary MODEL is ``MRCGNN(n_classes=1)`` (single-relation RGCN, 1-logit head);
    * the TrimNet FEATURES are built with the SAME event-supervised recipe as the
      multiclass baseline, from the SIBLING multiclass leaf's TRAIN positives
      (same dataset / regime / split_code / fold), using ``y_cls -> ddi_type`` and
      ``n_classes = K_multiclass``.
  Consequence: binary and multiclass SHARE the same TrimNet cache (identical key) for a
  given dataset/split/fold — efficient + consistent. Cold-start safe: only the sibling
  fold's TRAIN rows feed TrimNet (never val/test). If the sibling multiclass leaf is
  absent for a (dataset, split, fold), we RAISE (no silent fallback to binary-supervised
  TrimNet).

COLD-START: node/feature universe = ALL leaf drugs (seen + unseen). Unseen test drugs are
rows with valid TrimNet features but NO incident train DDI edge, so their RGCN branch
contributes little and the molecular features (RGCN input + skip) carry the signal.

NORMALIZATION: the TrimNet ``.npy`` from ``_shared.ensure_trimnet_features`` is ALREADY
row-normalized (Step-3 builder decision, build_trimnet_features.py:22-27); this core does
NOT re-normalize.
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

from baseline.base import BaselineModel, register, write_manifest
from baseline.mrcgnn._shared import ensure_trimnet_features
from baseline.mrcgnn.mol_features import build_drug_graphs
from baseline.mrcgnn.models import MRCGNN

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol

_PAIR = ["drug_a_id", "drug_b_id"]

# upstream paper dims (parms_setting.py:66-76) — task-invariant, preserved verbatim.
_FEATURE_DIM = 128
_HIDDEN1 = 64
_HIDDEN2 = 32
# binary loss ratios: keep View-A weight (multi_cls/baseline.py:73-74); View B dropped.
_LOSS_RATIO1 = 1.0     # BCE classification
_LOSS_RATIO2 = 0.05    # BCE(view A) contrastive


@register("mrcgnn_bin")
class MRCGNNBinaryBaseline(BaselineModel):
    """MRCGNN binary (interaction / no-interaction). Single-relation RGCN + View-A DGI
    contrastive + TrimNet skip. BCE over pos + per-epoch negatives; best-ckpt by val AUROC.
    """

    VERSION = "1.0-bin"

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        dropout: float = 0.5,
        batch_size: int = 256,
        n_epochs: int = 100,
        trimnet_epochs: int = 300,
        seed: int = 42,
        fold: object = 0,
        device: str = "auto",
        log_step_every: int = 50,
        run_dir: "str | Path | None" = None,
        #: sibling multiclass leaf inputs for the TrimNet feature builder (decision Q3).
        #: The binary wrapper resolves these from the sibling MC leaf and injects them.
        trimnet_train_tri: "list[tuple[str, str, int]] | None" = None,
        trimnet_n_classes: "int | None" = None,
        trimnet_source: str = "",
        **_ignored,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.trimnet_epochs = int(trimnet_epochs)
        self.seed = int(seed)
        self.fold = fold
        self.device = self._resolve_device(device)
        self.log_step_every = int(log_step_every)
        self.run_dir = run_dir
        # sibling-MC TrimNet supervision (decision Q3)
        self._trimnet_train_tri = trimnet_train_tri
        self._trimnet_n_classes = trimnet_n_classes
        self._trimnet_source = trimnet_source
        # lazy state
        self._model: MRCGNN | None = None
        self._graphs: dict | None = None
        self._missing: list[str] = []
        self._drug_order: list[str] | None = None
        self._drug_to_idx: dict[str, int] | None = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    # ------------------------------------------------------------------
    # Featurization
    # ------------------------------------------------------------------
    def _drug_smiles_dict(self, train: "PairDataset") -> dict[str, str]:
        if train.drugs is None or "smiles" not in train.drugs.columns:
            raise ValueError("MRCGNN requires `PairDataset.drugs` with a `smiles` column.")
        return {
            str(row["drugbank_id"]): ("" if pd.isna(row["smiles"]) else str(row["smiles"]))
            for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
        }

    def _build_graphs(self, train: "PairDataset") -> None:
        """Molecular graphs for ALL leaf drugs (seen + unseen). Drug index universe =
        sorted union of all leaf drugs (deterministic). Same as multi_cls/baseline.py:141-152."""
        smiles = self._drug_smiles_dict(train)
        self._graphs, self._missing = build_drug_graphs(smiles)
        self._drug_order = sorted(str(d) for d in train.drugs["drugbank_id"])
        self._drug_to_idx = {d: i for i, d in enumerate(self._drug_order)}
        if self._missing:
            print(f"[mrcgnn_bin] {len(self._missing)} drugs had unparseable SMILES; "
                  f"their pairs are filtered at train/inference.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Binary DDI graph + ONE corrupted view (View A only)
    # ------------------------------------------------------------------
    def _build_ddi_graph(self, train: "PairDataset"):
        """Build the true binary DDI graph (train POSITIVES as edges, single relation) and
        the View-A feature-permutation index. View B (relation-label shuffle) is DROPPED —
        with num_relations=1 it is degenerate (see module docstring; models.py:150-154).

        Returns (edge_index (2,2E), edge_type (2E,) ALL ZEROS, x_perm_idx (n_drugs,)).
        Adapted from multi_cls/baseline.py:157-210, minus the ddi_type edge labels and the
        View-B pair-shuffle."""
        pos = train.splits.train
        assert self._drug_to_idx is not None
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        edge_index: list[list[int]] = []
        a_ids = pos["drug_a_id"].astype(str).to_numpy()
        b_ids = pos["drug_b_id"].astype(str).to_numpy()
        for a, b in zip(a_ids, b_ids):
            if a not in idx or b not in idx or a not in graphs or b not in graphs:
                continue
            ia, ib = idx[a], idx[b]
            # bidirectional edges; edge_type is the single "interaction" relation (0)
            # (collapse of multi_cls/baseline.py:186-187, which used the global ddi_type id).
            edge_index.append([ia, ib])
            edge_index.append([ib, ia])

        if not edge_index:
            raise ValueError("MRCGNN binary: zero usable train DDI edges (all pairs filtered).")

        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # single-relation: all edge types = 0 (SSI-DDI rel_total=1 analogue, ssi_ddi/baseline.py:157,179).
        edge_type_t = torch.zeros(edge_index_t.shape[1], dtype=torch.long)

        # ---- View A: a random ROW-PERMUTATION of the node feature matrix
        # (multi_cls/baseline.py:205-208). Relation-count-agnostic, kept unchanged. ----
        rng = np.random.default_rng(self.seed)
        n_drugs = len(self._drug_order)
        x_perm_idx = torch.tensor(rng.permutation(n_drugs), dtype=torch.long)

        return edge_index_t, edge_type_t, x_perm_idx

    def _keep_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows whose BOTH drugs have a molecular graph AND are in the drug universe,
        with node indices attached (multi_cls/baseline.py:284-290)."""
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        out = df.copy()
        out["_a"] = out["drug_a_id"].astype(str)
        out["_b"] = out["drug_b_id"].astype(str)
        keep = out.apply(lambda r: r["_a"] in idx and r["_b"] in idx
                         and r["_a"] in graphs and r["_b"] in graphs, axis=1)
        out = out[keep].reset_index(drop=True)
        if len(out):
            out["_ia"] = out["_a"].map(idx).astype(int)
            out["_ib"] = out["_b"].map(idx).astype(int)
        return out

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        self._build_graphs(train)
        assert self._drug_order is not None

        # ---- TrimNet features from the SIBLING MULTICLASS leaf (decision Q3) ----
        # The event-supervised builder needs the sibling MC train triples + its K. These
        # are injected by the unified wrapper; refuse to run without them (no fallback to
        # a binary-supervised TrimNet).
        if self._trimnet_train_tri is None or self._trimnet_n_classes is None:
            raise ValueError(
                "MRCGNN binary requires sibling-multiclass TrimNet supervision "
                "(trimnet_train_tri + trimnet_n_classes). The unified wrapper resolves "
                "these from the sibling multiclass leaf (decision Q3). Absent -> the "
                "sibling MC leaf does not exist for this (dataset, split, fold); MRCGNN "
                "binary is unsupported there.")
        smiles = self._drug_smiles_dict(train)
        print(f"[mrcgnn_bin] TrimNet features from SIBLING MULTICLASS leaf: "
              f"{self._trimnet_source} (K={self._trimnet_n_classes}, "
              f"{len(self._trimnet_train_tri)} train triples)", file=sys.stderr, flush=True)
        feats = ensure_trimnet_features(
            self._drug_order, smiles, self._trimnet_train_tri,
            n_classes=int(self._trimnet_n_classes), fold=self.fold, seed=self.seed,
            epochs=self.trimnet_epochs, device=self.device)
        if feats.shape != (len(self._drug_order), _FEATURE_DIM):
            raise ValueError(f"TrimNet features shape {feats.shape} != "
                             f"({len(self._drug_order)}, {_FEATURE_DIM})")
        mol_features = torch.as_tensor(feats, dtype=torch.float)

        # ---- binary DDI graph + View-A permutation (BUILT ONCE; reused across epochs) ----
        edge_index, edge_type, x_perm_idx = self._build_ddi_graph(train)

        dev = self.device
        x_o = mol_features.to(dev)                       # RGCN node input = TrimNet feats
        x_a = x_o[x_perm_idx.to(dev)]                    # View A: feature row-permutation (once)
        edge_index = edge_index.to(dev)
        edge_type = edge_type.to(dev)                    # all zeros (single relation)

        # DGI contrastive target: [ones(n_drugs) | zeros(n_drugs)] (multi_cls/baseline.py:258-261).
        n_drugs = len(self._drug_order)
        dgi_target = torch.cat((torch.ones(n_drugs, 1), torch.zeros(n_drugs, 1)),
                               dim=1).to(dev)

        # cache forward-graph tensors for mid-training validation via predict_proba.
        # ``edge_type_shuf`` is FED THE SAME ``edge_type`` (all zeros) purely to satisfy the
        # model's forward signature (models.py:116-118) — View-B outputs are NOT used in the
        # loss (View B dropped), so this only wastes two RGCN forwards. To avoid that waste
        # we do NOT call the full forward in the training step; see the step loop below.
        self._fwd = {"x_o": x_o, "edge_index": edge_index, "edge_type": edge_type,
                     "x_a": x_a, "edge_type_shuf": edge_type}

        # ---- model: n_classes=1 collapses RGCN num_relations AND the head to 1 ----
        self._model = MRCGNN(
            feature=_FEATURE_DIM, n_classes=1, n_drugs=n_drugs,
            mol_features=mol_features, hidden1=_HIDDEN1, hidden2=_HIDDEN2,
            dropout=self.dropout).to(dev)
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)
        bce = torch.nn.BCEWithLogitsLoss()

        pos = self._keep_pairs(train.splits.train)
        if len(pos) == 0:
            raise ValueError("MRCGNN binary: zero usable train positives after graph filtering.")

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.task_eval import eval_binary
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.task_eval import eval_binary

        # steps-per-epoch is approximate (per-epoch negatives change the count slightly).
        steps_per_epoch = max(1, (2 * len(pos) + self.batch_size - 1) // self.batch_size)
        progress = TrainProgress(total_epochs=self.n_epochs,
                                 log_step_every=self.log_step_every,
                                 total_steps_per_epoch=steps_per_epoch,
                                 prefix="[mrcgnn_bin] ")
        best_val_auroc = -1.0
        best_state: dict | None = None

        def _val_metrics() -> dict:
            if val is None:
                return {}
            vpos = val.splits.val_s2[_PAIR]
            vneg = val.get_negatives("val_s2")[_PAIR]
            if len(vpos) == 0 or len(vneg) == 0:
                return {}
            y_score = np.concatenate([self.predict_proba(vpos), self.predict_proba(vneg)])
            y_true = np.concatenate([np.ones(len(vpos)), np.zeros(len(vneg))])
            m = eval_binary(y_score, y_true)   # positional (preds, labels); task_eval.py:26
            from sklearn.metrics import average_precision_score
            return {"val_auroc": m["auc"],
                    "val_auprc": float(average_precision_score(y_true, y_score))}

        for epoch in range(self.n_epochs):
            # ---- per-epoch fresh negatives (cold-start safe; ssi_ddi/baseline.py:196) ----
            neg = train.get_train_negatives(epoch, regenerate=True)
            neg = self._keep_pairs(neg[_PAIR]) if len(neg) else neg.iloc[0:0]
            pos_b = pos[["_ia", "_ib"]].assign(_y=1.0)
            neg_b = (neg[["_ia", "_ib"]].assign(_y=0.0) if len(neg)
                     else pd.DataFrame(columns=["_ia", "_ib", "_y"]))
            pairs_df = (pd.concat([pos_b, neg_b], ignore_index=True)
                        .sample(frac=1, random_state=epoch).reset_index(drop=True))

            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                ia = torch.tensor(batch["_ia"].to_numpy(), dtype=torch.long, device=dev)
                ib = torch.tensor(batch["_ib"].to_numpy(), dtype=torch.long, device=dev)
                y = torch.tensor(batch["_y"].to_numpy(), dtype=torch.float32, device=dev)
                opt.zero_grad(set_to_none=True)
                # full-graph forward once per step (multi_cls/baseline.py:330-332). The model
                # returns (log, ret_os (view A), ret_os_a (view B), x2). We use log + ret_os;
                # ret_os_a is IGNORED (View B dropped). The extra View-B forward inside the
                # model is unavoidable via this signature but its output is unused.
                logits, ret_a, _ret_b_unused, _ = self._model(
                    x_o, edge_index, edge_type, x_a, edge_type, ia, ib)
                loss1 = bce(logits.squeeze(-1), y)          # BCE classification
                loss2 = bce(ret_a, dgi_target)              # 0.05 * BCE(view A)
                loss = _LOSS_RATIO1 * loss1 + _LOSS_RATIO2 * loss2
                loss.backward()
                opt.step()
                progress.step(loss.item())

            extra: dict = {}
            if val is not None:
                self._model.eval()
                metrics = _val_metrics()
                self._model.train()
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_auroc", -1) > best_val_auroc:
                        best_val_auroc = metrics["val_auroc"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        # load_best_model_at_end (val AUROC primary; decision Q2).
        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[mrcgnn_bin] loaded best val_auroc={best_val_auroc:.4f} state",
                  flush=True)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self,
        pairs: "pd.DataFrame",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        """Return (n,) P(interaction). 0.5 for pairs whose drug lacks a molecular graph or
        is out of the drug universe (one row per input, like ssi_ddi/baseline.py:250-279)."""
        if self._model is None or self._drug_to_idx is None:
            raise RuntimeError("MRCGNNBinaryBaseline must be fitted before predict.")
        self._model.eval()
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        fwd = self._fwd
        out = np.full(len(pairs), 0.5, dtype=np.float32)
        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()

        keep_rows: list[int] = []
        ia_list: list[int] = []
        ib_list: list[int] = []
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            if a in idx and b in idx and a in graphs and b in graphs:
                keep_rows.append(i)
                ia_list.append(idx[a])
                ib_list.append(idx[b])
        if not keep_rows:
            return out

        dev = self.device
        for s in range(0, len(keep_rows), self.batch_size):
            rows = keep_rows[s:s + self.batch_size]
            ia = torch.tensor(ia_list[s:s + self.batch_size], dtype=torch.long, device=dev)
            ib = torch.tensor(ib_list[s:s + self.batch_size], dtype=torch.long, device=dev)
            logits, _, _, _ = self._model(
                fwd["x_o"], fwd["edge_index"], fwd["edge_type"],
                fwd["x_a"], fwd["edge_type_shuf"], ia, ib)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            for j, row in enumerate(rows):
                out[row] = probs[j]
        return out

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path: "Path | str") -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump({"graphs": self._graphs, "missing": self._missing,
                         "drug_order": self._drug_order,
                         "drug_to_idx": self._drug_to_idx,
                         "fwd": {k: v.cpu() for k, v in self._fwd.items()}}, f)
        write_manifest(
            out, baseline_name=self.name,
            extra={"version": self.VERSION, "task": "binary",
                   "trimnet_source": self._trimnet_source,
                   "trimnet_n_classes": self._trimnet_n_classes,
                   "hyperparameters": {
                       "feature": _FEATURE_DIM, "hidden1": _HIDDEN1, "hidden2": _HIDDEN2,
                       "learning_rate": self.learning_rate,
                       "weight_decay": self.weight_decay, "dropout": self.dropout,
                       "batch_size": self.batch_size, "n_epochs": self.n_epochs,
                       "trimnet_epochs": self.trimnet_epochs, "seed": self.seed,
                       "fold": self.fold}})

    @classmethod
    def load(cls, path: "Path | str") -> "MRCGNNBinaryBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hp = manifest.get("hyperparameters", {})
        inst = cls(learning_rate=hp.get("learning_rate", 1e-3),
                   weight_decay=hp.get("weight_decay", 5e-4),
                   dropout=hp.get("dropout", 0.5),
                   batch_size=hp.get("batch_size", 256),
                   n_epochs=hp.get("n_epochs", 100),
                   trimnet_epochs=hp.get("trimnet_epochs", 300),
                   seed=hp.get("seed", 42), fold=hp.get("fold", 0),
                   trimnet_source=manifest.get("trimnet_source", ""),
                   trimnet_n_classes=manifest.get("trimnet_n_classes"))
        with (p / "graphs.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._graphs = payload["graphs"]
        inst._missing = payload.get("missing", [])
        inst._drug_order = payload["drug_order"]
        inst._drug_to_idx = payload["drug_to_idx"]
        mol = payload["fwd"]["x_o"]
        inst._model = MRCGNN(
            feature=_FEATURE_DIM, n_classes=1,
            n_drugs=len(inst._drug_order), mol_features=mol,
            hidden1=_HIDDEN1, hidden2=_HIDDEN2, dropout=inst.dropout).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        inst._fwd = {k: v.to(inst.device) for k, v in payload["fwd"].items()}
        return inst


__all__ = ["MRCGNNBinaryBaseline"]
