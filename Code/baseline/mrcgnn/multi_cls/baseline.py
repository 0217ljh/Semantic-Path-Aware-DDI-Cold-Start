"""MRCGNN multiclass training core — K-way DDI-event prediction (AAAI-2023).

Ported + adapted from upstream
``Paper/Reference/Original-Code/MRCGNN/codes for MRCGNN/{data_preprocess,train,
instantiation,parms_setting}.py``. File-independence (CLAUDE.md §Baseline): COPY+adapt,
NO import of ``Paper/Reference/Original-Code/`` or ``reproductions/``. Reuses this
package's model core (``baseline.mrcgnn.models.MRCGNN``), featurizer
(``baseline.mrcgnn.mol_features``), and TrimNet feature cache
(``baseline.mrcgnn._shared``).

Follows the pattern of ``baseline/ssi_ddi/multi_cls/baseline.py`` and
``baseline/emergnn/multi_cls/baseline.py`` (ddi vocab, macro-F1 best-ckpt, TrainProgress
logging, per-input-row predict). Registered ``@register("mrcgnn_mc")``.

CASE-A faithfulness (task matches paper's multiclass; CLAUDE.md §Baseline §4):
architecture / 3-loss / optimizer / batch / dims preserved EXACTLY:
  * 2x RGCNConv over the DDI-event graph, num_relations = K_global (upstream ``layer.py:88-89``).
  * dims 128->64->32, decoder MLP 448->256->128->K (upstream ``parms_setting.py:66-76``,
    ``layer.py:100-107``). ``self.attt`` RAW learnable scalars, no softmax (``layer.py:91-94``).
  * dropout 0.5 (``parms_setting.py:41``); Adam lr 1e-3 weight_decay 5e-4
    (``instantiation.py:21-22`` / ``parms_setting.py:38,44``); batch 256, epochs 100
    (``parms_setting.py:47,50``).
  * 3-loss: 1.0*CE + 0.05*BCE(viewA) + 0.1*BCE(viewB) (``train.py:65-69`` /
    ``parms_setting.py:56-63``). Full-graph forward once per step; batch indexes node
    embeddings by pair idx (upstream ``train.py:51-73`` runs RGCN once per forward).

CASE-B deviations (allowed, for the unified benchmark contract):
  * cold-start splits (S0/S1/S2); global-K = ``resources.meta["labels"]["n_labels"]``
    (do NOT shrink to train-observed K — the RGCN edge_type ids are global);
  * macro-F1 primary metric + best-ckpt (upstream selects by acc&f1, ``train.py:106``);
  * per-input-row ``predict_proba`` (uniform 1/K for drugs lacking a molecular graph).

COLD-START: node/feature universe = ALL leaf drugs (seen + unseen). Unseen test drugs
are rows with valid TrimNet molecular features but NO incident train DDI edge, so their
RGCN branch contributes little and the molecular features (as both RGCN input AND the
skip) carry the signal. Modest S2 macro-F1 is FAITHFUL, not a bug.

NORMALIZATION: the TrimNet ``.npy`` from ``_shared.ensure_trimnet_features`` is ALREADY
row-normalized (Step-3 builder decision). This core does NOT normalize again (upstream
normalizes once at load, ``data_preprocess.py:150``; we moved it into the builder).
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

from baseline.base import BaselineModel, register, write_manifest
from baseline.mrcgnn._shared import ensure_trimnet_features
from baseline.mrcgnn.mol_features import build_drug_graphs
from baseline.mrcgnn.models import MRCGNN

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol

_PAIR = ["drug_a_id", "drug_b_id"]

# upstream paper dims (parms_setting.py:66-76) — CASE-A, preserved verbatim.
_FEATURE_DIM = 128
_HIDDEN1 = 64
_HIDDEN2 = 32
# upstream 3-loss ratios (parms_setting.py:56-63).
_LOSS_RATIO1 = 1.0
_LOSS_RATIO2 = 0.05
_LOSS_RATIO3 = 0.1


@register("mrcgnn_mc")
class MRCGNNMulticlassBaseline(BaselineModel):
    """MRCGNN multi-class (K DDI-event types). RGCN + DGI contrastive + TrimNet skip."""

    VERSION = "1.0-mc"

    def __init__(
        self,
        *,
        n_classes: int = 65,
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
        **_ignored,
    ) -> None:
        self.n_classes = int(n_classes)
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
        # lazy state
        self._model: MRCGNN | None = None
        self._graphs: dict | None = None
        self._missing: list[str] = []
        self._drug_order: list[str] | None = None
        self._drug_to_idx: dict[str, int] | None = None
        # identity vocab: model emits GLOBAL-K columns indexed by global class id,
        # so the wrapper scatter (`out[:, int(t)] = p[:, i]`) is an identity map.
        self._ddi_type_to_idx: dict[str, int] | None = None
        self._idx_to_ddi_type: list[str] | None = None

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
        sorted union of all leaf drugs (deterministic). Unseen drugs are rows with
        valid features but no incident train edge (cold-start)."""
        smiles = self._drug_smiles_dict(train)
        self._graphs, self._missing = build_drug_graphs(smiles)
        # drug ordering = ALL leaf drugs (from train.drugs), sorted for determinism.
        self._drug_order = sorted(str(d) for d in train.drugs["drugbank_id"])
        self._drug_to_idx = {d: i for i, d in enumerate(self._drug_order)}
        if self._missing:
            print(f"[mrcgnn_mc] {len(self._missing)} drugs had unparseable SMILES; "
                  f"their pairs are filtered at train/inference.", file=sys.stderr)

    # ------------------------------------------------------------------
    # DDI-event graph + two corrupted views
    # ------------------------------------------------------------------
    def _build_ddi_graph(self, train: "PairDataset"):
        """Build data_o (true graph), the view-A feature-permuted node matrix, and the
        view-B pair-shuffled edge_type — faithful to upstream ``data_preprocess.py:156-195``.

        Returns (edge_index (2,2E), edge_type (2E,), x_perm_idx (n_drugs,) row-perm for
        view A, edge_type_shuf (2E,) view-B labels). Node features (x_o) + the model's
        mol skip come from the TrimNet matrix (set in fit)."""
        pos = train.splits.train
        if "ddi_type" not in pos.columns:
            raise ValueError(
                "MRCGNN multi-class requires `ddi_type` in train.splits.train; got "
                + str(list(pos.columns)))
        assert self._drug_to_idx is not None

        # keep only train positives whose BOTH drugs have a molecular graph AND are in
        # the drug universe (defensive; drug_order covers all leaf drugs).
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        edge_index: list[list[int]] = []
        edge_type: list[int] = []
        pair_labels: list[list[int]] = []   # upstream label_list11: [r, r] PAIRS
        a_ids = pos["drug_a_id"].astype(str).to_numpy()
        b_ids = pos["drug_b_id"].astype(str).to_numpy()
        rels = pos["ddi_type"].astype(int).to_numpy()
        for a, b, r in zip(a_ids, b_ids, rels):
            if a not in idx or b not in idx or a not in graphs or b not in graphs:
                continue
            ia, ib, ir = idx[a], idx[b], int(r)
            # bidirectional edges, edge_type = GLOBAL class id (upstream :166-178)
            edge_index.append([ia, ib]); edge_type.append(ir)
            edge_index.append([ib, ia]); edge_type.append(ir)
            pair_labels.append([ir, ir])   # upstream :179-182 — the [r,r] PAIR

        if not edge_index:
            raise ValueError("MRCGNN: zero usable train DDI edges (all pairs filtered).")

        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type_t = torch.tensor(edge_type, dtype=torch.long)

        # ---- view B: shuffle the [r,r] PAIRS, THEN flatten (upstream :191-194) ----
        # This preserves the bidirectional-pair invariant: both directions of an edge
        # get the SAME shuffled label (NOT an independent flat-2E shuffle). Codex nit #2.
        rng = np.random.default_rng(self.seed)
        pair_arr = np.asarray(pair_labels, dtype=np.int64)          # (E, 2), each row [r,r]
        perm = rng.permutation(pair_arr.shape[0])
        shuffled_pairs = pair_arr[perm]                              # rows still [r',r']
        edge_type_shuf = torch.tensor(shuffled_pairs.reshape(-1), dtype=torch.long)  # (2E,)

        # ---- view A: a random ROW-PERMUTATION of the node feature matrix (upstream
        # data_preprocess.py:156-158 features_a = features_o[permuted ids]) ----
        n_drugs = len(self._drug_order)
        x_perm_idx = torch.tensor(rng.permutation(n_drugs), dtype=torch.long)

        return edge_index_t, edge_type_t, x_perm_idx, edge_type_shuf

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
        # GLOBAL-K vocab: identity map (model emits K_global columns by global id).
        self._ddi_type_to_idx = {str(g): g for g in range(self.n_classes)}
        self._idx_to_ddi_type = [str(g) for g in range(self.n_classes)]

        self._build_graphs(train)
        assert self._drug_order is not None

        # ---- TrimNet features for ALL leaf drugs (detect-and-build; ALREADY row-norm) ----
        smiles = self._drug_smiles_dict(train)
        train_tri = self._train_triples(train)
        feats = ensure_trimnet_features(
            self._drug_order, smiles, train_tri,
            n_classes=self.n_classes, fold=self.fold, seed=self.seed,
            epochs=self.trimnet_epochs, device=self.device)
        if feats.shape != (len(self._drug_order), _FEATURE_DIM):
            raise ValueError(f"TrimNet features shape {feats.shape} != "
                             f"({len(self._drug_order)}, {_FEATURE_DIM})")
        mol_features = torch.as_tensor(feats, dtype=torch.float)

        # ---- DDI graph + two corrupted views (BUILT ONCE; reused across ALL epochs) ----
        # Faithful to upstream: the two corrupted views (data_s feature-permutation,
        # data_a relation-label shuffle) are built ONCE in preprocessing
        # (data_preprocess.py:156-195) and reused across all epochs (upstream iterates a
        # shuffle=False loader, train.py:51). We therefore build x_a and edge_type_shuf
        # here, before the epoch loop, and pass the SAME tensors into every step. Only the
        # TRAINING PAIR / batch ORDER is reshuffled per epoch (pos.sample below) — that is
        # standard SGD, NOT view re-corruption.
        edge_index, edge_type, x_perm_idx, edge_type_shuf = self._build_ddi_graph(train)

        dev = self.device
        x_o = mol_features.to(dev)                       # RGCN node input = TrimNet feats
        x_a = x_o[x_perm_idx.to(dev)]                    # view A: feature row-permutation (once)
        edge_index = edge_index.to(dev)
        edge_type = edge_type.to(dev)
        edge_type_shuf = edge_type_shuf.to(dev)          # view B: relation shuffle (once)

        # DGI contrastive target: [ones(n_drugs) | zeros(n_drugs)] (upstream :159).
        n_drugs = len(self._drug_order)
        dgi_target = torch.cat((torch.ones(n_drugs, 1), torch.zeros(n_drugs, 1)),
                               dim=1).to(dev)

        # cache the forward-graph tensors NOW (before the loop) so mid-training
        # validation via predict_proba can run.
        self._fwd = {"x_o": x_o, "edge_index": edge_index, "edge_type": edge_type,
                     "x_a": x_a, "edge_type_shuf": edge_type_shuf}

        # ---- model ----
        self._model = MRCGNN(
            feature=_FEATURE_DIM, n_classes=self.n_classes, n_drugs=n_drugs,
            mol_features=mol_features, hidden1=_HIDDEN1, hidden2=_HIDDEN2,
            dropout=self.dropout).to(dev)
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)
        ce = torch.nn.CrossEntropyLoss()
        bce = torch.nn.BCEWithLogitsLoss()

        # ---- train positives (pairs whose both drugs have a graph) ----
        pos = train.splits.train.copy()
        pos["_a"] = pos["drug_a_id"].astype(str)
        pos["_b"] = pos["drug_b_id"].astype(str)
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        keep = pos.apply(lambda r: r["_a"] in idx and r["_b"] in idx
                         and r["_a"] in graphs and r["_b"] in graphs, axis=1)
        pos = pos[keep].reset_index(drop=True)
        if len(pos) == 0:
            raise ValueError("MRCGNN: zero usable train positives after graph filtering.")
        pos["_ia"] = pos["_a"].map(idx).astype(int)
        pos["_ib"] = pos["_b"].map(idx).astype(int)
        pos["_y"] = pos["ddi_type"].astype(int)

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.task_eval import eval_multiclass
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.task_eval import eval_multiclass

        steps_per_epoch = max(1, (len(pos) + self.batch_size - 1) // self.batch_size)
        progress = TrainProgress(total_epochs=self.n_epochs,
                                 log_step_every=self.log_step_every,
                                 total_steps_per_epoch=steps_per_epoch,
                                 prefix="[mrcgnn_mc] ")
        best_val_macro_f1 = -1.0
        best_state: dict | None = None

        def _val_metrics() -> dict:
            if val is None:
                return {}
            vp = val.splits.val_s2[["drug_a_id", "drug_b_id", "ddi_type"]]
            if len(vp) == 0:
                return {}
            preds = self.predict_proba(vp[_PAIR])          # (n, K_global)
            labels = vp["ddi_type"].astype(int).to_numpy()
            m = eval_multiclass(preds, labels, self.n_classes)
            return {"val_top1": m["top1_acc"], "val_macro_f1": m["macro_f1"]}

        for epoch in range(self.n_epochs):
            pairs_df = pos.sample(frac=1, random_state=epoch).reset_index(drop=True)
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                ia = torch.tensor(batch["_ia"].to_numpy(), dtype=torch.long, device=dev)
                ib = torch.tensor(batch["_ib"].to_numpy(), dtype=torch.long, device=dev)
                y = torch.tensor(batch["_y"].to_numpy(), dtype=torch.long, device=dev)
                opt.zero_grad(set_to_none=True)
                # full-graph forward once per step (upstream train.py:61)
                logits, ret_a, ret_b, _ = self._model(
                    x_o, edge_index, edge_type, x_a, edge_type_shuf, ia, ib)
                loss1 = ce(logits, y)
                loss2 = bce(ret_a, dgi_target)
                loss3 = bce(ret_b, dgi_target)
                loss = (_LOSS_RATIO1 * loss1 + _LOSS_RATIO2 * loss2
                        + _LOSS_RATIO3 * loss3)
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
                    if metrics.get("val_macro_f1", -1) > best_val_macro_f1:
                        best_val_macro_f1 = metrics["val_macro_f1"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        # load_best_model_at_end (macro-F1 primary) — CLAUDE.md training规范.
        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[mrcgnn_mc] loaded best val_macro_f1={best_val_macro_f1:.4f} state",
                  flush=True)

    def _train_triples(self, train: "PairDataset") -> list[tuple[str, str, int]]:
        pos = train.splits.train
        return [(str(a), str(b), int(r)) for a, b, r in
                zip(pos["drug_a_id"], pos["drug_b_id"], pos["ddi_type"].astype(int))]

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
        if self._model is None or self._drug_to_idx is None:
            raise RuntimeError("MRCGNNMulticlassBaseline must be fitted before predict.")
        self._model.eval()
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        fwd = self._fwd
        # uniform 1/K for pairs whose drug lacks a molecular graph OR is out of the
        # drug universe (one row per input, like ssi_ddi/multi_cls).
        out = np.full((len(pairs), self.n_classes), 1.0 / self.n_classes, dtype=np.float32)
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
            probs = F.softmax(logits, dim=-1).cpu().numpy()
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
                         "ddi_type_to_idx": self._ddi_type_to_idx,
                         "idx_to_ddi_type": self._idx_to_ddi_type,
                         "n_classes": self.n_classes,
                         "fwd": {k: v.cpu() for k, v in self._fwd.items()}}, f)
        write_manifest(
            out, baseline_name=self.name,
            extra={"version": self.VERSION, "task": "multiclass",
                   "n_classes": self.n_classes,
                   "hyperparameters": {
                       "feature": _FEATURE_DIM, "hidden1": _HIDDEN1, "hidden2": _HIDDEN2,
                       "learning_rate": self.learning_rate,
                       "weight_decay": self.weight_decay, "dropout": self.dropout,
                       "batch_size": self.batch_size, "n_epochs": self.n_epochs,
                       "trimnet_epochs": self.trimnet_epochs, "seed": self.seed,
                       "fold": self.fold}})

    @classmethod
    def load(cls, path: "Path | str") -> "MRCGNNMulticlassBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hp = manifest.get("hyperparameters", {})
        n_classes = manifest.get("n_classes", 65)
        inst = cls(n_classes=n_classes,
                   learning_rate=hp.get("learning_rate", 1e-3),
                   weight_decay=hp.get("weight_decay", 5e-4),
                   dropout=hp.get("dropout", 0.5),
                   batch_size=hp.get("batch_size", 256),
                   n_epochs=hp.get("n_epochs", 100),
                   trimnet_epochs=hp.get("trimnet_epochs", 300),
                   seed=hp.get("seed", 42), fold=hp.get("fold", 0))
        with (p / "graphs.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._graphs = payload["graphs"]
        inst._missing = payload.get("missing", [])
        inst._drug_order = payload["drug_order"]
        inst._drug_to_idx = payload["drug_to_idx"]
        inst._ddi_type_to_idx = payload["ddi_type_to_idx"]
        inst._idx_to_ddi_type = payload["idx_to_ddi_type"]
        inst.n_classes = int(payload.get("n_classes", n_classes))
        mol = payload["fwd"]["x_o"]
        inst._model = MRCGNN(
            feature=_FEATURE_DIM, n_classes=inst.n_classes,
            n_drugs=len(inst._drug_order), mol_features=mol,
            hidden1=_HIDDEN1, hidden2=_HIDDEN2, dropout=inst.dropout).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        inst._fwd = {k: v.to(inst.device) for k, v in payload["fwd"].items()}
        return inst


__all__ = ["MRCGNNMulticlassBaseline"]
