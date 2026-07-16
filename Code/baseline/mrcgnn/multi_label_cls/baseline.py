"""MRCGNN MULTILABEL baseline — TWOSIDES 200-side-effect prediction (Case-B).

NOTE — paper-vs-port task formulation:
  * Original paper (MRCGNN, AAAI-2023) does **multiclass** K-way DDI-EVENT
    prediction (softmax + CrossEntropy) over a multi-relational RGCN DDI graph
    with DGI-style contrastive learning + a TrimNet molecular skip.
  * THIS multilabel variant is a **Case-B adaptation** (CLAUDE.md §Baseline §4
    case B): the paper never did multilabel, but the paper's ALGORITHM CORE
    (2x RGCNConv over the DDI graph, two DGI contrastive views, TrimNet skip,
    the MLP pair head) is reused UNCHANGED. Only the TASK SURFACE — head width
    (200-logit), activation (sigmoid not softmax), classification loss (masked
    BCE not CE), targets (paired pos/neg 200-multihot not a single ddi_type
    idx), and the val metric (macro-AUPRC not macro-F1) — is swapped.

**AGREED DESIGN (coordinator + codex, 2026-07-01) — implemented exactly here:**

  * Q1 edge scheme = MULTI-RELATIONAL. ``num_relations = n_labels = 200``. The
    train DDI graph is built by EXPLODING each positive pair into ONE
    bidirectional edge-pair PER active side-effect label ``r`` in that pair's
    active columns -> edges ``(a,b,r)`` and ``(b,a,r)``. Parallel edges are OK.
    Reuses :class:`baseline.mrcgnn.models.MRCGNN` with ``n_classes=200`` UNCHANGED
    (num_relations=200 via models.py:85-86; head width 200 via models.py:104-105).

  * Q2 head + loss = 200-logit MLP head trained with MASKED BCEWithLogits on the
    paired pos/neg bundle. For each pos pair the target is 1 on its ACTIVE labels;
    the paired endpoint-corrupted negative gets target 0 on the SAME columns;
    inactive columns are ignored. Mirrors the EmerGNN TWOSIDES core
    (emergnn/multi_label_cls/_core_twoside.py:441-457) + the HDN-DDI ml core
    (hdn_ddi/multi_label_cls/baseline.py:213-229). Sigmoid only at inference.

  * Q3 contrastive views + 3-loss = KEEP BOTH DGI views + the 3-loss
    (``1.0 * BCE_cls + 0.05 * BCE_A + 0.1 * BCE_B``). CE_cls is replaced by the
    masked-BCE_cls above; the two contrastive terms are the SAME BCE(disc,
    dgi_target) as multiclass (multi_cls/baseline.py:334-337). Under 200 relations
    View B (relation-label shuffle) is MEANINGFUL — the exploded ``[r,r]``
    bidirectional edge-pairs are shuffled exactly like the multiclass core
    (multi_cls/baseline.py:196-208), just over the label INCIDENCES.

  * Q4 TrimNet = the ``_shared.ensure_trimnet_features`` builder is UNCHANGED.
    It is fed the EXPLODED train triples ``(a,b,r)`` from the multilabel
    positives + ``n_classes=200`` (CE over 200 label-relations, one relation per
    training example, per build_trimnet_features.py:196-215). The cache key
    naturally distinguishes this build (it bakes the triple set + n_classes,
    _shared.py:47-65).

  * Q5 cold-start = graph + views + TrimNet from TRAIN positives ONLY; the
    node/feature universe = ALL leaf drugs (from ``ds.drugs`` = resources.drugs);
    unseen drugs get TrimNet features but no incident edges. No val/test leakage.

  * Metric / best-ckpt = primary macro-AUPRC (multilabel), best-ckpt by val
    macro-AUPRC (like emergnn/hdn ml).

Independence (CLAUDE.md §文件级独立性): this module imports ONLY
``baseline.mrcgnn.*`` (its own shared model/featurizer/TrimNet cache) +
``data_utils`` — no ``reproductions/`` and no other baseline package (the
EmerGNN / HDN-DDI ml cores were READ as reference patterns, not imported).

The MRCGNN shared code (models.py / layers.py / _shared.py /
build_trimnet_features.py) and the multiclass / binary cores are UNCHANGED.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import optim

from baseline.base import register, write_manifest
from baseline.mrcgnn._shared import ensure_trimnet_features
from baseline.mrcgnn.models import MRCGNN
from baseline.mrcgnn.multi_cls.baseline import MRCGNNMulticlassBaseline

if TYPE_CHECKING:
    from data_utils.leaf_adapter import _LeafDataset
    from data_utils.protocols import KnowledgeGraphProtocol

_PAIR = ["drug_a_id", "drug_b_id"]

# upstream paper dims (parms_setting.py:66-76) — task-invariant, preserved verbatim.
_FEATURE_DIM = 128
_HIDDEN1 = 64
_HIDDEN2 = 32
# upstream 3-loss ratios (parms_setting.py:56-63) — Q3 keeps ALL THREE.
_LOSS_RATIO1 = 1.0    # (masked) BCE classification (replaces multiclass CE)
_LOSS_RATIO2 = 0.05   # BCE(view A) contrastive
_LOSS_RATIO3 = 0.1    # BCE(view B) contrastive


@register("mrcgnn_ml")
class MRCGNNMultilabelBaseline(MRCGNNMulticlassBaseline):
    """MRCGNN multilabel (fixed ``n_labels`` side-effect labels), Case-B.

    All MRCGNN paper contributions (2x RGCNConv multi-relational DDI graph, two
    DGI contrastive views, TrimNet molecular skip, MLP pair head, Adam
    lr=1e-3/wd=5e-4, dropout 0.5, batch 256, 3-loss) are inherited from
    :class:`MRCGNNMulticlassBaseline` / :class:`baseline.mrcgnn.models.MRCGNN`.
    Only the task surface (fixed 200-label head, sigmoid + masked BCE_cls loss,
    paired pos/neg bundle, label-explosion graph + TrimNet triples, macro-AUPRC
    selection) is overridden here.
    """

    VERSION = "1.0-ml"  # Case-B multilabel adaptation of the MRCGNN core

    def __init__(self, *, n_labels: int = 200, **kw) -> None:
        # Route the fixed label count through the parent's ``n_classes`` slot so
        # the inherited RGCN (num_relations) AND the MLP head are both sized to
        # ``n_labels`` (models.py:85-86, 104-105). NO per-fold re-vocab.
        super().__init__(n_classes=int(n_labels), **kw)
        self.n_labels = int(n_labels)

    # ------------------------------------------------------------------
    # Label-explosion DDI graph + two corrupted views (multilabel variant)
    # ------------------------------------------------------------------
    def _build_ddi_graph_ml(self, pos_ht: np.ndarray, pos_y: np.ndarray):
        """Build the exploded multi-relational DDI graph + the two DGI views from
        the multilabel TRAIN positives.

        EXPLOSION (Q1): each positive pair ``(a,b)`` with active labels
        ``{r : pos_y[i, r] > 0}`` becomes, per active ``r``, a bidirectional
        edge-pair ``(a,b,r)`` and ``(b,a,r)`` with ``edge_type = r``. Parallel
        edges across labels are OK (RGCNConv handles multi-edges). This is the
        multilabel analogue of the multiclass single-``ddi_type`` explosion
        (multi_cls/baseline.py:181-188) — one edge per (pair, active label).

        ``pos_ht`` carries INTEGER drug endpoints (bundle space); we map them
        through ``self._drug_to_idx`` (keyed on the STRING drugbank_id, set by the
        inherited ``_build_graphs``). Pairs whose endpoint is out of the drug
        universe OR lacks a molecular graph are skipped (defensive; cold-start
        safe — the drug universe already covers all leaf drugs).

        Returns (edge_index (2,2E), edge_type (2E,), x_perm_idx (n_drugs,) view-A
        row-perm, edge_type_shuf (2E,) view-B labels). Mirrors
        multi_cls/baseline.py:157-210 but over label incidences.
        """
        assert self._drug_to_idx is not None
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        edge_index: list[list[int]] = []
        edge_type: list[int] = []
        pair_labels: list[list[int]] = []   # upstream label_list11: [r, r] PAIRS
        for i in range(len(pos_ht)):
            a = str(int(pos_ht[i, 0]))
            b = str(int(pos_ht[i, 1]))
            if a not in idx or b not in idx or a not in graphs or b not in graphs:
                continue
            ia, ib = idx[a], idx[b]
            active = np.nonzero(pos_y[i])[0]
            for r in active:
                ir = int(r)
                # bidirectional edges, edge_type = label id (Q1 explosion;
                # multi_cls/baseline.py:186-187 used the single ddi_type, here one
                # edge per active label).
                edge_index.append([ia, ib]); edge_type.append(ir)
                edge_index.append([ib, ia]); edge_type.append(ir)
                pair_labels.append([ir, ir])   # the [r,r] PAIR (multi_cls:188)

        if not edge_index:
            raise ValueError("MRCGNN multilabel: zero usable train DDI edges "
                             "(all pairs filtered / no active labels).")

        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type_t = torch.tensor(edge_type, dtype=torch.long)

        # ---- view B: shuffle the [r,r] PAIRS, THEN flatten (multi_cls:196-203) ----
        # preserves the bidirectional-pair invariant (both directions get the SAME
        # shuffled label). Under 200 relations this is a meaningful relation-label
        # corruption (unlike the binary num_relations=1 degenerate case).
        rng = np.random.default_rng(self.seed)
        pair_arr = np.asarray(pair_labels, dtype=np.int64)          # (E, 2)
        perm = rng.permutation(pair_arr.shape[0])
        shuffled_pairs = pair_arr[perm]
        edge_type_shuf = torch.tensor(shuffled_pairs.reshape(-1), dtype=torch.long)

        # ---- view A: a random ROW-PERMUTATION of the node feature matrix
        # (multi_cls/baseline.py:205-208). ----
        n_drugs = len(self._drug_order)
        x_perm_idx = torch.tensor(rng.permutation(n_drugs), dtype=torch.long)

        return edge_index_t, edge_type_t, x_perm_idx, edge_type_shuf

    # ------------------------------------------------------------------
    # fit: multilabel training on the paired pos/neg bundle
    # ------------------------------------------------------------------
    def fit(  # type: ignore[override]
        self,
        ds: "_LeafDataset",
        bundle: dict,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        """Train on the ``make_multilabel_bundle`` output.

        ``ds`` is the duck-typed leaf dataset (used for the inherited
        ``_build_graphs`` / ``_drug_smiles_dict``, i.e. the drug universe +
        molecular graphs from ``drugs[[drugbank_id, smiles]]``). ``bundle``
        carries the paired pos/neg multihot targets: keys
        ``{train,val}_{pos,neg}_{ht,y}`` (``*_ht`` (n,2) int endpoints, ``*_y``
        (n, n_labels) float32 multihot; neg_y == pos_y == the ACTIVE-LABEL vector).
        """
        # ---- drug universe + molecular graphs for ALL leaf drugs (inherited) ----
        self._build_graphs(ds)
        assert self._drug_order is not None

        train_pos_ht = np.asarray(bundle["train_pos_ht"], dtype=np.int64)
        train_pos_y = np.asarray(bundle["train_pos_y"], dtype=np.float32)
        train_neg_ht = np.asarray(bundle["train_neg_ht"], dtype=np.int64)
        val_pos_ht = np.asarray(bundle["val_pos_ht"], dtype=np.int64)
        val_pos_y = np.asarray(bundle["val_pos_y"], dtype=np.float32)
        val_neg_ht = np.asarray(bundle["val_neg_ht"], dtype=np.int64)

        # ---- TrimNet features (Q4): exploded train triples + n_classes=200 ----
        # one (a,b,r) per active label of each pos pair (CE-over-200-label recipe;
        # build_trimnet_features.py:196-215). String ids to match _drug_order.
        smiles = self._drug_smiles_dict(ds)
        train_tri = self._train_triples_ml(train_pos_ht, train_pos_y)
        if not train_tri:
            raise ValueError("MRCGNN multilabel: zero TrimNet train triples "
                             "(no active labels in train positives).")
        feats = ensure_trimnet_features(
            self._drug_order, smiles, train_tri,
            n_classes=self.n_labels, fold=self.fold, seed=self.seed,
            epochs=self.trimnet_epochs, device=self.device)
        if feats.shape != (len(self._drug_order), _FEATURE_DIM):
            raise ValueError(f"TrimNet features shape {feats.shape} != "
                             f"({len(self._drug_order)}, {_FEATURE_DIM})")
        mol_features = torch.as_tensor(feats, dtype=torch.float)

        # ---- exploded multi-relational DDI graph + two DGI views (BUILT ONCE) ----
        edge_index, edge_type, x_perm_idx, edge_type_shuf = self._build_ddi_graph_ml(
            train_pos_ht, train_pos_y)

        dev = self.device
        x_o = mol_features.to(dev)                       # RGCN node input = TrimNet feats
        x_a = x_o[x_perm_idx.to(dev)]                    # view A: feature row-permutation
        edge_index = edge_index.to(dev)
        edge_type = edge_type.to(dev)
        edge_type_shuf = edge_type_shuf.to(dev)          # view B: relation shuffle

        # DGI contrastive target: [ones(n_drugs) | zeros(n_drugs)] (multi_cls:258-261).
        n_drugs = len(self._drug_order)
        dgi_target = torch.cat((torch.ones(n_drugs, 1), torch.zeros(n_drugs, 1)),
                               dim=1).to(dev)

        # cache forward-graph tensors for mid-training validation via predict_proba.
        self._fwd = {"x_o": x_o, "edge_index": edge_index, "edge_type": edge_type,
                     "x_a": x_a, "edge_type_shuf": edge_type_shuf}

        # ---- model: n_classes=n_labels sizes RGCN num_relations AND head to 200 ----
        self.n_classes = self.n_labels
        self._model = MRCGNN(
            feature=_FEATURE_DIM, n_classes=self.n_labels, n_drugs=n_drugs,
            mol_features=mol_features, hidden1=_HIDDEN1, hidden2=_HIDDEN2,
            dropout=self.dropout).to(dev)
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)
        bce = torch.nn.BCEWithLogitsLoss()

        # ---- paired pos/neg training rows: map int endpoints -> node idx, keep
        # only pairs whose BOTH pos endpoints AND both paired-neg endpoints are in
        # the drug universe with a graph (strict pair alignment). ----
        keep_idx, pos_ia, pos_ib, neg_ia, neg_ib = self._map_paired(
            train_pos_ht, train_neg_ht)
        if len(keep_idx) == 0:
            raise ValueError("MRCGNN multilabel: zero usable paired pos/neg rows "
                             "after graph filtering.")
        y_kept = train_pos_y[keep_idx]                   # (M, n_labels) active mask

        try:
            from my_code.utils.train_progress import TrainProgress
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress

        n_train = len(keep_idx)
        steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size)
        progress = TrainProgress(total_epochs=self.n_epochs,
                                 log_step_every=self.log_step_every,
                                 total_steps_per_epoch=steps_per_epoch,
                                 prefix="[mrcgnn_ml] ")
        best_val_macro_auprc = -1.0
        best_state: dict | None = None

        def _val_metrics() -> dict:
            if len(val_pos_ht) == 0:
                return {}
            roc, pr = self._eval_paired(val_pos_ht, val_pos_y, val_neg_ht)
            return {"val_macro_auroc": roc, "val_macro_auprc": pr}

        for epoch in range(self.n_epochs):
            perm = np.random.RandomState(epoch).permutation(n_train)
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, n_train, self.batch_size):
                sel = perm[start:start + self.batch_size]
                p_ia = torch.tensor(pos_ia[sel], dtype=torch.long, device=dev)
                p_ib = torch.tensor(pos_ib[sel], dtype=torch.long, device=dev)
                n_ia = torch.tensor(neg_ia[sel], dtype=torch.long, device=dev)
                n_ib = torch.tensor(neg_ib[sel], dtype=torch.long, device=dev)
                y_mask = torch.from_numpy(y_kept[sel]).float().to(dev) > 0  # (b, n_labels)

                opt.zero_grad(set_to_none=True)
                # full-graph forward for the POS pairs (gives the two DGI views too).
                pos_logits, ret_a, ret_b, _ = self._model(
                    x_o, edge_index, edge_type, x_a, edge_type_shuf, p_ia, p_ib)
                # second forward for the paired NEG pairs (same graph/views; we only
                # use the classification logits for the masked-BCE_cls term).
                neg_logits, _, _, _ = self._model(
                    x_o, edge_index, edge_type, x_a, edge_type_shuf, n_ia, n_ib)

                # masked BCE_cls: active columns only. pos active -> 1, neg active -> 0.
                pos_sel = pos_logits[y_mask]              # (M_active,)
                neg_sel = neg_logits[y_mask]              # (M_active,)
                if pos_sel.numel() == 0:
                    continue
                loss1 = (bce(pos_sel, torch.ones_like(pos_sel))
                         + bce(neg_sel, torch.zeros_like(neg_sel)))
                loss2 = bce(ret_a, dgi_target)            # 0.05 * BCE(view A)
                loss3 = bce(ret_b, dgi_target)            # 0.1  * BCE(view B)
                loss = (_LOSS_RATIO1 * loss1 + _LOSS_RATIO2 * loss2
                        + _LOSS_RATIO3 * loss3)
                loss.backward()
                opt.step()
                progress.step(loss.item())

            extra: dict = {}
            if len(val_pos_ht):
                self._model.eval()
                metrics = _val_metrics()
                self._model.train()
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_macro_auprc", -1) > best_val_macro_auprc:
                        best_val_macro_auprc = metrics["val_macro_auprc"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        # load_best_model_at_end (macro-AUPRC primary; matches the test metric).
        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[mrcgnn_ml] loaded best val_macro_auprc="
                  f"{best_val_macro_auprc:.4f} state", flush=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _train_triples_ml(self, pos_ht: np.ndarray, pos_y: np.ndarray
                          ) -> list[tuple[str, str, int]]:
        """Exploded (str a, str b, int r) triples for the TrimNet builder: one per
        (pos pair, active label). Endpoints kept as strings to match ``drug_order``
        (build_trimnet_features filters to pairs whose both drugs have a graph)."""
        tri: list[tuple[str, str, int]] = []
        for i in range(len(pos_ht)):
            a = str(int(pos_ht[i, 0]))
            b = str(int(pos_ht[i, 1]))
            for r in np.nonzero(pos_y[i])[0]:
                tri.append((a, b, int(r)))
        return tri

    def _map_paired(self, pos_ht: np.ndarray, neg_ht: np.ndarray):
        """Map int endpoints -> node idx for paired pos/neg rows; keep only rows
        where ALL FOUR endpoints (pos_h, pos_t, neg_h, neg_t) are in the drug
        universe with a graph (strict pair alignment). Returns
        (keep_idx, pos_ia, pos_ib, neg_ia, neg_ib) as np.int64 arrays."""
        idx = self._drug_to_idx or {}
        graphs = self._graphs or {}

        def _ok(x: int) -> bool:
            s = str(int(x))
            return s in idx and s in graphs

        keep, p_ia, p_ib, n_ia, n_ib = [], [], [], [], []
        for i in range(len(pos_ht)):
            pa, pb = pos_ht[i, 0], pos_ht[i, 1]
            na, nb = neg_ht[i, 0], neg_ht[i, 1]
            if _ok(pa) and _ok(pb) and _ok(na) and _ok(nb):
                keep.append(i)
                p_ia.append(idx[str(int(pa))]); p_ib.append(idx[str(int(pb))])
                n_ia.append(idx[str(int(na))]); n_ib.append(idx[str(int(nb))])
        return (np.asarray(keep, dtype=np.int64), np.asarray(p_ia, dtype=np.int64),
                np.asarray(p_ib, dtype=np.int64), np.asarray(n_ia, dtype=np.int64),
                np.asarray(n_ib, dtype=np.int64))

    # ------------------------------------------------------------------
    # Paired eval (macro AUROC / AUPRC over active labels) — matches test metric
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_paired(self, pos_ht: np.ndarray, pos_y: np.ndarray,
                     neg_ht: np.ndarray) -> tuple[float, float]:
        """Per-label ROC-AUC / PR-AUC over paired pos/neg, averaged over labels
        with >=1 active positive. Mirrors the runner + emergnn ml eval
        (_core_twoside.py:498-516): positives = pos-pair score at column r for
        rows with multihot[r]>0; negatives = paired neg-pair score at column r for
        the SAME rows. STRICT pair alignment: only rows where BOTH pos and its
        paired neg kept a graph are scored."""
        pos_scores, pos_kept = self._score_pairs(pos_ht)
        neg_scores, neg_kept = self._score_pairs(neg_ht)
        keep = pos_kept & neg_kept
        if not keep.any():
            return float("nan"), float("nan")
        pos_scores = pos_scores[keep]
        neg_scores = neg_scores[keep]
        y = np.asarray(pos_y, dtype=np.float32)[keep]
        rocs, prs = [], []
        for r in range(self.n_labels):
            sel = y[:, r] > 0
            k = int(sel.sum())
            if k == 0:
                continue
            score = np.concatenate([pos_scores[sel, r], neg_scores[sel, r]])
            label = np.concatenate([np.ones(k), np.zeros(k)])
            if label.min() == label.max():
                continue
            try:
                rocs.append(roc_auc_score(label, score))
            except ValueError:
                pass
            prs.append(average_precision_score(label, score))
        return (float(np.mean(rocs)) if rocs else float("nan"),
                float(np.mean(prs)) if prs else float("nan"))

    @torch.no_grad()
    def _score_pairs(self, ht: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n,2) int endpoints -> ((n, n_labels) sigmoid probs, keep_mask). Rows
        whose drug is out of universe / lacks a graph get a 0.5 row (keep=False so
        callers can drop them for strict pair alignment)."""
        if self._model is None or self._drug_to_idx is None:
            raise RuntimeError("MRCGNNMultilabelBaseline must be fitted before scoring.")
        self._model.eval()
        idx = self._drug_to_idx
        graphs = self._graphs or {}
        fwd = self._fwd
        out = np.full((len(ht), self.n_labels), 0.5, dtype=np.float32)
        keep = np.zeros(len(ht), dtype=bool)

        keep_rows: list[int] = []
        ia_list: list[int] = []
        ib_list: list[int] = []
        for i in range(len(ht)):
            a = str(int(ht[i, 0]))
            b = str(int(ht[i, 1]))
            if a in idx and b in idx and a in graphs and b in graphs:
                keep_rows.append(i)
                ia_list.append(idx[a])
                ib_list.append(idx[b])
        if not keep_rows:
            return out, keep

        dev = self.device
        for s in range(0, len(keep_rows), self.batch_size):
            rows = keep_rows[s:s + self.batch_size]
            ia = torch.tensor(ia_list[s:s + self.batch_size], dtype=torch.long, device=dev)
            ib = torch.tensor(ib_list[s:s + self.batch_size], dtype=torch.long, device=dev)
            logits, _, _, _ = self._model(
                fwd["x_o"], fwd["edge_index"], fwd["edge_type"],
                fwd["x_a"], fwd["edge_type_shuf"], ia, ib)
            probs = torch.sigmoid(logits).cpu().numpy()
            for j, row in enumerate(rows):
                out[row] = probs[j]
                keep[row] = True
        return out, keep

    # ------------------------------------------------------------------
    # Override predict_proba: (n, n_labels) sigmoid probabilities
    # ------------------------------------------------------------------
    def predict_proba(  # type: ignore[override]
        self,
        pairs: "pd.DataFrame | np.ndarray",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None or self._drug_to_idx is None:
            raise RuntimeError("MRCGNNMultilabelBaseline.fit() must be called before predict.")
        if isinstance(pairs, pd.DataFrame):
            ht = pairs[_PAIR].to_numpy()
        else:
            ht = np.asarray(pairs)
        ht_int = np.empty((len(ht), 2), dtype=np.int64)
        for i in range(len(ht)):
            ht_int[i, 0] = int(ht[i, 0])
            ht_int[i, 1] = int(ht[i, 1])
        scores, _keep = self._score_pairs(ht_int)
        return scores

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path: "Path | str") -> None:  # type: ignore[override]
        if self._model is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump({"graphs": self._graphs, "missing": self._missing,
                         "drug_order": self._drug_order,
                         "drug_to_idx": self._drug_to_idx,
                         "n_labels": self.n_labels,
                         "fwd": {k: v.cpu() for k, v in self._fwd.items()}}, f)
        write_manifest(
            out, baseline_name=self.name,
            extra={"version": self.VERSION, "task": "multilabel",
                   "n_labels": self.n_labels,
                   "hyperparameters": {
                       "feature": _FEATURE_DIM, "hidden1": _HIDDEN1, "hidden2": _HIDDEN2,
                       "learning_rate": self.learning_rate,
                       "weight_decay": self.weight_decay, "dropout": self.dropout,
                       "batch_size": self.batch_size, "n_epochs": self.n_epochs,
                       "trimnet_epochs": self.trimnet_epochs, "seed": self.seed,
                       "fold": self.fold}})

    @classmethod
    def load(cls, path: "Path | str") -> "MRCGNNMultilabelBaseline":  # type: ignore[override]
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hp = manifest.get("hyperparameters", {})
        n_labels = int(manifest.get("n_labels", 200))
        inst = cls(n_labels=n_labels,
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
        inst.n_labels = int(payload.get("n_labels", n_labels))
        inst.n_classes = inst.n_labels
        mol = payload["fwd"]["x_o"]
        inst._model = MRCGNN(
            feature=_FEATURE_DIM, n_classes=inst.n_labels,
            n_drugs=len(inst._drug_order), mol_features=mol,
            hidden1=_HIDDEN1, hidden2=_HIDDEN2, dropout=inst.dropout).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        inst._fwd = {k: v.to(inst.device) for k, v in payload["fwd"].items()}
        return inst


__all__ = ["MRCGNNMultilabelBaseline"]
