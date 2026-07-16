"""SumGNN multiclass baseline for the UNIFIED benchmark (decision B).

SumGNN is KG-subgraph based -> applies to KG datasets (drugbank_latest_* via the merged
DrugBank+Hetionet+PrimeKG KG). One leaf = one regime -> a SINGLE SumGNN model
(subgraph-extraction + layer-independent self-attention KG summarization + Morgan-feature
multi-channel head), CrossEntropy over the train-observed DDI-event vocab. NO cross-regime
routing.

Pipeline per leaf:
  1. Build SumGNN's on-disk int-indexed layout (train/dev/test.txt, entity.txt,
     relations_2hop.txt, DB_molecular_feats.pkl) from the leaf's string-id drugs + pairs +
     the merged KG, via `_data/necessary/build_sumgnn_data.build_sumgnn_data` (drugs indexed
     first; kg_scope="full" by default = the WHOLE merged KG, standing decision 2026-07-01).
     Cached under `_data/necessary/mc__<hash>__<kg_scope>/`.
  2. Extract per-pair enclosing subgraphs (lmdb) + train the ported DGL core (`_core`).
  3. `predict` scores each test pair's subgraph -> softmax over the TRAIN relation vocab,
     scattered to the leaf's GLOBAL class axis -> (n, n_labels_global).

`predict` returns (n, n_labels_GLOBAL): SumGNN's fc head outputs `train_rels` logits (one per
train-observed relation id). We map each train relation id back to its global class and
scatter. Test rows whose gold class was unseen-in-train get ~0 mass -> counted wrong (same
convention as EmerGNN multiclass; surfaced via the runner's oov_target_rate).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

# baseline-local ported SumGNN core (independent copy; NOT the reproduction).
_CORE = Path(__file__).resolve().parents[1] / "_core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.sumgnn._data.necessary.build_sumgnn_data import build_sumgnn_data  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd  # noqa: F811

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "_data" / "necessary"


def _leaf_hash(meta: dict, fold: str) -> str:
    key = f"{meta.get('dataset_id')}|{meta.get('split_type')}|{meta.get('split_code')}|{fold}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@register_unified("sumgnn")
class SumGNNUnifiedMulticlass(UnifiedBaseline):
    task = "multiclass"

    def __init__(self, *, emb_dim: int = 32, hop: int = 2, num_bases: int = 4,
                 n_epochs: int = 50, learning_rate: float = 5e-3, batch_size: int = 128,
                 max_links: int = 250000, num_workers: int = 8, eval_every_iter: int = 526,
                 device: str = "auto", log_step_every: int = 50, materialize: bool = False,
                 kg_scope: str = "full",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.emb_dim = emb_dim
        self.hop = hop
        self.num_bases = num_bases
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_links = max_links
        self.num_workers = num_workers
        self.eval_every_iter = eval_every_iter
        # standing decision 2026-07-01: default to the FULL merged KG (not drug-incident).
        self.kg_scope = kg_scope
        # perf (codex 019f1fdc): materialize train/valid/test subgraphs into RAM once
        # (faithful; eliminates per-epoch reconstruction). Forces num_workers=0.
        self.materialize = materialize
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._fold = "fold0"
        self._n_global: int | None = None
        self._params: SimpleNamespace | None = None
        self._clf = None
        self._train = None
        self._idx_to_global: list[int] | None = None

    # ---- KG / data prep -------------------------------------------------
    @staticmethod
    def _merged_edges_path(resources: "LeafResources") -> Path:
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("SumGNN requires a KG; this leaf declares none.")
        edges = Path(kg.source) / MERGED_EDGES
        if not edges.is_file():
            raise FileNotFoundError(f"merged KG edges not found: {edges}")
        return edges

    def _ensure_data(self, train_df, val_df, test_df, resources) -> Path:
        meta = resources.meta
        # scope in the cache dir so full-KG vs drug-incident caches never collide.
        data_dir = _CACHE_ROOT / f"mc__{_leaf_hash(meta, self._fold)}__{self.kg_scope}"
        stamp = data_dir / "_build_stats.json"
        if stamp.is_file():
            logging.info(f"[sumgnn] data cache HIT: {data_dir}")
            return data_dir
        logging.info(f"[sumgnn] BUILDING SumGNN data at {data_dir} (kg_scope={self.kg_scope})")
        edges = pd.read_parquet(self._merged_edges_path(resources))
        stats = build_sumgnn_data(
            data_dir, resources.drugs, edges, train_df, val_df, test_df,
            label_col="y_cls", morgan_from="smiles", kg_scope=self.kg_scope,
        )
        logging.info(f"[sumgnn] build stats: {stats}")
        return data_dir

    def fit_leaf(self, leaf) -> None:  # capture fold for the cache key
        self._fold = leaf.fold
        self.fit(leaf.train, leaf.val, resources=leaf.resources, _test=leaf.test)

    def fit(self, train_df, val_df=None, *, resources, _test=None) -> None:
        from subgraph_extraction.datasets import (SubgraphDataset,
                                                  generate_subgraph_datasets)
        from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
        from model.dgl.graph_classifier import GraphClassifier as dgl_model
        from managers.evaluator import Evaluator
        from managers.trainer import Trainer

        meta = resources.meta
        self._n_global = int(meta["labels"]["n_labels"])
        if val_df is None:
            val_df = train_df
        # test pairs must be in the on-disk test.txt so their subgraphs get extracted
        test_df = _test if _test is not None else val_df
        data_dir = self._ensure_data(train_df, val_df, test_df, resources)

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        params = SimpleNamespace(
            experiment_name=f"sumgnn_{_leaf_hash(meta, self._fold)}",
            dataset="drugbank", main_dir=".",  # resolved against the chdir sandbox (data/drugbank symlink)
            train_file="train", valid_file="dev", test_file="test",
            num_epochs=self.n_epochs, eval_every=3, eval_every_iter=self.eval_every_iter,
            save_every=10_000, early_stop=100, optimizer="Adam", lr=self.learning_rate,
            clip=1000, l2=1e-5, max_links=self.max_links, hop=self.hop, max_nodes_per_hop=200,
            use_kge_embeddings=False, kge_model="TransE", model_type="dgl",
            constrained_neg_prob=0.0, batch_size=self.batch_size, num_neg_samples_per_link=0,
            num_workers=self.num_workers, add_traspose_rels=False, enclosing_sub_graph=True,
            rel_emb_dim=32, attn_rel_emb_dim=32, emb_dim=self.emb_dim, num_gcn_layers=2,
            num_bases=self.num_bases, dropout=0.3, edge_dropout=0.4, gnn_agg_type="sum",
            add_ht_emb=True, add_sb_emb=True, has_attn=True, has_kg=True, feat="morgan",
            feat_dim=1024, add_feat_emb=True, add_transe_emb=True, gamma=0.2,
            device=torch.device(dev), collate_fn=collate_dgl,
            move_batch_to_device=move_batch_to_device_dgl,
        )
        # absolute on-disk paths for the ported pipeline (which otherwise reads
        # data/{dataset} relative to cwd). We point every reader at data_dir.
        params.file_paths = {"train": str(data_dir / "train.txt"),
                             "valid": str(data_dir / "dev.txt"),
                             "test": str(data_dir / "test.txt")}
        params.db_path = str(data_dir / f"subgraphs_en_True_neg_0_hop_{self.hop}_max{self.max_links}")
        params.exp_dir = str(data_dir / "exp")
        os.makedirs(params.exp_dir, exist_ok=True)
        self._params = params

        # the ported datasets.py reads 'data/{dataset}/...' relative to CWD for
        # entity.txt / relations_2hop.txt / relation2id.json — chdir into the build
        # dir's grandparent and symlink so 'data/drugbank' resolves to data_dir.
        self._prepare_cwd(data_dir)

        if not os.path.isdir(params.db_path):
            generate_subgraph_datasets(params)

        train = SubgraphDataset(params.db_path, "train_pos", "train_neg", params.file_paths,
                                add_traspose_rels=False, num_neg_samples_per_link=0,
                                use_kge_embeddings=False, dataset="drugbank",
                                kge_model="TransE", file_name="train")
        test = SubgraphDataset(params.db_path, "test_pos", "test_neg", params.file_paths,
                               add_traspose_rels=False, num_neg_samples_per_link=0,
                               use_kge_embeddings=False, dataset="drugbank",
                               kge_model="TransE", file_name="test",
                               ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                               id2relation=train.id2relation, rel=train.num_rels, graph=train.graph)
        # dense train class -> global class map (builder remapped y_cls to a
        # contiguous [0, K_train) block; fc head index == dense id).
        import json
        map_path = data_dir / "train_class_to_global.json"
        self._idx_to_global = json.loads(map_path.read_text())

        params.num_rels = train.num_rels
        params.aug_num_rels = train.aug_num_rels
        params.inp_dim = train.n_feat_dim
        # fc head width = # dense train classes (== builder's contiguous vocab). Use
        # the map length so the head matches the on-disk dense label range exactly.
        params.train_rels = len(self._idx_to_global)
        params.num_nodes = 200000
        params.max_label_value = train.max_n_label

        clf = self._build_model(params, dgl_model, data_dir)
        # Model selection on VALIDATION (dev), NOT test — the original SumGNN builds
        # separate valid + test evaluators (train.py) and picks best on valid 'auc'
        # (= macro-F1). Wiring test as the valid evaluator was eval leakage.
        valid = SubgraphDataset(params.db_path, "valid_pos", "valid_neg", params.file_paths,
                                add_traspose_rels=False, num_neg_samples_per_link=0,
                                use_kge_embeddings=False, dataset="drugbank",
                                kge_model="TransE", file_name="valid",
                                ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                                id2relation=train.id2relation, rel=train.num_rels, graph=train.graph)
        if self.materialize:
            from subgraph_extraction.datasets_materialized import MaterializedSubgraphDataset
            train = MaterializedSubgraphDataset(train)
            valid = MaterializedSubgraphDataset(valid)
            test = MaterializedSubgraphDataset(test)
            params.num_workers = 0   # cache is in-process; forked workers would duplicate it
            logging.info("[sumgnn] materialized train/valid/test subgraphs -> num_workers=0")
        valid_ev = Evaluator(params, clf, valid)   # best-ckpt selection (validation)
        test_ev = Evaluator(params, clf, test)     # logging only
        trainer = Trainer(params, clf, train, valid_ev, valid_ev, test_ev)
        logging.info(f"[sumgnn] training {self.n_epochs} epochs on {len(train)} subgraphs "
                     f"(num_rels={params.num_rels}, inp_dim={params.inp_dim})")
        trainer.train()
        # load_best_model_at_end: reload the best (valid-selected) checkpoint if the
        # trainer saved one (eval fired >=1x); else keep the final in-memory model.
        best = Path(params.exp_dir) / "best_graph_classifier.pth"
        if best.is_file():
            try:
                clf = torch.load(best, map_location=params.device, weights_only=False)
                logging.info(f"[sumgnn] loaded best valid-selected checkpoint: {best}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[sumgnn] best-ckpt reload failed ({e}); using final model")
        self._clf = clf
        self._train = train
        self._test_ds = test

    def _prepare_cwd(self, data_dir: Path) -> None:
        """The ported datasets.py loads entity.txt/relations_2hop.txt via
        'data/drugbank/...' relative to CWD. Create a CWD sandbox with a
        data/drugbank -> data_dir symlink and chdir into it."""
        sandbox = data_dir / "_cwd"
        (sandbox / "data").mkdir(parents=True, exist_ok=True)
        link = sandbox / "data" / "drugbank"
        try:
            if link.is_symlink() or link.exists():
                if link.is_symlink():
                    link.unlink()
            if not link.exists():
                os.symlink(data_dir, link, target_is_directory=True)
        except OSError:
            # fallback: copy the small text files the pipeline reads by rel path
            import shutil
            link.mkdir(parents=True, exist_ok=True)
            for f in ("entity.txt", "relations_2hop.txt"):
                shutil.copy(data_dir / f, link / f)
        os.chdir(sandbox)

    def _build_model(self, params, dgl_model, data_dir):
        # initialize_model reads relation2id.json (written by the pipeline). Build
        # the classifier directly to avoid its cwd-relative path assumptions.
        import json
        r2i_path = data_dir / "relation2id.json"
        relation2id = json.loads(r2i_path.read_text()) if r2i_path.is_file() else {}
        clf = dgl_model(params, relation2id).to(params.device)
        mfeat_path = data_dir / "DB_molecular_feats.pkl"
        with open(mfeat_path, "rb") as f:
            x = pickle.load(f, encoding="utf-8")
        mfeat = np.array([y for y in x["Morgan_Features"]])
        clf.drug_feat(torch.FloatTensor(mfeat).to(params.device))
        return clf

    # ---- prediction -----------------------------------------------------
    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._clf is None:
            raise RuntimeError("fit() before predict().")
        from torch.utils.data import DataLoader
        import torch.nn.functional as F

        p = self._params
        loader = DataLoader(self._test_ds, batch_size=p.batch_size, shuffle=False,
                            num_workers=p.num_workers, collate_fn=p.collate_fn)
        self._clf.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                data_pos, r_labels_pos, targets_pos = p.move_batch_to_device(batch, p.device)
                logits = self._clf(data_pos)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(self._idx_to_global)))

        out = np.zeros((prob.shape[0], self._n_global), dtype=np.float32)
        for i, g in enumerate(self._idx_to_global):
            if 0 <= g < self._n_global:
                out[:, g] = prob[:, i]
        # align to test_df row order: the on-disk test.txt was written in the same
        # order as leaf.test after dropping unknown-drug pairs. Reconstruct the keep
        # mask to place predictions; dropped rows get uniform ~0 (counted wrong).
        return self._align_to_test(out, test_df)

    def _align_to_test(self, out: np.ndarray, test_df: "pd.DataFrame") -> np.ndarray:
        """The extracted test subgraphs cover only pairs whose BOTH drugs were known
        (build dropped the rest). Map predictions back onto the full test_df row order."""
        data_dir = Path(self._params.file_paths["test"]).parent
        with open(data_dir / "entity2id.pkl", "rb") as f:
            entity2id = pickle.load(f)
        a = test_df["drug_a_id"].astype(str).map(entity2id)
        b = test_df["drug_b_id"].astype(str).map(entity2id)
        keep = (a.notna() & b.notna()).to_numpy()
        full = np.zeros((len(test_df), self._n_global), dtype=np.float32)
        if keep.sum() == out.shape[0]:
            full[keep] = out
        else:  # defensive: shape mismatch -> best-effort truncate/pad
            m = min(keep.sum(), out.shape[0])
            idx = np.where(keep)[0][:m]
            full[idx] = out[:m]
        return full


__all__ = ["SumGNNUnifiedMulticlass"]
