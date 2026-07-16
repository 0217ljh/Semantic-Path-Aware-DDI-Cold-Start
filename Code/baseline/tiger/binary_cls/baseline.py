"""TIGER thin adapter — implementation of :class:`BaselineModel`.

NOTE — paper-vs-port task formulation:
  * This binary adapter **matches the original paper** (Su et al.,
    AAAI 2024): binary classification (pos pair vs random negative),
    softmax CE on (B, 2) head, metrics = ACC / F1 / AUC / AUPR. Paper
    DOES NOT do 86-class — see Su et al. §Evaluation Metrics p.5.
  * Cold-start ``_patch_unseen_center_nodes`` (in model.py) is a
    Code-Released ColdDDI EXTENSION beyond Blair1213/TIGER — used to
    keep the KG channel usable for unseen drugs under S1/S2.
  * The companion ``baseline_multiclass.py`` is an MC adaptation we
    added; NOT in the original paper.

Default mode: **dual-channel** (mol + BKG subgraph). Wraps:

  * :class:`baseline.tiger.model.TIGER` — the dual-channel model
    with cold-start center-node patching for unseen drugs.
  * :mod:`baseline.tiger.mol_features` — SMILES → atom features.
  * :mod:`baseline.tiger.kg_builder` — BKG construction from
    our merged KG parquet (DrugBank + Hetionet + PrimeKG).
  * :mod:`baseline.tiger.subgraph_features` — per-drug DeepWalk
    subgraph extraction over the BKG.

Pairs are scored by a 2-class softmax head returning the class-1
probability as the binary score (mirrors upstream's forward pass).

Cold-start handling: g2 (unseen) drug indices are computed from
``train.splits.g2_drugs`` and passed into the model's forward as
``unseen_ids``. The model then projects ``mol_graph_emb`` into the
KG-node space to replace the center-node row of the BKG subgraph
encoder. This is the Code-Released cold-start extension, NOT in
upstream Blair1213/TIGER.

Backward-compat: setting ``mol_only=True`` reproduces the previous
mol-only adapter behaviour (SMILES branch only, no BKG).
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
from torch_geometric.data import Batch

from baseline.base import BaselineModel, register, write_manifest
from baseline.tiger._shared import (
    DEFAULT_MOL_PKL_PATH as _DEFAULT_MOL_PKL_PATH,
    bkg_cache_key_for as _bkg_cache_key_for,
    ensure_bkg_cache as _ensure_bkg_cache,
    ensure_mol_pkl as _ensure_mol_pkl,
    ensure_subgraph_cache as _ensure_subgraph_cache,
)
from baseline.tiger.mol_features import (
    ATOM_FEATURE_DIM,
    build_drug_graphs,
)
from baseline.tiger.model import TIGER

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


def _drug_smiles_dict(train: "PairDataset") -> dict[str, str]:
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "TIGER requires `PairDataset.drugs` with a `smiles` column."
        )
    return {
        str(row["drugbank_id"]): "" if pd.isna(row["smiles"]) else str(row["smiles"])
        for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
    }


@register("tiger")
class TIGERBaseline(BaselineModel):
    """TIGER (dual-channel by default): mol GraphTransformer + BKG-subgraph
    GraphTransformer + cold-start center-node patch + 2-class softmax head.
    """

    VERSION = "2.0"  # dual-channel; v1.0 was mol-only

    def __init__(
        self,
        *,
        # Model arch
        max_layer: int = 4,
        output_dim: int = 64,
        max_degree_graph: int = 100,
        max_degree_node: int = 100,
        num_relations_mol: int | None = None,
        num_relations_graph: int | None = None,
        sub_coeff: float = 0.2,
        mi_coeff: float = 0.5,
        dropout: float = 0.2,
        mol_only: bool = False,
        # KG / BKG source
        #   "merged"   : use merged DrugBank+Hetionet+PrimeKG parquet
        #                (requires ``merged_kg_path``). Default.
        #   "none"     : force mol-only (alias for ``mol_only=True``)
        kg_source: str = "merged",
        merged_kg_path: str | Path | None = None,
        merged_kg_blocklist: tuple[str, ...] = (),
        # Subgraph extractor (paper Sec "Biomedical Knowledge Graph
        # Channel" / Su et al. AAAI 2024). Choose one of:
        #   "randomWalk"   — TIGER-DW (default; back-compat with old runs)
        #   "khop-subtree" — TIGER-KS (paper recommends k=2 fan-out=4)
        #   "probability"  — TIGER-P (personalized PageRank, fixed_num=32)
        # All 3 restored 2026-05-18 (codex round-1 fix #3); previous
        # version supported only randomWalk which violated CLAUDE.md
        # §"Baseline 规范" §4 ("不允许借口'新任务'省略 paper 算法核心").
        extractor: str = "randomWalk",
        khop: int = 2,
        khop_fanout: int = 4,
        prob_fixed_num: int = 32,
        rw_num_walks: int = 1,
        rw_walk_length: int = 32,
        # Training
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 64,
        n_epochs: int = 5,
        device: str = "auto",
        # Training progress logging (per CLAUDE.md)
        log_step_every: int = 50,
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        save_strategy: str = "no",
        save_steps: int = 500,
        save_total_limit: int = 3,
        load_best_model_at_end: bool = True,
        run_dir: str | None = None,
        # Cold-start center-node patch — PROJECT EXTENSION, NOT in paper
        # (Su et al. AAAI 2024 doesn't address unseen drugs at all). When
        # True, BKG-channel center-node embedding of every drug whose
        # global idx is in ``self._unseen_ids`` is replaced with
        # ``cold_start_proj(mol_graph_emb) + degree_encoder(center_degree)``
        # before further propagation. Per codex review 2026-05-18 +
        # CLAUDE.md §"Baseline 规范" §4: "5. 将 cold-start patch 作为显式
        # adaptation 开关 ... 主 ``tiger`` 保持 paper-faithful 默认".
        # Default False → paper-faithful (matches Su et al. 2024 exactly,
        # only meaningful behaviour on warm-start S0). Set True to enable
        # the extension for S1/S2 cold-start runs.
        cold_start_patch: bool = False,
        # Path to the mol-graph pkl (CLAUDE.md §"Baseline 规范" §2
        # detect-and-build). Default = canonical local path
        # ``_data/necessary/mol_pkl__mine.pkl``. ``_build_mol_graphs``
        # routes through ``_shared.ensure_mol_pkl`` which auto-detects
        # (cache HIT) or auto-builds (subprocess invokes the local
        # ``_data/necessary/build_mol_pkl.py``). Pass ``mol_pkl_path=None``
        # to opt out and use the in-memory legacy build path (NOT
        # recommended for production; kept for debugging only).
        mol_pkl_path: str | Path | None = _DEFAULT_MOL_PKL_PATH,
    ) -> None:
        self.max_layer = max_layer
        self.output_dim = output_dim
        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node
        self.num_relations_mol = num_relations_mol
        self.num_relations_graph = num_relations_graph
        self.sub_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout
        self.mol_only = bool(mol_only) or (kg_source == "none")
        self.kg_source = kg_source
        self.merged_kg_path = (
            Path(merged_kg_path) if merged_kg_path is not None else None
        )
        self.merged_kg_blocklist = tuple(merged_kg_blocklist)
        if extractor not in {"randomWalk", "khop-subtree", "probability"}:
            raise ValueError(
                f"unknown extractor={extractor!r}; expected "
                "'randomWalk' / 'khop-subtree' / 'probability'"
            )
        self.extractor = extractor
        self.khop = int(khop)
        self.khop_fanout = int(khop_fanout)
        self.prob_fixed_num = int(prob_fixed_num)
        self.rw_num_walks = int(rw_num_walks)
        self.rw_walk_length = int(rw_walk_length)
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
        self.cold_start_patch = bool(cold_start_patch)
        self.mol_pkl_path = Path(mol_pkl_path) if mol_pkl_path is not None else None

        if self.kg_source not in {"merged", "none"}:
            raise ValueError(
                f"unknown kg_source={self.kg_source!r}; expect 'merged' or 'none'"
            )
        if (
            self.kg_source == "merged"
            and not self.mol_only
            and self.merged_kg_path is None
        ):
            raise ValueError(
                "kg_source='merged' (dual-channel) requires merged_kg_path to be set"
            )

        # Fitted state
        self._model: TIGER | None = None
        self._mol_graphs: dict | None = None
        self._mol_missing: list[str] = []
        self._bkg: dict | None = None
        self._subgraphs: dict[int, "Data"] | None = None
        self._drug_to_idx: dict[str, int] | None = None
        self._unseen_ids: set[int] = set()
        self._effective_num_rel_mol: int | None = None
        self._effective_num_rel_graph: int | None = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    # ------------------------------------------------------------------
    # Internal build helpers
    # ------------------------------------------------------------------

    def _build_mol_graphs(self, train: "PairDataset") -> int:
        """Parse SMILES → ``{drug_id: Data}``. Returns max_sp_rel for
        sizing ``num_relations_mol``.

        Preferred path (CLAUDE.md §"Baseline 规范" §2 step 3): call
        :func:`baseline.tiger._shared.ensure_mol_pkl` which detects an
        existing ``__mine`` pkl and reuses it, or auto-builds via
        subprocess if missing / stale / corrupt. Falls back to the
        legacy in-memory build only if ``self.mol_pkl_path is None``.
        """
        if self.mol_pkl_path is not None:
            self._mol_graphs = _ensure_mol_pkl(train, self.mol_pkl_path)
            # ``ensure_mol_pkl`` guarantees coverage when it auto-builds,
            # but a stale user-supplied pkl could still leave gaps —
            # surface them as a warning + recompute max_rel from the
            # loaded graphs (pkl doesn't store it explicitly).
            train_drug_ids = (
                {str(d) for d in train.drugs["drugbank_id"]}
                if train.drugs is not None
                else set()
            )
            self._mol_missing = sorted(
                train_drug_ids - {str(k) for k in self._mol_graphs.keys()}
            )
            if self._mol_missing:
                print(
                    f"[tiger] WARNING: {len(self._mol_missing)} drugs from "
                    f"train.drugs missing in mol_pkl ({self.mol_pkl_path}). "
                    f"Delete the pkl to force rebuild.",
                    file=sys.stderr,
                )
            max_rel = 0
            for data in self._mol_graphs.values():
                if data is None or not hasattr(data, "sp_edge_rel"):
                    continue
                max_rel = max(max_rel, int(data.sp_edge_rel.max().item()))
            return max_rel + 1

        # Legacy in-memory build (only reachable via explicit opt-out
        # ``mol_pkl_path=None``). NOT recommended for production — see
        # CLAUDE.md §"Baseline 规范" §4 anti-pattern.
        smiles = _drug_smiles_dict(train)
        self._mol_graphs, self._mol_missing, max_rel = build_drug_graphs(smiles)
        if not self._mol_graphs:
            raise ValueError(
                "TIGER built zero molecular graphs — every drug failed to parse."
            )
        if self._mol_missing:
            print(
                f"[tiger] {len(self._mol_missing)} drugs skipped (no parseable SMILES).",
                file=sys.stderr,
            )
        return max_rel + 1

    def _build_bkg_and_subgraphs(
        self, train: "PairDataset", verbose: bool = True
    ) -> tuple[int, int, int]:
        """Build BKG from merged KG + per-drug subgraphs via DeepWalk.

        Returns ``(num_rel_graph_required, max_degree_node_observed, n_total_nodes)``.
        """
        # Drug universe — union of all split drugs to keep BKG consistent
        all_drugs = (
            set(str(d) for d in train.splits.train["drug_a_id"])
            | set(str(d) for d in train.splits.train["drug_b_id"])
            | set(str(d) for d in train.splits.val_s2["drug_a_id"])
            | set(str(d) for d in train.splits.val_s2["drug_b_id"])
            | set(str(d) for d in train.splits.test_s2["drug_a_id"])
            | set(str(d) for d in train.splits.test_s2["drug_b_id"])
        )
        if train.splits.g1_drugs:
            all_drugs |= set(map(str, train.splits.g1_drugs))
        if train.splits.g2_drugs:
            all_drugs |= set(map(str, train.splits.g2_drugs))

        g2_drugs = list(map(str, train.splits.g2_drugs or []))

        # CLAUDE.md §"Baseline 规范" §2 step 3: detect-and-build BKG
        # cache via subprocess. Cache key encodes drug pool, DDI edges,
        # g2 drugs, blocklist, and merged_kg_path so cross-fold/-seed
        # runs get distinct cache files (no silent stale).
        self._bkg = _ensure_bkg_cache(
            train,
            self.merged_kg_path,
            g2_drugs=g2_drugs,
            blocklist=self.merged_kg_blocklist,
        )
        # Pre-compute the same cache key for subgraph cache invalidation
        # — if BKG inputs change, ``bkg_cache_key`` differs → subgraph
        # cache lookup misses → subprocess rebuilds correctly.
        self._bkg_cache_key = _bkg_cache_key_for(
            train,
            self.merged_kg_path,
            g2_drugs=g2_drugs,
            blocklist=self.merged_kg_blocklist,
        )
        self._drug_to_idx = self._bkg["drug_to_idx"]
        # Cold-start patch is opt-in (paper-faithful default = off). When
        # disabled, ``_unseen_ids`` stays empty → model.forward sees
        # ``unseen_ids=set()`` → ``_patch_unseen_center_nodes`` no-ops →
        # behaviour exactly matches Su et al. AAAI 2024.
        if self.cold_start_patch:
            self._unseen_ids = set(self._bkg["g2_idx"])
            print(
                f"[tiger] cold-start patch ENABLED — {len(self._unseen_ids)} "
                f"unseen drugs will use mol-channel center-node projection",
                file=sys.stderr,
            )
        else:
            self._unseen_ids = set()

        # CLAUDE.md §"Baseline 规范" §2 step 3: detect-and-build
        # subgraph cache via subprocess. Cache key bundles bkg_cache_key
        # + extractor name + extractor hparams + seed so different
        # extractor configs coexist as separate cache files.
        extractor_params = {
            "num_walks": self.rw_num_walks,
            "walk_length": self.rw_walk_length,
            "khop": self.khop,
            "khop_fanout": self.khop_fanout,
            "fixed_num": self.prob_fixed_num,
        }
        sg_payload = _ensure_subgraph_cache(
            self._bkg,
            self._bkg_cache_key,
            self.extractor,
            extractor_params,
            seed=42,
        )
        self._subgraphs = sg_payload["subgraphs"]
        max_deg = sg_payload["max_degree"]
        max_sp_rel = sg_payload["max_sp_rel"]
        return max_sp_rel + 1, max_deg + 1, self._bkg["n_total_nodes"]

    def _make_pair_batch(
        self, pairs: pd.DataFrame, labels: np.ndarray | None = None
    ):
        """Build 4-tuple PyG batches (mol1, sub1, mol2, sub2) plus
        per-sample (batch_idx1, batch_idx2) tensors for cold-start
        identification. Returns ``None`` placeholders if no pair is
        scoreable."""
        keep_idx: list[int] = []
        mol1_list, mol2_list = [], []
        sub1_list, sub2_list = [], []
        idx1_list, idx2_list = [], []
        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            mga = self._mol_graphs.get(a)
            mgb = self._mol_graphs.get(b)
            if mga is None or mgb is None:
                continue
            if not self.mol_only:
                ia = self._drug_to_idx.get(a) if self._drug_to_idx else None
                ib = self._drug_to_idx.get(b) if self._drug_to_idx else None
                sga = self._subgraphs.get(ia) if ia is not None else None
                sgb = self._subgraphs.get(ib) if ib is not None else None
                if sga is None or sgb is None:
                    continue
            else:
                ia = ib = -1
                sga = sgb = None
            mga = mga.clone()
            mgb = mgb.clone()
            if labels is not None:
                mga.y = torch.tensor([int(labels[i])], dtype=torch.long)
                mgb.y = torch.tensor([int(labels[i])], dtype=torch.long)
            else:
                mga.y = torch.tensor([0], dtype=torch.long)
                mgb.y = torch.tensor([0], dtype=torch.long)
            mol1_list.append(mga)
            mol2_list.append(mgb)
            if self.mol_only:
                # Fill subgraph slots with mol so signature stays
                # consistent; model's mol_only branch ignores them.
                sub1_list.append(mga.clone())
                sub2_list.append(mgb.clone())
            else:
                sub1_list.append(sga.clone())
                sub2_list.append(sgb.clone())
            idx1_list.append(int(ia) if ia is not None else -1)
            idx2_list.append(int(ib) if ib is not None else -1)
            keep_idx.append(i)
        if not keep_idx:
            mask = np.zeros(len(pairs), dtype=bool)
            return None, None, None, None, None, None, mask
        mol1_batch = Batch.from_data_list(mol1_list)
        mol2_batch = Batch.from_data_list(mol2_list)
        sub1_batch = Batch.from_data_list(sub1_list)
        sub2_batch = Batch.from_data_list(sub2_list)
        batch_idx1 = torch.tensor(idx1_list, dtype=torch.long)
        batch_idx2 = torch.tensor(idx2_list, dtype=torch.long)
        mask = np.zeros(len(pairs), dtype=bool)
        mask[np.asarray(keep_idx)] = True
        return mol1_batch, sub1_batch, mol2_batch, sub2_batch, batch_idx1, batch_idx2, mask

    def _forward(self, mol1, sub1, mol2, sub2, idx1, idx2):
        if self.mol_only:
            # mol_only branch: pass mol twice (subgraph slot is ignored)
            return self._model(
                mol1.to(self.device),
                mol1.to(self.device),
                mol2.to(self.device),
                mol2.to(self.device),
            )
        return self._model(
            mol1.to(self.device),
            sub1.to(self.device),
            mol2.to(self.device),
            sub2.to(self.device),
            batch_idx1=idx1,
            batch_idx2=idx2,
            unseen_ids=self._unseen_ids,
        )

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
        # ── 1. SMILES graphs ──────────────────────────────────────────
        mol_rel_required = self._build_mol_graphs(train)
        if self.num_relations_mol is None:
            num_rel_mol = max(mol_rel_required + 4, 32)
        else:
            num_rel_mol = self.num_relations_mol
            if num_rel_mol < mol_rel_required:
                raise ValueError(
                    f"num_relations_mol={num_rel_mol} but observed "
                    f"sp_edge_rel requires >= {mol_rel_required}"
                )
        self._effective_num_rel_mol = num_rel_mol

        # ── 2. BKG + per-drug subgraphs (only when dual-channel) ──────
        if self.mol_only:
            num_rel_graph = 2
            n_total_nodes = 2
            max_deg_node = 2
        else:
            graph_rel_required, max_deg_node_obs, n_total_nodes = (
                self._build_bkg_and_subgraphs(train, verbose=True)
            )
            if self.num_relations_graph is None:
                num_rel_graph = max(graph_rel_required + 4, 32)
            else:
                num_rel_graph = self.num_relations_graph
                if num_rel_graph < graph_rel_required:
                    raise ValueError(
                        f"num_relations_graph={num_rel_graph} but observed "
                        f"max sp_rel requires >= {graph_rel_required}"
                    )
            max_deg_node = max(self.max_degree_node, max_deg_node_obs + 1)
        self._effective_num_rel_graph = num_rel_graph

        # ── 3. Build model ────────────────────────────────────────────
        self._model = TIGER(
            max_layer=self.max_layer,
            num_features_drug=ATOM_FEATURE_DIM,
            num_nodes=n_total_nodes,
            num_relations_mol=num_rel_mol,
            num_relations_graph=num_rel_graph,
            output_dim=self.output_dim,
            max_degree_graph=self.max_degree_graph,
            max_degree_node=max_deg_node,
            sub_coeff=self.sub_coeff,
            mi_coeff=self.mi_coeff,
            dropout=self.dropout,
            device=self.device,
            mol_only=self.mol_only,
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        pos = train.splits.train.copy()
        best_val_auc = -1.0
        best_state: dict | None = None

        # Training progress logger (per CLAUDE.md 训练进度/eval/save 规范)
        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        except ImportError:
            import sys as _sys
            from pathlib import Path as _P
            _sys.path.insert(0, str(_P(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        n_pos_plus_neg = 2 * len(pos)
        steps_per_epoch = (n_pos_plus_neg + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[tiger] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        ckpt_root = (
            (self.run_dir / "checkpoints") if self.run_dir is not None else None
        )

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
            # Per-epoch fresh negatives via deterministic sub-seed
            # ``base_seed + 1000 + epoch`` (aligned with hdn_ddi / emergnn /
            # ssi_ddi / dsn_ddi; see CLAUDE.md §"Baseline 规范" §4).
            # The legacy PKL bundle's ``train_neg_epochs`` only ships
            # 4 fixed epochs (and tne[0]..tne[3] are the same pair-set),
            # so ``regenerate=False`` would lock first 4 epochs to a
            # near-fixed neg pool — not what we want for 100-epoch runs.
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
                mol1, sub1, mol2, sub2, idx1, idx2, mask = self._make_pair_batch(
                    batch, labels=batch["label"].to_numpy()
                )
                if mol1 is None:
                    continue
                opt.zero_grad(set_to_none=True)
                _probs, loss = self._forward(mol1, sub1, mol2, sub2, idx1, idx2)
                loss.backward()
                opt.step()
                progress.step(loss.item())

                # mid-epoch eval + save
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

            # end-of-epoch eval + save
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

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[tiger] loaded best val_auc={best_val_auc:.4f} state",
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
        if self._model is None or self._mol_graphs is None:
            raise RuntimeError(
                "TIGERBaseline must be fitted (or loaded) before prediction."
            )
        self._model.eval()
        out = np.full(len(pairs), 0.5, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            zero_labels = np.zeros(len(batch), dtype=np.int64)
            mol1, sub1, mol2, sub2, idx1, idx2, mask = self._make_pair_batch(
                batch, labels=zero_labels
            )
            if mol1 is None:
                continue
            probs, _loss = self._forward(mol1, sub1, mol2, sub2, idx1, idx2)
            # ``probs`` is (B, n_classes) softmax. For binary take class-1
            # column; multi-class subclass overrides this method.
            probs = probs[:, 1].detach().cpu().numpy()
            out_slice = np.full(len(batch), 0.5, dtype=np.float32)
            out_slice[mask] = probs
            out[start : start + len(batch)] = out_slice
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._mol_graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump(
                {
                    "mol_graphs": self._mol_graphs,
                    "mol_missing": self._mol_missing,
                    "subgraphs": self._subgraphs,
                    "drug_to_idx": self._drug_to_idx,
                    "unseen_ids": list(self._unseen_ids),
                    "n_total_nodes": (
                        self._bkg["n_total_nodes"] if self._bkg else 2
                    ),
                    "num_rel_graph": self._effective_num_rel_graph,
                },
                f,
            )
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "hyperparameters": {
                    "max_layer": self.max_layer,
                    "output_dim": self.output_dim,
                    "max_degree_graph": self.max_degree_graph,
                    "max_degree_node": self.max_degree_node,
                    "num_relations_mol": self._effective_num_rel_mol,
                    "num_relations_graph": self._effective_num_rel_graph,
                    "sub_coeff": self.sub_coeff,
                    "mi_coeff": self.mi_coeff,
                    "dropout": self.dropout,
                    "mol_only": self.mol_only,
                    "kg_source": self.kg_source,
                    "rw_num_walks": self.rw_num_walks,
                    "rw_walk_length": self.rw_walk_length,
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
                    "n_drugs_with_mol_graph": len(self._mol_graphs),
                    "n_drugs_missing_mol_graph": len(self._mol_missing),
                    "n_drugs_with_subgraph": (
                        len(self._subgraphs) if self._subgraphs else 0
                    ),
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "TIGERBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        inst = cls(**hparams)
        with (p / "graphs.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._mol_graphs = payload["mol_graphs"]
        inst._mol_missing = payload.get("mol_missing", [])
        inst._subgraphs = payload.get("subgraphs")
        inst._drug_to_idx = payload.get("drug_to_idx")
        inst._unseen_ids = set(payload.get("unseen_ids", []))
        inst._effective_num_rel_mol = hparams.get("num_relations_mol")
        inst._effective_num_rel_graph = hparams.get("num_relations_graph")
        n_total_nodes = payload.get("n_total_nodes", 2)
        inst._model = TIGER(
            max_layer=inst.max_layer,
            num_features_drug=ATOM_FEATURE_DIM,
            num_nodes=n_total_nodes,
            num_relations_mol=inst._effective_num_rel_mol,
            num_relations_graph=inst._effective_num_rel_graph,
            output_dim=inst.output_dim,
            max_degree_graph=inst.max_degree_graph,
            max_degree_node=inst.max_degree_node,
            sub_coeff=inst.sub_coeff,
            mi_coeff=inst.mi_coeff,
            dropout=inst.dropout,
            device=inst.device,
            mol_only=inst.mol_only,
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()
        return inst
