"""TIGERRankWrapper - TIGER (dual-channel: molecular GraphTransformer + fixed
biomedical-KG DeepWalk subgraph) backbone plugged into the code-adapter harness.
Written fresh (no rank_analysis wrapper existed), reusing the verified paper-faithful
TIGER core (baseline.tiger.binary_cls.TIGERBaseline / multi_cls.TIGERMulticlassBaseline)
UNCHANGED: it drives the core's setup helpers + model per-epoch, but owns the protocol
epoch loop itself (the core's own fit() is NOT called).

TIGER is KG-FREE w.r.t. the DDI facts: it scores a pair from the two drugs' molecular
graphs + their FIXED biomedical-KG (BKG) subgraphs (BKG carries merged-KG BIO edges, NO
DDI edges). So encode_pairs and train_epoch IGNORE the ScoringContext -> leak-free by
construction (a pair's own DDI edge is never an input), exactly like ssi_ddi. The BKG is
loaded directly from the merged-KG parquet (hp['merged_kg_path']-overridable, mirroring
emergnn's MERGED_EDGES), bypassing leaf.kg. known_drugs = drugs with BOTH a molecular
graph AND a BKG subgraph, so the harness drops uncovered pairs and _make_pair_batch's
keep-mask is all-True.

The 2-class (binary) / K-class (multiclass) softmax head is converted to logits so the
harness metrics (sigmoid for binary, softmax for multi) recover the same probabilities.

Needs hp['dataset']+hp['fold'] (to load the leaf for per-drug SMILES + the drug pool);
the entrypoint injects them. "wrapper" = harness glue, NOT the M_A.M_B adapter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "Code"))
sys.path.insert(0, str(_ROOT / "Code" / "scripts"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402
from data_utils.unified import DATASET_DIRS  # noqa: E402
from baseline.tiger.binary_cls.baseline import TIGERBaseline  # noqa: E402
from baseline.tiger.multi_cls.baseline import TIGERMulticlassBaseline  # noqa: E402
from baseline.tiger.mol_features import ATOM_FEATURE_DIM  # noqa: E402
from baseline.tiger.model import TIGER  # noqa: E402

from ..contracts import (EpochData, PairEncoding, RankModel, ScoringContext,  # noqa: E402
                         TrainEpochOutput)

#: merged-KG edge file under Code/data/KG/_merged_kg/ (dataset-independent), same as emergnn
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"

#: pair-frame columns the TIGER core's _make_pair_batch reads
_PAIR = ["drug_a_id", "drug_b_id"]

#: tunable hyperparameters (exposed via hp). Defaults mirror TIGERBaseline.__init__
#: (binary_cls/baseline.py:93-221); extractor="randomWalk" = paper TIGER-DW default.
_DEFAULTS = dict(
    max_layer=4, output_dim=64, max_degree_graph=100, max_degree_node=100,
    sub_coeff=0.2, mi_coeff=0.5, dropout=0.2,
    extractor="randomWalk", khop=2, khop_fanout=4, prob_fixed_num=32,
    rw_num_walks=1, rw_walk_length=32,
    learning_rate=1e-3, weight_decay=5e-4, batch_size=64, seed=42,
)


class TIGERRankWrapper(RankModel):
    # -- build ---------------------------------------------------------------
    def setup(self, task, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_str = "cuda" if self.device.type == "cuda" else "cpu"
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
        ds = make_dataset(leaf.train, leaf.val, leaf.resources, task=leaf_task)
        # The core's _build_bkg_and_subgraphs reads splits.test_s2 in a DEAD all_drugs
        # union (binary_cls/baseline.py:307-309); make_dataset's _LeafSplits has none.
        # g1_drugs|g2_drugs already covers every leaf drug, so an EMPTY test_s2 is
        # faithful (no universe change) — this shim only avoids the dead-code error
        # (same shim as baseline_unified.py:_shim_test_s2).
        if not hasattr(ds.splits, "test_s2"):
            ds.splits.test_s2 = pd.DataFrame(columns=_PAIR)

        merged_kg_path = Path(self.hp.get("merged_kg_path")
                              or _ROOT / "Code" / "data" / "KG" / "_merged_kg" / MERGED_EDGES)
        if not merged_kg_path.is_file():
            raise FileNotFoundError(
                f"merged KG edges not found at {merged_kg_path}; pass hp['merged_kg_path']")

        core_cls = TIGERBaseline if task.is_binary else TIGERMulticlassBaseline
        core_kw = dict(
            max_layer=self.hp["max_layer"], output_dim=self.hp["output_dim"],
            max_degree_graph=self.hp["max_degree_graph"],
            max_degree_node=self.hp["max_degree_node"],
            sub_coeff=self.hp["sub_coeff"], mi_coeff=self.hp["mi_coeff"],
            dropout=self.hp["dropout"],
            kg_source="merged", merged_kg_path=merged_kg_path,
            extractor=self.hp["extractor"], khop=self.hp["khop"],
            khop_fanout=self.hp["khop_fanout"], prob_fixed_num=self.hp["prob_fixed_num"],
            rw_num_walks=self.hp["rw_num_walks"], rw_walk_length=self.hp["rw_walk_length"],
            learning_rate=self.hp["learning_rate"], weight_decay=self.hp["weight_decay"],
            batch_size=self.hp["batch_size"],
            # HARNESS owns the epoch loop -> the core's own n_epochs is unused.
            n_epochs=1, device=dev_str,
            # paper-faithful default: cold-start center-node patch OFF (project extension).
            cold_start_patch=False,
        )
        # n_classes for the K-class softmax head (multiclass core only; binary head=2).
        self.n_classes = 2 if task.is_binary else int(task.n_classes)
        if not task.is_binary:
            core_kw["n_classes"] = self.n_classes
        self.core = core_cls(**core_kw)

        self._build_model(ds)

    def _build_model(self, ds) -> None:
        """Replicate the core fit()'s dim-sizing + TIGER(...) construction
        (binary_cls/baseline.py:468-521, multi_cls/baseline.py:105-159) by REUSING
        the core's setup helpers, then own the optimizer."""
        core = self.core
        # -- mol channel dims (populates core._mol_graphs / _mol_missing) --
        mol_rel_required = core._build_mol_graphs(ds)
        if core.num_relations_mol is None:
            num_rel_mol = max(mol_rel_required + 4, 32)
        else:
            num_rel_mol = core.num_relations_mol
        core._effective_num_rel_mol = num_rel_mol

        # -- BKG + per-drug subgraph dims (populates core._bkg / _subgraphs /
        #    _drug_to_idx / _unseen_ids) --
        graph_rel_required, max_deg_node_obs, n_total_nodes = (
            core._build_bkg_and_subgraphs(ds, verbose=True)
        )
        if core.num_relations_graph is None:
            num_rel_graph = max(graph_rel_required + 4, 32)
        else:
            num_rel_graph = core.num_relations_graph
        max_deg_node = max(core.max_degree_node, max_deg_node_obs + 1)
        core._effective_num_rel_graph = num_rel_graph

        model = TIGER(
            max_layer=core.max_layer,
            num_features_drug=ATOM_FEATURE_DIM,
            num_nodes=n_total_nodes,
            num_relations_mol=num_rel_mol,
            num_relations_graph=num_rel_graph,
            output_dim=core.output_dim,
            max_degree_graph=core.max_degree_graph,
            max_degree_node=max_deg_node,
            sub_coeff=core.sub_coeff,
            mi_coeff=core.mi_coeff,
            dropout=core.dropout,
            device=core.device,
            mol_only=core.mol_only,
            n_classes=self.n_classes,
        ).to(self.device)
        core._model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=core.learning_rate,
                                    weight_decay=core.weight_decay)

    # -- helpers -------------------------------------------------------------
    def _pair_frame(self, pairs: np.ndarray) -> pd.DataFrame:
        pairs = np.asarray(pairs)
        return pd.DataFrame({"drug_a_id": pairs[:, 0].astype(str),
                             "drug_b_id": pairs[:, 1].astype(str)})

    # -- RankModel contract (KG-free: ScoringContext ignored) ----------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        pairs = np.asarray(epoch.target_pairs)
        y = np.asarray(epoch.target_labels)
        n = len(y)
        if n == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)
        order = rng.permutation(n)
        pairs, y = pairs[order], y[order]
        df = self._pair_frame(pairs)
        bs = int(self.hp["batch_size"])
        self.core._model.train()
        total = 0.0
        for s in range(0, n, bs):
            batch_df = df.iloc[s:s + bs]
            batch_labels = y[s:s + bs]
            mol1, sub1, mol2, sub2, idx1, idx2, mask = self.core._make_pair_batch(
                batch_df, labels=batch_labels)
            if mol1 is None:                       # no scoreable pair in this slice
                continue
            self.opt.zero_grad(set_to_none=True)
            # The model computes its OWN loss (softmax CE/NLL + MI) internally.
            _probs, loss = self.core._forward(mol1, sub1, mol2, sub2, idx1, idx2)
            loss.backward()
            self.opt.step()
            total += float(loss.item()) * len(batch_df)
        return TrainEpochOutput(mean_loss=total / max(n, 1), n_targets=n)

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        self.core._model.eval()
        n = len(pairs)
        df = self._pair_frame(pairs)
        bs = int(self.hp["batch_size"])
        if self.task.is_binary:
            out = np.zeros((n,), dtype=np.float32)                 # dropped -> 0 logit (p=0.5)
        else:
            out = np.zeros((n, self.n_classes), dtype=np.float32)  # dropped -> uniform softmax
        for s in range(0, n, bs):
            batch_df = df.iloc[s:s + bs]
            zero_labels = np.zeros(len(batch_df), dtype=np.int64)
            mol1, sub1, mol2, sub2, idx1, idx2, mask = self.core._make_pair_batch(
                batch_df, labels=zero_labels)
            if mol1 is None:
                continue
            probs, _loss = self.core._forward(mol1, sub1, mol2, sub2, idx1, idx2)
            probs = probs.detach().cpu().numpy()                   # (B_kept, n_classes) softmax
            if self.task.is_binary:
                p1 = np.clip(probs[:, 1], 1e-6, 1.0 - 1e-6)
                kept_logits = np.log(p1) - np.log(1.0 - p1)        # (B_kept,)
                slot = np.zeros((len(batch_df),), dtype=np.float32)
            else:
                kept_logits = np.log(np.clip(probs, 1e-9, 1.0))    # (B_kept, K)
                slot = np.zeros((len(batch_df), self.n_classes), dtype=np.float32)
            slot[mask] = kept_logits
            out[s:s + len(batch_df)] = slot
        return PairEncoding(pair_ids=pairs, pair_repr=None, logits=out.astype(np.float32),
                            repr_kind="tiger_dualchannel", repr_stage="softmax_to_logit")

    @property
    def model(self):
        """The trainable nn.Module (composer contract). TIGER keeps it at core._model."""
        return self.core._model

    # -- adapter-composition interface (BKG-KG channel ONLY; molecular skipped) ----------
    def _kg_pair_batched(self, pairs) -> "torch.Tensor":
        """Grad-enabled KG-only pre-scorer pair rep [N, 2*output_dim] = [drug1_node_emb ||
        drug2_node_emb], SKIPPING the molecular channel per the multimodal integration rule.
        Runs ONLY the model's BKG branch (drug_node_feature -> node_representation_learning);
        the molecular GraphTransformer + the fc1/fc2 fusion are never invoked, so molecular +
        fusion params receive no gradient in joint mode (molecular left untouched, by design).
        Reuses core._make_pair_batch for faithful subgraph batching (its molecular tensors are
        built but unused). The cold-start center-node patch is NOT applied (it mixes molecular;
        the wrapper runs cold_start_patch=False). Respects the caller's train/eval mode."""
        m = self.core._model
        if getattr(m, "mol_only", False):
            raise NotImplementedError("TIGER KG-channel rep needs mol_only=False (no BKG branch)")
        pairs = np.asarray(pairs)
        df = self._pair_frame(pairs)
        bs = int(self.hp["batch_size"]); outs = []
        for s in range(0, len(pairs), bs):
            batch_df = df.iloc[s:s + bs]
            zero = np.zeros(len(batch_df), dtype=np.int64)
            mol1, sub1, mol2, sub2, idx1, idx2, mask = self.core._make_pair_batch(batch_df, labels=zero)
            if mol1 is None or not bool(np.asarray(mask).all()):
                raise ValueError("TIGER pair_forward/pair_streams got unscoreable/OOV pairs; "
                                 "the composer must filter to known_drugs() first")
            sub1 = sub1.to(self.device); sub2 = sub2.to(self.device)   # core._forward moves too
            nf1 = m.drug_node_feature(sub1); nf2 = m.drug_node_feature(sub2)
            node_emb1, _, _ = m.node_representation_learning(nf1, sub1)   # BKG branch only
            node_emb2, _, _ = m.node_representation_learning(nf2, sub2)
            outs.append(torch.cat([node_emb1, node_emb2], dim=-1))       # [b, 2*output_dim]
        if not outs:
            raise ValueError("TIGER pair_forward: empty pairs")
        return torch.cat(outs, dim=0)

    def pair_forward(self, pairs, context: ScoringContext) -> "torch.Tensor":
        """Grad-enabled KG-only pair rep [N, 2*output_dim] for design-R composition (molecular
        channel dropped per the multimodal rule). ScoringContext ignored (BKG is fixed bio-only,
        leak-free) like encode_pairs."""
        return self._kg_pair_batched(pairs)

    def pair_streams(self, pairs, context: ScoringContext) -> "list":
        """Design-R 'default' streams. After dropping the molecular channel TIGER has a SINGLE
        BKG-KG channel with one pooled readout (the GraphTransformer's per-layer states are not
        exposed), so this is a single stream [kg_pair] -> 'default' degenerates to 'last' for
        TIGER. Returned as a 1-element list for the uniform composer interface.
        NOTE (codex 019f5d73): pair_forward and pair_streams each recompute _kg_pair_batched, and
        the KG GraphTransformer has train-mode dropout, so cat(pair_streams) != pair_forward across
        the two APIs at train time. Harmless: the composer calls exactly ONE per mode (last->
        pair_forward, default->pair_streams), never both, never expecting equality."""
        return [self._kg_pair_batched(pairs)]

    def effective_hp(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.hp.items()}

    def known_drugs(self) -> set:
        # Drugs the core can score a pair from: molecular graph AND a BKG subgraph
        # (mirrors the KEEP condition in _make_pair_batch, so the mask is all-True).
        idx = self.core._drug_to_idx or {}
        sub = self.core._subgraphs or {}
        return {d for d in self.core._mol_graphs
                if idx.get(d) is not None and sub.get(idx[d]) is not None}

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.core._model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.core._model.load_state_dict(state)


__all__ = ["TIGERRankWrapper"]
