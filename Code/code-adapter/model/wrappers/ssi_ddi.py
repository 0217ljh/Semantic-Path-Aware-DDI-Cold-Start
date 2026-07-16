"""SSIDDIRankWrapper - SSI-DDI (molecular substructure paradigm) backbone plugged
into the code-adapter harness. Written fresh (no rank_analysis wrapper existed),
reusing the verified SSI-DDI model/featurizer (baseline.ssi_ddi): GAT substructure
blocks + co-attention + RESCAL head, per-drug PyG graphs from SMILES.

SSI-DDI is KG-FREE: it scores a pair purely from the two drugs' molecular graphs and
does NOT use the KG DDI facts. So encode_pairs IGNORES the ScoringContext -> leak-free
by construction (a pair's own DDI edge is never an input). known_drugs = drugs with a
parseable molecular graph, so the harness drops pairs whose drug lacks SMILES. Binary
uses SSI_DDI(rel_total=1); multiclass uses SSI_DDI_MC.forward_all -> (B,K). Native P1.

Needs hp['dataset']+hp['fold'] (to load the leaf for per-drug SMILES); the entrypoint
injects them. "wrapper" = harness glue, NOT the M_A·M_B adapter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "Code"))
sys.path.insert(0, str(_ROOT / "Code" / "scripts"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402
from data_utils.unified import DATASET_DIRS  # noqa: E402
from baseline.ssi_ddi.mol_features import ATOM_FEATURE_DIM, build_drug_graphs  # noqa: E402
from baseline.ssi_ddi.models import SSI_DDI  # noqa: E402
from baseline.ssi_ddi.multi_cls.model import SSI_DDI_MC  # noqa: E402

from ..contracts import (EpochData, PairEncoding, RankModel, ScoringContext,  # noqa: E402
                         TrainEpochOutput)

#: tunable hyperparameters (exposed via hp). RESCAL requires head_out*n_heads==kge_dim
#: for every block (SSIDDIBaseline validates this).
_DEFAULTS = dict(hidd_dim=64, kge_dim=64, heads_out_feat_params=(32, 32, 32, 32),
                 blocks_params=(2, 2, 2, 2), lr=1e-3, weight_decay=5e-4,
                 batch_size=256, seed=42)


def _smiles_dict(ds) -> dict[str, str]:
    if ds.drugs is None or "smiles" not in ds.drugs.columns:
        raise ValueError("SSI-DDI requires the dataset's drugs table with a 'smiles' column")
    return {str(r["drugbank_id"]): ("" if pd.isna(r["smiles"]) else str(r["smiles"]))
            for _, r in ds.drugs[["drugbank_id", "smiles"]].iterrows()}


class SSIDDIRankWrapper(RankModel):
    def setup(self, task, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        dataset = self.hp["dataset"]; fold = self.hp["fold"]
        dir2group = {v: k for k, v in DATASET_DIRS.items()}
        if dataset not in dir2group:
            raise ValueError(f"dataset dir {dataset!r} not in DATASET_DIRS values")
        group = dir2group[dataset]
        leaf_task = "binary" if task.is_binary else "multiclass"
        unified_root = _ROOT / "Code" / "data" / "ddi_unified"
        leaf = U.load_leaf(str(unified_root), group, leaf_task, "cold_s2", fold)
        ds = make_dataset(leaf.train, leaf.val, leaf.resources, task=leaf_task)
        self._graphs, missing = build_drug_graphs(_smiles_dict(ds))
        if not self._graphs:
            raise ValueError("SSI-DDI built zero molecular graphs (check the smiles column)")
        if missing:
            print(f"[ssi_ddi] {len(missing)} drugs skipped (no parseable SMILES); their "
                  f"pairs are filtered by the harness known-drug filter", flush=True)

        common = dict(in_features=ATOM_FEATURE_DIM, hidd_dim=self.hp["hidd_dim"],
                      kge_dim=self.hp["kge_dim"],
                      heads_out_feat_params=list(self.hp["heads_out_feat_params"]),
                      blocks_params=list(self.hp["blocks_params"]))
        if task.is_binary:
            self.model = SSI_DDI(rel_total=1, **common).to(self.device)
        else:
            self.model = SSI_DDI_MC(rel_total=task.n_classes, **common).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["lr"],
                                    weight_decay=self.hp["weight_decay"])

    # -- molecular pair batch (all pairs have graphs after the known-drug filter) --
    def _pair_batch(self, pairs: np.ndarray):
        h = Batch.from_data_list([self._graphs[str(a)].clone() for a, _ in pairs]).to(self.device)
        t = Batch.from_data_list([self._graphs[str(b)].clone() for _, b in pairs]).to(self.device)
        return h, t

    def _logits(self, h, t):
        if self.task.is_binary:
            rels = torch.zeros(h.num_graphs, dtype=torch.long, device=self.device)
            return self.model((h, t, rels))                 # (B,)
        return self.model.forward_all((h, t))               # (B, K)

    # -- RankModel contract (KG-free: context ignored) -----------------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        pairs = np.asarray(epoch.target_pairs)
        y = np.asarray(epoch.target_labels)
        n = len(y)
        if n == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)
        order = rng.permutation(n)
        pairs, y = pairs[order], y[order]
        bs = int(self.hp["batch_size"]); self.model.train(); total = 0.0
        for s in range(0, n, bs):
            pb, yb = pairs[s:s + bs], y[s:s + bs]
            h, t = self._pair_batch(pb)
            yt = torch.as_tensor(yb, device=self.device)
            self.opt.zero_grad(set_to_none=True)
            logits = self._logits(h, t)
            if self.task.is_binary:
                loss = F.binary_cross_entropy_with_logits(logits, yt.float())
            else:
                loss = F.cross_entropy(logits, yt.long())
            loss.backward(); self.opt.step()
            total += float(loss.item()) * len(yb)
        return TrainEpochOutput(mean_loss=total / max(n, 1), n_targets=n)

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        self.model.eval()
        bs = int(self.hp["batch_size"]); logits = []
        for s in range(0, len(pairs), bs):
            h, t = self._pair_batch(pairs[s:s + bs])
            logits.append(self._logits(h, t).float().cpu().numpy())
        out = (np.concatenate(logits) if logits
               else np.zeros((0,) if self.task.is_binary else (0, self.task.n_classes), np.float32))
        return PairEncoding(pair_ids=pairs, pair_repr=None, logits=out.astype(np.float32),
                            repr_kind="mol_substructure", repr_stage="rescal_logit")

    # -- adapter-composition interface (molecular EXCEPTION: correct DOWNWARD at the
    #    per-GAT-block substructure level, since SSI-DDI is molecular-only with no KG channel) --
    def _block_reprs(self, h_data, t_data):
        """Per-GAT-block substructure embeddings (repr_h, repr_t), each [B, L, kge_dim], by
        replicating SSI_DDI.forward's block loop (models.py:54-72) HERE -- does NOT touch the
        byte-exact model, uses model.blocks / initial_norm / net_norms (public). Stops BEFORE
        co_attention + RESCAL (the native interaction head, which the composer head replaces).
        Grad-enabled; respects caller train/eval mode. Works for binary SSI_DDI + SSI_DDI_MC
        (MC subclasses SSI_DDI, same block attributes)."""
        m = self.model
        h_data.x = m.initial_norm(h_data.x, h_data.batch)
        t_data.x = m.initial_norm(t_data.x, t_data.batch)
        repr_h, repr_t = [], []
        for i, block in enumerate(m.blocks):
            out1, out2 = block(h_data), block(t_data)
            h_data, t_data = out1[0], out2[0]
            repr_h.append(out1[1]); repr_t.append(out2[1])
            h_data.x = F.elu(m.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(m.net_norms[i](t_data.x, t_data.batch))
        return torch.stack(repr_h, dim=-2), torch.stack(repr_t, dim=-2)   # [B,L,d], [B,L,d]

    def _block_streams(self, pairs) -> list:
        """Grad-enabled per-block pair streams [ [repr_h_i || repr_t_i], .. ], each [N, 2*kge_dim],
        one per GAT block. Batched over hp.batch_size and concatenated (pair order preserved)."""
        pairs = np.asarray(pairs)
        bs = int(self.hp["batch_size"])
        L = self.model.n_blocks
        chunks = [[] for _ in range(L)]
        for s in range(0, len(pairs), bs):
            h, t = self._pair_batch(pairs[s:s + bs])
            rh, rt = self._block_reprs(h, t)                 # [b, L, d] each
            for i in range(L):
                chunks[i].append(torch.cat([rh[:, i, :], rt[:, i, :]], dim=-1))   # [b, 2d]
        return [torch.cat(c, dim=0) for c in chunks] if chunks[0] else \
            [torch.zeros((0, 2 * self.model.kge_dim), device=self.device) for _ in range(L)]

    def pair_forward(self, pairs, context: ScoringContext) -> "torch.Tensor":
        """Grad-enabled pre-scorer pair rep [N, 2*L*kge_dim] = concat of the per-block streams
        ([repr_h||repr_t] over all L GAT blocks), for design-R composition. The native co-
        attention + RESCAL head is dropped (the composer head replaces it, as for every
        backbone). ScoringContext ignored (molecular-only, leak-free) like encode_pairs."""
        return torch.cat(self._block_streams(pairs), dim=-1)     # [N, 2*L*d]

    def pair_streams(self, pairs, context: ScoringContext) -> "list":
        """Design-R 'default' streams: one per GAT block (SSI-DDI's 4 substructure blocks), each
        [N, 2*kge_dim] = [repr_h_i || repr_t_i]. This is the molecular 'downward' correction (at
        the substructure-block level). cat(pair_streams) == pair_forward by construction."""
        return self._block_streams(pairs)

    def effective_hp(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.hp.items()}

    def known_drugs(self) -> set:
        return set(self._graphs.keys())

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)


__all__ = ["SSIDDIRankWrapper"]
