"""MKG-FENN binary baseline for the UNIFIED benchmark (decision B).

Case-B binary (DDI-existence). One leaf = one regime -> a SINGLE regime-aware
``MKGFENNBinaryBaseline``:
  * S0 (transductive) -> the warm 4-channel model (``baseline.mkg_fenn.model.MKGFENN``,
    event_num=2, via the core's ``cold=False`` path).
  * S1/S2 (inductive) -> the cold 3-channel model + nearest-seen imputation
    (``baseline.mkg_fenn.multi_cls.model_cold``), with drug_sim1..4 + test_adj built
    from the leaf's seen/unseen drug split. (The dead binary adapter WRONGLY used the
    4ch warm model for cold; this does NOT repeat that.)

KG1 (drug->entity) is loaded from the NATIVE filtered tables at
``Code/data/KG/drugbank/filtered/`` via ``KnowledgeGraph.from_filtered_dir`` — NOT the
leaf's merged-KG ``resources.kg.source`` (MKG-FENN's KG1 is intrinsic drug side-info,
cold-start safe). KG2/KG3/KG4 are built from SMILES + train DDI inside the core
(kg_builder.build_all_kgs, UNCHANGED).

Negatives are the leaf adapter's per-epoch DETERMINISTIC train-negatives
(``get_train_negatives(epoch, regenerate=True)``, design B), identical across baselines.

``predict`` returns ``(n,)`` = P(interaction) aligned to test rows.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.mkg_fenn.binary_cls.baseline import (  # noqa: E402
    MKGFENNBinaryBaseline, PAPER_HYPERPARAMS,
)
from data_utils.kg import KnowledgeGraph  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]

#: Native KG1 source (drug->{enzyme,target,transporter,carrier,pathway}) — intrinsic
#: drug side-info, NOT the merged triple KG. Relative to the repo root.
_KG1_FILTERED_DIR = "Code/data/KG/drugbank/filtered"


def _project_root(start: Path) -> Path:
    """Walk up to the repo root (dir holding Code/data/KG)."""
    p = Path(start).resolve()
    for c in [p, *p.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    return p


@register_unified("mkg_fenn")
class MKGFENNUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(
        self, *, n_epochs: int = 50, embedding_num: int = 128,
        neighbor_sample_size: int = 6, dropout: float = 0.3,
        learning_rate: float = 1e-2, weight_decay: float = 1e-8,
        batch_size: int = 256, seed: int = 1, device: str = "auto",
        log_step_every: int = 50, run_dir: "str | Path | None" = None,
        **_ignored,
    ) -> None:
        self.n_epochs = n_epochs
        self.embedding_num = embedding_num
        self.neighbor_sample_size = neighbor_sample_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: MKGFENNBinaryBaseline | None = None

    @staticmethod
    def _load_kg1(resources: "LeafResources") -> KnowledgeGraph:
        """Load native KG1 tables from the filtered dir (not resources.kg.source)."""
        root = _project_root(Path(__file__))
        kg_dir = root / _KG1_FILTERED_DIR
        if not kg_dir.is_dir():
            raise FileNotFoundError(f"native KG1 filtered dir not found: {kg_dir}")
        return KnowledgeGraph.from_filtered_dir(kg_dir)

    @staticmethod
    def _is_cold(resources: "LeafResources") -> bool:
        """Cold iff inductive regime (S1/S2). Transductive S0 -> warm."""
        return str(resources.meta.get("regime", "")).lower() == "inductive"

    def _build_dict1(self, resources: "LeafResources") -> dict[str, int]:
        """All drugs -> contiguous idx (sorted). Covers seen + unseen (keeps cold
        drugs reachable; matches core._build_dict1 / baseline.py:_build_dict1)."""
        drug_ids = set(resources.drugs["drug_id"].astype(str))
        return {did: i for i, did in enumerate(sorted(drug_ids))}

    @staticmethod
    def _unseen_ids(resources: "LeafResources", dict1: dict[str, int]) -> list[int]:
        """Unseen (test/eval) drug ids in dict1 index space (cold regimes)."""
        roles = resources.drug_split
        g2 = roles[roles["role"].isin(["test", "val", "eval_unseen"])]["drug_id"].astype(str)
        return sorted({dict1[d] for d in g2 if d in dict1})

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        cold = self._is_cold(resources)
        dict1 = self._build_dict1(resources)
        unseen_ids = self._unseen_ids(resources, dict1) if cold else []
        kg1 = self._load_kg1(resources)
        print(f"[mkg_fenn_bin] regime={resources.meta.get('regime')} "
              f"split={resources.meta.get('split_code')} cold={cold} "
              f"n_drug={len(dict1)} n_unseen={len(unseen_ids)}", flush=True)

        self._core = MKGFENNBinaryBaseline(
            cold=cold,
            embedding_num=self.embedding_num,
            neighbor_sample_size=self.neighbor_sample_size,
            dropout=self.dropout, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, batch_size=self.batch_size,
            n_epochs=self.n_epochs, seed=self.seed, device=self.device,
            log_step_every=self.log_step_every, run_dir=self.run_dir,
        )
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="binary")
        self._core.fit(ds, ds, kg=kg1, dict1=dict1, unseen_ids=unseen_ids)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[_PAIR])   # (n,) P(interaction)


__all__ = ["MKGFENNUnifiedBinary", "PAPER_HYPERPARAMS"]
