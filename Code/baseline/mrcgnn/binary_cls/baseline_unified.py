"""MRCGNN BINARY baseline for the UNIFIED benchmark (case-B added task).

MRCGNN's paper is multiclass-only; BINARY is a case-B task (like the other molecular
baselines). One leaf = one regime -> a SINGLE ``MRCGNNBinaryBaseline`` (single-relation
RGCN over the leaf's train POSITIVES + View-A DGI contrastive + TrimNet molecular skip;
BCE over pos + per-epoch negatives; best-ckpt by val AUROC). KG-FREE (molecular + DDI
graph), so applies to every dataset. `predict` returns (n,) P(interaction).

TrimNet FEATURES come from the SIBLING MULTICLASS leaf (coordinator+codex decision Q3):
the TrimNet builder is a SUPERVISED multiclass event trainer, so a binary-supervised
TrimNet would be a DIFFERENT featurizer. This wrapper locates the sibling multiclass leaf
(same dataset / regime / split_code / fold), reads its train positives (`y_cls ->
ddi_type`) + its K (`labels.n_labels`), and injects them into the binary core so binary
and multiclass SHARE the same event-supervised TrimNet cache. Cold-start safe: only that
fold's train rows feed TrimNet. If the sibling MC leaf is ABSENT, we RAISE (no silent
fallback to a binary-supervised TrimNet).

The sibling MC leaf dir is derived from the binary leaf's ``resources.fold_dir`` by
swapping the task folder ``binary_cls -> multi_cls`` (unified.TASK_DIRS, data_utils/
unified.py:37-38). This avoids re-deriving the dataset_group->dir map and stays valid
regardless of dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

import pandas as pd  # noqa: E402

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.mrcgnn.binary_cls.baseline import MRCGNNBinaryBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import numpy as np

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]


def _sibling_mc_fold_dir(binary_fold_dir: Path) -> Path:
    """Map the BINARY leaf's fold_dir to the sibling MULTICLASS leaf's fold_dir by
    swapping the task folder (binary_cls -> multi_cls). fold_dir layout:
    <root>/<task_dir>/<dataset>/<regime>/<split_code>/<fold> (unified_loader.py:47)."""
    parts = list(binary_fold_dir.resolve().parts)
    # replace the LAST occurrence of 'binary_cls' with 'multi_cls' (task folder segment).
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "binary_cls":
            parts[i] = "multi_cls"
            return Path(*parts)
    raise ValueError(f"cannot locate 'binary_cls' segment in fold_dir {binary_fold_dir}")


def _load_sibling_mc_supervision(resources: "LeafResources"):
    """Return (train_tri, n_classes, source_str) from the sibling multiclass leaf, or
    raise if it does not exist (decision Q3: no silent fallback)."""
    fold_dir = getattr(resources, "fold_dir", None)
    if fold_dir is None:
        raise ValueError(
            "MRCGNN binary needs resources.fold_dir to locate the sibling multiclass "
            "leaf for TrimNet supervision (decision Q3); it was None.")
    mc_fold_dir = _sibling_mc_fold_dir(Path(fold_dir))
    mc_train = mc_fold_dir / "train.parquet"
    mc_meta = mc_fold_dir.parent / "meta.json"      # meta lives at the leaf-parent level
    if not mc_train.is_file() or not mc_meta.is_file():
        raise FileNotFoundError(
            "MRCGNN binary is UNSUPPORTED for this leaf: the sibling multiclass leaf is "
            f"absent (need {mc_train} + {mc_meta}). TrimNet features must come from the "
            "sibling MC leaf's event supervision (decision Q3); no binary-supervised "
            "fallback is allowed.")
    meta = json.loads(mc_meta.read_text())
    n_classes = int(meta["labels"]["n_labels"])
    df = pd.read_parquet(mc_train, columns=["drug_a_id", "drug_b_id", "y_cls"])
    train_tri = [(str(a), str(b), int(r))
                 for a, b, r in zip(df["drug_a_id"], df["drug_b_id"], df["y_cls"])]
    return train_tri, n_classes, str(mc_fold_dir)


@register_unified("mrcgnn")
class MRCGNNUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(self, *, n_epochs: int = 100, batch_size: int = 256, device: str = "auto",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.run_dir = run_dir
        self._core: MRCGNNBinaryBaseline | None = None

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        train_tri, n_classes, source = _load_sibling_mc_supervision(resources)
        self._core = MRCGNNBinaryBaseline(
            n_epochs=self.n_epochs, batch_size=self.batch_size, device=self.device,
            fold=resources.meta.get("split_code", 0), run_dir=self.run_dir,
            trimnet_train_tri=train_tri, trimnet_n_classes=n_classes,
            trimnet_source=source)
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="binary")
        self._core.fit(ds, ds)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[_PAIR])


__all__ = ["MRCGNNUnifiedBinary"]
