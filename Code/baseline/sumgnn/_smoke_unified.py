"""Smoke test for the SumGNN unified wrappers on a SUBSAMPLED leaf (1 epoch).

Checks: build -> subgraph extraction on OUR merged KG -> fit -> predict shape/non-degeneracy.
NOT a metric gate. Multiclass on drugbank_latest_partial (transductive S0), multilabel on
twoside (transductive S0), both subsampled small.

Run:
  python Code/baseline/sumgnn/_smoke_unified.py --task multiclass --n 300 --test-n 120
  python Code/baseline/sumgnn/_smoke_unified.py --task multilabel --n 300 --test-n 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Code"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.unified_loader import Leaf, LeafResources  # noqa: E402


def _subsample_leaf(leaf, n: int, test_n: int, seed: int = 0):
    tr = leaf.train.sample(min(n, len(leaf.train)), random_state=seed).reset_index(drop=True)
    va = leaf.val.sample(min(max(50, n // 4), len(leaf.val)), random_state=seed).reset_index(drop=True)
    te = leaf.test.sample(min(test_n, len(leaf.test)), random_state=seed).reset_index(drop=True)
    return Leaf(task=leaf.task, dataset=leaf.dataset, regime=leaf.regime,
                split_code=leaf.split_code, fold=leaf.fold, train=tr, val=va, test=te,
                resources=leaf.resources)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["multiclass", "multilabel"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--test-n", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    if args.task == "multiclass":
        leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), "ddi800", "multiclass",
                           "transductive", "fold0")
        from baseline.sumgnn.multi_cls.baseline_unified import SumGNNUnifiedMulticlass as Cls
        n_out = int(leaf.resources.meta["labels"]["n_labels"])
    else:
        leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), "twosides", "multilabel",
                           "transductive", "fold0")
        from baseline.sumgnn.multi_label_cls.baseline_unified import SumGNNUnifiedMultilabel as Cls
        n_out = int(leaf.resources.meta["labels"]["n_labels"])

    sub = _subsample_leaf(leaf, args.n, args.test_n)
    print(f"[smoke] task={args.task} train={sub.train.shape} val={sub.val.shape} "
          f"test={sub.test.shape} n_out={n_out} regime={sub.regime}/{sub.split_code}", flush=True)

    model = Cls(n_epochs=args.epochs, device="auto", num_workers=6, max_links=args.n,
                eval_every_iter=10_000_000)
    model.fit_leaf(sub)
    scores = np.asarray(model.predict(sub.test))
    print(f"[smoke] predict shape={scores.shape} expected=({len(sub.test)},{n_out})", flush=True)
    assert scores.shape == (len(sub.test), n_out), "predict shape mismatch"
    nz = int((scores.sum(axis=1) > 1e-6).sum())
    print(f"[smoke] non-degenerate rows (mass>0): {nz}/{len(sub.test)}  "
          f"score[min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.5f}]", flush=True)
    print(f"[smoke-{args.task}] PASS shape_ok non_degenerate={nz>0}", flush=True)


if __name__ == "__main__":
    main()
