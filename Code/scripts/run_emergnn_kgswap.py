"""idea2 v2 — run EmerGNN with a SWAPPED KG (lever-A KG-swap experiment, codex 019f1dd2).

Identical to run_baseline_unified's emergnn-binary path EXCEPT it overrides
leaf.resources.kg.source to a chosen KG-variant dir (each dir holds a file named
MERGED_EDGES) and sets a global seed, so we can compare {A1 old-flat, A2 v15-micro,
A3 v15-micro+meso-native, B v15-shortcut} under an otherwise-identical EmerGNN setup
(same split/optimizer/epochs/negatives/hidden/path-length — only the KG changes).

New file; changes no existing signatures. Writes results.json per (kg, seed).

Usage (WSL conda project_1, from project root; run ONLY when GPU free):
  PYTHONPATH=Code/code-idea-2 python -u Code/scripts/run_emergnn_kgswap.py \
      --kg-dir Code/data/KG/_v15_micro_meso_native --tag A3 --seed 0 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))

from data_utils import unified_loader as U  # noqa: E402
from baseline.unified_base import get_unified  # noqa: E402

_MODULE = "baseline.emergnn.binary_cls.baseline_unified"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _binary_metrics(y, scores) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"auroc": round(float(roc_auc_score(y, scores)), 4),
            "auprc": round(float(average_precision_score(y, scores)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", required=True, help="KG-variant dir holding the MERGED_EDGES parquet")
    ap.add_argument("--tag", required=True, help="short label e.g. A1/A2/A3/B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="ddi_full")
    ap.add_argument("--split", default="cold_s2")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--root", default=str(ROOT / "Code/data/ddi_unified"))
    args = ap.parse_args()

    _seed_everything(args.seed)
    __import__(_MODULE)
    cls = get_unified("emergnn", "binary")

    kg_dir = Path(args.kg_dir)
    if not kg_dir.is_absolute():
        kg_dir = (ROOT / kg_dir).resolve()
    if not kg_dir.is_dir():
        raise SystemExit(f"kg-dir not found: {kg_dir}")

    run_id = f"emergnn_kgswap__{args.tag}__seed{args.seed}"
    run_dir = ROOT / "Code/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[kgswap] loading leaf {args.dataset}/binary/{args.split}/{args.fold} | kg={kg_dir.name} seed={args.seed}", flush=True)
    leaf = U.load_leaf(args.root, args.dataset, "binary", args.split, args.fold)
    # ---- override KG source (robust to frozen dataclass) ----
    try:
        leaf.resources.kg.source = str(kg_dir)
    except Exception:
        object.__setattr__(leaf.resources.kg, "source", str(kg_dir))
    if not getattr(leaf.resources.kg, "present", False):
        object.__setattr__(leaf.resources.kg, "present", True)
    assert str(kg_dir) == str(leaf.resources.kg.source), "kg.source override failed"
    print(f"[kgswap] kg.source -> {leaf.resources.kg.source}", flush=True)
    print(f"[kgswap] train={leaf.train.shape} val={leaf.val.shape} test={leaf.test.shape}", flush=True)

    model = cls(n_epochs=args.epochs, device=args.device, run_dir=run_dir)
    t0 = time.time()
    model.fit_leaf(leaf)
    fit_s = time.time() - t0
    scores = np.asarray(model.predict(leaf.test))
    metrics = _binary_metrics(leaf.test["y_bin"].to_numpy(), scores)

    results = {"run_id": run_id, "tag": args.tag, "kg_dir": str(kg_dir), "seed": args.seed,
               "dataset": args.dataset, "split": args.split, "fold": args.fold,
               "epochs": args.epochs, "fit_time_s": round(fit_s, 1), "metrics": metrics}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[kgswap] DONE {run_id} metrics={metrics} fit={fit_s:.1f}s -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
