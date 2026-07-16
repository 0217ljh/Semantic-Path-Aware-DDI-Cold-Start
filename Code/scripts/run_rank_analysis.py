"""Unified CLI for the cold-start (S2) rank-analysis pipeline.

One command = one (model x dataset x task) run through the shared harness:
  python Code/scripts/run_rank_analysis.py --model rgcn --dataset drugbank_ryu \
         --task multi_cls --fold fold0 --epochs 1500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))

from my_code.rank_analysis import (TaskSpec, S2ProtocolSpec, RankHarness,  # noqa: E402
                                   load_rank_data)
from my_code.rank_analysis.wrappers import WRAPPERS  # noqa: E402


def _multiclass_k(dataset: str, fold: str) -> int:
    p = (ROOT / "Code/data/ddi_unified/multi_cls" / dataset / "inductive" / "S2" / fold
         / "train.parquet")
    tr = pd.read_parquet(p)
    return int(tr[tr["y_cls_train"] >= 0]["y_cls_train"].max()) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(WRAPPERS.keys()))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True, choices=["binary", "multi_cls"])
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.task == "binary":
        task = TaskSpec.binary()
    else:
        K = _multiclass_k(args.dataset, args.fold)
        task = TaskSpec.multiclass(K)
        print(f"[task] multiclass K={K}", flush=True)

    data = load_rank_data(args.dataset, task, args.fold)
    if not task.is_binary:                                   # fail loud on out-of-range labels
        for nm, lab in [("val", data.cold_val_labels), ("test", data.cold_test_labels)]:
            assert (lab >= 0).all() and (lab < task.n_classes).all(), \
                f"cold {nm} label out of [0,{task.n_classes})"
    print(f"[data] {args.dataset}/{args.task}/{args.fold}: train={len(data.train_pairs)} "
          f"cold_val={len(data.cold_val_pairs)} cold_test={len(data.cold_test_pairs)} "
          f"drugs={len(data.train_drugs)}", flush=True)

    model = WRAPPERS[args.model]()
    harness = RankHarness(S2ProtocolSpec(seed=args.seed))
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (ROOT / "Code" / "runs" /
               f"{ts}__rankpipe__{args.model}_{args.dataset}_{args.task}_{args.fold}__seed{args.seed}")
    results = harness.run(model, data, run_dir,
                          hp={"seed": args.seed, "dataset": args.dataset, "fold": args.fold},
                          epochs=args.epochs, eval_every=args.eval_every,
                          patience=args.patience)

    print(f"\n### {args.model} | {args.dataset} | {args.task} (dim={results['repr_dim']})")
    print(f" head_cold={results['head_cold']}")
    print(f" degree_only={results['degree_only']}")
    for key in ("within_cold", "transfer_mean"):
        cur = results.get(key) or {}
        for metric, vals in cur.items():
            import numpy as np
            if vals and not all(v != v for v in vals):
                print(f" {key}.{metric}: peak={np.nanmax(vals):.3f}@rank"
                      f"{results['ranks'][int(np.nanargmax(vals))]}")
    print(f"[done] {run_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
