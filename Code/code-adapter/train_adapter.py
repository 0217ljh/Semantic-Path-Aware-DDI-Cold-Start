"""Adapter (DDI-LoRA) entrypoint - train the M_A.M_B adapter through the harness.

Step-3 milestone: ADAPTER-ALONE (lightweight structural H_base, no backbone).
Mirrors train_baseline.py's wiring (data.loader -> protocol -> model/runner) but
drives adapter/rank_model.AdapterRankModel. The adapter's own protocol is the
method protocol (model_meta.MethodMeta); --protocol overrides it.

Usage (project root, WSL conda env):
  python Code/code-adapter/train_adapter.py --dataset drugbank_ryu --task binary \
      --fold fold0 --epochs 30 --eval-every 5 --set d=256 lam=4 batch_size=64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(_ADAPTER))

from adapter.rank_model import AdapterRankModel            # noqa: E402
from data.loader import load_rank_data                     # noqa: E402
from model.contracts import CheckpointPolicy               # noqa: E402
from model.runner import run_training                      # noqa: E402
from model_meta import Protocol, METHOD                    # noqa: E402
from specs import TaskSpec                                  # noqa: E402

_RUNS = _ADAPTER.parent / "runs"


def _parse_hp(pairs: list[str]) -> dict:
    hp: dict = {}
    for kv in pairs or []:
        k, _, v = kv.partition("=")
        if v.lower() in ("true", "false"):
            hp[k] = v.lower() == "true"
        else:
            try:
                hp[k] = int(v)
            except ValueError:
                try:
                    hp[k] = float(v)
                except ValueError:
                    hp[k] = v
    return hp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)
    ap.add_argument("--n-classes", type=int, default=0, help="required for multiclass")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--protocol", choices=["native", "p1", "p2"], default="native",
                    help="override the cold-start protocol (default: method protocol)")
    ap.add_argument("--emerging-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="hp overrides k=v")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    task = (TaskSpec.binary() if args.task == "binary"
            else TaskSpec.multiclass(args.n_classes))
    data = load_rank_data(args.dataset, task, args.fold)
    if args.protocol == "native":
        protocol = METHOD.protocol
    else:
        protocol = Protocol.P1_FIXED if args.protocol == "p1" else Protocol.P2_EMERGING

    hp = {"seed": args.seed, "dataset": args.dataset, "fold": args.fold,
          **_parse_hp(args.set)}
    run_id = (f"{time.strftime('%Y-%m-%d_%H-%M-%S')}__adapter_alone"
              f"__{args.dataset}_{args.task}_{args.fold}__{protocol.value}"
              f"{('__' + args.tag) if args.tag else ''}__seed{args.seed}")
    run_dir = _RUNS / run_id
    ckpt = CheckpointPolicy(save_best=True, to_disk=args.save_ckpt, save_dir=run_dir)

    print(f"[run] adapter-alone | {args.dataset} {args.task} {args.fold} | "
          f"protocol={protocol.value} | epochs={args.epochs} | run_dir={run_dir}")
    model = AdapterRankModel()
    res = run_training(model, data, protocol, run_dir, hp,
                       epochs=args.epochs, eval_every=args.eval_every,
                       patience=args.patience, ckpt=ckpt,
                       emerging_ratio=args.emerging_ratio, seed=args.seed)
    print(f"[result] cold_test={res['cold_test']} best_epoch={res['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
