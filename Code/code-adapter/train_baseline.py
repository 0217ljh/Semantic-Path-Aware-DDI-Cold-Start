"""Baseline entrypoint - train a backbone (backbone-alone) through the harness.

Wires: data.loader -> model_meta protocol -> model/runner. The protocol is the
baseline's own (set_protocol, defaults to native); switching it only changes the
per-epoch data, not the loop.

Usage (project root, WSL conda env):
  python Code/code-adapter/train_baseline.py --baseline rgcn --dataset drugbank_ryu \
      --task multiclass --n-classes 86 --fold fold0 --epochs 100 --eval-every 5 \
      --set lr=1e-3 hidden=128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(_ADAPTER))

from data.loader import load_rank_data            # noqa: E402
from model.contracts import CheckpointPolicy      # noqa: E402
from model.runner import run_training             # noqa: E402
from model.wrappers import WRAPPERS               # noqa: E402
from model_meta import BASELINES, Protocol        # noqa: E402
from specs import TaskSpec                         # noqa: E402

_RUNS = _ADAPTER.parent / "runs"                   # Code/runs (per project convention)


def _parse_hp(pairs: list[str]) -> dict:
    """--set k=v k2=v2 -> {k: parsed}. int/float/bool/str auto-typed."""
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
    ap.add_argument("--baseline", required=True, choices=sorted(WRAPPERS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)
    ap.add_argument("--n-classes", type=int, default=0, help="required for multiclass")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--protocol", choices=["native", "p1", "p2"], default="native",
                    help="override the cold-start protocol (default: baseline's native)")
    ap.add_argument("--emerging-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-ckpt", action="store_true", help="write best.pt bundle to run_dir")
    ap.add_argument("--backbone-ckpt", default="",
                    help="path to a saved backbone best.pt to load+freeze (FROZEN adapter mode)")
    ap.add_argument("--set", nargs="*", default=[], help="hp overrides k=v")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    task = (TaskSpec.binary() if args.task == "binary"
            else TaskSpec.multiclass(args.n_classes))
    data = load_rank_data(args.dataset, task, args.fold)
    if args.protocol == "native":
        protocol = BASELINES[args.baseline].set_protocol
    else:
        protocol = Protocol.P1_FIXED if args.protocol == "p1" else Protocol.P2_EMERGING

    # dataset/fold are part of the run config; wrappers that build their own leaf
    # (e.g. EmerGNN's entity vocab + Morgan features) read them from hp.
    hp = {"seed": args.seed, "dataset": args.dataset, "fold": args.fold,
          **_parse_hp(args.set)}
    if args.backbone_ckpt:
        hp["backbone_ckpt"] = args.backbone_ckpt   # consumed by frozen-adapter wrappers
    run_id = (f"{time.strftime('%Y-%m-%d_%H-%M-%S')}__baseline_{args.baseline}"
              f"__{args.dataset}_{args.task}_{args.fold}__{protocol.value}"
              f"{('__' + args.tag) if args.tag else ''}__seed{args.seed}")
    run_dir = _RUNS / run_id
    ckpt = CheckpointPolicy(save_best=True, to_disk=args.save_ckpt, save_dir=run_dir)

    print(f"[run] {args.baseline} | {args.dataset} {args.task} {args.fold} | "
          f"protocol={protocol.value} | epochs={args.epochs} | run_dir={run_dir}")
    model = WRAPPERS[args.baseline]()
    res = run_training(model, data, protocol, run_dir, hp,
                       epochs=args.epochs, eval_every=args.eval_every,
                       patience=args.patience, ckpt=ckpt,
                       emerging_ratio=args.emerging_ratio, seed=args.seed)
    print(f"[result] cold_test={res['cold_test']} best_epoch={res['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
