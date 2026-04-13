#!/usr/bin/env python3
"""CLI entry: --run or --data/--model/--training, --mode train|predict. See docs/4-scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from my_code.pipeline.run import main as pipeline_main


def parse_args():
    p = argparse.ArgumentParser(description="Train or predict (unified pipeline).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="Run config path or label (e.g. cora_gcn or configs/runs/cora_gcn.yaml)")
    g.add_argument("--data", help="Data config path (use with --model and --training)")
    p.add_argument("--model", help="Model config path")
    p.add_argument("--training", help="Training config path")
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--ckpt", help="Checkpoint path (required when --mode predict)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override config (repeatable)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--exp_name", help="Experiment name")
    args = p.parse_args()
    if args.run is None and (not args.data or not args.model or not args.training):
        p.error("Either --run or all of --data, --model, --training are required")
    if args.mode == "predict" and not args.ckpt and not args.run:
        p.error("--ckpt required when --mode predict (or provide --run and set ckpt in config)")
    return args


if __name__ == "__main__":
    args = parse_args()
    sys.exit(pipeline_main(args))
