"""Fair before/after timing for the SumGNN subgraph-materialization accelerator.

Protocol (codex 019f1fdc): real leaf (ddi800 multiclass S0), FIXED max_links so the
subsample size and extraction are identical across configs; extraction runs ONCE (cache
reused), excluded from the per-epoch comparison; eval DISABLED during timed epochs. Runs a
3-way table so overlap effects are visible:
  (1) original dataset, num_workers=8   (production baseline)
  (2) original dataset, num_workers=0   (isolate worker-overlap contribution)
  (3) materialized dataset, num_workers=0 (accelerated)
Reports per-epoch wall (steady state = epoch 2+) + the one-time materialization pass.

Usage (WSL conda project_1, from project root; GPU):
  python Code/scripts/time_sumgnn_accel.py --n 2000 --epochs 3
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.unified_loader import Leaf  # noqa: E402

_EPOCH_RE = re.compile(r"Epoch (\d+) with loss.*? in ([\d.]+)")
_MAT_RE = re.compile(r"materialized (\d+) subgraphs")


def _subsample(leaf, n, seed=0):
    tr = leaf.train.sample(min(n, len(leaf.train)), random_state=seed).reset_index(drop=True)
    va = leaf.val.sample(min(max(50, n // 6), len(leaf.val)), random_state=seed).reset_index(drop=True)
    te = leaf.test.sample(min(max(50, n // 6), len(leaf.test)), random_state=seed).reset_index(drop=True)
    return Leaf(task=leaf.task, dataset=leaf.dataset, regime=leaf.regime,
                split_code=leaf.split_code, fold=leaf.fold, train=tr, val=va, test=te,
                resources=leaf.resources)


def _run(tag, sub, *, materialize, num_workers, epochs, max_links):
    """Fit once; return (epoch_times list, materialize_pass_seconds|None) parsed from logs."""
    import torch
    from baseline.sumgnn.multi_cls.baseline_unified import SumGNNUnifiedMulticlass as Cls
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    root = logging.getLogger(); root.setLevel(logging.INFO); root.addHandler(h)
    t0 = time.time()
    model = Cls(n_epochs=epochs, device="auto", num_workers=num_workers, max_links=max_links,
                eval_every_iter=10_000_000, materialize=materialize)  # eval off
    model.fit_leaf(sub)
    wall = time.time() - t0
    root.removeHandler(h)
    log = buf.getvalue()
    epochs_t = [float(m.group(2)) for m in _EPOCH_RE.finditer(log)]
    print(f"[{tag}] fit wall={wall:.1f}s  per-epoch={['%.1f'%e for e in epochs_t]}", flush=True)
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return epochs_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), "ddi800", "multiclass",
                       "transductive", "fold0")
    sub = _subsample(leaf, args.n)
    print(f"[time] leaf ddi800/mc/S0 subsample train={sub.train.shape} val={sub.val.shape} "
          f"test={sub.test.shape} max_links={args.n} epochs={args.epochs}", flush=True)

    # fresh cache so the subsample+max_links are actually what gets extracted
    import hashlib
    meta = leaf.resources.meta
    key = f"{meta.get('dataset_id')}|{meta.get('split_type')}|{meta.get('split_code')}|{leaf.fold}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    cache_dir = ROOT / "Code/baseline/sumgnn/_data/necessary" / f"mc__{h}"
    if cache_dir.exists():
        print(f"[time] removing stale cache {cache_dir.name} for a clean {args.n}-link build", flush=True)
        shutil.rmtree(cache_dir)

    def steady(ts):
        return (sum(ts[1:]) / len(ts[1:])) if len(ts) > 1 else (ts[0] if ts else float("nan"))

    print("\n=== (1) original dataset, num_workers=8 (extraction happens here, once) ===", flush=True)
    c1 = _run("orig-w8", sub, materialize=False, num_workers=8, epochs=args.epochs, max_links=args.n)
    print("\n=== (2) original dataset, num_workers=0 ===", flush=True)
    c2 = _run("orig-w0", sub, materialize=False, num_workers=0, epochs=args.epochs, max_links=args.n)
    print("\n=== (3) materialized dataset, num_workers=0 ===", flush=True)
    c3 = _run("mat-w0", sub, materialize=True, num_workers=0, epochs=args.epochs, max_links=args.n)

    s1, s2, s3 = steady(c1), steady(c2), steady(c3)
    print("\n================ SUMMARY (steady-state per-epoch wall, epoch 2+) ================", flush=True)
    print(f"(1) orig + workers=8 : {s1:.1f} s/epoch", flush=True)
    print(f"(2) orig + workers=0 : {s2:.1f} s/epoch", flush=True)
    print(f"(3) mat  + workers=0 : {s3:.1f} s/epoch", flush=True)
    if s3 > 0:
        print(f"speedup 1->3 (production before/after): {s1/s3:.2f}x", flush=True)
        print(f"speedup 2->3 (isolated materialization): {s2/s3:.2f}x", flush=True)
        print(f"workers were hiding 1->2: {s1/s2:.2f}x", flush=True)


if __name__ == "__main__":
    main()
