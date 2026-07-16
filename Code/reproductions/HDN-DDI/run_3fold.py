"""Run HDN-DDI faithful reproduction on fold 0/1/2 and aggregate.

Mirrors jcsun-00's ``repeat.sh`` loop (which runs fold 0, 1, 2 with
seed 42 each), then averages ``Ds1`` / ``Ds2`` metrics across the three
``results.json`` files — matching how paper Table 3 numbers are
produced (``evaluate.ipynb`` reads ``log/cold-start/*.log`` and means
across runs).

Usage:
    python Code/reproductions/HDN-DDI/run_3fold.py

    # Override e.g. n_workers for Windows:
    python Code/reproductions/HDN-DDI/run_3fold.py -- --n_workers 0

Everything after ``--`` is forwarded verbatim to each per-fold
``run_faithful.py`` invocation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent  # ../../.. → project root
_RUNS_DIR = _PROJECT_ROOT / "Code" / "runs"


# ─────────────────────────── per-fold launcher ───────────────────────

def _run_one_fold(fold: int, extra_args: list[str]) -> dict:
    """Launch run_faithful.py for one fold and return its results.json."""
    pre_run_ids = set(p.name for p in _RUNS_DIR.glob("*hdn_ddi_faithful*"))
    cmd = [
        sys.executable,
        str(_THIS_DIR / "run_faithful.py"),
        "--fold", str(fold),
        *extra_args,
    ]
    print(f"\n{'='*72}\n[3fold] FOLD {fold}   launching: {' '.join(cmd)}\n{'='*72}",
          flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"fold {fold} run_faithful.py exited with code {rc}")
    print(f"[3fold] FOLD {fold} done in {elapsed/60:.1f} min", flush=True)

    # Find newly-created run_dir matching this fold's tag.
    post_run_ids = set(p.name for p in _RUNS_DIR.glob("*hdn_ddi_faithful*"))
    new_ids = post_run_ids - pre_run_ids
    matching = [rid for rid in new_ids if f"fold{fold}_faithful" in rid]
    if not matching:
        raise RuntimeError(
            f"fold {fold}: could not find new run_dir under {_RUNS_DIR}; "
            f"new ids = {sorted(new_ids)}"
        )
    run_id = sorted(matching)[-1]  # newest if multiple
    results_path = _RUNS_DIR / run_id / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"results.json missing at {results_path}")
    with results_path.open() as f:
        return json.load(f)


# ─────────────────────────── aggregator ──────────────────────────────

_METRIC_KEYS = ("acc", "auroc", "f1", "precision", "recall", "int_ap", "ap")


def _mean_std(xs: list[float]) -> tuple[float, float]:
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def _aggregate(results_per_fold: list[dict]) -> dict:
    """Per-bucket mean + std across folds (matches paper Table 3 format)."""
    agg: dict = {}
    for split in ("s1", "s2"):
        agg[split] = {}
        for k in _METRIC_KEYS:
            vals = [r[split][k] for r in results_per_fold if k in r[split]]
            if vals:
                mean, std = _mean_std(vals)
                agg[split][k] = {"mean": mean, "std": std, "values": vals}
    agg["folds"] = [r.get("fold", "?") for r in results_per_fold]
    agg["best_epochs"] = [r.get("best_epoch", "?") for r in results_per_fold]
    return agg


def _print_table(agg: dict) -> None:
    print(f"\n{'='*72}")
    print("  HDN-DDI faithful reproduction — 3-fold mean (paper Table 3 format)")
    print(f"{'='*72}")
    print(f"  folds:       {agg['folds']}")
    print(f"  best_epochs: {agg['best_epochs']}")
    print()
    print(f"  {'Split':<10}{'ACC':>14}{'AUROC':>14}{'AUPR':>14}{'F1':>14}")
    print(f"  {'-'*10}{'-'*14:>14}{'-'*14:>14}{'-'*14:>14}{'-'*14:>14}")
    for split in ("s1", "s2"):
        row = f"  {split.upper():<10}"
        for paper_key, our_key in (("ACC", "acc"), ("AUROC", "auroc"),
                                    ("AUPR", "ap"), ("F1", "f1")):
            m = agg[split].get(our_key, {})
            if m:
                row += f"{m['mean']*100:>9.2f}±{m['std']*100:<3.2f}"
            else:
                row += f"{'--':>14}"
        print(row)
    print()
    print("  Paper Table 3 (cold-start mean over 3 folds, %):")
    print(f"  {'Ds1':<10}{79.84:>9.2f}±{'?':<4}{89.48:>9.2f}±{'?':<4}{89.65:>9.2f}±{'?':<4}{77.42:>9.2f}±{'?':<4}")
    print(f"  {'Ds2':<10}{89.43:>9.2f}±{'?':<4}{95.92:>9.2f}±{'?':<4}{95.95:>9.2f}±{'?':<4}{89.34:>9.2f}±{'?':<4}")
    print(f"{'='*72}\n")


# ─────────────────────────── main ────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # Split argv at "--": before are this script's flags, after are forwarded.
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own_args = argv[:i]
        forward = argv[i + 1 :]
    else:
        own_args = argv
        forward = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds", type=int, nargs="+", default=[0, 1, 2],
        help="Which folds to run (default: 0 1 2).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Aggregated-results JSON output path "
             "(default: Code/runs/_logs/hdn_ddi_faithful_3fold_<timestamp>.json).",
    )
    args = parser.parse_args(own_args)

    results = []
    for fold in args.folds:
        r = _run_one_fold(fold, forward)
        results.append(r)

    agg = _aggregate(results)
    _print_table(agg)

    out_path = args.out or (
        _RUNS_DIR / "_logs" /
        f"hdn_ddi_faithful_3fold_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"per_fold": results, "aggregate": agg}, f, indent=2)
    print(f"[3fold] saved aggregated results to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
