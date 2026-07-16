"""Baseline campaign orchestrator - run the backbone-alone main-table grid.

Grid (General Settings, tex): 6 backbones x {drugbank_ryu, drugbank_latest_partial}
x {binary, multiclass} x fold0..4 = 120 runs. Each baseline uses its NATIVE protocol
(emergnn P2, the other five P1). 5-fold CV is the mean+/-std source (no extra seeds).

Resumable: skips a cell whose run_dir (matched by the train_baseline run_id pattern)
already holds results.json. Appends a one-row summary per completed run to
Code/runs/_logs/index.csv (per the logging convention). Never overwrites old runs.

Usage (project root, WSL conda env):
  python Code/code-adapter/run_baselines.py --dry-run
  python Code/code-adapter/run_baselines.py --datasets drugbank_latest_partial --tasks binary
  python Code/code-adapter/run_baselines.py --baselines rgcn emergnn --folds fold0
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

_ADAPTER = Path(__file__).resolve().parent
_RUNS = _ADAPTER.parent / "runs"
_LOGS = _RUNS / "_logs"

BASELINES = ["rgcn", "ssiddi", "emergnn", "mkgfenn", "knowddi", "tiger"]
DATASETS = ["drugbank_ryu", "drugbank_latest_partial"]
TASKS = ["binary", "multiclass"]
FOLDS = [f"fold{i}" for i in range(5)]
#: multiclass class counts per dataset (tex General Settings); binary needs none
N_CLASSES = {("drugbank_ryu", "multiclass"): 86,
             ("drugbank_latest_partial", "multiclass"): 165}


def _done(baseline: str, dataset: str, task: str, fold: str) -> str | None:
    """Return the run_dir if this cell already produced results.json, else None."""
    pat = f"*__baseline_{baseline}__{dataset}_{task}_{fold}__*"
    for d in glob.glob(str(_RUNS / pat)):
        if (Path(d) / "results.json").is_file():
            return d
    return None


def _append_index(run_dir: Path) -> None:
    """Append a summary row for a finished run to Code/runs/_logs/index.csv."""
    rj = run_dir / "results.json"
    if not rj.is_file():
        return
    r = json.loads(rj.read_text(encoding="utf-8"))
    ct = r.get("cold_test", {})
    row = {"run_id": run_dir.name, "model": r.get("model"), "dataset": r.get("dataset"),
           "task": r.get("task"), "n_classes": r.get("n_classes"), "fold": r.get("fold"),
           "protocol": r.get("protocol"), "best_epoch": r.get("best_epoch"),
           "cold_best_val": r.get("cold_best_val"),
           **{f"test_{k}": v for k, v in ct.items()}, "run_dir": str(run_dir)}
    _LOGS.mkdir(parents=True, exist_ok=True)
    idx = _LOGS / "index.csv"
    new = not idx.is_file()
    with open(idx, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="*", default=BASELINES, choices=BASELINES)
    ap.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    ap.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    ap.add_argument("--folds", nargs="*", default=FOLDS, choices=FOLDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="list the grid + done/pending, run nothing")
    args = ap.parse_args()

    grid = list(itertools.product(args.baselines, args.datasets, args.tasks, args.folds))
    pending = []
    print(f"[grid] {len(grid)} cells "
          f"({len(args.baselines)}b x {len(args.datasets)}ds x {len(args.tasks)}t x {len(args.folds)}f)")
    for baseline, dataset, task, fold in grid:
        done = _done(baseline, dataset, task, fold)
        tag = "DONE" if done else "pending"
        print(f"  [{tag}] {baseline:9s} {dataset:24s} {task:10s} {fold}")
        if not done:
            pending.append((baseline, dataset, task, fold))
    print(f"[grid] {len(grid) - len(pending)} done, {len(pending)} pending")
    if args.dry_run:
        return 0

    for i, (baseline, dataset, task, fold) in enumerate(pending, 1):
        cmd = [sys.executable, "-u", str(_ADAPTER / "train_baseline.py"),
               "--baseline", baseline, "--dataset", dataset, "--task", task,
               "--fold", fold, "--protocol", "native", "--epochs", str(args.epochs),
               "--eval-every", str(args.eval_every), "--patience", str(args.patience)]
        if task == "multiclass":
            cmd += ["--n-classes", str(N_CLASSES[(dataset, task)])]
        print(f"\n[{i}/{len(pending)}] {baseline} {dataset} {task} {fold} @ {time.strftime('%H:%M:%S')}")
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode         # train_baseline writes results.json + logs
        done = _done(baseline, dataset, task, fold)
        if done:
            _append_index(Path(done))
            print(f"    ok ({time.perf_counter() - t0:.0f}s) rc={rc} -> {Path(done).name}")
        else:
            print(f"    NO results.json (rc={rc}) - check the run log; continuing")
    print(f"\n[campaign] done; results in {_RUNS}, summary in {_LOGS}/index.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
