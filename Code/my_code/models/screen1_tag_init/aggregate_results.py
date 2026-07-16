"""Aggregate Screen 1 results across variants into a comparison table.

Scans `Code/runs/` for completed screen1 runs, extracts test_s0/s1/s2 AUC
from each results.json, and prints a comparison table + writes a markdown
report.

Usage:
  python Code/my_code/models/screen1_tag_init/aggregate_results.py \\
    --output Notes/Experiments/_results/screen1/2026-05-20__screen1_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
RUNS_DIR = PROJECT_ROOT / "Code" / "runs"


def _find_screen1_runs() -> list[dict]:
    """Walk runs/ to find dirs containing screen1 results.json."""
    out = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or "run_screen1" not in run_dir.name:
            continue
        rj = run_dir / "results.json"
        if not rj.exists():
            continue
        try:
            payload = json.loads(rj.read_text())
        except Exception:
            continue
        out.append({"run_dir": run_dir, "payload": payload})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to write summary markdown")
    args = p.parse_args()

    runs = _find_screen1_runs()
    print(f"Found {len(runs)} screen1 runs")
    if not runs:
        print("No results yet. Variant training in progress?")
        return

    # Anchor for comparison
    ANCHOR_NAME = "EmerGNN multimode (drugbank KG, seed 42)"
    ANCHOR = {"test_s0": 0.9895, "test_s1": 0.8328, "test_s2": 0.7462}

    rows = []
    for r in runs:
        cfg = r["payload"]["config"]
        m = r["payload"]["metrics"]
        rows.append({
            "tag": cfg.get("tag", "?"),
            "variant": cfg.get("variant", "?"),
            "projection": cfg.get("projection", "?"),
            "epochs": cfg.get("epochs", "?"),
            "fit_h": r["payload"].get("fit_sec", 0) / 3600,
            "test_s0_auc": m.get("test_s0", {}).get("auc", float("nan")),
            "test_s1_auc": m.get("test_s1", {}).get("auc", float("nan")),
            "test_s2_auc": m.get("test_s2", {}).get("auc", float("nan")),
        })

    rows.sort(key=lambda r: (r["variant"], r["projection"]))

    # Print to console
    print()
    print(f"{'Variant':<10}{'Proj':<8}{'Epochs':<8}{'Fit(h)':<10}{'s0 AUC':<12}{'s1 AUC':<12}{'s2 AUC':<12}")
    print("-" * 78)
    print(f"{'(anchor)':<10}{'-':<8}{'-':<8}{'-':<10}"
          f"{ANCHOR['test_s0']:.4f}      {ANCHOR['test_s1']:.4f}      {ANCHOR['test_s2']:.4f}")
    print("-" * 78)
    for r in rows:
        print(f"{r['variant']:<10}{r['projection']:<8}{str(r['epochs']):<8}{r['fit_h']:<10.2f}"
              f"{r['test_s0_auc']:<12.4f}{r['test_s1_auc']:<12.4f}{r['test_s2_auc']:<12.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Screen 1 Results Summary",
            "",
            f"**Anchor**: {ANCHOR_NAME}: s0={ANCHOR['test_s0']:.4f}, "
            f"s1={ANCHOR['test_s1']:.4f}, s2={ANCHOR['test_s2']:.4f}",
            "",
            "| Variant | Proj | Epochs | Fit (h) | test_s0 AUC | test_s1 AUC | test_s2 AUC | Δs2 vs anchor |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            ds2 = r["test_s2_auc"] - ANCHOR["test_s2"]
            sign = "+" if ds2 >= 0 else ""
            lines.append(
                f"| {r['variant']} | {r['projection']} | {r['epochs']} | "
                f"{r['fit_h']:.2f} | {r['test_s0_auc']:.4f} | {r['test_s1_auc']:.4f} | "
                f"{r['test_s2_auc']:.4f} | {sign}{ds2:.4f} |"
            )
        lines.extend([
            "",
            "## Gating",
            "",
            "Primary claim (per first_step_plan.md §4.5):",
            "- D > E by ≥ 2pt test_s2 AUC, paired bootstrap CI excluding 0 → "
            "PubMedBERT semantic content matters beyond text-length/template",
            "- D > D-name by ≥ 1pt → profile richness adds value",
            "- D > F by ≥ 1pt → text content beats type label",
            "",
            "Gating results will be added once D and E both have results.",
        ])
        out.write_text("\n".join(lines))
        print(f"\nReport saved -> {out}")


if __name__ == "__main__":
    main()
