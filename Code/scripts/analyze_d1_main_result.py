"""D1 main-run analyzer + hard-stop decision tool.

Reads the results.json of a D1 main run (run_v3_llm_edge.py output), compares
against the v2i4 anchor 0.7804 (run_id 2026-05-29_22-02-27), applies the
round4_backbone_diff_plan.md §6 hard-stop rule, and prints next-step commands.

Per CLAUDE.md "禁止编造事实": ALL numbers must come from results.json files;
the v2i4 anchor is hard-coded from
`Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json`
(verified 2026-05-30).

Usage (from project root):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\\
Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/analyze_d1_main_result.py \\
--run-dir Code/runs/<d1_main_run_dir>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Verified at 2026-05-30 from
# Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json
V2I4_ANCHOR = {
    "run_id": "2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42",
    "auc": 0.7803698295832711,
    "auprc": 0.7921942420186752,
    "nll": 1.4722882456069026,
    "auc_emergnn": 0.7405175365730533,
    "auc_count_only": 0.6687667902853477,
    "auc_i4_only": 0.6002510209606847,
}

HARD_STOP_THRESH = 0.775
SWEEP_BAND_LOW = 0.775
SWEEP_BAND_HIGH = 0.785
MULTI_SEED_THRESH = 0.785


def fmt_delta(d: float, pp: bool = True) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.2f} pp" if pp else f"{sign}{d:.4f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir", required=True, type=Path,
        help="Path to the D1 main run directory (containing results.json).",
    )
    args = p.parse_args()

    rj = args.run_dir / "results.json"
    if not rj.is_file():
        print(f"ERROR results.json not found at {rj}", file=sys.stderr)
        return 2

    data = json.loads(rj.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {}).get("test_s2", {})
    if "auc" not in metrics:
        print(f"ERROR results.json missing metrics.test_s2.auc", file=sys.stderr)
        return 2

    d1_auc = float(metrics["auc"])
    d1_emergnn = float(metrics.get("auc_emergnn", float("nan")))
    d1_count = float(metrics.get("auc_count_only", float("nan")))
    d1_i4 = float(metrics.get("auc_i4_only", float("nan")))
    d1_nll = float(metrics.get("nll", float("nan")))
    d1_auprc = float(metrics.get("auprc", float("nan")))
    coverage = metrics.get("coverage_breakdown", {})

    fit_sec = float(data.get("fit_sec", 0.0))
    inj = data.get("d1_injection_summary") or {}

    print("=" * 76)
    print("D1 main run analysis")
    print("=" * 76)
    print(f"run_dir            {args.run_dir}")
    if inj:
        print(f"injection summary  n_llm_edges={inj.get('n_llm_edges_injected')} "
              f"n_new_rel={inj.get('n_new_relations')} n_ent_after={inj.get('n_ent_after')}")
    print(f"fit time           {fit_sec:.0f}s = {fit_sec/60:.1f} min")
    print()
    print(f"{'branch':<22}{'D1 (this run)':<18}{'v2i4 anchor':<18}{'delta':<12}")
    print("-" * 76)
    print(f"{'combined AUC':<22}{d1_auc:<18.4f}{V2I4_ANCHOR['auc']:<18.4f}"
          f"{fmt_delta(d1_auc - V2I4_ANCHOR['auc']):<12}")
    print(f"{'emergnn (backbone)':<22}{d1_emergnn:<18.4f}{V2I4_ANCHOR['auc_emergnn']:<18.4f}"
          f"{fmt_delta(d1_emergnn - V2I4_ANCHOR['auc_emergnn']):<12}")
    print(f"{'count_only (i2)':<22}{d1_count:<18.4f}{V2I4_ANCHOR['auc_count_only']:<18.4f}"
          f"{fmt_delta(d1_count - V2I4_ANCHOR['auc_count_only']):<12}")
    print(f"{'i4_only (readout)':<22}{d1_i4:<18.4f}{V2I4_ANCHOR['auc_i4_only']:<18.4f}"
          f"{fmt_delta(d1_i4 - V2I4_ANCHOR['auc_i4_only']):<12}")
    print(f"{'AUPRC':<22}{d1_auprc:<18.4f}{V2I4_ANCHOR['auprc']:<18.4f}"
          f"{fmt_delta(d1_auprc - V2I4_ANCHOR['auprc']):<12}")
    print(f"{'NLL':<22}{d1_nll:<18.4f}{V2I4_ANCHOR['nll']:<18.4f}"
          f"{fmt_delta(d1_nll - V2I4_ANCHOR['nll'], pp=False):<12}")

    if coverage:
        print()
        print("Coverage-stratified S2 AUC:")
        for bucket in ("both_covered", "one_covered", "neither_covered"):
            b = coverage.get(bucket)
            if not b:
                continue
            auc = b.get("auc")
            n_pos = b.get("n_pos")
            n_neg = b.get("n_neg")
            auc_s = f"{auc:.4f}" if auc is not None else "n/a"
            print(f"  {bucket:<20} AUC={auc_s}  n_pos={n_pos}  n_neg={n_neg}")
        print("Expected pattern (CP-1 covered-vs-uncovered check): both > one > neither.")

    print()
    print("=" * 76)
    print("Hard-stop decision (round4 plan §6 + d1_llm_edge_design.md §4)")
    print("=" * 76)
    if d1_auc < HARD_STOP_THRESH:
        print(f"VERDICT: STOP — combined {d1_auc:.4f} < hard-stop threshold {HARD_STOP_THRESH}")
        print("ACTION: do NOT run controls; do NOT sweep; open")
        print(
            "        Notes/Log/round4_d1_failure.md and summarize. "
            "Discuss whether design is wrong (e.g. token vocab explosion, "
            "audit too aggressive, backbone not absorbing signal) before any new run."
        )
        return 0
    if d1_auc < MULTI_SEED_THRESH:
        print(
            f"VERDICT: LIMITED SWEEP — combined {d1_auc:.4f} in band "
            f"[{SWEEP_BAND_LOW}, {MULTI_SEED_THRESH}); above hard-stop but below the "
            f"multi-seed bar."
        )
        print("ACTION: run K1 shuf-token control + 1 hyperparam sweep; do NOT multi-seed yet.")
        print()
        _print_k1_only(args.run_dir)
        return 0
    # combined >= 0.785
    print(f"VERDICT: PASS — combined {d1_auc:.4f} >= multi-seed threshold {MULTI_SEED_THRESH}")
    print(
        "ACTION: run K1/K2/K3 controls; pending CP-3 PASS, run multi-seed "
        "(seed=43, seed=44 same setting)."
    )
    print()
    _print_full_controls(args.run_dir)
    return 0


def _wsl_prefix() -> str:
    return (
        "wsl bash -ic \"conda activate project_1 && cd "
        "/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && "
    )


def _print_k1_only(run_dir: Path) -> None:
    print("Command (K1 shuf-token, ~50 min):")
    print(
        f"  {_wsl_prefix()}python -u "
        f"Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py "
        f"--epochs 100 --tag d1_shuf_token_seed42 --seed 42 --d1-shuf-token\""
    )


def _print_full_controls(run_dir: Path) -> None:
    print("Commands (K1, K2, K3 sequentially; each ~50 min; total ~2.5 h):")
    for tag, flag in [
        ("d1_shuf_token_seed42", "--d1-shuf-token"),
        ("d1_rand_token_seed42", "--d1-rand-token"),
        ("d1_drop_i4head_seed42", "--d1-drop-i4-head"),
    ]:
        print(
            f"  {_wsl_prefix()}python -u "
            f"Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py "
            f"--epochs 100 --tag {tag} --seed 42 {flag}\""
        )
    print()
    print("After CP-3 PASS, multi-seed (~1.5h total parallel-launchable):")
    for seed in (43, 44):
        print(
            f"  {_wsl_prefix()}python -u "
            f"Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py "
            f"--epochs 100 --tag d1_main_seed{seed} --seed {seed}\""
        )


if __name__ == "__main__":
    sys.exit(main())
