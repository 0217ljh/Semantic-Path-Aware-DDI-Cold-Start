"""Round 1 ARIS analyzer: R1 (gated fusion) + R3 (PMP) results vs v2i4 anchor and D2 K3 deterministic baseline.

Reads results.json from R1/R3 main runs (and optionally K-controls), prints
a comparison table + verdict.

Per CLAUDE.md anti-fabrication: all anchor numbers verified from
`Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json`
on 2026-05-30/2026-06-01.

Usage:
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\\
Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/summarize_r1_r3.py"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V2I4_ANCHOR = {
    "auc": 0.7803698295832711,
    "auc_emergnn": 0.7405175365730533,
    "nll": 1.4722882456069026,
}
D2_K3_DETERMINISTIC_BASELINE = {
    "auc": 0.7843,
    "auc_emergnn": 0.7497,
    "note": "verified 2026-06-02 from run 2026-06-02_00-52-02__run_v3_meet_mask__d2_freeze_alpha_seed42__seed42; v2i4-equivalent forward + deterministic seed",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "Code" / "runs"

HARD_STOP = 0.775
LIMITED_SWEEP_HIGH = 0.785
PROGRESS_DELTA_VS_K3 = 0.005  # R1/R3 must beat K3 by at least 0.5pp


def _find_latest(tag: str) -> Path | None:
    cands = sorted(RUNS.glob(f"*{tag}*"))
    return cands[-1] if cands else None


def _load(run_dir: Path) -> dict | None:
    rj = run_dir / "results.json"
    if not rj.is_file():
        return None
    return json.loads(rj.read_text(encoding="utf-8"))


def _key_metrics(d: dict | None) -> dict:
    if d is None:
        return {}
    t = d.get("metrics", {}).get("test_s2", {})
    return {
        "combined": t.get("auc"),
        "emergnn": t.get("auc_emergnn"),
        "count_only": t.get("auc_count_only"),
        "i4_only": t.get("auc_i4_only"),
        "nll": t.get("nll"),
        "n_pos": t.get("n_pos"),
        "n_neg": t.get("n_neg"),
        "auc_residual": t.get("auc_residual"),
    }


def _pp(a: float | None, b: float) -> str:
    if a is None:
        return "—"
    d = (a - b) * 100
    return f"{d:+.2f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--r1-tag", default="r1_main_seed42")
    p.add_argument("--r3-tag", default="r3_main_seed42")
    args = p.parse_args()

    r1_dir = _find_latest(args.r1_tag)
    r3_dir = _find_latest(args.r3_tag)
    r1 = _key_metrics(_load(r1_dir)) if r1_dir else {}
    r3 = _key_metrics(_load(r3_dir)) if r3_dir else {}

    print("=" * 78)
    print("ARIS Round 1 — R1 (gated fusion) + R3 (PMP) results")
    print("=" * 78)
    print(f"R1 run_dir : {r1_dir if r1_dir else '(no match)'}")
    print(f"R3 run_dir : {r3_dir if r3_dir else '(no match)'}")
    print()

    rows = [
        ("combined", r1.get("combined"), r3.get("combined"), V2I4_ANCHOR["auc"], D2_K3_DETERMINISTIC_BASELINE["auc"]),
        ("emergnn",  r1.get("emergnn"),  r3.get("emergnn"),  V2I4_ANCHOR["auc_emergnn"], D2_K3_DETERMINISTIC_BASELINE["auc_emergnn"]),
        ("nll",      r1.get("nll"),      r3.get("nll"),      V2I4_ANCHOR["nll"], None),
    ]

    print(f"{'branch':<10}{'R1':<10}{'R3':<10}{'v2i4':<10}{'D2-K3-det':<12}{'R1 vs anchor (pp)':<20}{'R3 vs anchor (pp)':<20}")
    print("-" * 100)
    for name, r1v, r3v, anchor, k3 in rows:
        r1s = f"{r1v:.4f}" if isinstance(r1v, (int, float)) else "—"
        r3s = f"{r3v:.4f}" if isinstance(r3v, (int, float)) else "—"
        anchor_s = f"{anchor:.4f}" if isinstance(anchor, (int, float)) else "—"
        k3_s = f"{k3:.4f}" if isinstance(k3, (int, float)) else "—"
        if name == "nll":
            r1_d = "—" if r1v is None else f"{(r1v - anchor):+.4f}"
            r3_d = "—" if r3v is None else f"{(r3v - anchor):+.4f}"
        else:
            r1_d = _pp(r1v, anchor)
            r3_d = _pp(r3v, anchor)
        print(f"{name:<10}{r1s:<10}{r3s:<10}{anchor_s:<10}{k3_s:<12}{r1_d:<20}{r3_d:<20}")

    print()

    # Verdict suggestion
    print("=" * 78)
    print("Verdict suggestion (per Notes/Log/ARIS_R1_R3_CONTROL_PLAN.md band rules)")
    print("=" * 78)

    def verdict(name: str, combined: float | None):
        if combined is None:
            print(f"{name}: NO RESULT (run not complete)")
            return
        delta_anchor = combined - V2I4_ANCHOR["auc"]
        delta_k3 = combined - D2_K3_DETERMINISTIC_BASELINE["auc"]
        if combined < HARD_STOP:
            band = "STOP (below hard-stop 0.775; debug)"
        elif combined < LIMITED_SWEEP_HIGH:
            band = f"LIMITED SWEEP ({HARD_STOP} <= combined < {LIMITED_SWEEP_HIGH}); K1 only"
        else:
            band = f"FULL CONTROLS (combined >= {LIMITED_SWEEP_HIGH}); K1+K2+K3+multi-seed"
        progress = "PROGRESS" if delta_k3 >= PROGRESS_DELTA_VS_K3 else ("PARTIAL" if combined >= V2I4_ANCHOR["auc"] else "NO_PROGRESS")
        print(f"{name}: combined={combined:.4f} | vs anchor {delta_anchor*100:+.2f}pp | vs D2-K3-det {delta_k3*100:+.2f}pp")
        print(f"   band: {band}")
        print(f"   ARIS verdict: {progress}")

    verdict("R1", r1.get("combined"))
    verdict("R3", r3.get("combined"))

    print()
    print("Next step (per ARIS_R1_R3_CONTROL_PLAN.md):")
    print("- If PROGRESS: launch K1/K2/K3 controls + multi-seed for winner")
    print("- If PARTIAL: launch K1 only + queue R2 (hypernet) for attempt 4")
    print("- If NO_PROGRESS: archive results + transition to D3 (alignment InfoNCE init)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
