"""D2 multi-run summary — D2 main + K1 shuf-mediators + K2b rand-uniform + K3 freeze-alpha.

Mirrors summarize_d1_controls.py for the D2 cycle. Reads each run's results.json,
builds a 4-way comparison vs v2i4 anchor, and prints CP-3 verdict suggestion
based on design §3-§4 falsification ladder.

Per CLAUDE.md "禁止编造事实": all numbers from results.json files. The v2i4 anchor
is hard-coded from
`Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json`
(verified 2026-05-30 + 2026-05-31).

Usage (from project root):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\\
Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/summarize_d2_controls.py"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V2I4_ANCHOR = {
    "run_id": "2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42",
    "auc": 0.7803698295832711,
    "auc_emergnn": 0.7405175365730533,
    "auc_count_only": 0.6687667902853477,
    "auc_i4_only": 0.6002510209606847,
    "nll": 1.4722882456069026,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "Code" / "runs"

# Hard-stop / band thresholds (round4 plan §6).
HARD_STOP_THRESH = 0.775
MULTI_SEED_THRESH = 0.785


def _find_latest(tag_substring: str) -> Path | None:
    candidates = sorted(RUNS.glob(f"*{tag_substring}*"))
    return candidates[-1] if candidates else None


def _load(run_dir: Path) -> dict | None:
    rj = run_dir / "results.json"
    if not rj.is_file():
        return None
    return json.loads(rj.read_text(encoding="utf-8"))


def _branches(m: dict) -> dict:
    t = m.get("metrics", {}).get("test_s2", {})
    return {
        "combined": float(t.get("auc", float("nan"))),
        "emergnn": float(t.get("auc_emergnn", float("nan"))),
        "count_only": float(t.get("auc_count_only", float("nan"))),
        "i4_only": float(t.get("auc_i4_only", float("nan"))),
        "nll": float(t.get("nll", float("nan"))),
        "card_breakdown": t.get("mediator_cardinality_breakdown", {}),
        "coverage": t.get("coverage_breakdown", {}),
    }


def _delta_pp(a: float, b: float) -> str:
    if a != a or b != b:
        return "n/a"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.2f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main", default="d2_main_seed42")
    p.add_argument("--k1", default="d2_shuf_mediators_seed42")
    p.add_argument("--k2b", default="d2_rand_uniform_seed42")
    p.add_argument("--k3", default="d2_freeze_alpha_seed42")
    args = p.parse_args()

    runs = {
        "D2 main": _find_latest(args.main),
        "K1 shuf-mediators": _find_latest(args.k1),
        "K2b rand-uniform": _find_latest(args.k2b),
        "K3 freeze-alpha": _find_latest(args.k3),
    }
    print("Run discovery:")
    for name, rd in runs.items():
        status = (rd / "results.json").is_file() if rd else False
        ready = "READY" if status else "in-flight or missing"
        rd_label = rd.name if rd else "(no match)"
        print(f"  {name:<22} {ready:<22} {rd_label}")
    print()

    data: dict[str, dict | None] = {}
    for name, rd in runs.items():
        if rd is None or not (rd / "results.json").is_file():
            data[name] = None
            continue
        loaded = _load(rd)
        data[name] = _branches(loaded) if loaded else None

    anchor = {
        "combined": V2I4_ANCHOR["auc"],
        "emergnn": V2I4_ANCHOR["auc_emergnn"],
        "count_only": V2I4_ANCHOR["auc_count_only"],
        "i4_only": V2I4_ANCHOR["auc_i4_only"],
        "nll": V2I4_ANCHOR["nll"],
    }

    print("=" * 100)
    print(
        f"{'branch':<14}{'D2 main':<11}{'K1 shuf':<11}{'K2b rand':<11}"
        f"{'K3 frz-α':<11}{'v2i4':<10}{'pp Δ vs anchor (D2/K1/K2b/K3)':<32}"
    )
    print("-" * 100)
    for k in ("combined", "emergnn", "count_only", "i4_only", "nll"):
        row = [k]
        for nm in ("D2 main", "K1 shuf-mediators", "K2b rand-uniform", "K3 freeze-alpha"):
            v = (data.get(nm) or {}).get(k, float("nan"))
            row.append(f"{v:.4f}" if v == v else "—")
        row.append(f"{anchor[k]:.4f}")
        deltas = []
        for nm in ("D2 main", "K1 shuf-mediators", "K2b rand-uniform", "K3 freeze-alpha"):
            v = (data.get(nm) or {}).get(k, float("nan"))
            deltas.append(_delta_pp(v, anchor[k]))
        row.append("/".join(deltas))
        print(f"{row[0]:<14}{row[1]:<11}{row[2]:<11}{row[3]:<11}{row[4]:<11}"
              f"{row[5]:<10}{row[6]:<32}")
    print()

    # Cardinality breakdown for D2 main if present.
    # Key schema from run_v3_meet_mask.py:140 — `q0`, `q1`, `q2`, `q3`
    # (quartile buckets, q0 = lowest-cardinality bucket).
    cb = (data.get("D2 main") or {}).get("card_breakdown") or {}
    if cb:
        print("Mediator-cardinality-stratified S2 AUC (D2 main):")
        for bucket_name in sorted(cb.keys()):
            b = cb.get(bucket_name)
            if b and b.get("auc") is not None:
                print(
                    f"  {bucket_name:<6} AUC={b['auc']:.4f}  "
                    f"n_pos={b.get('n_pos')}  n_neg={b.get('n_neg')}  "
                    f"card_range={b.get('card_range', 'n/a')}"
                )
        print()

    # CP-3 verdict suggestion when all 4 ready.
    if all(data.get(nm) is not None for nm in
           ("D2 main", "K1 shuf-mediators", "K2b rand-uniform", "K3 freeze-alpha")):
        d2 = data["D2 main"]
        k1 = data["K1 shuf-mediators"]
        k2b = data["K2b rand-uniform"]
        k3 = data["K3 freeze-alpha"]

        print("=" * 100)
        print("Design §3-§4 falsification ladder check")
        print("=" * 100)

        # K1: combined drop ≥ 1.5pp, emergnn ≤ anchor 0.7405
        k1_combined_drop = d2["combined"] - k1["combined"]
        print(
            f"K1 shuf-mediators expected: combined drops ≥ 1.50pp; emergnn ≤ 0.7405"
        )
        print(
            f"  observed: combined drop = {k1_combined_drop*100:+.2f}pp; "
            f"emergnn = {k1['emergnn']:.4f}"
        )
        k1_pass = (k1_combined_drop >= 0.015) and (k1["emergnn"] <= V2I4_ANCHOR["auc_emergnn"])
        print(f"  K1 binding-falsification: {'PASS' if k1_pass else 'FAIL'}")
        print()

        k2_combined_drop = d2["combined"] - k2b["combined"]
        print(f"K2b rand-uniform expected: combined drops more than K1; emergnn ≤ 0.7405")
        print(
            f"  observed: combined drop = {k2_combined_drop*100:+.2f}pp; "
            f"emergnn = {k2b['emergnn']:.4f}"
        )
        k2_pass = (k2_combined_drop >= k1_combined_drop) and (k2b["emergnn"] <= V2I4_ANCHOR["auc_emergnn"])
        print(f"  K2b identity-falsification: {'PASS' if k2_pass else 'FAIL'}")
        print()

        # K3: combined ≈ v2i4 anchor (since alpha_meet=0, bonus path disabled)
        print(f"K3 freeze-alpha expected: combined ≈ v2i4 anchor 0.7804 (mathematically equivalent)")
        print(f"  observed: combined = {k3['combined']:.4f}")
        k3_pass = abs(k3["combined"] - V2I4_ANCHOR["auc"]) <= 0.005  # within ±0.5pp
        print(f"  K3 backbone-parity: {'PASS' if k3_pass else 'FAIL (>0.5pp deviation)'}")
        print()

        print("=" * 100)
        print("Suggested CP-3 verdict input")
        print("=" * 100)
        if k1_pass and k2_pass and k3_pass:
            print("STRONG POSITIVE: all three controls behave as designed; D2 architectural claim well-supported.")
            print("Action: stage CP-3 codex, then multi-seed (seed=43, 44).")
        elif (not k1_pass) and (not k2_pass):
            print("STRONG NEGATIVE: K1 + K2b both failed falsification — same D1 capacity story.")
            print("Action: stage CP-3 with NOT_PASS framing. Pivot to D3/D4.")
        elif k1_pass and not k2_pass:
            print("MIXED: K1 confirmed binding matters but K2b didn't drop further.")
            print("Action: stage CP-3, explain mechanism in K2b terms.")
        else:
            print("PARTIAL: see per-control verdicts above. CP-3 with 4-cell table.")
    else:
        print("(Not all 4 runs loaded yet — partial summary above.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
