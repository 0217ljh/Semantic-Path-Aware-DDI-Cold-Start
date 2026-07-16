"""Multi-run D1 result summary — D1 main + K1 shuf-token + K2 rand-token + K3 drop-i4-head.

Reads results.json from each run, builds a 4-way comparison table vs v2i4 anchor,
and prints CP-3 verdict suggestion based on the design §3 falsification ladder.

Per CLAUDE.md "禁止编造事实": all numbers from results.json. The v2i4 anchor is
hard-coded from
`Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json`
(verified 2026-05-30 + 2026-05-31).

Usage (from project root):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\\
Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/summarize_d1_controls.py"
"""
from __future__ import annotations

import argparse
import glob
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


def _find_latest(tag_substring: str) -> Path | None:
    candidates = sorted(RUNS.glob(f"*{tag_substring}*"))
    if not candidates:
        return None
    return candidates[-1]


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
        "coverage": t.get("coverage_breakdown", {}),
    }


def _delta(a: float, b: float) -> str:
    if a != a or b != b:
        return "n/a"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.2f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main", default="d1_main_seed42")
    p.add_argument("--k1", default="d1_shuf_token_seed42")
    p.add_argument("--k2", default="d1_rand_token_seed42")
    p.add_argument("--k3", default="d1_drop_i4head_seed42")
    args = p.parse_args()

    runs = {
        "D1 main": _find_latest(args.main),
        "K1 shuf-token": _find_latest(args.k1),
        "K2 rand-token": _find_latest(args.k2),
        "K3 drop-i4head": _find_latest(args.k3),
    }
    print("Run discovery:")
    for name, rd in runs.items():
        status = (rd / "results.json").is_file() if rd else False
        ready = "READY" if status else "in-flight or missing"
        rd_label = rd.name if rd else "(no match)"
        print(f"  {name:<16} {ready:<22} {rd_label}")
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

    print("=" * 92)
    print(
        f"{'branch':<14}{'D1 main':<11}{'K1 shuf':<11}{'K2 rand':<11}"
        f"{'K3 drop':<11}{'v2i4':<10}{'pp Δ vs anchor (D1/K1/K2/K3)':<26}"
    )
    print("-" * 92)
    for k in ("combined", "emergnn", "count_only", "i4_only", "nll"):
        row = [k]
        for nm in ("D1 main", "K1 shuf-token", "K2 rand-token", "K3 drop-i4head"):
            v = (data.get(nm) or {}).get(k, float("nan"))
            row.append(f"{v:.4f}" if v == v else "—")
        row.append(f"{anchor[k]:.4f}")
        deltas = []
        for nm in ("D1 main", "K1 shuf-token", "K2 rand-token", "K3 drop-i4head"):
            v = (data.get(nm) or {}).get(k, float("nan"))
            deltas.append(_delta(v, anchor[k]))
        row.append("/".join(deltas))
        print(f"{row[0]:<14}{row[1]:<11}{row[2]:<11}{row[3]:<11}{row[4]:<11}"
              f"{row[5]:<10}{row[6]:<26}")
    print()

    # Coverage breakdown (D1 main only — keep brief).
    cv = (data.get("D1 main") or {}).get("coverage") or {}
    if cv:
        print("Coverage breakdown (D1 main):")
        for bucket in ("both_covered", "one_covered", "neither_covered"):
            b = cv.get(bucket)
            if b and b.get("auc") is not None:
                print(f"  {bucket:<18} AUC={b['auc']:.4f}  n_pos={b.get('n_pos')}  n_neg={b.get('n_neg')}")
        print()

    # CP-3 verdict suggestion (only when all four are loaded).
    if all(data.get(nm) is not None for nm in
           ("D1 main", "K1 shuf-token", "K2 rand-token", "K3 drop-i4head")):
        d1 = data["D1 main"]
        k1 = data["K1 shuf-token"]
        k2 = data["K2 rand-token"]
        k3 = data["K3 drop-i4head"]
        print("=" * 92)
        print("Design §3 falsification ladder check")
        print("=" * 92)
        # K1: combined drop ≥ 1.5pp from D1 main, emergnn back to ≤ 0.7405
        k1_combined_drop = d1["combined"] - k1["combined"]
        k1_emergnn_drop = d1["emergnn"] - k1["emergnn"]
        print(
            f"K1 shuf-token expected: combined drops ≥ 1.50pp; emergnn ≤ 0.7405"
        )
        print(
            f"  observed: combined drop = {k1_combined_drop*100:+.2f}pp; "
            f"emergnn = {k1['emergnn']:.4f}"
        )
        k1_pass = (k1_combined_drop >= 0.015) and (k1["emergnn"] <= V2I4_ANCHOR["auc_emergnn"])
        print(f"  K1 falsification: {'PASS (signal confirmed)' if k1_pass else 'FAIL (signal not from drug-token binding)'}")
        print()

        # K2: combined drop ≥ K1 drop, emergnn back to ≤ anchor
        k2_combined_drop = d1["combined"] - k2["combined"]
        k2_emergnn_drop = d1["emergnn"] - k2["emergnn"]
        print(
            f"K2 rand-token expected: combined drops more than K1; emergnn ≤ 0.7405"
        )
        print(
            f"  observed: combined drop = {k2_combined_drop*100:+.2f}pp; "
            f"emergnn = {k2['emergnn']:.4f}"
        )
        k2_pass = (k2_combined_drop >= k1_combined_drop) and (k2["emergnn"] <= V2I4_ANCHOR["auc_emergnn"])
        print(f"  K2 bridge-removal: {'PASS (cross-drug bridges carry the lift)' if k2_pass else 'FAIL (lift survives bridge removal)'}")
        print()

        # K3: combined ≥ v2i4 anchor (0.7804) — readout drop doesn't kill us
        print(f"K3 drop-i4-head expected: combined ≥ v2i4 anchor 0.7804")
        print(f"  observed: combined = {k3['combined']:.4f}")
        k3_pass = k3["combined"] >= V2I4_ANCHOR["auc"]
        print(f"  K3 backbone-carry: {'PASS (backbone replaces readout)' if k3_pass else 'FAIL (backbone cannot carry alone)'}")
        print()

        print("=" * 92)
        print("Suggested CP-3 verdict input")
        print("=" * 92)
        if k1_pass and k2_pass and k3_pass:
            print("STRONG POSITIVE: all three controls behave as designed; D1 architectural claim well-supported.")
            print("Action: stage CP-3 codex review, then multi-seed (seed=43, seed=44).")
        elif (not k1_pass) and (not k2_pass):
            print("STRONG NEGATIVE: K1 + K2 both failed the falsification — D1 lift is capacity/density, not LLM evidence.")
            print("Action: stage CP-3 codex with NOT_PASS framing on the original D1 thesis.")
            print("Then pivot decision: drop D1, move to D2 (meeting-node-aware propagation) or D3 (alignment init).")
        elif k1_pass and not k2_pass:
            print("MIXED: K1 confirmed binding matters, but K2 didn't drop further. The bridges are not the mechanism.")
            print("Action: stage CP-3, explain mechanism in K2 terms.")
        elif not k1_pass and k2_pass:
            print("MIXED: K1 didn't drop (binding doesn't matter at edge level) but K2 did (bridges do).")
            print("Action: stage CP-3 with refined attribution; mechanism is bridge-routing not binding.")
        else:
            print("PARTIAL: see per-control verdicts above.")
            print("Action: stage CP-3 with the exact 4-cell falsification table.")
    else:
        print("(Not all 4 runs loaded yet — partial summary above.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
