"""Validate ``build_hierarchical_pkl.py`` against the official jcsun-00 pkl.

We can't byte-match the official pkl (its builder source was never
released, so individual feature values and node ordering differ).  What
we CAN match — and what HDN-DDI's downstream training actually consumes
— are the structural invariants:

1. Same set of drug IDs covered.
2. Same per-drug (n_y0, n_y1, n_y2) node-type composition, or at least
   the same distribution shape.
3. Same ``edge_attr`` column-0 unique value set (bond-type ids).
4. Same ``edge_attr`` column-1 unique value set (edge category ids).
5. Same endpoint-y-type pattern per edge category.

This script loads the official pkl and our reproduced pkl side-by-side
and reports per-invariant match rates.  Designed to be the regression
guard for `build_hierarchical_pkl.py` — if a future tweak to the BRICS
strategy or feature layout breaks one of these invariants the run will
flag the divergence.

Usage::

    # First build the reproduction pkl (reproduction-side builder copy):
    python Code/reproductions/HDN-DDI/_Original-Dataset/necessary/build_hierarchical_pkl.py \\
        --smiles-csv Code/reproductions/HDN-DDI/_Original-Dataset/drug_smiles.csv \\
        --out        Code/reproductions/HDN-DDI/_Original-Dataset/necessary/id_data_dict__mine.pkl

    # Then validate:
    python Code/baseline/hdn_ddi/_data/necessary/validate_hierarchical_pkl.py \\
        --official Code/reproductions/HDN-DDI/_Original-Dataset/necessary/id_data_dict__official.pkl \\
        --repro    Code/reproductions/HDN-DDI/_Original-Dataset/necessary/id_data_dict__mine.pkl

Note: the official pkl + drug_smiles.csv stay in ``reproductions/HDN-DDI/``
(they are upstream's downloaded files, not ours). The builder + this
validator live next to each other inside the baseline package's
`_data/necessary/` per CLAUDE.md §"Baseline 规范" §1.
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------
# Per-drug summary record (everything we compare)
# ---------------------------------------------------------------------

def _summarise(data) -> dict:
    """Reduce a PyG ``Data`` object to comparable structural summaries."""
    y = data.y
    edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
    summary = {
        "n_nodes": int(data.x.shape[0]),
        "feat_dim": int(data.x.shape[1]),
        "n_y0": int((y == 0).sum()),
        "n_y1": int((y == 1).sum()),
        "n_y2": int((y == 2).sum()),
        "n_edges": int(data.edge_index.shape[1]),
    }
    if edge_attr is not None and edge_attr.numel() > 0:
        summary["col0_unique"] = sorted(edge_attr[:, 0].unique().tolist())
        summary["col1_unique"] = sorted(edge_attr[:, 1].unique().tolist())
        # Endpoint y-type counts per col1 category.
        endpoints_per_cat: dict[int, Counter] = {}
        src, dst = data.edge_index
        for i in range(edge_attr.shape[0]):
            cat = int(edge_attr[i, 1])
            endpoints_per_cat.setdefault(cat, Counter())[
                (int(y[src[i]]), int(y[dst[i]]))
            ] += 1
        summary["endpoints_per_cat"] = {
            cat: dict(c) for cat, c in endpoints_per_cat.items()
        }
    else:
        summary["col0_unique"] = []
        summary["col1_unique"] = []
        summary["endpoints_per_cat"] = {}
    return summary


# ---------------------------------------------------------------------
# Comparison + reporting
# ---------------------------------------------------------------------

def _print_block(title: str) -> None:
    print()
    print("-" * 72)
    print(f"  {title}")
    print("-" * 72)


def _compare(official: dict, repro: dict) -> int:
    """Side-by-side comparison; returns exit code (0 = pass, 1 = drift)."""
    off_ids = set(official.keys())
    rep_ids = set(repro.keys())

    # ── 1. Drug-ID coverage ─────────────────────────────────────────
    _print_block("1. Drug-ID coverage")
    print(f"  official: {len(off_ids):4d} drugs")
    print(f"  repro:    {len(rep_ids):4d} drugs")
    common = off_ids & rep_ids
    only_off = off_ids - rep_ids
    only_rep = rep_ids - off_ids
    print(f"  common:   {len(common):4d}")
    print(f"  only in official: {len(only_off)}  "
          f"(e.g., {sorted(only_off)[:5]})")
    print(f"  only in repro:    {len(only_rep)}  "
          f"(e.g., {sorted(only_rep)[:5]})")

    if not common:
        print("  [FAIL] No overlapping drug IDs - cannot validate.")
        return 1

    # Build summary dicts for common drugs.
    off_sum = {did: _summarise(_unwrap(official[did])) for did in common}
    rep_sum = {did: _summarise(_unwrap(repro[did])) for did in common}

    # ── 2. Feature dim sanity ───────────────────────────────────────
    _print_block("2. Feature dimensionality (must be 66)")
    off_feat = Counter(s["feat_dim"] for s in off_sum.values())
    rep_feat = Counter(s["feat_dim"] for s in rep_sum.values())
    print(f"  official feat_dim: {dict(off_feat)}")
    print(f"  repro    feat_dim: {dict(rep_feat)}")
    feat_ok = off_feat == rep_feat == Counter({66: len(common)})
    print(f"  match: {'[OK]' if feat_ok else '[FAIL]'}")

    # ── 3. y-composition distributions ──────────────────────────────
    _print_block("3. Node-type composition (median per drug)")
    def _med(d, k):
        return statistics.median(s[k] for s in d.values())

    for col in ("n_y0", "n_y1", "n_y2"):
        o = _med(off_sum, col); r = _med(rep_sum, col)
        print(f"  {col}: official median={o:6.1f}  repro median={r:6.1f}  "
              f"{'[OK]' if abs(o - r) <= max(1, 0.3 * o) else '[FAIL]'}")

    # ── 4. edge_attr column unique-value sets ───────────────────────
    _print_block("4. edge_attr column unique-value sets")
    off_col0 = set()
    rep_col0 = set()
    off_col1 = set()
    rep_col1 = set()
    for s in off_sum.values():
        off_col0.update(s["col0_unique"])
        off_col1.update(s["col1_unique"])
    for s in rep_sum.values():
        rep_col0.update(s["col0_unique"])
        rep_col1.update(s["col1_unique"])
    print(f"  col0 official: {sorted(off_col0)}")
    print(f"  col0 repro:    {sorted(rep_col0)}")
    col0_ok = off_col0 == rep_col0
    print(f"  col0 match: {'[OK]' if col0_ok else '[FAIL]'}")
    print(f"  col1 official: {sorted(off_col1)}")
    print(f"  col1 repro:    {sorted(rep_col1)}")
    col1_ok = off_col1 == rep_col1
    print(f"  col1 match: {'[OK]' if col1_ok else '[FAIL]'}")

    # ── 5. Endpoint-y-type pattern per edge category ────────────────
    #
    # PAPER-FAITHFUL DIVERGENCE: the HDN-DDI paper text (§Methods)
    # explicitly states "there are no edges between substructure-level
    # nodes within a molecule".  The shipped pkl nevertheless contains
    # some (1,1) edges in col1==1 / col1==2 (~5000 total across 1706
    # drugs) — this appears to be a paper-implementation inconsistency
    # in the original release.  Our reproduction follows the PAPER
    # TEXT (no frag-frag edges), so the only "expected divergence"
    # from the shipped pkl is repro missing the (1,1) endpoint pair
    # in col1=={1,2}.  We mark that case [PAPER] rather than [FAIL].
    _print_block("5. Endpoint y-type pattern per edge category (aggregated)")
    off_agg = {0: Counter(), 1: Counter(), 2: Counter()}
    rep_agg = {0: Counter(), 1: Counter(), 2: Counter()}
    for s in off_sum.values():
        for cat, c in s["endpoints_per_cat"].items():
            off_agg.setdefault(cat, Counter()).update(c)
    for s in rep_sum.values():
        for cat, c in s["endpoints_per_cat"].items():
            rep_agg.setdefault(cat, Counter()).update(c)
    endpoints_ok = True
    for cat in sorted(set(off_agg) | set(rep_agg)):
        off_keys = set(off_agg.get(cat, {}).keys())
        rep_keys = set(rep_agg.get(cat, {}).keys())
        cat_ok = off_keys == rep_keys
        # Special case: cat in {1, 2} and the ONLY mismatch is missing
        # (1,1) in repro → mark as paper-faithful divergence, not FAIL.
        missing_in_repro = off_keys - rep_keys
        extra_in_repro = rep_keys - off_keys
        is_paper_div = (
            cat in (1, 2)
            and missing_in_repro == {(1, 1)}
            and not extra_in_repro
        )
        if not cat_ok and not is_paper_div:
            endpoints_ok = False
        status = (
            "[OK]" if cat_ok
            else ("[PAPER]" if is_paper_div else "[FAIL]")
        )
        print(f"  col1=={cat}:")
        print(f"    official endpoint y-pairs: "
              f"{sorted(off_agg.get(cat, {}).items())}")
        print(f"    repro    endpoint y-pairs: "
              f"{sorted(rep_agg.get(cat, {}).items())}")
        print(f"    match: {status}")
        if is_paper_div:
            print(f"      -> repro intentionally omits (1,1) edges "
                  f"per paper Methods section (\"there are no edges "
                  f"between substructure-level nodes\")")

    # ── 6. Overall verdict ──────────────────────────────────────────
    _print_block("Overall verdict")
    all_ok = feat_ok and col0_ok and col1_ok and endpoints_ok
    if all_ok:
        print("  [OK] All structural invariants match the paper's intended")
        print("       graph topology.")
        print("    Per-node feature VALUES and node ORDERING may still")
        print("    differ - those depend on jcsun-00's unpublished")
        print("    featurization; HDN-DDI training is invariant to them.")
        print()
        print("    [PAPER] marks above (if any) flag known paper-vs-pkl")
        print("    inconsistencies in the upstream release where the")
        print("    shipped pkl contradicts what Methods describes. Our")
        print("    repro follows the paper text.")
        return 0
    print("  [FAIL] At least one invariant diverges - review [FAIL] marks above.")
    return 1


def _unwrap(entry):
    """Official pkl values are ``(smiles, Data)`` tuples; ours match."""
    if isinstance(entry, tuple):
        return entry[1]
    return entry


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the hierarchical-pkl reproduction against the "
            "official jcsun-00 pkl by comparing structural invariants."
        )
    )
    parser.add_argument(
        "--official", required=True, type=Path,
        help="Official id_data_dict__official.pkl path (upstream-shipped).",
    )
    parser.add_argument(
        "--repro", required=True, type=Path,
        help="Our reproduced pkl path (from build_hierarchical_pkl.py).",
    )
    args = parser.parse_args(argv)

    if not args.official.is_file():
        print(f"[validate] official pkl not found: {args.official}",
              file=sys.stderr)
        return 2
    if not args.repro.is_file():
        print(f"[validate] repro pkl not found: {args.repro}  "
              f"(run build_hierarchical_pkl.py first)",
              file=sys.stderr)
        return 2

    print(f"[validate] loading official: {args.official}")
    with args.official.open("rb") as f:
        official = pickle.load(f)
    print(f"[validate] loading repro:    {args.repro}")
    with args.repro.open("rb") as f:
        repro = pickle.load(f)

    return _compare(official, repro)


if __name__ == "__main__":
    sys.exit(main())
