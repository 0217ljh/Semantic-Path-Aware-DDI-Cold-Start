"""Builder for TIGER BKG (Background KG) cache.

Per CLAUDE.md §"Baseline 规范" §2 step 3: this is the baseline-side
CLI builder invoked via subprocess by
:func:`baseline.tiger._shared.ensure_bkg_cache`. Takes a JSON manifest
file (containing all the complex args that don't fit easily in a CLI),
loads inputs, calls :func:`baseline.tiger.kg_builder.build_bkg_from_merged_parquet`,
saves the resulting dict to ``--out`` pkl.

Why a manifest file: BKG construction needs (merged_kg_path, drug pool
[hundreds of ids], DDI edges [10K+ rows], g2_drugs [hundreds of ids],
blocklist). These don't fit in a simple flag-based CLI. Manifest is
JSON; subprocess reads it from disk.

CLI::

    python build_bkg_cache.py \\
        --manifest <path-to-manifest.json> \\
        --out _data/necessary/bkg_cache__<hash>__mine.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_CODE_ROOT = _HERE.parents[4]  # necessary -> _data -> tiger -> baseline -> Code
sys.path.insert(0, str(_CODE_ROOT))

from baseline.tiger.kg_builder import build_bkg_from_merged_parquet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build TIGER BKG cache from a manifest JSON."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="JSON file with keys: merged_kg_path, drugs (list[str]), "
             "ddi_edges_path (CSV with drug_a_id, drug_b_id, ddi_type), "
             "g2_drugs (list[str]), blocklist (list[str]).",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output pkl path.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    with args.manifest.open("r", encoding="utf-8") as f:
        m = json.load(f)
    merged_kg_path = Path(m["merged_kg_path"])
    drugs = set(m["drugs"])
    ddi_edges = pd.read_csv(m["ddi_edges_path"])
    g2_drugs = m.get("g2_drugs", [])
    blocklist = tuple(m.get("blocklist", []))
    kg_scope = m.get("kg_scope", "full")

    bkg = build_bkg_from_merged_parquet(
        merged_kg_path,
        drugs,
        ddi_edges,
        g2_drug_ids=g2_drugs,
        blocklist=blocklist,
        kg_scope=kg_scope,
        verbose=not args.quiet,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(bkg, f)
    n_nodes = bkg.get("n_total_nodes", "?")
    n_edges = len(bkg.get("edge_list", []))
    print(f"Wrote {args.out}  ({n_nodes} nodes, {n_edges} edges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
