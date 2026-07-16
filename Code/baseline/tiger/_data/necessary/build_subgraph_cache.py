"""Builder for per-drug subgraph cache (one of 3 paper extractors).

Per CLAUDE.md §"Baseline 规范" §2 step 3: subprocess CLI for
:func:`baseline.tiger._shared.ensure_subgraph_cache`. Loads a saved BKG
pkl, calls :func:`baseline.tiger.subgraph_features.build_drug_subgraphs`
with the requested extractor + hparams, saves the resulting subgraph
dict + sizing info to ``--out`` pkl.

Cache file is named with both extractor and hparams encoded in the
hash so different (extractor, params) combos coexist.

CLI::

    python build_subgraph_cache.py \\
        --bkg-pkl <path>/bkg_cache__<hash>__mine.pkl \\
        --extractor randomWalk \\
        --extractor-params '{"num_walks": 1, "walk_length": 32}' \\
        --seed 42 \\
        --out <path>/subgraph_cache__<hash>__mine.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_CODE_ROOT = _HERE.parents[4]
sys.path.insert(0, str(_CODE_ROOT))

from baseline.tiger.subgraph_features import build_drug_subgraphs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build per-extractor subgraph cache from a saved BKG pkl."
    )
    parser.add_argument(
        "--bkg-pkl", required=True, type=Path,
        help="Path to saved BKG pkl (output of build_bkg_cache.py).",
    )
    parser.add_argument(
        "--extractor", required=True,
        choices=["randomWalk", "khop-subtree", "probability"],
        help="Paper subgraph extractor selector.",
    )
    parser.add_argument(
        "--extractor-params", default="{}",
        help="JSON dict of per-extractor hparams. Recognised keys: "
             "num_walks, walk_length (rw); khop, khop_fanout (subtree); "
             "fixed_num (probability).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output pkl path.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    with args.bkg_pkl.open("rb") as f:
        bkg = pickle.load(f)

    params = json.loads(args.extractor_params)
    subgraphs, max_deg, max_sp_rel = build_drug_subgraphs(
        bkg,
        extractor=args.extractor,
        num_walks=int(params.get("num_walks", 1)),
        walk_length=int(params.get("walk_length", 32)),
        khop=int(params.get("khop", 2)),
        khop_fanout=int(params.get("khop_fanout", 4)),
        fixed_num=int(params.get("fixed_num", 32)),
        seed=args.seed,
        verbose=not args.quiet,
    )

    payload = {
        "subgraphs": subgraphs,
        "max_degree": max_deg,
        "max_sp_rel": max_sp_rel,
        "extractor": args.extractor,
        "params": params,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(payload, f)
    print(
        f"Wrote {args.out}  ({len(subgraphs)} drugs, "
        f"max_deg={max_deg}, max_sp_rel={max_sp_rel})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
