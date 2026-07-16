"""Builder for TIGER mol-graph pkl on this project's drug pool.

Per CLAUDE.md §"Baseline 规范" §2 step 2: this is the baseline-side
copy of the mol-graph builder. The reproduction-side has its own
equivalent (currently inline in
``Code/reproductions/TIGER/data_process.py:single_smile_to_graph``;
a future ``Code/reproductions/TIGER/_Original-Dataset/necessary/build_mol_sp.py``
will be the canonical paper-side builder). The two copies are maintained
independently per §文件级独立性; ``baseline/tiger/_shared.py:ensure_mol_pkl``
subprocesses THIS script, NEVER the reproduction-side one.

Output format::

    {drug_id_str: torch_geometric.data.Data}

where each Data carries: ``x`` (atom features), ``edge_index`` (bonds),
``sp_edge_index`` (shortest-path edges over molecule), ``sp_value``
(SP lengths), ``sp_edge_rel`` (bond type for length=1, length+offset
otherwise) — same schema consumed by
:func:`baseline.tiger.binary_cls.baseline.TIGERBaseline._make_pair_batch`.

CLI::

    python build_mol_pkl.py \\
        --smiles-csv <path-to-project-drug-smiles.csv> \\
        --id-col drugbank_id --smiles-col smiles \\
        --out _data/necessary/mol_pkl__mine.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

# We import the baseline's existing molecular featurizer rather than
# duplicating BRICS + SP logic here. This keeps a single source of truth
# inside the baseline package — paper-vs-impl divergence (if any) lives
# in one place, not two.
_HERE = Path(__file__).resolve()
# parents math from ``Code/baseline/tiger/_data/necessary/build_mol_pkl.py``:
#   parents[0] = necessary/   parents[3] = baseline/
#   parents[1] = _data/       parents[4] = Code/
#   parents[2] = tiger/       parents[5] = <project root>
# We need ``Code/`` to be on sys.path so ``from baseline.tiger.mol_features
# import build_drug_graphs`` works when this script is invoked via
# subprocess from arbitrary cwd.
_CODE_ROOT = _HERE.parents[4]
sys.path.insert(0, str(_CODE_ROOT))

from baseline.tiger.mol_features import build_drug_graphs  # noqa: E402


def build_mol_pkl(smiles_map: dict[str, str], *, verbose: bool = True) -> dict:
    """Build ``{drug_id: Data}`` mol-graph cache from a SMILES dict.

    Drugs whose SMILES fail RDKit parsing are silently dropped (matches
    upstream Blair1213/TIGER behaviour in
    ``data_process.py:smile_to_graph``).
    """
    graphs, missing, max_rel = build_drug_graphs(smiles_map)
    if verbose:
        print(
            f"[build_mol_pkl] built {len(graphs)} / {len(smiles_map)} "
            f"(skipped {len(missing)} invalid SMILES); max_sp_rel={max_rel}",
            file=sys.stderr,
            flush=True,
        )
    return graphs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build TIGER mol-graph pkl from a SMILES CSV."
    )
    parser.add_argument(
        "--smiles-csv", required=True, type=Path,
        help="CSV with drug-id + SMILES columns.",
    )
    parser.add_argument(
        "--id-col", default="drugbank_id",
        help="Drug-id column in --smiles-csv (default: drugbank_id).",
    )
    parser.add_argument(
        "--smiles-col", default="smiles",
        help="SMILES column in --smiles-csv (default: smiles).",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output pkl path (e.g. _data/necessary/mol_pkl__mine.pkl).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.smiles_csv, usecols=[args.id_col, args.smiles_col])
    smi_map = {str(k): (v if isinstance(v, str) else "") for k, v in
               zip(df[args.id_col], df[args.smiles_col])}

    graphs = build_mol_pkl(smi_map, verbose=not args.quiet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(graphs, f)
    print(f"Wrote {args.out}  ({len(graphs)} drugs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
