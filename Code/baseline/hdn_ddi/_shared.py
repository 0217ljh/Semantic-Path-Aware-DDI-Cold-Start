"""Cross-task shared helpers for HDN-DDI baselines.

These helpers are used by both ``binary_cls/baseline.py`` and
``multi_cls/baseline.py``.  Per CLAUDE.md §"Baseline 目录布局规范",
跨 task 共享的代码必须留在顶层 (而不是放在某个 task 子文件夹下被
其他 task 跨子文件夹 import — 后者是 layout 反例).
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch

# Canonical absolute path to the builder script (single source of truth
# for the 3-level hierarchical graph algorithm in this baseline). The
# script lives **inside this package** at ``_data/necessary/`` per
# CLAUDE.md §"Baseline 规范" §1 (builder co-located with its outputs
# inside _data/necessary/). An independent twin lives on the reproduction
# side at ``reproductions/HDN-DDI/_Original-Dataset/necessary/`` per
# §"文件级独立性" (same-named files allowed, sha may drift).
_BASELINE_HDN_DIR = Path(__file__).resolve().parent
_BUILDER_SCRIPT = (
    _BASELINE_HDN_DIR / "_data" / "necessary" / "build_hierarchical_pkl.py"
)

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset


def drug_smiles_dict(train: "PairDataset") -> dict[str, str]:
    """Extract ``{drug_id: smiles}`` from a PairDataset's drugs table.

    Raises ValueError if the dataset has no SMILES column.
    """
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "HDN-DDI requires `PairDataset.drugs` with a `smiles` column."
        )
    return {
        str(row["drugbank_id"]): (
            "" if pd.isna(row["smiles"]) else str(row["smiles"])
        )
        for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows()
    }


def bipartite_edge_index_y1(ga, gb) -> torch.Tensor:
    """Substructure-level bipartite edge list (paper §HDN Encoder).

    Only the y==1 nodes of drug A connect to y==1 nodes of drug B,
    matching paper's "Unlike existing methods that consider all atomic
    nodes, HDN-DDI constructs bipartite graphs exclusively comprising
    substructure-level nodes" claim.

    Returned indices are LOCAL to each drug's full node list (matching
    the atom-frag-super ordering produced by
    :func:`baseline.hdn_ddi.mol_features.mol_to_data`), so
    :class:`InterGraphAttention` can do ``ga.x[ei[0]]`` / ``gb.x[ei[1]]``
    directly with no remapping.

    Falls back to shape ``[2, 0]`` for the rare "small rigid molecule
    with no atom layer" case where ``mol_to_data`` ships frag+super only
    and the y==1 fragment is the whole molecule (no cross-product to
    build).
    """
    src_idx = (ga.y == 1).nonzero(as_tuple=False).flatten()
    dst_idx = (gb.y == 1).nonzero(as_tuple=False).flatten()
    n_s, n_t = src_idx.numel(), dst_idx.numel()
    if n_s == 0 or n_t == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    src = src_idx.repeat_interleave(n_t)
    dst = dst_idx.repeat(n_s)
    return torch.stack([src, dst], dim=0)


def load_mol_graphs_pkl(pkl_path: "str | Path") -> dict[str, object]:
    """Load a pre-built 3-level hierarchical molecular graph pkl produced by
    ``baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py``.

    The builder lives in this package (not in reproductions/) per CLAUDE.md
    §"Baseline 规范" §1 (each side owns its own builder copy at
    `_data/necessary/`).

    The builder writes ``{drug_id: (smiles, Data)}`` (see builder
    ``build_id_data_dict``).  We strip the smiles and keep only
    ``{drug_id: Data}`` for the baseline's downstream
    ``_make_pair_batch`` lookup pattern (``self._graphs.get(drug_id)``).

    Returns:
        graphs: ``{drug_id_str: torch_geometric.data.Data}``
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"HDN-DDI mol graph pkl not found: {pkl_path}. "
            f"Generate it via: python Code/baseline/hdn_ddi/_data/necessary/"
            f"build_hierarchical_pkl.py --smiles-csv <project drug csv> "
            f"--id-col drugbank_id --smiles-col smiles --out {pkl_path}"
        )
    with pkl_path.open("rb") as f:
        raw = pickle.load(f)
    graphs: dict[str, object] = {}
    for did, payload in raw.items():
        # builder format: (smiles, Data); be tolerant of legacy {did: Data}
        if isinstance(payload, tuple) and len(payload) == 2:
            graphs[str(did)] = payload[1]
        else:
            graphs[str(did)] = payload
    return graphs


def ensure_mol_graphs_pkl(
    train: "PairDataset",
    pkl_path: "str | Path",
    *,
    force_rebuild: bool = False,
) -> dict[str, object]:
    """Auto-detect or auto-build the 3-level hierarchical mol-graphs pkl.

    Behaviour:
      1. If ``pkl_path`` exists AND the cached drug set ⊇ train.drugs →
         load and return.
      2. Otherwise (missing OR stale OR ``force_rebuild=True``) →
         (a) dump ``train.drugs[["drugbank_id", "smiles"]]`` to a CSV
             next to the pkl (``<pkl_path_stem>__smiles.csv``);
         (b) invoke ``baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py``
             via ``subprocess.run([sys.executable, ...])`` —— same Python
             interpreter, so conda env is preserved;
         (c) load and return.

    The builder is in this same package now (not in reproductions/),
    so there is no cross-folder dependency to worry about. Subprocess
    invocation is kept (rather than direct Python import) only to keep
    the builder's CLI as the single canonical entry point — exactly
    what the user runs by hand when debugging.

    Raises ``FileNotFoundError`` if the builder script is missing,
    ``RuntimeError`` if the subprocess returns non-zero exit.
    """
    pkl_path = Path(pkl_path).resolve()

    # 1. Detect: pkl exists AND covers all train drugs?
    train_drug_ids = {str(d) for d in train.drugs["drugbank_id"]} \
        if train.drugs is not None else set()
    if not pkl_path.is_file():
        print(
            f"[hdn_ddi] mol_pkl MISSING at {pkl_path} -- will build",
            file=sys.stderr,
        )
    if pkl_path.is_file() and not force_rebuild:
        # Tolerate partial / truncated pkl from a previously-crashed
        # builder: catch unpickling failures and treat as stale →
        # auto-rebuild instead of hard-failing.
        try:
            cached = load_mol_graphs_pkl(pkl_path)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(
                f"[hdn_ddi] mol_pkl CORRUPT at {pkl_path} ({type(e).__name__}: "
                f"{e}); treating as stale, will rebuild",
                file=sys.stderr,
            )
        else:
            cached_ids = set(cached.keys())
            missing_from_cache = train_drug_ids - cached_ids
            if not missing_from_cache:
                print(
                    f"[hdn_ddi] mol_pkl cache HIT: {pkl_path} "
                    f"({len(cached)} drugs cached, {len(train_drug_ids)} "
                    f"train drugs all covered)",
                    file=sys.stderr,
                )
                return cached
            print(
                f"[hdn_ddi] mol_pkl cache STALE: {pkl_path} missing "
                f"{len(missing_from_cache)} train drugs -- rebuilding",
                file=sys.stderr,
            )

    # 2. Build: dump SMILES CSV → subprocess builder → load output
    if not _BUILDER_SCRIPT.is_file():
        raise FileNotFoundError(
            f"HDN-DDI builder script not found at {_BUILDER_SCRIPT}. "
            f"Expected the local builder copy in this baseline package "
            f"(CLAUDE.md §'Baseline 规范' §1 — builder lives at "
            f"`_data/necessary/build_hierarchical_pkl.py`)."
        )
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "auto-build of mol_pkl requires train.drugs with 'smiles' column"
        )
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = pkl_path.with_name(pkl_path.stem + "__smiles.csv")
    train.drugs[["drugbank_id", "smiles"]].to_csv(csv_path, index=False)
    print(
        f"[hdn_ddi] dumped {len(train.drugs)} drug SMILES → {csv_path}",
        file=sys.stderr,
    )

    cmd = [
        sys.executable,
        str(_BUILDER_SCRIPT),
        "--smiles-csv", str(csv_path),
        "--id-col", "drugbank_id",
        "--smiles-col", "smiles",
        "--out", str(pkl_path),
    ]
    print(
        f"[hdn_ddi] invoking builder subprocess:\n    {' '.join(cmd)}",
        file=sys.stderr,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"HDN-DDI builder subprocess failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    print(
        f"[hdn_ddi] mol_pkl BUILT at {pkl_path}. tail stderr:\n"
        f"{proc.stderr.splitlines()[-1] if proc.stderr else '(empty)'}",
        file=sys.stderr,
    )
    return load_mol_graphs_pkl(pkl_path)
