"""Cross-task shared helpers for TIGER baselines.

Per CLAUDE.md §"Baseline 规范" §1: helpers used by both ``binary_cls/``
and ``multi_cls/`` live at the top level. This file is the canonical
home for the **detect-and-build** pipeline that backs all baseline-side
``__mine`` artefacts (mol-graph pkl today; BKG cache + subgraph caches
later in Round 3).

Detect-and-build contract (per CLAUDE.md §"Baseline 规范" §2 step 3):
  - input: a `PairDataset` (or any object exposing the data needed by
    the builder) + a target ``pkl_path``
  - if ``pkl_path`` exists AND cached set ⊇ data set → load + return
  - else (missing / stale / corrupt) → subprocess-invoke the local
    ``_data/necessary/build_<name>.py`` (via ``sys.executable`` so the
    conda env is preserved) → load + return

Independence: the subprocess always targets the **local** builder in
``baseline/tiger/_data/necessary/``, NEVER the reproduction-side
script — §文件级独立性 forbids cross-folder borrowing.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset


# Canonical absolute paths to baseline-local builders + their default
# output pkl. ``_PROJECT_ROOT`` is unused inside this file (we only need
# absolute paths under the baseline package) but kept for readability.
_BASELINE_TIGER_DIR = Path(__file__).resolve().parent
_NECESSARY_DIR = _BASELINE_TIGER_DIR / "_data" / "necessary"
_BUILD_MOL_PKL_SCRIPT = _NECESSARY_DIR / "build_mol_pkl.py"
_BUILD_BKG_CACHE_SCRIPT = _NECESSARY_DIR / "build_bkg_cache.py"
_BUILD_SUBGRAPH_CACHE_SCRIPT = _NECESSARY_DIR / "build_subgraph_cache.py"

# Default output paths (consumed by both task baselines via __init__
# kwargs). All baseline-built artefacts get the `__mine` suffix per
# CLAUDE.md §1 line 482-484.
DEFAULT_MOL_PKL_PATH = _NECESSARY_DIR / "mol_pkl__mine.pkl"


# ---------------------------------------------------------------------
# Loader (pure load, no build)
# ---------------------------------------------------------------------

def load_mol_pkl(pkl_path: "str | Path") -> dict:
    """Load a TIGER mol-graph pkl produced by ``build_mol_pkl.py``.

    The pkl format is ``{drug_id_str: torch_geometric.data.Data}``
    (see ``build_mol_pkl.py`` docstring for Data schema).

    Raises ``FileNotFoundError`` if the file is missing — callers that
    want auto-build behaviour should use :func:`ensure_mol_pkl` instead.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"TIGER mol-graph pkl not found: {pkl_path}. Either run the "
            f"local builder ({_BUILD_MOL_PKL_SCRIPT}) or call "
            f"``ensure_mol_pkl(train, pkl_path)`` which auto-builds."
        )
    with pkl_path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# Detect-and-build (load if fresh, subprocess-build otherwise)
# ---------------------------------------------------------------------

def ensure_mol_pkl(
    train: "PairDataset",
    pkl_path: "str | Path" = DEFAULT_MOL_PKL_PATH,
    *,
    force_rebuild: bool = False,
) -> dict:
    """Auto-detect or auto-build the TIGER mol-graph pkl.

    Behaviour (mirrors :func:`baseline.hdn_ddi._shared.ensure_mol_graphs_pkl`):

      1. If ``pkl_path`` exists AND cached drug set ⊇ train.drugs →
         load + return (log ``cache HIT``).
      2. Otherwise (missing / stale / corrupt / ``force_rebuild``):
         (a) dump ``train.drugs[["drugbank_id", "smiles"]]`` to a CSV
             next to the pkl (``<pkl_stem>__smiles.csv``);
         (b) ``subprocess.run([sys.executable, build_mol_pkl.py, ...])``;
         (c) load + return.

    Raises ``FileNotFoundError`` if the builder script is missing,
    ``RuntimeError`` if the subprocess returns non-zero exit.
    """
    pkl_path = Path(pkl_path).resolve()

    # 1. Detect: pkl exists + covers train drugs?
    train_drug_ids = (
        {str(d) for d in train.drugs["drugbank_id"]}
        if train.drugs is not None
        else set()
    )
    if pkl_path.is_file() and not force_rebuild:
        try:
            cached = load_mol_pkl(pkl_path)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(
                f"[tiger] mol_pkl CORRUPT at {pkl_path} ({type(e).__name__}: "
                f"{e}); treating as stale, will rebuild",
                file=sys.stderr,
            )
        else:
            cached_ids = {str(k) for k in cached.keys()}
            missing = train_drug_ids - cached_ids
            if not missing:
                print(
                    f"[tiger] mol_pkl cache HIT: {pkl_path} "
                    f"({len(cached)} drugs cached, {len(train_drug_ids)} "
                    f"train drugs all covered)",
                    file=sys.stderr,
                )
                return cached
            print(
                f"[tiger] mol_pkl cache STALE: {pkl_path} missing "
                f"{len(missing)} train drugs -- rebuilding",
                file=sys.stderr,
            )

    # 2. Build: dump SMILES CSV → subprocess builder → load output
    if not _BUILD_MOL_PKL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"TIGER mol-pkl builder not found at {_BUILD_MOL_PKL_SCRIPT}. "
            f"Per CLAUDE.md §'Baseline 规范' §2 step 2 the baseline-side "
            f"builder must live under ``_data/necessary/``."
        )
    if train.drugs is None or "smiles" not in train.drugs.columns:
        raise ValueError(
            "auto-build of TIGER mol_pkl requires train.drugs with "
            "a 'smiles' column."
        )

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = pkl_path.with_name(pkl_path.stem + "__smiles.csv")
    train.drugs[["drugbank_id", "smiles"]].to_csv(csv_path, index=False)
    print(
        f"[tiger] dumped {len(train.drugs)} drug SMILES → {csv_path}",
        file=sys.stderr,
    )

    cmd = [
        sys.executable,
        str(_BUILD_MOL_PKL_SCRIPT),
        "--smiles-csv", str(csv_path),
        "--id-col", "drugbank_id",
        "--smiles-col", "smiles",
        "--out", str(pkl_path),
    ]
    print(
        f"[tiger] invoking mol-pkl builder subprocess:\n    {' '.join(cmd)}",
        file=sys.stderr,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"TIGER mol-pkl builder subprocess failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    last_line = proc.stderr.splitlines()[-1] if proc.stderr else "(empty)"
    print(f"[tiger] builder OK. tail stderr: {last_line}", file=sys.stderr)
    return load_mol_pkl(pkl_path)


# ---------------------------------------------------------------------
# BKG cache: detect-and-build via subprocess (CLAUDE.md §"Baseline 规范"
# §2 step 3). Cache key encodes everything that affects BKG output so
# different fold/seed/drug-pool combinations get distinct cache files.
# ---------------------------------------------------------------------

def _stable_hash(obj) -> str:
    """16-hex-char stable hash of a JSON-serialisable object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _bkg_cache_key(
    drugs: list[str],
    ddi_edges: pd.DataFrame,
    g2_drugs: list[str],
    blocklist: tuple[str, ...],
    merged_kg_path: Path,
) -> str:
    """Derive a stable hash from all inputs that affect BKG construction.
    Any change → different cache file (no silent stale).

    Includes KG file (size, mtime_ns) so an in-place parquet rewrite at
    the same path also invalidates the cache. Falls back to path-only
    string when stat fails (e.g. file deleted between callers)."""
    drugs_sorted = sorted(map(str, drugs))
    g2_sorted = sorted(map(str, g2_drugs))
    blocklist_sorted = sorted(map(str, blocklist))
    edges_signature = sorted(
        f"{r.drug_a_id}|{r.drug_b_id}|{getattr(r, 'ddi_type', '')}"
        for r in ddi_edges.itertuples()
    )
    try:
        st = merged_kg_path.stat()
        kg_sig = {"path": str(merged_kg_path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        kg_sig = {"path": str(merged_kg_path), "size": None, "mtime_ns": None}
    # KG scope (standing decision 2026-07-01: full merged KG by default). Read from env
    # so both this internal key AND the public bkg_cache_key_for (which wraps this fn) stay
    # consistent without a signature change, and full vs drug_incident caches never collide.
    kg_scope = os.environ.get("TIGER_KG_SCOPE", "full")
    return _stable_hash({
        "kg": kg_sig,
        "drugs": drugs_sorted,
        "g2": g2_sorted,
        "blocklist": blocklist_sorted,
        "edges": edges_signature,
        "kg_scope": kg_scope,
    })


def ensure_bkg_cache(
    train: "PairDataset",
    merged_kg_path: "str | Path",
    *,
    g2_drugs: list[str] | None = None,
    blocklist: tuple[str, ...] = (),
    force_rebuild: bool = False,
    cache_dir: "str | Path | None" = None,
) -> dict:
    """Auto-detect or auto-build the BKG dict cache.

    Cache key = sha256 of (drug pool, DDI edges, g2_drugs, blocklist,
    merged_kg_path). Any change → distinct cache file. Returns the same
    dict as :func:`baseline.tiger.kg_builder.build_bkg_from_merged_parquet`.
    """
    merged_kg_path = Path(merged_kg_path).resolve()
    cache_dir = Path(cache_dir).resolve() if cache_dir else _NECESSARY_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect inputs for cache-key derivation.
    all_drugs: set[str] = set()
    for _name, df in train.splits.items():
        if "drug_a_id" in df.columns:
            all_drugs |= set(map(str, df["drug_a_id"]))
            all_drugs |= set(map(str, df["drug_b_id"]))
    if train.splits.g1_drugs:
        all_drugs |= set(map(str, train.splits.g1_drugs))
    if train.splits.g2_drugs:
        all_drugs |= set(map(str, train.splits.g2_drugs))
    g2_list = list(map(str, g2_drugs if g2_drugs is not None
                       else (train.splits.g2_drugs or [])))

    key = _bkg_cache_key(
        list(all_drugs), train.splits.train, g2_list, blocklist, merged_kg_path
    )
    pkl_path = cache_dir / f"bkg_cache__{key}__mine.pkl"

    if pkl_path.is_file() and not force_rebuild:
        try:
            with pkl_path.open("rb") as f:
                cached = pickle.load(f)
            print(
                f"[tiger] bkg_cache HIT: {pkl_path} (key {key})",
                file=sys.stderr,
            )
            return cached
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(
                f"[tiger] bkg_cache CORRUPT ({type(e).__name__}: {e}); rebuilding",
                file=sys.stderr,
            )

    # Build via subprocess. Write a JSON manifest + an intermediate CSV
    # for DDI edges (too many rows for inline JSON).
    print(
        f"[tiger] bkg_cache MISS (key {key}); building via subprocess",
        file=sys.stderr,
    )
    if not _BUILD_BKG_CACHE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"BKG cache builder not found at {_BUILD_BKG_CACHE_SCRIPT}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        edges_csv = tmp_dir / "ddi_edges.csv"
        manifest_path = tmp_dir / "manifest.json"
        train.splits.train.to_csv(edges_csv, index=False)
        manifest = {
            "merged_kg_path": str(merged_kg_path),
            "drugs": sorted(all_drugs),
            "ddi_edges_path": str(edges_csv),
            "g2_drugs": sorted(g2_list),
            "blocklist": list(blocklist),
            "kg_scope": os.environ.get("TIGER_KG_SCOPE", "full"),
        }
        manifest_path.write_text(json.dumps(manifest))
        cmd = [
            sys.executable,
            str(_BUILD_BKG_CACHE_SCRIPT),
            "--manifest", str(manifest_path),
            "--out", str(pkl_path),
        ]
        print(
            f"[tiger] invoking BKG builder subprocess:\n    {' '.join(cmd[:3])} ...",
            file=sys.stderr,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"BKG cache builder failed (exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
    with pkl_path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# Subgraph cache: detect-and-build via subprocess. Cache key includes
# the BKG cache key (BKG change → subgraph invalidated) PLUS extractor
# name + hparams (different extractors → different subgraphs).
# ---------------------------------------------------------------------

def _subgraph_cache_key(
    bkg_key: str,
    extractor: str,
    extractor_params: dict,
    seed: int,
) -> str:
    return _stable_hash({
        "bkg": bkg_key,
        "extractor": extractor,
        "params": extractor_params,
        "seed": seed,
    })


def ensure_subgraph_cache(
    bkg: dict,
    bkg_cache_key: str,
    extractor: str,
    extractor_params: dict,
    *,
    seed: int = 42,
    force_rebuild: bool = False,
    cache_dir: "str | Path | None" = None,
) -> dict:
    """Auto-detect or auto-build the per-extractor subgraph cache.

    Returns a dict with keys ``subgraphs`` (the dict of Data per drug),
    ``max_degree``, ``max_sp_rel`` — same triple as
    :func:`baseline.tiger.subgraph_features.build_drug_subgraphs`.
    """
    cache_dir = Path(cache_dir).resolve() if cache_dir else _NECESSARY_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _subgraph_cache_key(bkg_cache_key, extractor, extractor_params, seed)
    pkl_path = cache_dir / f"subgraph_cache__{extractor}__{key}__mine.pkl"

    if pkl_path.is_file() and not force_rebuild:
        try:
            with pkl_path.open("rb") as f:
                cached = pickle.load(f)
            print(
                f"[tiger] subgraph_cache HIT ({extractor}): {pkl_path}",
                file=sys.stderr,
            )
            return cached
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(
                f"[tiger] subgraph_cache CORRUPT ({type(e).__name__}: {e}); rebuilding",
                file=sys.stderr,
            )

    print(
        f"[tiger] subgraph_cache MISS ({extractor}, key {key}); "
        f"building via subprocess",
        file=sys.stderr,
    )
    if not _BUILD_SUBGRAPH_CACHE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Subgraph cache builder not found at {_BUILD_SUBGRAPH_CACHE_SCRIPT}"
        )

    # Subprocess needs BKG on disk → dump it to a temp pkl, then call.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        bkg_pkl = tmp_dir / "bkg.pkl"
        with bkg_pkl.open("wb") as f:
            pickle.dump(bkg, f)
        cmd = [
            sys.executable,
            str(_BUILD_SUBGRAPH_CACHE_SCRIPT),
            "--bkg-pkl", str(bkg_pkl),
            "--extractor", extractor,
            "--extractor-params", json.dumps(extractor_params),
            "--seed", str(seed),
            "--out", str(pkl_path),
        ]
        print(
            f"[tiger] invoking subgraph builder subprocess:\n    "
            f"{' '.join(cmd[:3])} ...",
            file=sys.stderr,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subgraph cache builder failed (exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
    with pkl_path.open("rb") as f:
        return pickle.load(f)


# Re-export the bkg cache key helper so callers (binary_cls.baseline)
# can pair-up bkg_cache_key with the matching subgraph cache derivation.
def bkg_cache_key_for(
    train: "PairDataset",
    merged_kg_path: "str | Path",
    *,
    g2_drugs: list[str] | None = None,
    blocklist: tuple[str, ...] = (),
) -> str:
    """Compute the BKG cache key without building/loading the BKG.

    Used by the consumer side to feed the same key into
    :func:`ensure_subgraph_cache` so the subgraph cache is correctly
    invalidated when the BKG changes.
    """
    all_drugs: set[str] = set()
    for _name, df in train.splits.items():
        if "drug_a_id" in df.columns:
            all_drugs |= set(map(str, df["drug_a_id"]))
            all_drugs |= set(map(str, df["drug_b_id"]))
    if train.splits.g1_drugs:
        all_drugs |= set(map(str, train.splits.g1_drugs))
    if train.splits.g2_drugs:
        all_drugs |= set(map(str, train.splits.g2_drugs))
    g2_list = list(map(str, g2_drugs if g2_drugs is not None
                       else (train.splits.g2_drugs or [])))
    return _bkg_cache_key(
        list(all_drugs), train.splits.train, g2_list, blocklist,
        Path(merged_kg_path).resolve(),
    )
