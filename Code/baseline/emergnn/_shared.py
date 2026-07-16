"""Cross-task shared helpers for EmerGNN baselines.

Per CLAUDE.md §"Baseline 规范" §1 line 475 "可选 共享工具 (e.g.
detect-and-build pipeline)" — this file holds the detect-and-build
pipeline backing all baseline-side `__mine` cache artefacts.

Currently supported cache:
  - **kg_setup** (merged-KG source only): combined cache of
    entity2id + n_ent + n_base_rel + kg_triplets + kg_entity_set +
    edge_src/dst/rel + morgan_mat + drug_id_list + missing_smiles.
    File: ``_data/necessary/kg_setup__<hash>__mine.pkl``

For the legacy ``backbone_kg_source="drugbank"`` path (5-bucket schema from
``train.kg``), caching is NOT implemented — the input is a transient
Python object with 5 DataFrames whose content hashing would be
complex. The in-memory fallback path in
``binary_cls/baseline.py._setup_graph`` covers that case.

Per CLAUDE.md §"Baseline 规范" §2 step 3: detect-and-build via
subprocess. Cache key encodes all inputs that affect setup output
so cross-fold/seed runs get distinct caches (no silent stale).
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

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset


_BASELINE_EMERGNN_DIR = Path(__file__).resolve().parent
_NECESSARY_DIR = _BASELINE_EMERGNN_DIR / "_data" / "necessary"
_BUILD_KG_SETUP_SCRIPT = _NECESSARY_DIR / "build_kg_setup_cache.py"


# ---------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------

def _stable_hash(obj) -> str:
    """16-hex-char stable sha256 of a JSON-serialisable object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _kg_setup_cache_key(
    backbone_kg_source: str,
    drug_id_list: list[str],
    smiles_map: dict[str, str],
    merged_kg_path: Path | None,
    blocklist: tuple[str, ...],
) -> str:
    """Derive a stable hash from all inputs that affect kg_setup output.

    Includes (size, mtime_ns) of merged_kg_path so in-place file rewrite
    also invalidates the cache (matches TIGER/HDN pattern)."""
    drugs_sorted = sorted(map(str, drug_id_list))
    blocklist_sorted = sorted(map(str, blocklist))
    smiles_signature = sorted(
        f"{d}|{smiles_map.get(d, '')}" for d in drugs_sorted
    )
    kg_sig: dict
    if merged_kg_path is not None:
        try:
            st = merged_kg_path.stat()
            kg_sig = {
                "path": str(merged_kg_path),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        except OSError:
            kg_sig = {"path": str(merged_kg_path), "size": None, "mtime_ns": None}
    else:
        kg_sig = {"path": None}
    return _stable_hash({
        "backbone_kg_source": backbone_kg_source,
        "drugs": drugs_sorted,
        "smiles": smiles_signature,
        "kg": kg_sig,
        "blocklist": blocklist_sorted,
    })


# ---------------------------------------------------------------------
# Loader (pure load, no build)
# ---------------------------------------------------------------------

def load_kg_setup_cache(pkl_path: "str | Path") -> dict:
    """Load a pre-built kg_setup pkl produced by ``build_kg_setup_cache.py``.

    Cache dict shape:
      - entity2id: dict[str, int]
      - n_ent: int
      - n_base_rel: int
      - kg_triplets: np.ndarray (h, t, r) shape (N, 3) int64
      - kg_entity_set: list[int] (cast to set by consumer)
      - edge_src, edge_dst, edge_rel: np.ndarray int64 (cast to torch by consumer)
      - morgan_mat: np.ndarray float32 shape (n_ent, 1024)
      - drug_id_list: list[str] sorted
      - missing_smiles: list[str]
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"EmerGNN kg_setup pkl not found: {pkl_path}. Run the local "
            f"builder ({_BUILD_KG_SETUP_SCRIPT}) or call "
            f"``ensure_kg_setup_cache(...)`` which auto-builds."
        )
    with pkl_path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# Detect-and-build via subprocess
# ---------------------------------------------------------------------

def ensure_kg_setup_cache(
    train: "PairDataset",
    *,
    backbone_kg_source: str,
    merged_kg_path: "str | Path | None" = None,
    blocklist: tuple[str, ...] = (),
    force_rebuild: bool = False,
    cache_dir: "str | Path | None" = None,
) -> dict:
    """Auto-detect or auto-build the EmerGNN kg_setup cache.

    Only supports ``backbone_kg_source="merged"``; for "drugbank" raises
    NotImplementedError (caller should fall back to in-memory build).

    Cache key encodes drug pool, SMILES, KG file (path+size+mtime),
    blocklist. Cache file: ``_data/necessary/kg_setup__<hash>__mine.pkl``.
    """
    if backbone_kg_source != "merged":
        raise NotImplementedError(
            f"ensure_kg_setup_cache only supports backbone_kg_source='merged'; "
            f"got {backbone_kg_source!r}. Use the in-memory _setup_graph fallback."
        )
    if merged_kg_path is None:
        raise ValueError("ensure_kg_setup_cache requires merged_kg_path")
    merged_kg_path = Path(merged_kg_path).resolve()
    cache_dir = Path(cache_dir).resolve() if cache_dir else _NECESSARY_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Inputs for cache-key derivation.
    drug_ids: set[str] = set()
    for _name, df in train.splits.items():
        if "drug_a_id" in df.columns:
            drug_ids.update(map(str, df["drug_a_id"]))
            drug_ids.update(map(str, df["drug_b_id"]))
    drug_id_list = sorted(drug_ids)

    smiles_map: dict[str, str] = {}
    if train.drugs is not None and "smiles" in train.drugs.columns:
        for _, row in train.drugs[["drugbank_id", "smiles"]].iterrows():
            smiles_map[str(row["drugbank_id"])] = (
                "" if pd.isna(row["smiles"]) else str(row["smiles"])
            )

    # KG scope selector (env-var driven so no signature change ripples through
    # _per_mode/core): "drug_incident" (default, 1-hop drug neighborhood) or
    # "full" (whole merged KG). Folded into the cache key ONLY for the non-default
    # scope, so existing drug_incident caches keep their exact hash/path.
    kg_scope = os.environ.get("EMERGNN_KG_SCOPE", "drug_incident")
    if kg_scope not in ("drug_incident", "full"):
        raise ValueError(
            f"EMERGNN_KG_SCOPE must be 'drug_incident' or 'full'; got {kg_scope!r}"
        )

    key = _kg_setup_cache_key(
        backbone_kg_source, drug_id_list, smiles_map, merged_kg_path, blocklist
    )
    if kg_scope != "drug_incident":
        key = _stable_hash({"base": key, "kg_scope": kg_scope})
    pkl_path = cache_dir / f"kg_setup__{key}__mine.pkl"

    if pkl_path.is_file() and not force_rebuild:
        try:
            cached = load_kg_setup_cache(pkl_path)
            print(
                f"[emergnn] kg_setup cache HIT: {pkl_path} (key {key})",
                file=sys.stderr,
            )
            return cached
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(
                f"[emergnn] kg_setup CORRUPT ({type(e).__name__}: {e}); rebuilding",
                file=sys.stderr,
            )

    # Distinguish "first-time miss" from "stale rebuild" for clearer
    # diagnostics — per CLAUDE.md §"必需文件 detection pipeline" line 635
    # baseline log labels: "cache HIT / STALE rebuild / BUILT".
    # STALE = sibling cache files exist with DIFFERENT hash (inputs
    # changed since last build). Excludes the current key's path so
    # ``CORRUPT`` rebuild + ``force_rebuild=True`` don't mislabel as STALE.
    sibling_stale = [
        p for p in cache_dir.glob("kg_setup__*__mine.pkl")
        if p.name != pkl_path.name
    ]
    if sibling_stale:
        print(
            f"[emergnn] kg_setup STALE (key {key} not in cache; "
            f"{len(sibling_stale)} sibling kg_setup pkl(s) exist with "
            f"different inputs — rebuilding)",
            file=sys.stderr,
        )
    else:
        print(
            f"[emergnn] kg_setup MISS first-time (key {key}); building via subprocess",
            file=sys.stderr,
        )
    if not _BUILD_KG_SETUP_SCRIPT.is_file():
        raise FileNotFoundError(
            f"kg_setup builder missing at {_BUILD_KG_SETUP_SCRIPT}. Per "
            f"CLAUDE.md §'Baseline 规范' §2 step 2 the baseline-side "
            f"builder must live under ``_data/necessary/``."
        )

    # Subprocess via JSON manifest (drugs list + smiles map can be 1000s
    # of entries — too long for argparse).
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "manifest.json"
        manifest = {
            "backbone_kg_source": backbone_kg_source,
            "merged_kg_path": str(merged_kg_path),
            "blocklist": list(blocklist),
            "kg_scope": kg_scope,
            "drug_id_list": drug_id_list,
            "smiles_map": smiles_map,
        }
        manifest_path.write_text(json.dumps(manifest))
        cmd = [
            sys.executable,
            str(_BUILD_KG_SETUP_SCRIPT),
            "--manifest", str(manifest_path),
            "--out", str(pkl_path),
        ]
        print(
            f"[emergnn] invoking kg_setup builder subprocess:\n    "
            f"{' '.join(cmd[:3])} ...",
            file=sys.stderr,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"kg_setup builder failed (exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
    print(
        f"[emergnn] kg_setup BUILT (key {key}); wrote {pkl_path}",
        file=sys.stderr,
    )

    return load_kg_setup_cache(pkl_path)
