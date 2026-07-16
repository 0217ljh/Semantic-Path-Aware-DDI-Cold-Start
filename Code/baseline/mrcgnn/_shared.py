"""Cross-task shared helpers for the MRCGNN baseline: TrimNet feature cache.

Used by the (later-step) multi_cls training core. Per CLAUDE.md §Baseline §1, shared
code lives at the package top level. This module implements the baseline-side
detect-and-autobuild pipeline (§Baseline §2 step 3): detect the cache -> missing /
stale / corrupt -> subprocess-build THIS package's builder -> load; log HIT / STALE /
BUILT. Builder subprocess uses ``sys.executable`` so the conda env is preserved.

CACHE KEY (no cross-leaf leakage): the cache filename encodes a hash of
  (ORDERED (drug_id, smiles) pairs — as given, NOT sorted, so a permuted drug_order
     or an edited SMILES under a stable id invalidates,
   sorted TRAIN triples (drug_a, drug_b, global ddi_type),
   n_classes = K_global,
   epochs — the training recipe (a 5-epoch smoke must NOT be reused as a 300-epoch build),
   fold id,
   seed)
so different leaves / folds / train-splits / recipes / drug-orders get DISTINCT caches.
Because TrimNet is trained on TRAIN pairs only (never val/test), and the key includes
the exact train triple set, two leaves that differ only in their test split (but share
train + drug order + SMILES + recipe) still correctly share a cache — while any change
to the train pairs, the drug ORDER, the SMILES text, or the epoch count invalidates it.

Row-order contract: the emitted matrix rows follow ``drug_order`` AS GIVEN, so the order
is load-bearing. A permuted drug_order must yield a distinct key (else the cache would
return row-misaligned features). We ALSO persist ``drug_order`` in a JSON sidecar next
to the .npy and validate it on load (defence-in-depth against a hash collision).

The builder writes ``mrcgnn_trimnet_features__<hash>__mine.npy`` (``__mine`` suffix per
§Baseline §1: OUR builder product, no upstream official counterpart).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent
_BUILDER_SCRIPT = _PKG_DIR / "_data" / "necessary" / "build_trimnet_features.py"
_CACHE_DIR = _PKG_DIR / "_data" / "necessary"


def trimnet_cache_key(drug_order: list[str], smiles_map: dict[str, str],
                      train_tri: list[tuple[str, str, int]],
                      *, n_classes: int, epochs: int, fold: object, seed: int) -> str:
    """Stable short hash of the cold-start-relevant inputs (see module docstring).

    Encodes the ORDERED (drug_id, smiles) sequence (NOT sorted — order is load-bearing
    and SMILES text can change under a stable id), the sorted TRAIN triples, and the
    training recipe (n_classes, epochs, fold, seed)."""
    h = hashlib.sha1()
    # ORDERED (drug_id, smiles) — fix #2 (order) + fix #1 (SMILES content).
    ordered = "|".join(f"{d}\t{smiles_map.get(str(d), '')}" for d in drug_order)
    h.update(ordered.encode())
    h.update(b"\x00")
    tri_norm = sorted((str(a), str(b), int(r)) for a, b, r in train_tri)
    h.update(("|".join(f"{a},{b},{r}" for a, b, r in tri_norm)).encode())
    h.update(b"\x00")
    # training recipe — fix #1 (epochs) so a smoke build never masquerades as the real one.
    h.update(f"K={int(n_classes)};epochs={int(epochs)};fold={fold};seed={int(seed)}".encode())
    return h.hexdigest()[:16]


def cache_path_for(key: str) -> Path:
    return _CACHE_DIR / f"mrcgnn_trimnet_features__{key}__mine.npy"


def _sidecar_path_for(key: str) -> Path:
    return _CACHE_DIR / f"mrcgnn_trimnet_features__{key}__drug_order.json"


def ensure_trimnet_features(
    drug_order: list[str],
    smiles_map: dict[str, str],
    train_tri: list[tuple[str, str, int]],
    *,
    n_classes: int,
    fold: object = 0,
    seed: int = 42,
    epochs: int = 300,
    device: str = "auto",
    force_rebuild: bool = False,
) -> np.ndarray:
    """Detect-or-build the TrimNet feature matrix; return (len(drug_order), 128).

    Detect: cache exists AND has the right #rows -> HIT (load). Otherwise (missing /
    wrong-shape / corrupt / ``force_rebuild``) -> dump the builder's CSV inputs,
    subprocess the local builder, load. Logs a stderr line for the branch taken.
    """
    key = trimnet_cache_key(drug_order, smiles_map, train_tri, n_classes=n_classes,
                            epochs=epochs, fold=fold, seed=seed)
    out_path = cache_path_for(key).resolve()
    sidecar = _sidecar_path_for(key).resolve()

    if out_path.is_file() and not force_rebuild:
        try:
            feats = np.load(out_path)
        except (ValueError, OSError) as e:
            print(f"[mrcgnn] trimnet cache CORRUPT at {out_path} "
                  f"({type(e).__name__}: {e}); rebuilding", file=sys.stderr)
        else:
            # Validate the persisted drug_order sidecar (defence-in-depth vs a hash
            # collision): rows must align to THIS drug_order exactly.
            cached_order = None
            if sidecar.is_file():
                try:
                    cached_order = json.loads(sidecar.read_text())
                except (ValueError, OSError):
                    cached_order = None
            order_ok = cached_order == [str(d) for d in drug_order]
            if feats.shape[0] == len(drug_order) and order_ok:
                print(f"[mrcgnn] trimnet cache HIT: {out_path} shape={feats.shape}",
                      file=sys.stderr)
                return feats.astype(np.float32)
            if not order_ok:
                print(f"[mrcgnn] trimnet cache STALE: {out_path} drug_order sidecar "
                      f"mismatch (or missing); rebuilding", file=sys.stderr)
            else:
                print(f"[mrcgnn] trimnet cache STALE: {out_path} rows={feats.shape[0]} "
                      f"!= n_drugs={len(drug_order)}; rebuilding", file=sys.stderr)
    else:
        print(f"[mrcgnn] trimnet cache MISSING at {out_path} -- will build",
              file=sys.stderr)

    if not _BUILDER_SCRIPT.is_file():
        raise FileNotFoundError(
            f"MRCGNN TrimNet builder not found at {_BUILDER_SCRIPT} "
            f"(CLAUDE.md §Baseline §1 — builder lives at _data/necessary/).")

    # Dump the builder's CSV inputs next to the cache (deterministic from the key).
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    smiles_csv = _CACHE_DIR / f"_trimnet_input__{key}__smiles.csv"
    pairs_csv = _CACHE_DIR / f"_trimnet_input__{key}__train_pairs.csv"
    pd.DataFrame({"drug_id": [str(d) for d in drug_order],
                  "smiles": [smiles_map.get(str(d), "") for d in drug_order]}
                 ).to_csv(smiles_csv, index=False)
    pd.DataFrame({"drug_a_id": [str(a) for a, _, _ in train_tri],
                  "drug_b_id": [str(b) for _, b, _ in train_tri],
                  "ddi_type": [int(r) for _, _, r in train_tri]}
                 ).to_csv(pairs_csv, index=False)

    cmd = [
        sys.executable, str(_BUILDER_SCRIPT),
        "--smiles-csv", str(smiles_csv), "--id-col", "drug_id", "--smiles-col", "smiles",
        "--train-pairs-csv", str(pairs_csv),
        "--a-col", "drug_a_id", "--b-col", "drug_b_id", "--type-col", "ddi_type",
        "--n-classes", str(int(n_classes)), "--epochs", str(int(epochs)),
        "--seed", str(int(seed)), "--device", str(device),
        "--out", str(out_path),
    ]
    print(f"[mrcgnn] invoking trimnet builder:\n    {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MRCGNN TrimNet builder failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    # Persist the drug_order sidecar so a later load can validate row alignment.
    sidecar.write_text(json.dumps([str(d) for d in drug_order]))
    print(f"[mrcgnn] trimnet cache BUILT at {out_path}. tail stderr:\n"
          f"{proc.stderr.splitlines()[-1] if proc.stderr else '(empty)'}",
          file=sys.stderr)
    return np.load(out_path).astype(np.float32)


__all__ = ["trimnet_cache_key", "cache_path_for", "ensure_trimnet_features"]
