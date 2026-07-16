"""Builder for EmerGNN kg_setup combined cache.

Per CLAUDE.md §"Baseline 规范" §2 step 3: this is the baseline-side
CLI builder invoked via subprocess by
:func:`baseline.emergnn._shared.ensure_kg_setup_cache`. Takes a JSON
manifest, runs the same logic as
``baseline.emergnn.binary_cls.baseline.EmerGNNBaseline._setup_graph``
(merged-KG path only — drugbank legacy 5-bucket path uses in-memory
fallback), saves the combined dict to ``--out`` pkl.

Manifest schema::

    {
      "backbone_kg_source": "merged",
      "merged_kg_path": "<absolute path to parquet>",
      "blocklist": [<str>, ...],
      "kg_scope": "drug_incident" | "full",          # optional, default drug_incident
      "drug_id_list": [<drugbank_id_str>, ...],     # sorted
      "smiles_map": {<drugbank_id_str>: <smiles>}    # all drugs in pool
    }

Output pkl shape::

    {
      "entity2id": dict[str, int],
      "n_ent": int,
      "n_base_rel": int,
      "kg_triplets": np.ndarray int64 shape (N, 3),  # (h, t, r)
      "kg_entity_set": list[int],
      "edge_src": np.ndarray int64,
      "edge_dst": np.ndarray int64,
      "edge_rel": np.ndarray int64,
      "morgan_mat": np.ndarray float32 shape (n_ent, 1024),
      "drug_id_list": list[str],
      "missing_smiles": list[str],
    }

CLI::

    python build_kg_setup_cache.py \\
        --manifest <path-to-manifest.json> \\
        --out _data/necessary/kg_setup__<hash>__mine.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
# parents math from ``Code/baseline/emergnn/_data/necessary/build_kg_setup_cache.py``:
#   parents[0] = necessary/   parents[3] = baseline/
#   parents[1] = _data/       parents[4] = Code/
#   parents[2] = emergnn/
_CODE_ROOT = _HERE.parents[4]
sys.path.insert(0, str(_CODE_ROOT))

from baseline.emergnn.kg_builder import (  # noqa: E402
    build_kg_from_kb,
    build_sparse_adj,
    edges_as_dense_lists,
    N_BASE_REL,
)
from baseline.emergnn.kg_builder_merged import (  # noqa: E402
    build_kg_from_merged_parquet,
    build_full_kg_from_merged_parquet,
)
from baseline.emergnn.morgan_features import compute_morgan_matrix  # noqa: E402


def build_kg_setup(
    backbone_kg_source: str,
    drug_id_list: list[str],
    smiles_map: dict[str, str],
    *,
    merged_kg_path: Path | None = None,
    blocklist: tuple[str, ...] = (),
    kg_scope: str = "drug_incident",
    verbose: bool = True,
) -> dict:
    """Run the same logic as
    ``EmerGNNBaseline._setup_graph`` and package the result as a
    pickle-able dict.

    ``kg_scope`` selects the merged-KG builder:
      - ``"drug_incident"`` (default): 1-hop drug neighborhood
        (:func:`build_kg_from_merged_parquet`) — original behavior.
      - ``"full"``: the whole merged KG
        (:func:`build_full_kg_from_merged_parquet`).
    """
    if backbone_kg_source != "merged":
        raise NotImplementedError(
            f"builder only supports backbone_kg_source='merged'; got {backbone_kg_source!r}"
        )
    if merged_kg_path is None:
        raise ValueError("merged_kg_path required for backbone_kg_source='merged'")
    if kg_scope not in ("drug_incident", "full"):
        raise ValueError(
            f"kg_scope must be 'drug_incident' or 'full'; got {kg_scope!r}"
        )

    _builder = (
        build_full_kg_from_merged_parquet
        if kg_scope == "full"
        else build_kg_from_merged_parquet
    )
    kg_artifacts = _builder(
        merged_kg_path,
        drug_id_list,
        blocklist=blocklist,
        verbose=verbose,
    )
    n_base_rel = kg_artifacts["n_rel"]

    entity2id = kg_artifacts["entity2id"]
    n_ent = kg_artifacts["n_ent"]
    triplets_arr = np.asarray(kg_artifacts["triplets"], dtype=np.int64)

    # KG entity union (every entity that appears in any KG triplet
    # head/tail), used by shuffle_train extra_kg_ent at training time.
    if len(triplets_arr):
        kg_entity_set_list = sorted(
            int(x) for x in np.unique(triplets_arr[:, :2]).astype(np.int64).tolist()
        )
    else:
        kg_entity_set_list = []

    # Edge tensors (forward + reverse + self-loop) — same as
    # ``_setup_graph`` lines 250-253.
    adj = build_sparse_adj(kg_artifacts["triplets"], n_ent, n_base_rel)
    src_t, dst_t, rel_t = edges_as_dense_lists(adj)
    edge_src = np.asarray(src_t.cpu().numpy() if hasattr(src_t, "cpu") else src_t, dtype=np.int64)
    edge_dst = np.asarray(dst_t.cpu().numpy() if hasattr(dst_t, "cpu") else dst_t, dtype=np.int64)
    edge_rel = np.asarray(rel_t.cpu().numpy() if hasattr(rel_t, "cpu") else rel_t, dtype=np.int64)

    # Morgan features (zeros for non-drug entities).
    morgan_mat = np.zeros((n_ent, 1024), dtype=np.float32)
    drug_only_mat, missing = compute_morgan_matrix(drug_id_list, smiles_map)
    for did, row in zip(drug_id_list, drug_only_mat):
        if did in entity2id:
            morgan_mat[entity2id[did]] = row
    if verbose and missing:
        print(
            f"[build_kg_setup] {len(missing)} drugs lacked parseable SMILES "
            f"and were zero-filled.",
            file=sys.stderr,
        )

    return {
        "entity2id": entity2id,
        "n_ent": int(n_ent),
        "n_base_rel": int(n_base_rel),
        "kg_triplets": triplets_arr,
        "kg_entity_set": kg_entity_set_list,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_rel": edge_rel,
        "morgan_mat": morgan_mat,
        "drug_id_list": list(drug_id_list),
        "missing_smiles": list(missing),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build EmerGNN kg_setup combined cache from a manifest JSON."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="JSON file: see module docstring for schema.",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output pkl path.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    with args.manifest.open("r", encoding="utf-8") as f:
        m = json.load(f)

    payload = build_kg_setup(
        backbone_kg_source=m["backbone_kg_source"],
        drug_id_list=m["drug_id_list"],
        smiles_map=m["smiles_map"],
        merged_kg_path=Path(m["merged_kg_path"]) if m.get("merged_kg_path") else None,
        blocklist=tuple(m.get("blocklist", [])),
        kg_scope=m.get("kg_scope", "drug_incident"),
        verbose=not args.quiet,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(payload, f)
    print(
        f"Wrote {args.out}  (n_ent={payload['n_ent']}, "
        f"n_base_rel={payload['n_base_rel']}, "
        f"triplets={len(payload['kg_triplets'])}, "
        f"morgan_shape={payload['morgan_mat'].shape})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
