"""Precompute per-drug mediator dictionary cache for PMP v1.

Each drug d gets:
  n1[d] = list of (mediator_id_str, type_id_int, rel_id_int)
          for 1-hop non-drug neighbors with their incident edge relation type
  n2[d] = list of (mediator_id_str, type_id_int)
          for 2-hop non-drug nodes (no single rel; treated as REL_2HOP special index)

KG semantics:
  - merged KG = DrugBank + Hetionet + PrimeKG (178k nodes)
  - DDI edges are ALREADY MASKED in the input parquet (`mask1` suffix), so
    nothing extra to remove
  - non-drug = node.kind not in {"Drug", "drug"}
  - n2 path: drug -> m1 -> m2 where neither m1 nor m2 is a drug; we record m2

Cache schema (single pickle):
  {
    "n1": {drug_id: [(mediator_id, type_id, rel_id), ...]},
    "n2": {drug_id: [(mediator_id, type_id), ...]},
    "type2id": {kind_group_str: type_id_int},
    "rel2id":  {relation_str: rel_id_int},
    "n_types": int,            # |type2id| (11 KG kind groups + "other")
    "n_rels": int,             # |rel2id|
    "rel_2hop_id": int,        # special index for 2-hop "pseudo-relation"
                               #   = n_rels (i.e., last index)
    "n_rels_plus_special": int # n_rels + 1
  }

Type grouping is same as `precompute_meet_features.py` (11 kind groups + "other"
catch-all), so PMP types are comparable to MNAH 22-d count types.

Output: Code/data/_cache/pmp_mediator_cache_{backbone_kg_source}_seed{seed}_kgonly_v1.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]  # pmp_v1/ directly under Code/my_code/models/

# Module-level defaults (also re-used by run_pmp_v1.py auto-cache detection).
DEFAULT_NODES_PATH = (
    PROJECT_ROOT / "Code/data/KG/_merged_kg/"
    "nodes__drugbank_hetionet_primekg.parquet"
)
DEFAULT_EDGES_PATH = (
    PROJECT_ROOT / "Code/data/KG/_merged_kg/"
    "edges__drugbank_hetionet_primekg__mask1.parquet"
)
DEFAULT_CACHE_PATH = (
    PROJECT_ROOT / "Code/data/_cache/"
    "pmp_mediator_cache_drugbank_seed42_kgonly_v1.pkl"
)


def ensure_pmp_cache(
    cache_path: str | Path | None = None,
    nodes_path: str | Path | None = None,
    edges_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Detect-and-build helper. If cache exists, return its path (HIT). Otherwise
    build it from KG parquet (MISS -> BUILT).

    Used by run_pmp_v1.py to consolidate the build + train commands. Direct
    function call (no subprocess) so caller can also use this from any script.
    """
    cache_p = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
    if cache_p.is_file() and not force:
        print(f"[pmp-cache] HIT: {cache_p} ({cache_p.stat().st_size / 1e6:.1f} MB)",
              flush=True)
        return cache_p
    print(f"[pmp-cache] {'FORCE-REBUILD' if force else 'MISS'} at {cache_p}; "
          f"building from merged KG ...", flush=True)
    nodes_p = Path(nodes_path) if nodes_path is not None else DEFAULT_NODES_PATH
    edges_p = Path(edges_path) if edges_path is not None else DEFAULT_EDGES_PATH
    if not nodes_p.is_file():
        raise FileNotFoundError(f"KG nodes parquet missing: {nodes_p}")
    if not edges_p.is_file():
        raise FileNotFoundError(f"KG edges parquet missing: {edges_p}")
    build_pmp_cache(nodes_p, edges_p, cache_p)
    print(f"[pmp-cache] BUILT: {cache_p}", flush=True)
    return cache_p


# Same 11 kind groups as MNAH (Code/my_code/models/screen_s2_v2_meetnode/precompute_meet_features.py)
KIND_GROUPS = {
    "protein_gene": ["Gene", "gene/protein", "Protein"],
    "pathway": ["Pathway", "pathway"],
    "side_effect": ["Side Effect", "effect/phenotype", "Symptom"],
    "disease": ["Disease", "disease"],
    "anatomy": ["Anatomy", "anatomy"],
    "compound": ["Compound"],
    "biological_process": ["Biological Process", "biological_process"],
    "molecular_function": ["Molecular Function", "molecular_function"],
    "cellular_component": ["Cellular Component", "cellular_component"],
    "pharmacologic_class": ["Pharmacologic Class"],
    "exposure": ["exposure"],
}
KIND_ORDER = list(KIND_GROUPS.keys()) + ["other"]
KIND_TO_GROUP: dict[str, str] = {}
for grp, kinds in KIND_GROUPS.items():
    for k in kinds:
        KIND_TO_GROUP[k] = grp


def _kind_to_type_id(kind: str, type2id: dict[str, int]) -> int:
    grp = KIND_TO_GROUP.get(kind, "other")
    return type2id[grp]


def build_pmp_cache(
    nodes_path: Path,
    edges_path: Path,
    out_path: Path,
) -> None:
    print(f"[pmp-pre] loading nodes {nodes_path}", flush=True)
    nodes = pd.read_parquet(nodes_path)
    print(f"[pmp-pre] loading edges {edges_path}", flush=True)
    edges = pd.read_parquet(edges_path)
    print(f"[pmp-pre] nodes={len(nodes)} edges={len(edges)}", flush=True)

    id2kind = dict(zip(nodes["id"].astype(str), nodes["kind"].astype(str)))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"].astype(str))
    print(f"[pmp-pre] |drugs| = {len(drug_set)}", flush=True)

    # Type vocabulary: 11 KG kind groups + "other" catch-all
    type2id = {k: i for i, k in enumerate(KIND_ORDER)}
    n_types = len(type2id)

    # Relation vocabulary: sorted unique edge relations from the parquet
    rel_values = sorted(edges["relation"].astype(str).unique().tolist())
    rel2id = {r: i for i, r in enumerate(rel_values)}
    n_rels = len(rel2id)
    rel_2hop_id = n_rels   # special pseudo-relation for 2-hop mediators
    n_rels_plus_special = n_rels + 1
    print(f"[pmp-pre] n_types={n_types} n_rels={n_rels} rel_2hop_id={rel_2hop_id}", flush=True)

    # Build n1: drug -> [(mediator_id_str, type_id, rel_id), ...]
    #          and fwd: node_id -> set of neighbor_id (no rel; for 2-hop traversal)
    print("[pmp-pre] building n1 (1-hop drug -> non-drug) ...", flush=True)
    n1_tmp: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    fwd: dict[str, set[str]] = defaultdict(set)
    t0 = time.time()
    for i, (src, dst, rel, directed) in enumerate(zip(
        edges["src"].astype(str),
        edges["dst"].astype(str),
        edges["relation"].astype(str),
        edges["directed"].astype(bool),
    )):
        # always populate fwd (undirected adds reverse direction)
        fwd[src].add(dst)
        if not directed:
            fwd[dst].add(src)
        src_is_drug = src in drug_set
        dst_is_drug = dst in drug_set
        rel_id = rel2id[rel]
        # drug -> non-drug 1-hop
        if src_is_drug and not dst_is_drug:
            type_id = _kind_to_type_id(id2kind.get(dst, ""), type2id)
            n1_tmp[src].add((dst, type_id, rel_id))
        # non-drug -> drug 1-hop only if undirected (drug as dst of incoming edge)
        if dst_is_drug and (not src_is_drug) and (not directed):
            type_id = _kind_to_type_id(id2kind.get(src, ""), type2id)
            n1_tmp[dst].add((src, type_id, rel_id))
        if (i + 1) % 1_000_000 == 0:
            print(f"[pmp-pre]   processed {i + 1}/{len(edges)} edges  "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    print(f"[pmp-pre] n1 built: drugs with >=1 1-hop non-drug nbr = "
          f"{sum(1 for d in drug_set if d in n1_tmp)}/{len(drug_set)}  "
          f"time={time.time() - t0:.1f}s", flush=True)

    # Build n2: 2-hop non-drug nodes via drug -> m1 -> m2 where neither m1, m2 is drug
    print("[pmp-pre] building n2 (2-hop drug -> non-drug via non-drug pivot) ...", flush=True)
    n2_tmp: dict[str, set[tuple[str, int]]] = defaultdict(set)
    t0 = time.time()
    for di, drug in enumerate(drug_set):
        for m1_id, _, _ in n1_tmp.get(drug, ()):
            for m2_id in fwd.get(m1_id, ()):
                if m2_id == drug or m2_id in drug_set:
                    continue
                type_id = _kind_to_type_id(id2kind.get(m2_id, ""), type2id)
                n2_tmp[drug].add((m2_id, type_id))
        if (di + 1) % 1000 == 0:
            print(f"[pmp-pre]   n2 progress {di + 1}/{len(drug_set)}  "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    print(f"[pmp-pre] n2 built: drugs with >=1 2-hop non-drug = "
          f"{sum(1 for d in drug_set if d in n2_tmp)}/{len(drug_set)}  "
          f"time={time.time() - t0:.1f}s", flush=True)

    # Convert sets -> sorted lists for deterministic ordering
    n1_final = {d: sorted(tuples) for d, tuples in n1_tmp.items()}
    n2_final = {d: sorted(tuples) for d, tuples in n2_tmp.items()}

    # Build PMP's own mediator vocabulary (codex r2 fix #2: decouple from
    # backbone entity2id). Union of all mediator node IDs across n1 and n2 of
    # all drugs. PMP head will allocate a trainable embedding table sized to
    # |pmp_mediator_set| so that ALL merged-KG mediators get a learnable
    # representation, regardless of whether they appear in the backbone's
    # entity vocab (which may be smaller, e.g., drugbank-only).
    pmp_mediator_set: set[str] = set()
    for drug_id, tuples in n1_final.items():
        for m_id, _, _ in tuples:
            pmp_mediator_set.add(m_id)
    for drug_id, tuples in n2_final.items():
        for m_id, _ in tuples:
            pmp_mediator_set.add(m_id)
    pmp_mediator2id = {m_id: i for i, m_id in enumerate(sorted(pmp_mediator_set))}
    n_mediators = len(pmp_mediator2id)
    print(f"[pmp-pre] PMP mediator vocab: |pmp_mediator2id|={n_mediators}", flush=True)

    payload = {
        "n1": n1_final,
        "n2": n2_final,
        "type2id": type2id,
        "rel2id": rel2id,
        "n_types": n_types,
        "n_rels": n_rels,
        "rel_2hop_id": rel_2hop_id,
        "n_rels_plus_special": n_rels_plus_special,
        # PMP-internal mediator vocabulary (decoupled from backbone entity2id)
        "pmp_mediator2id": pmp_mediator2id,
        "n_mediators": n_mediators,
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "schema_version": "pmp_v1_codex_r2",  # bumped for the fix
    }
    print(f"[pmp-pre] writing -> {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[pmp-pre] DONE. cache size: {out_path.stat().st_size / 1e6:.1f} MB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=str, default=str(DEFAULT_NODES_PATH))
    p.add_argument("--edges", type=str, default=str(DEFAULT_EDGES_PATH))
    p.add_argument("--out",   type=str, default=str(DEFAULT_CACHE_PATH))
    p.add_argument("--force", action="store_true",
                   help="rebuild even if cache exists")
    args = p.parse_args()
    ensure_pmp_cache(
        cache_path=args.out,
        nodes_path=args.nodes,
        edges_path=args.edges,
        force=args.force,
    )


if __name__ == "__main__":
    main()
