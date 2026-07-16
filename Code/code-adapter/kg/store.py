"""Stage 2 / atom A1 - KG store (adjacency + node types + directed edges).

Single responsibility (codex-reviewed): load ONE merged-KG version into an
in-memory store exposing two views built from a single edge pass -

* UNDIRECTED reachability view (symmetrize every edge + dedup neighbor nodes)
  for mediator finding downstream. `directed=False` edges are stored once so we
  add both directions for all edges; direction is a semantic property handled
  later by the relation term, not a reachability property.
* DIRECTED multigraph view (all edge rows kept, with relation + raw `directed`)
  for the later relation-term / path atom. Parallel edges preserved.

NOT A1's job: L-hop traversal, shared-mediator M_uv, semantics, path scoring,
interpreting the `directed` flag.

Configurable: `kg_dir` (which merged-KG version) and `kind_map` (canonicalization
in kinds.py, overridable). Output is a versioned cache dir that IS the downstream
interface; switching KG = rebuild, downstream reads whichever.

Cache layout: `kg/_cache/kg_store__<kgname>__<fp>/{arrays.npz, meta.json, idmap.parquet}`.
Fingerprint covers normalized kind_map + nodes/edges size+mtime + schema version.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kg.kinds import DRUG_CANONICAL, KIND_CANONICAL, UNKNOWN_CANONICAL

_ROOT = Path(__file__).resolve().parents[3]                       # project root
_KG_ROOT = _ROOT / "Code" / "data" / "KG"
_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache"    # code-adapter/kg/_cache
DEFAULT_KG_DIR = _KG_ROOT / "_merged_kg"
SCHEMA_VERSION = "kgstore_v1"


def _kg_files(kg_dir: Path) -> tuple[Path, Path]:
    nodes = sorted(kg_dir.glob("nodes*.parquet"))
    edges = sorted(kg_dir.glob("edges*.parquet"))
    if len(nodes) != 1 or len(edges) != 1:
        raise FileNotFoundError(
            f"KG dir must contain exactly one nodes*.parquet and one edges*.parquet: "
            f"{kg_dir} (nodes={[p.name for p in nodes]}, edges={[p.name for p in edges]})")
    return nodes[0], edges[0]


def _fingerprint(nodes_p: Path, edges_p: Path, kind_map: dict) -> str:
    def finfo(p: Path):
        st = p.stat()
        return [p.name, st.st_size, int(st.st_mtime)]
    payload = json.dumps({"schema": SCHEMA_VERSION, "kind_map": kind_map,
                          "nodes": finfo(nodes_p), "edges": finfo(edges_p)},
                         sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


class KGStore:
    """String-id API over int-remapped arrays. Built by build_kg_store."""

    def __init__(self, idx2node: np.ndarray, type_id: np.ndarray, type_names: list[str],
                 drug_mask: np.ndarray, u_indptr: np.ndarray, u_indices: np.ndarray,
                 d_indptr: np.ndarray, d_dst: np.ndarray, d_rel: np.ndarray,
                 d_directed: np.ndarray, rel_names: list[str], meta: dict):
        self.idx2node = idx2node
        self.node2idx = {n: i for i, n in enumerate(idx2node.tolist())}
        self.type_id = type_id; self.type_names = type_names
        self.drug_mask = drug_mask
        self.u_indptr = u_indptr; self.u_indices = u_indices
        self.d_indptr = d_indptr; self.d_dst = d_dst; self.d_rel = d_rel
        self.d_directed = d_directed; self.rel_names = rel_names
        self.meta = meta

    @property
    def n_nodes(self) -> int:
        return len(self.idx2node)

    def get_idx(self, node_id: str):
        return self.node2idx.get(str(node_id))

    def get_id(self, idx: int) -> str:
        return str(self.idx2node[idx])

    def neighbors(self, node_id: str) -> np.ndarray:
        """Undirected neighbor node-ids (deduped). String-space convenience wrapper."""
        i = self.node2idx.get(str(node_id))
        if i is None:
            return np.array([], dtype=object)
        return self.idx2node[self.u_indices[self.u_indptr[i]:self.u_indptr[i + 1]]]

    def neighbors_idx(self, i: int) -> np.ndarray:
        """Undirected neighbor INDICES (idx-space; hot-path for A2/A3 traversal)."""
        return self.u_indices[self.u_indptr[i]:self.u_indptr[i + 1]]

    def edges_from_idx(self, i: int):
        """Directed out-edges of node idx i as idx-space arrays (dst_idx, rel_idx,
        directed), all rows kept. Hot-path for the later relation/path atom."""
        s, e = self.d_indptr[i], self.d_indptr[i + 1]
        return self.d_dst[s:e], self.d_rel[s:e], self.d_directed[s:e]

    def node_type(self, node_id: str) -> str | None:
        i = self.node2idx.get(str(node_id))
        return self.type_names[self.type_id[i]] if i is not None else None

    def is_drug(self, node_id: str) -> bool:
        i = self.node2idx.get(str(node_id))
        return bool(self.drug_mask[i]) if i is not None else False

    def drug_ids(self) -> np.ndarray:
        return self.idx2node[self.drug_mask]

    def edge_rows_from(self, node_id: str):
        """Directed edges out of node: array of (dst_id, relation, directed), all kept."""
        i = self.node2idx.get(str(node_id))
        if i is None:
            return []
        s, e = self.d_indptr[i], self.d_indptr[i + 1]
        return [(str(self.idx2node[self.d_dst[k]]), self.rel_names[self.d_rel[k]],
                 bool(self.d_directed[k])) for k in range(s, e)]


def build_kg_store(kg_dir: str | Path = DEFAULT_KG_DIR, kind_map: dict | None = None,
                   rebuild: bool = False, log=print) -> KGStore:
    kg_dir = Path(kg_dir)
    km = dict(kind_map) if kind_map is not None else dict(KIND_CANONICAL)
    nodes_p, edges_p = _kg_files(kg_dir)
    fp = _fingerprint(nodes_p, edges_p, km)
    cache_dir = _CACHE / f"kg_store__{kg_dir.name}__{fp}"
    if cache_dir.is_dir() and not rebuild:
        log(f"[A1] cache HIT: {cache_dir}")
        return _load(cache_dir)
    log(f"[A1] cache MISS -> building from {kg_dir.name} (fp={fp}) ...")

    nodes = pd.read_parquet(nodes_p, columns=["id", "kind"])
    nodes["id"] = nodes["id"].astype(str)
    nodes = nodes.drop_duplicates("id", keep="first").reset_index(drop=True)
    n = len(nodes)
    node2idx = {nid: i for i, nid in enumerate(nodes["id"].tolist())}
    idx2node = nodes["id"].to_numpy().astype(object)

    # canonical types (stable name order = first appearance), unmapped -> Other (counted)
    canon = nodes["kind"].astype(str).map(lambda k: km.get(k, UNKNOWN_CANONICAL)).to_numpy()
    n_unmapped = int((canon == UNKNOWN_CANONICAL).sum())
    unmapped_kinds = sorted(set(nodes["kind"].astype(str)[canon == UNKNOWN_CANONICAL])) if n_unmapped else []
    type_names: list[str] = []
    tname2id: dict[str, int] = {}
    type_id = np.empty(n, dtype=np.int32)
    for i, c in enumerate(canon):
        if c not in tname2id:
            tname2id[c] = len(type_names); type_names.append(c)
        type_id[i] = tname2id[c]
    drug_mask = np.array([type_names[t] == DRUG_CANONICAL for t in type_id], dtype=bool)

    edges = pd.read_parquet(edges_p, columns=["src", "dst", "relation", "directed"])
    edges["src"] = edges["src"].astype(str); edges["dst"] = edges["dst"].astype(str)
    rel_str = edges["relation"].astype(str)
    # relation vocab from ALL edges (tex reports 59; keep even if a dropped edge
    # was the only carrier of a relation). first-seen order, no normalization.
    rel_names = pd.unique(rel_str).tolist()
    rname2id = {r: i for i, r in enumerate(rel_names)}
    in_src = edges["src"].isin(node2idx); in_dst = edges["dst"].isin(node2idx)
    keep = (in_src & in_dst).to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        log(f"[A1] WARN dropped {n_dropped} edges with endpoint absent from nodes")
    e_src = edges["src"][keep].map(node2idx).to_numpy(np.int32)
    e_dst = edges["dst"][keep].map(node2idx).to_numpy(np.int32)
    e_rel = rel_str[keep].map(rname2id).to_numpy(np.int32)
    e_directed = edges["directed"][keep].astype(bool).to_numpy()

    # UNDIRECTED CSR: symmetrize + dedup neighbors (drop self-loops)
    u = np.concatenate([e_src, e_dst]).astype(np.int64)
    v = np.concatenate([e_dst, e_src]).astype(np.int64)
    nz = u != v
    u, v = u[nz], v[nz]
    key = np.unique(u * n + v)                       # sorted by (u, v) -> CSR order
    uu = (key // n).astype(np.int64); vv = (key % n).astype(np.int32)
    u_indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(u_indptr, uu + 1, 1); u_indptr = np.cumsum(u_indptr)
    u_indices = vv                                   # already ordered by uu

    # DIRECTED CSR: source-sorted, ALL edges kept (multigraph)
    order = np.argsort(e_src, kind="stable")
    d_indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(d_indptr, e_src.astype(np.int64) + 1, 1); d_indptr = np.cumsum(d_indptr)
    d_dst = e_dst[order]; d_rel = e_rel[order]; d_directed = e_directed[order]

    meta = {"schema_version": SCHEMA_VERSION, "fingerprint": fp, "kg_dir": str(kg_dir),
            "kg_name": kg_dir.name, "n_nodes": int(n), "n_edges": int(len(e_src)),
            "n_edges_dropped": n_dropped, "n_unmapped_kinds": n_unmapped,
            "unmapped_kinds": unmapped_kinds, "type_names": type_names,
            "rel_names": rel_names, "kind_map": km}
    _save(cache_dir, idx2node, type_id, drug_mask, u_indptr, u_indices,
          d_indptr, d_dst, d_rel, d_directed, meta)
    log(f"[A1] built + cached: {cache_dir}  (nodes={n} edges={len(e_src)} "
        f"types={len(type_names)} rels={len(rel_names)} unmapped={n_unmapped})")
    return KGStore(idx2node, type_id, type_names, drug_mask, u_indptr, u_indices,
                   d_indptr, d_dst, d_rel, d_directed, rel_names, meta)


def _save(cache_dir, idx2node, type_id, drug_mask, u_indptr, u_indices,
          d_indptr, d_dst, d_rel, d_directed, meta):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / "arrays.npz", type_id=type_id, drug_mask=drug_mask,
             u_indptr=u_indptr, u_indices=u_indices, d_indptr=d_indptr,
             d_dst=d_dst, d_rel=d_rel, d_directed=d_directed)
    pd.DataFrame({"idx": np.arange(len(idx2node)), "node_id": idx2node}).to_parquet(
        cache_dir / "idmap.parquet")
    (cache_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _load(cache_dir: Path) -> KGStore:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"cache schema mismatch at {cache_dir}; rebuild")
    a = np.load(cache_dir / "arrays.npz")
    idx2node = pd.read_parquet(cache_dir / "idmap.parquet")["node_id"].to_numpy().astype(object)
    return KGStore(idx2node, a["type_id"], meta["type_names"], a["drug_mask"],
                   a["u_indptr"], a["u_indices"], a["d_indptr"], a["d_dst"],
                   a["d_rel"], a["d_directed"], meta["rel_names"], meta)


__all__ = ["KGStore", "build_kg_store", "DEFAULT_KG_DIR", "SCHEMA_VERSION"]
