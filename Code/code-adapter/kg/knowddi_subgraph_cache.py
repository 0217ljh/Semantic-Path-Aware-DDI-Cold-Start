"""Persistent disk cache for KnowDDI per-pair enclosing subgraphs (node-idx keyed).

Byte-identical reuse across runs: the wrapper's per-pair extraction is DETERMINISTIC
(rng seeded by (seed,u,v)) and pair-INDEPENDENT, so a cached (nodes,labels) equals a
fresh extraction exactly. A `signature` guards against stale caches (different KG /
hop / max_nodes_per_hop / seed / extraction code). One .npz per signature holds all
pairs extracted so far (all folds/splits share it since keys are node indices + the KG
identity is in the signature). NOT built by the pilot's build_subgraph_cache (that uses
a single RNG stream over a list -> order-dependent -> would differ from the wrapper).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# bump ONLY if extract_enclosing_subgraph / _neighbor_nodes / _bfs_relational / _node_label change
_CODE_VERSION = "knowddi-subgraph-v1"
_CACHE_DIR = Path(__file__).resolve().parent / "_cache" / "knowddi_subgraph"


def signature(*, hop: int, max_nodes_per_hop: int, seed: int, n_nodes: int, inc) -> str:
    """Stable JSON signature. KG identity = sha1 of the CSR indptr AND indices (indptr alone
    is insufficient: same indptr + different indices = different neighbors = different subgraphs,
    codex 019f5f9e). data is binarized by build_graph_tensors so it is not part of the key."""
    h = hashlib.sha1()
    h.update(np.asarray(inc.indptr).tobytes())
    h.update(np.asarray(inc.indices).tobytes())
    kg_hash = h.hexdigest()[:16]
    return json.dumps({"code": _CODE_VERSION, "hop": int(hop), "mnph": int(max_nodes_per_hop),
                       "seed": int(seed), "n_nodes": int(n_nodes), "kg": kg_hash},
                      sort_keys=True)


def cache_path(sig: str) -> Path:
    return _CACHE_DIR / f"knowddi_subgraph__{hashlib.sha1(sig.encode()).hexdigest()[:16]}.npz"


def save(path, nodeset: dict, sig: str) -> None:
    """nodeset: {(u,v): (nodes list, labels (k,2))}. Flat-array shard (no pickle)."""
    keys = list(nodeset.keys())
    pairs = np.array(keys, dtype=np.int64).reshape(-1, 2) if keys else np.zeros((0, 2), np.int64)
    lens = np.array([len(nodeset[k][0]) for k in keys], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64) if keys else np.zeros(1, np.int64)
    nodes = (np.concatenate([np.asarray(nodeset[k][0], np.int64).reshape(-1) for k in keys])
             if keys else np.zeros(0, np.int64))
    labels = (np.concatenate([np.asarray(nodeset[k][1], np.int64).reshape(-1, 2) for k in keys])
              if keys else np.zeros((0, 2), np.int64))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pairs=pairs, offsets=offsets, nodes=nodes, labels=labels, sig=np.array(sig))


def load(path, sig: str) -> dict | None:
    """Return {(u,v): (nodes list, labels (k,2) array)} or None on miss/signature mismatch."""
    p = Path(path)
    if not p.is_file():
        return None
    d = np.load(p, allow_pickle=False)
    if str(d["sig"]) != sig:
        return None
    pairs, offs, nodes, labels = d["pairs"], d["offsets"], d["nodes"], d["labels"]
    out = {}
    for i in range(len(pairs)):
        s, e = int(offs[i]), int(offs[i + 1])
        out[(int(pairs[i, 0]), int(pairs[i, 1]))] = (nodes[s:e].tolist(), labels[s:e])
    return out


__all__ = ["signature", "cache_path", "save", "load"]
