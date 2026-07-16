"""Stage 2 / Part B / atoms B3 (combine) + B4 (cache) - frozen node embedding z_m.

z_m is built by encoding the three node fields SEPARATELY with the frozen encoder,
L2-normalizing each, combining with fixed weights, and normalizing again:

    z_m = normalize( w_name * n_hat + w_type * t_hat + w_desc * d_hat )

(codex Q2 ruling). Separate-field + per-field normalization keeps a short name and
a long description on equal footing (a single concatenated string would let the
long description dominate the pooled vector). A node with no description just drops
that term (its weight goes to 0), so the relative name/type balance is preserved by
the final normalization. Fixed dim = encoder.dim, so U_t / k-means stay simple.

Cache (B4): kg/_cache/node_embed/zm__<enc>__<fp>.npz  (z + node_ids), keyed by
encoder id, weights, name-schema and description prompt-version.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from kg.node_names import NAMES_SCHEMA, build_node_names
from kg import node_desc

_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache" / "node_embed"
EMBED_SCHEMA = "node_embed_v1"
DEFAULT_WEIGHTS = {"name": 0.55, "type": 0.20, "desc": 0.25}


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def load_descriptions() -> dict[str, str]:
    """node_id -> description (status 'ok', non-empty) from the final v4 JSONL."""
    out: dict[str, str] = {}
    jl = node_desc.default_jsonl()
    if not jl.is_file():
        return out
    for line in jl.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("description"):
            out[r["node_id"]] = r["description"]
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")[:40]


def _content_fp(node_ids, name_texts, type_texts, desc_texts, weights, enc_name) -> str:
    """Fingerprint the ACTUAL encode inputs, so any change to a resolved name,
    type, description, the node set, order, weights or encoder invalidates the
    cache (subsumes KG / NAMES_SCHEMA / PROMPT_VERSION - they only matter through
    these values). Delimited to avoid concatenation collisions."""
    h = hashlib.sha1()
    h.update(enc_name.encode()); h.update(b"\x00")
    h.update(json.dumps(weights, sort_keys=True).encode()); h.update(b"\x00")
    h.update(EMBED_SCHEMA.encode()); h.update(b"\x00")
    for nid, n, t, d in zip(node_ids, name_texts, type_texts, desc_texts):
        h.update(f"{nid}\x1f{n}\x1f{t}\x1f{d}\x1e".encode())
    return h.hexdigest()[:12]


def build_node_embeddings(kg, node_ids, encoder, names=None,
                          descriptions=None, weights=None, batch_size: int = 64,
                          rebuild: bool = False, log=print):
    """Return (Z float32 [N, dim], node_ids list). Cached (B4)."""
    node_ids = [str(x) for x in node_ids]
    weights = weights or DEFAULT_WEIGHTS
    if not node_ids:
        return np.zeros((0, encoder.dim), dtype=np.float32), []

    names = names if names is not None else build_node_names(kg)
    desc_map = descriptions if descriptions is not None else load_descriptions()
    idx = [kg.get_idx(nid) for nid in node_ids]
    if any(i is None for i in idx):
        raise KeyError("some node_ids are not in the KG")

    name_texts = [str(names[i]) for i in idx]
    type_texts = [kg.type_names[kg.type_id[i]] for i in idx]
    desc_texts = [desc_map.get(nid, "") for nid in node_ids]
    has_d = np.array([bool(d and d.strip()) for d in desc_texts])

    fp = _content_fp(node_ids, name_texts, type_texts, desc_texts, weights,
                     f"{encoder.name}|{getattr(encoder, 'pooling', '')}")
    cache = _CACHE / f"zm__{_slug(encoder.name)}__{fp}.npz"
    if cache.is_file() and not rebuild:
        log(f"[B3] cache HIT: {cache}")
        d = np.load(cache, allow_pickle=True)
        return d["z"], list(d["node_ids"])
    log(f"[B3] {len(node_ids)} nodes | name {len(name_texts)}, "
        f"types {len(set(type_texts))} unique, desc {int(has_d.sum())}")

    # type: encode the ~12 unique canonical kinds once, then map
    uniq = sorted(set(type_texts))
    tmap = {t: v for t, v in zip(uniq, _l2(encoder.encode(uniq)))}
    T = np.stack([tmap[t] for t in type_texts]).astype(np.float32)
    N = _l2(encoder.encode(name_texts, batch_size=batch_size, log=log)).astype(np.float32)
    D = np.zeros((len(node_ids), encoder.dim), dtype=np.float32)
    nz = np.flatnonzero(has_d)
    if len(nz):
        D[nz] = _l2(encoder.encode([desc_texts[i] for i in nz],
                                   batch_size=batch_size, log=log)).astype(np.float32)

    wn, wt, wd = weights["name"], weights["type"], weights["desc"]
    Z = _l2(wn * N + wt * T + (wd * has_d[:, None]) * D).astype(np.float32)

    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, z=Z, node_ids=np.array(node_ids, dtype=object))
    meta = {"schema": EMBED_SCHEMA, "encoder": encoder.name,
            "pooling": getattr(encoder, "pooling", ""), "dim": int(encoder.dim),
            "weights": weights, "names_schema": NAMES_SCHEMA,
            "desc_prompt": node_desc.PROMPT_VERSION, "n": len(node_ids),
            "n_with_desc": int(has_d.sum()), "fingerprint": fp}
    cache.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[B3] built z_m {Z.shape} -> {cache}")
    return Z, node_ids


__all__ = ["build_node_embeddings", "load_descriptions", "DEFAULT_WEIGHTS",
           "EMBED_SCHEMA"]
