"""Stage 2 / Part B / atom B2b - frozen relation embedding z_r.

Encode each canonical relation's typed-triple sentence with the SAME frozen encoder
used for z_m, L2-normalize -> z_r. Only 47 relations, so this is a single small
batch. Cached per-encoder under kg/_cache/relation_embed/.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from kg import relations as rel

_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache" / "relation_embed"
RELEMBED_SCHEMA = "relation_embed_v1"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")[:40]


def validate_coverage(kg_rel_names) -> None:
    """Assert every raw KG relation maps to a canonical with a sentence (or is dropped)."""
    missing = []
    for r in kg_rel_names:
        c = rel.canonical_relation(r)
        if c is not None and c not in rel.RELATION_SENTENCE:
            missing.append((r, c))
    if missing:
        raise ValueError(f"relations with no sentence: {missing}")


def build_relation_embeddings(encoder, rebuild: bool = False, log=print):
    """Return (z_r float32 [47, dim], canonical_keys list). Cached (per encoder)."""
    keys = rel.canonical_relations()
    sentences = [rel.relation_sentence(k) for k in keys]
    h = hashlib.sha1()
    h.update(f"{encoder.name}|{getattr(encoder, 'pooling', '')}".encode()); h.update(b"\x00")
    h.update(RELEMBED_SCHEMA.encode()); h.update(b"\x00")
    for k, s in zip(keys, sentences):
        h.update(f"{k}\x1f{s}\x1e".encode())
    fp = h.hexdigest()[:12]
    cache = _CACHE / f"zr__{_slug(encoder.name)}__{fp}.npz"
    if cache.is_file() and not rebuild:
        log(f"[B2] cache HIT: {cache}")
        d = np.load(cache, allow_pickle=True)
        return d["z"], list(d["keys"])

    Z = _l2(encoder.encode(sentences)).astype(np.float32)
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, z=Z, keys=np.array(keys, dtype=object),
             sentences=np.array(sentences, dtype=object))
    meta = {"schema": RELEMBED_SCHEMA, "encoder": encoder.name,
            "pooling": getattr(encoder, "pooling", ""), "dim": int(encoder.dim),
            "n_relations": len(keys), "fingerprint": fp}
    cache.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[B2] built z_r {Z.shape} ({len(keys)} relations) -> {cache}")
    return Z, keys


__all__ = ["build_relation_embeddings", "validate_coverage", "RELEMBED_SCHEMA"]
