"""Stage 2 / Part B / atom B1b-disease - PrimeKG disease FEATURE source.

PrimeKG ships `disease_features.tab` keyed by MONDO (mondo_name + free-text
definitions). The merged-KG disease nodes (mostly `prime:disease:*`) carry a
readable `name` that matches `mondo_name` for ~97% of nodes, so we join on the
lowercased name and expose the richest definition text (mondo_definition, then
umls_description) as the raw source later compressed into a node description.

Read-only over Code/data. Cached to kg/_cache.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from kg.store import DEFAULT_KG_DIR

_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache"
_DISEASE_TAB = DEFAULT_KG_DIR.parent / "primekg" / "disease_features.tab"
DISEASE_SCHEMA = "disease_text_v1"
#: definition columns, richest first; joined for the source text
_DEF_COLS = ["mondo_definition", "umls_description", "orphanet_definition",
             "orphanet_clinical_description"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def build_disease_text(disease_tab: Path = _DISEASE_TAB, rebuild: bool = False,
                       log=print) -> dict[str, str]:
    """Return {lowercased mondo_name -> source definition text} (non-empty only)."""
    if not disease_tab.is_file():
        raise FileNotFoundError(f"PrimeKG disease_features not found: {disease_tab}")
    st = disease_tab.stat()
    fp = f"{int(st.st_mtime)}_{st.st_size}"
    cache = _CACHE / f"disease_text__{DISEASE_SCHEMA}__{fp}.parquet"
    if cache.is_file() and not rebuild:
        log(f"[B1b/dis] cache HIT: {cache}")
        df = pd.read_parquet(cache)
        return dict(zip(df["key"], df["source_text"]))
    log("[B1b/dis] building disease text map ...")
    raw = pd.read_csv(disease_tab, sep="\t", dtype=str)
    raw["key"] = raw["mondo_name"].astype(str).str.strip().str.lower()
    out: dict[str, str] = {}
    for key, grp in raw.groupby("key"):
        if not key or key == "nan":
            continue
        parts: list[str] = []
        for c in _DEF_COLS:
            if c not in grp.columns:
                continue
            for v in grp[c].dropna().astype(str):
                v = _norm(v)
                if v and v.lower() != "nan" and v not in parts:
                    parts.append(v)
                    break  # one value per column is enough
        text = " ".join(parts).strip()
        if text:
            out[key] = text
    df = pd.DataFrame({"key": list(out), "source_text": list(out.values())})
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    (cache.with_suffix(".meta.json")).write_text(
        json.dumps({"schema": DISEASE_SCHEMA, "tab_fp": fp, "n": int(len(df))}),
        encoding="utf-8")
    log(f"[B1b/dis] built + cached: {len(out)} diseases with text -> {cache}")
    return out


__all__ = ["build_disease_text", "DISEASE_SCHEMA"]
