"""Stage 2 / Part B / atom B1a - node NAME resolution (idx-aligned to A1).

The merged KG left the 2482 DrugBank BE-protein nodes (db:target/enzyme/carrier/
transporter:BE...) with an empty `name` (the merge never joined the protein name
that DrugBank ships in `drugbank/filtered/drug_{targets,enzymes,carriers,
transporters}.csv`). Since those BE-proteins are exactly the drug's direct PK
mediators, their name matters most. This atom backfills them from those CSVs
(verified 100% coverage) WITHOUT touching the (tex-aligned) `_merged_kg`.

Output: `names[i]` (object array) = resolved display name of node `kg.idx2node[i]`.
Precedence: nodes.parquet name -> BE crosswalk -> "" (empty; caller decides fallback).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from kg.store import DEFAULT_KG_DIR, KGStore

_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache"
_DB_FILTERED = DEFAULT_KG_DIR.parent / "drugbank" / "filtered"
NAMES_SCHEMA = "node_names_v2"

#: DrugBank BE-name sources: (file, id col, name col)
_BE_SOURCES = [
    ("drug_targets.csv", "target_id", "target_name"),
    ("drug_enzymes.csv", "enzyme_id", "enzyme_name"),
    ("drug_carriers.csv", "carrier_id", "carrier_name"),
    ("drug_transporters.csv", "transporter_id", "transporter_name"),
]
#: DrugBank SMPDB pathway name source (same blank-name gap as BE-proteins:
#: the merge left db:pathway:SMP* nodes nameless; drug_pathways.csv carries them)
_PATHWAY_SOURCE = ("drug_pathways.csv", "pathway_id", "pathway_name")
_BE_STRIP = re.compile(r"^db:[a-z]+:")


def _be_crosswalk(db_filtered: Path) -> dict:
    m: dict[str, str] = {}
    for fn, idc, namec in _BE_SOURCES:
        p = db_filtered / fn
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        for i, nm in zip(df[idc].astype(str), df[namec].astype(str)):
            i, nm = i.strip(), nm.strip()
            if i and nm and i not in m:
                m[i] = nm
    return m


def _pathway_crosswalk(db_filtered: Path) -> dict:
    fn, idc, namec = _PATHWAY_SOURCE
    p = db_filtered / fn
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    m: dict[str, str] = {}
    for i, nm in zip(df[idc].astype(str), df[namec].astype(str)):
        i, nm = i.strip(), nm.strip()
        if i and nm and i not in m:
            m[i] = nm
    return m


def build_node_names(kg: KGStore, kg_dir: Path = DEFAULT_KG_DIR,
                     db_filtered: Path = _DB_FILTERED, rebuild: bool = False,
                     log=print) -> np.ndarray:
    """Return object array names[i] aligned to kg.idx2node (BE-backfilled)."""
    fp = kg.meta.get("fingerprint", "nofp")
    cache = _CACHE / f"node_names__{kg.meta.get('kg_name','kg')}__{fp}__{NAMES_SCHEMA}.parquet"
    if cache.is_file() and not rebuild:
        log(f"[B1a] cache HIT: {cache}")
        return pd.read_parquet(cache)["name"].to_numpy().astype(object)
    log("[B1a] building node names (BE backfill) ...")

    nodes_p = sorted(kg_dir.glob("nodes*.parquet"))[0]
    nd = pd.read_parquet(nodes_p, columns=["id", "name"])
    id2name = {str(i): (str(n).strip() if pd.notna(n) else "")
               for i, n in zip(nd["id"], nd["name"])}
    be2name = _be_crosswalk(db_filtered)
    smp2name = _pathway_crosswalk(db_filtered)

    n_be_fixed = 0
    n_path_fixed = 0
    names = np.empty(kg.n_nodes, dtype=object)
    for i, nid in enumerate(kg.idx2node.tolist()):
        nm = id2name.get(nid, "")
        if not nm and nid.startswith("db:") and ":BE" in nid:
            cand = be2name.get(_BE_STRIP.sub("", nid), "")
            if cand:
                nm = cand; n_be_fixed += 1
        if not nm and nid.startswith("db:pathway:"):
            cand = smp2name.get(_BE_STRIP.sub("", nid), "")
            if cand:
                nm = cand; n_path_fixed += 1
        names[i] = nm

    nonempty = int((names != "").sum())
    meta = {"schema": NAMES_SCHEMA, "kg_fingerprint": fp, "n_nodes": kg.n_nodes,
            "n_nonempty": nonempty, "n_be_backfilled": n_be_fixed,
            "n_pathway_backfilled": n_path_fixed}
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": names}).to_parquet(cache)
    (cache.with_suffix(".meta.json")).write_text(json.dumps(meta), encoding="utf-8")
    log(f"[B1a] built: nonempty {nonempty}/{kg.n_nodes} "
        f"({round(100*nonempty/kg.n_nodes,1)}%), BE backfilled {n_be_fixed}, "
        f"pathway backfilled {n_path_fixed}")
    return names


__all__ = ["build_node_names", "NAMES_SCHEMA"]
