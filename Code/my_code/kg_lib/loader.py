"""Load and merge multiple biomedical KGs into a unified edge table.

Design:
- Drug nodes share a single namespace (DrugBank ID, e.g. `DB00006`).
- Non-drug entities are namespaced by source KG to avoid ID collisions:
    Hetionet : `het:<kind>:<id>`     e.g. `het:Gene::1234`
    PrimeKG  : `prime:<type>:<id>`   e.g. `prime:effect/phenotype:4308`
    DrugBank : `db:<role>:<id>`      e.g. `db:target:BE0001234`
- Each edge keeps its source-KG-prefixed relation name and a directed flag.
- The returned `edges` DataFrame is the single source-of-truth for path queries.

Relations summary (after merge, prefixed):
    het:CcSE, het:CbG, het:CdG, het:CuG, het:CrC, het:CtD, het:CpD, het:PCiC
    prime:drug_effect, prime:drug_protein, prime:contraindication,
        prime:indication, prime:off-label
    db:target, db:enzyme, db:transporter, db:carrier, db:pathway

`directed` boolean column: True means src→dst direction is meaningful.
For symmetric relations (e.g. `het:CrC`, `het:GiG` would be in there if kept),
each undirected edge is stored as a single row with directed=False.
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

# Locate project root robustly: walk up until we find Code/data/KG
def _find_project_root() -> Path:
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root with Code/data/KG not found from {cur}")

PROJECT_ROOT = _find_project_root()
KG_ROOT = PROJECT_ROOT / "Code" / "data" / "KG"


# -----------------------------------------------------------------------------
# Drug-side helpers
# -----------------------------------------------------------------------------
def load_drug_set() -> set[str]:
    """Return the 1900 canonical DrugBank IDs (the working set)."""
    drugs = pd.read_csv(KG_ROOT / "drugbank" / "filtered" / "drugs.csv")
    return set(drugs["drugbank_id"])


# -----------------------------------------------------------------------------
# Per-KG loaders → list of edge dicts (src, src_kind, dst, dst_kind, rel, dir)
# -----------------------------------------------------------------------------
def _load_drugbank_edges(drug_set: set[str]) -> pd.DataFrame:
    """Drug → protein/pathway edges from drugbank/filtered/*.csv (PK-only)."""
    base = KG_ROOT / "drugbank" / "filtered"
    files = [
        ("drug_targets.csv",      "target_id",      "target",      "Protein"),
        ("drug_enzymes.csv",      "enzyme_id",      "enzyme",      "Protein"),
        ("drug_transporters.csv", "transporter_id", "transporter", "Protein"),
        ("drug_carriers.csv",     "carrier_id",     "carrier",     "Protein"),
        ("drug_pathways.csv",     "pathway_id",     "pathway",     "Pathway"),
    ]
    edges = []
    for fn, id_col, rel_role, kind in files:
        df = pd.read_csv(base / fn)
        df = df[df["drugbank_id"].isin(drug_set)]
        df = df[["drugbank_id", id_col]].dropna().drop_duplicates()
        for d, p in zip(df["drugbank_id"], df[id_col]):
            edges.append(dict(
                src=d, src_kind="Drug",
                dst=f"db:{rel_role}:{p}", dst_kind=kind,
                relation=f"db:{rel_role}",
                source_kg="drugbank",
                directed=True,
            ))
    return pd.DataFrame(edges)


def _load_hetionet_edges(drug_set: set[str]) -> tuple[pd.DataFrame, dict]:
    """All Hetionet edges (we keep the full graph for indirect path support)."""
    base = KG_ROOT / "hetionet"
    nodes = pd.read_csv(base / "hetionet-v1.0-nodes.tsv", sep="\t")
    raw_edges = pd.read_csv(
        base / "hetionet-v1.0-edges.sif", sep="\t",
        header=None, names=["head", "relation", "tail"],
    )
    kind_of = dict(zip(nodes["id"], nodes["kind"]))
    name_of = dict(zip(nodes["id"], nodes["name"]))

    # Symmetric relations in Hetionet (preserve symmetry but mark directed=False)
    SYMMETRIC = {"CrC", "GiG", "DrD"}

    def to_node_id(orig_id: str) -> str:
        """Map Hetionet id to unified namespace.
        Drug (Compound) → DrugBank ID; everything else → het:<kind>:<original_id>.
        """
        kind = kind_of.get(orig_id, "Unknown")
        if kind == "Compound":
            # Compound::DBXXXXX → DBXXXXX
            db_id = orig_id.split("::", 1)[-1]
            return db_id, "Drug" if db_id in drug_set else "Drug_outside"
        return f"het:{kind}:{orig_id}", kind

    rows = []
    for head, rel, tail in zip(raw_edges["head"], raw_edges["relation"], raw_edges["tail"]):
        src_id, src_kind = to_node_id(head)
        dst_id, dst_kind = to_node_id(tail)
        rows.append(dict(
            src=src_id, src_kind=src_kind,
            dst=dst_id, dst_kind=dst_kind,
            relation=f"het:{rel}",
            source_kg="hetionet",
            directed=(rel not in SYMMETRIC),
        ))
    edges_df = pd.DataFrame(rows)

    # Node attribute table (id, kind, name) — used for later path readout
    het_node_attrs = []
    for nid, kind in kind_of.items():
        het_id, _ = to_node_id(nid)
        het_node_attrs.append(dict(id=het_id, kind=kind, name=name_of.get(nid, ""), source_kg="hetionet"))
    return edges_df, het_node_attrs


def _load_primekg_edges(drug_set: set[str]) -> tuple[pd.DataFrame, list]:
    """PrimeKG edges with TWO de-duplication groups:

    1) Same-type symmetric (protein_protein, disease_disease, ...): single
       undirected logical edge per pair.
    2) Cross-type bidirectional-listed (drug_effect, contraindication,
       indication, off-label use, drug_protein, drug_drug): PrimeKG stores
       each pair as TWO rows (x→y AND y→x) even though semantically it's a
       single undirected relationship. We keep only the row where one
       canonical side appears as `x` and mark as undirected.

    `drug_drug` is INCLUDED but masked later (DDI leak risk).
    """
    base = KG_ROOT / "primekg"
    prime = pd.read_csv(
        base / "kg.csv", low_memory=False,
        usecols=["x_id", "x_type", "x_name", "y_id", "y_type", "y_name", "relation"],
    )

    # Same-type symmetric relations: stored once but conceptually undirected
    SYMMETRIC_SAMETYPE = {"protein_protein", "drug_drug", "anatomy_anatomy",
                          "bioprocess_bioprocess", "molfunc_molfunc",
                          "cellcomp_cellcomp", "pathway_pathway",
                          "phenotype_phenotype", "disease_disease", "exposure_exposure"}
    # Cross-type bidirectionally-listed relations: PrimeKG lists each pair twice
    BIDIR_LISTED = {"drug_effect", "contraindication", "indication",
                    "off-label use", "drug_protein"}

    # For BIDIR_LISTED: keep only rows where x_type comes "canonical first".
    # We use the convention: x must be 'drug' when one side is drug;
    # otherwise alphabetical type order.
    def is_canonical_side(xt, yt):
        if xt == yt:
            return True  # one-side symmetric: keep both for now, dedup below
        if "drug" in (xt, yt):
            return xt == "drug"
        return xt < yt

    keep_mask = []
    for xt, yt, rel in zip(prime["x_type"], prime["y_type"], prime["relation"]):
        if rel in BIDIR_LISTED:
            keep_mask.append(is_canonical_side(xt, yt))
        else:
            keep_mask.append(True)
    prime = prime[keep_mask].reset_index(drop=True)

    # For SYMMETRIC_SAMETYPE that may still be listed twice (depending on PrimeKG),
    # additional dedup by canonical-ordered pair
    sym_mask = prime["relation"].isin(SYMMETRIC_SAMETYPE)
    if sym_mask.any():
        sym = prime[sym_mask].copy()
        sym["_pair"] = sym.apply(
            lambda r: tuple(sorted([str(r["x_id"]) + r["x_type"],
                                     str(r["y_id"]) + r["y_type"]])), axis=1)
        sym = sym.drop_duplicates(subset=["_pair", "relation"]).drop(columns="_pair")
        prime = pd.concat([prime[~sym_mask], sym], ignore_index=True)

    def to_node_id(orig_id: str, ntype: str) -> str:
        if ntype == "drug":
            sid = str(orig_id)
            kind = "Drug" if sid in drug_set else "Drug_outside"
            return sid, kind
        return f"prime:{ntype}:{orig_id}", ntype

    src_ids, src_kinds, dst_ids, dst_kinds = [], [], [], []
    for xi, xt, yi, yt in zip(prime["x_id"], prime["x_type"], prime["y_id"], prime["y_type"]):
        si, sk = to_node_id(xi, xt)
        di, dk = to_node_id(yi, yt)
        src_ids.append(si); src_kinds.append(sk)
        dst_ids.append(di); dst_kinds.append(dk)
    # Both same-type symmetric AND bidir-listed are now undirected
    UNDIRECTED = SYMMETRIC_SAMETYPE | BIDIR_LISTED
    edges_df = pd.DataFrame({
        "src": src_ids, "src_kind": src_kinds,
        "dst": dst_ids, "dst_kind": dst_kinds,
        "relation": ["prime:" + r for r in prime["relation"]],
        "source_kg": "primekg",
        "directed": [r not in UNDIRECTED for r in prime["relation"]],
    })

    # Node attribute table
    prime_node_attrs_a = prime[["x_id", "x_type", "x_name"]].drop_duplicates().rename(
        columns={"x_id":"orig_id", "x_type":"type", "x_name":"name"})
    prime_node_attrs_b = prime[["y_id", "y_type", "y_name"]].drop_duplicates().rename(
        columns={"y_id":"orig_id", "y_type":"type", "y_name":"name"})
    all_nodes = pd.concat([prime_node_attrs_a, prime_node_attrs_b]).drop_duplicates(subset=["orig_id","type"])
    prime_node_attrs = []
    for _, r in all_nodes.iterrows():
        nid, _ = to_node_id(r["orig_id"], r["type"])
        prime_node_attrs.append(dict(id=nid, kind=r["type"], name=r["name"], source_kg="primekg"))
    return edges_df, prime_node_attrs


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_merged_kg(
    include: tuple[str, ...] = ("drugbank", "hetionet", "primekg"),
    mask_primekg_drugdrug: bool = True,
    cache: bool = True,
) -> dict:
    """Build the unified KG and return as dict of {edges_df, nodes_df}.

    Caches to `KG_ROOT/_merged_kg/{edges.parquet, nodes.parquet}`.

    Args:
        include: which KGs to load (subset of drugbank / hetionet / primekg).
        mask_primekg_drugdrug: drop `prime:drug_drug` edges (leak risk).
        cache: load from cache if exists (else rebuild and save).
    """
    cache_dir = KG_ROOT / "_merged_kg"
    cache_dir.mkdir(exist_ok=True)
    edges_cache = cache_dir / f"edges__{'_'.join(sorted(include))}__mask{int(mask_primekg_drugdrug)}.parquet"
    nodes_cache = cache_dir / f"nodes__{'_'.join(sorted(include))}.parquet"

    if cache and edges_cache.exists() and nodes_cache.exists():
        edges = pd.read_parquet(edges_cache)
        nodes = pd.read_parquet(nodes_cache)
        return {"edges": edges, "nodes": nodes, "cache_hit": True}

    drug_set = load_drug_set()

    all_edges = []
    all_node_attrs = []

    if "drugbank" in include:
        e = _load_drugbank_edges(drug_set)
        all_edges.append(e)
        # drugbank protein/pathway nodes don't have clean names — leave name blank
        for n in pd.concat([e[["dst","dst_kind"]].rename(columns={"dst":"id","dst_kind":"kind"})]).drop_duplicates().to_dict("records"):
            n["name"] = ""
            n["source_kg"] = "drugbank"
            all_node_attrs.append(n)

    if "hetionet" in include:
        e, n = _load_hetionet_edges(drug_set)
        all_edges.append(e)
        all_node_attrs.extend(n)

    if "primekg" in include:
        e, n = _load_primekg_edges(drug_set)
        if mask_primekg_drugdrug:
            e = e[e["relation"] != "prime:drug_drug"]
        all_edges.append(e)
        all_node_attrs.extend(n)

    edges = pd.concat(all_edges, ignore_index=True)
    nodes = pd.DataFrame(all_node_attrs).drop_duplicates(subset=["id"])

    # Drug-node entries (always populated)
    drug_rows = []
    id2name = json.load(open(KG_ROOT / "drugbank" / "filtered" / "id2name.json"))
    for d in drug_set:
        drug_rows.append(dict(id=d, kind="Drug", name=id2name.get(d, ""), source_kg="drugbank"))
    drug_nodes = pd.DataFrame(drug_rows)
    nodes = pd.concat([drug_nodes, nodes[nodes["kind"] != "Drug"]], ignore_index=True)
    nodes = nodes.drop_duplicates(subset=["id"]).reset_index(drop=True)

    if cache:
        edges.to_parquet(edges_cache, index=False)
        nodes.to_parquet(nodes_cache, index=False)

    return {"edges": edges, "nodes": nodes, "cache_hit": False}


if __name__ == "__main__":
    print("Building merged KG (this may take a minute on first run)...")
    out = build_merged_kg()
    e, n = out["edges"], out["nodes"]
    print(f"Cache hit: {out['cache_hit']}")
    print(f"Edges: {len(e):,}")
    print(f"Nodes: {len(n):,}")
    print(f"\nNode kinds:")
    print(n["kind"].value_counts().head(15).to_string())
    print(f"\nEdge relations (top 20):")
    print(e["relation"].value_counts().head(20).to_string())
