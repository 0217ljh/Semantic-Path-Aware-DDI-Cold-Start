"""idea2 v2 — v15 KG loader + typed drug adjacency + PK typed-motif features.

Foundation for L2 (pure PK typed model). Canonical KG = _merged_kg_dedup_v15.
The PK causal signal is SHARED metabolizing enzyme/transporter (db:enzyme AUROC ~0.85,
shared target inert ~0.52), so drug adjacency is kept TYPED by pharmacology role, not
aggregated. Read-only over Code/data/. WSL conda env project_1.

Usage (smoke, from project root):
    PYTHONPATH=Code/code-idea-2 python -u Code/code-idea-2/kg_v15.py
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pandas as pd

V15 = "Code/data/KG/_merged_kg_dedup_v15"

# drug->protein edge relations, kept typed by pharmacology role (PK causal = enzyme/transporter)
DRUG_PROT_REL = {"db:target": "target", "db:enzyme": "enzyme",
                 "db:transporter": "transporter", "db:carrier": "carrier"}
# generic drug-protein (all sources) for the "any protein" fallback feature
GENERIC_DP_REL = {"prime:drug_protein", "het:CdG", "het:CuG", "het:CbG"}
PROT_MESO_REL = {"prime:pathway_protein": ("path:", "pathway"),
                 "prime:bioprocess_protein": ("go:", "gobp"),
                 "prime:anatomy_protein_present": ("uberon:", "anatomy"),
                 "prime:disease_protein": ("mondo:", "disease"),
                 "prime:phenotype_protein": ("hp:", "phenotype")}


def load_v15(kg_dir: str = V15):
    nd = pd.read_parquet(f"{kg_dir}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{kg_dir}/edges__dedup.parquet")
    return nd, ed


def build_typed_drug_adj(nd: pd.DataFrame, ed: pd.DataFrame):
    """Return {slot: {drug_id: set(protein/pclass ids)}} for slots
    target/enzyme/transporter/carrier/protein(any)/pclass."""
    drugs = set(nd[nd["kind"].astype(str) == "drug"]["id"])
    adj = {k: defaultdict(set) for k in ("target", "enzyme", "transporter", "carrier", "protein", "pclass")}
    s, d, r = ed["src"].astype(str), ed["dst"].astype(str), ed["relation"].astype(str)
    for a, b, rel in zip(s, d, r):
        dr = a if a in drugs else (b if b in drugs else None)
        if dr is None:
            continue
        o = b if dr == a else a
        if o.startswith("prot:"):
            adj["protein"][dr].add(o)
            slot = DRUG_PROT_REL.get(rel)
            if slot:
                adj[slot][dr].add(o)
            elif rel in GENERIC_DP_REL:
                pass  # already in "protein"
        elif o.startswith("pclass:") and rel == "pclass:includes_drug":
            adj["pclass"][dr].add(o)
    return {k: dict(v) for k, v in adj.items()}, drugs


def build_protein_meso(ed: pd.DataFrame):
    """Return {meso_key: {protein: set(meso ids)}} for pathway/gobp/anatomy/disease/phenotype."""
    mp = {key: defaultdict(set) for _, key in PROT_MESO_REL.values()}
    r = ed["relation"].astype(str)
    for rel, (pref, key) in PROT_MESO_REL.items():
        sub = ed[r == rel]
        for a, b in zip(sub["src"].astype(str), sub["dst"].astype(str)):
            p, o = (a, b) if a.startswith("prot:") else (b, a)
            if p.startswith("prot:") and o.startswith(pref):
                mp[key][p].add(o)
    return {k: dict(v) for k, v in mp.items()}


def pk_pair_feats(a: str, b: str, adj: dict) -> dict:
    """PK typed-motif shared-count features for a drug pair. The discriminative PK signal
    is shared enzyme/transporter (NOT shared target — inert). Keep typed, do not aggregate."""
    f = {}
    for slot in ("target", "enzyme", "transporter", "carrier", "protein", "pclass"):
        sa = adj[slot].get(a, set())
        sb = adj[slot].get(b, set())
        inter = sa & sb
        f[f"shared_{slot}"] = len(inter)
        f[f"jacc_{slot}"] = len(inter) / len(sa | sb) if (sa or sb) else 0.0
    return f


def main() -> None:
    nd, ed = load_v15()
    print(f"v15: {len(nd)} nodes, {len(ed)} edges")
    adj, drugs = build_typed_drug_adj(nd, ed)
    print(f"drugs: {len(drugs)} | with target {len(adj['target'])} | enzyme {len(adj['enzyme'])} "
          f"| transporter {len(adj['transporter'])} | carrier {len(adj['carrier'])} | pclass {len(adj['pclass'])}")
    nm = dict(zip(nd["id"], nd["name"]))
    # smoke: a known PK pair should share an enzyme; print feats for 2 drugs sharing CYP3A4
    cyp = "prot:1576"  # CYP3A4
    cyp_drugs = [dr for dr, ps in adj["enzyme"].items() if cyp in ps][:2]
    if len(cyp_drugs) == 2:
        f = pk_pair_feats(cyp_drugs[0], cyp_drugs[1], adj)
        print(f"sample PK pair {nm.get(cyp_drugs[0])} + {nm.get(cyp_drugs[1])}: {f}")
    print("kg_v15 loader OK (CPU-only)")


if __name__ == "__main__":
    main()
