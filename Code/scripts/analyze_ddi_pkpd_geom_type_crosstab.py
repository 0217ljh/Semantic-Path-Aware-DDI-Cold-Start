"""PK vs PD cross-table: meeting-geometry category (A/B/C) x mediator type.

Geometry categories (by how many drugs directly own the mediator M):
  A 共有 confluent : (1,1)                         -> both drugs 1-hop to M
  B 单边 one-sided : (1,2),(2,1),(1,3),(3,1)        -> exactly one drug 1-hop to M
  C 远端 distal    : (2,2)                          -> neither drug 1-hop to M
(plus severed = no mediator within dA+dB<=4.)

Mediator type:
  - node KIND group (11 groups covering all merged-KG kinds)
  - for category A only: relation CHANNEL (both drugs via same relation r),
    which resolves enzyme/transporter/target (all are protein/gene by kind).

For 100 PK + 100 PD positive pairs (seed=42), reports mean mediators/pair in
each (category x type) cell, PK vs PD, to characterize how the two classes
differ. Undirected reachability (directed ~= undirected shown earlier).

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_ddi_pkpd_geom_type_crosstab.py --n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"

MAX_DEPTH = 3
BUDGET = 4

KIND_GROUPS = {
    "protein_gene": ["Gene", "gene/protein", "Protein"],
    "side_effect_pheno": ["Side Effect", "effect/phenotype", "Symptom"],
    "disease": ["disease", "Disease"],
    "drug": ["Drug", "drug", "Compound"],
    "pathway": ["Pathway", "pathway"],
    "anatomy": ["anatomy", "Anatomy"],
    "biological_process": ["biological_process", "Biological Process"],
    "molecular_function": ["molecular_function", "Molecular Function"],
    "cellular_component": ["cellular_component", "Cellular Component"],
    "pharmacologic_class": ["Pharmacologic Class"],
    "exposure": ["exposure"],
}
GROUP_ORDER = list(KIND_GROUPS.keys())

CHANNELS = [
    "db:target", "db:enzyme", "db:transporter", "db:carrier",
    "het:CcSE", "prime:drug_effect", "prime:drug_protein",
    "het:CbG", "het:CdG", "het:CuG",
    "prime:contraindication", "prime:indication", "het:CrC",
]


def log(m: str) -> None:
    print(m, flush=True)


def pkpd_bucket(t: str) -> str:
    t = str(t).lower()
    if any(k in t for k in ("metabolism", "excretion", "serum concentration",
                            "absorption", "protein binding", "clearance")):
        return "PK"
    if any(k in t for k in ("risk or severity", "activities", "efficacy",
                            "cns depression", "qtc", "hypertension",
                            "hypotensive", "sedative", "adverse effects")):
        return "PD"
    return "other"


def bfs_depth(A_csr, source, n_nodes, max_depth=MAX_DEPTH):
    dist = np.full(n_nodes, -1, dtype=np.int16)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    for d in range(1, max_depth + 1):
        if len(frontier) == 0:
            break
        nbrs = np.unique(A_csr[frontier].indices)
        new = nbrs[dist[nbrs] == -1]
        if len(new) == 0:
            break
        dist[new] = d
        frontier = new
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_ddi_pkpd_geom_type_crosstab"))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    nodes = pd.read_parquet(NODES_PARQUET)
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)

    # group code per node
    kind2grp = {}
    for g, ks in KIND_GROUPS.items():
        for k in ks:
            kind2grp[k] = g
    grp2col = {g: i for i, g in enumerate(GROUP_ORDER)}
    node_grp = np.array([grp2col.get(kind2grp.get(k, None), -1) for k in nodes["kind"].to_numpy()], dtype=np.int32)
    n_grp = len(GROUP_ORDER)

    drug_db = nodes["id"].to_numpy()[nodes["kind"].to_numpy() == "Drug"]
    drug_idx = np.array([id2idx[d] for d in drug_db], dtype=np.int64)
    d2glob = {db: id2idx[db] for db in drug_db}

    # undirected adjacency
    src = edges["src"].map(id2idx).to_numpy(); dst = edges["dst"].map(id2idx).to_numpy()
    rel = edges["relation"].to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64); dst = dst[keep].astype(np.int64); rel = rel[keep]
    nz = src != dst
    s2, d2 = src[nz], dst[nz]
    rows = np.concatenate([s2, d2]); cols = np.concatenate([d2, s2])
    A = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.data[:] = 1
    log(f"[csr] undirected nnz={A.nnz}")

    # channel incidence (drug_local x node)
    g2l = {int(g): i for i, g in enumerate(drug_idx)}
    dset = set(int(g) for g in drug_idx)
    inc = {}
    for ch in CHANNELS:
        m = rel == ch; s, d = src[m], dst[m]
        sd = np.array([x in dset for x in s]); dd = np.array([x in dset for x in d]) if len(d) else np.array([], bool)
        rr = np.concatenate([np.array([g2l[x] for x in s[sd]], dtype=np.int64),
                             np.array([g2l[x] for x in d[dd]], dtype=np.int64)])
        cc = np.concatenate([d[sd], s[dd]])
        M = sp.coo_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(len(drug_idx), n_nodes)).tocsr()
        M.data[:] = 1
        inc[ch] = M

    pos = pd.read_csv(DDI_POS_CSV)
    pos = pos[pos["drug_a_id"].isin(d2glob) & pos["drug_b_id"].isin(d2glob)].reset_index(drop=True)
    bk_arr = np.array([pkpd_bucket(t) for t in pos["ddi_type"].to_numpy()])
    samples = {}
    for bk in ["PK", "PD"]:
        idx = np.where(bk_arr == bk)[0]
        sel = rng.choice(idx, size=min(args.n, len(idx)), replace=False)
        samples[bk] = pos.iloc[sel][["drug_a_id", "drug_b_id"]].to_numpy()
        log(f"[sample] {bk}: {len(sel)} pairs")

    dist_cache = {}

    def get_dist(g):
        if g not in dist_cache:
            dist_cache[g] = bfs_depth(A, g, n_nodes)
        return dist_cache[g]

    # crosstab[cat][bucket] = (n_grp,) mean mediators/pair ; cat in A,B,C
    cross = {c: {bk: np.zeros(n_grp) for bk in ["PK", "PD"]} for c in ["A", "B", "C"]}
    cat_total = {c: {bk: 0.0 for bk in ["PK", "PD"]} for c in ["A", "B", "C"]}
    severed = {bk: 0 for bk in ["PK", "PD"]}

    for bk in ["PK", "PD"]:
        npairs = len(samples[bk])
        for (a, b) in samples[bk]:
            dA = get_dist(d2glob[a]); dB = get_dist(d2glob[b])
            mask = (dA >= 1) & (dB >= 1) & ((dA + dB) <= BUDGET)
            if mask.sum() == 0:
                severed[bk] += 1
            catA = mask & (dA == 1) & (dB == 1)
            catC = mask & (dA == 2) & (dB == 2)
            catB = mask & ~catA & ~catC
            for cat, cm in [("A", catA), ("B", catB), ("C", catC)]:
                gc = node_grp[cm]
                gc = gc[gc >= 0]
                np.add.at(cross[cat][bk], gc, 1.0)
                cat_total[cat][bk] += int(cm.sum())
        for cat in ["A", "B", "C"]:
            cross[cat][bk] /= npairs
            cat_total[cat][bk] /= npairs

    # category-A relation channel (both drugs via same r) = confluent typed by relation
    chan = {bk: {ch: 0.0 for ch in CHANNELS} for bk in ["PK", "PD"]}
    chan_cov = {bk: {ch: 0 for ch in CHANNELS} for bk in ["PK", "PD"]}
    for bk in ["PK", "PD"]:
        npairs = len(samples[bk])
        for (a, b) in samples[bk]:
            la, lb = g2l[d2glob[a]], g2l[d2glob[b]]
            for ch in CHANNELS:
                M = inc[ch]
                sh = int(M[la].multiply(M[lb]).sum())
                chan[bk][ch] += sh
                chan_cov[bk][ch] += int(sh > 0)
        for ch in CHANNELS:
            chan[bk][ch] /= npairs

    # ===================== PRINT =====================
    log("\n############ CROSS-TABLE  geometry category x node-kind group  (mean mediators / pair) ############")
    log(f"severed (no mediator in budget):  PK={severed['PK']}/{len(samples['PK'])}  PD={severed['PD']}/{len(samples['PD'])}")
    for cat, name in [("A", "A 共有 confluent (1,1)"), ("B", "B 单边 one-sided (1,2/2,1/1,3/3,1)"), ("C", "C 远端 distal (2,2)")]:
        log(f"\n===== {name}  | category total/pair: PK={cat_total[cat]['PK']:.1f}  PD={cat_total[cat]['PD']:.1f} =====")
        log(f"{'kind group':22s} {'PK/pair':>9s} {'PD/pair':>9s} {'PD-PK':>9s} {'PD/PK':>7s}")
        pk = cross[cat]["PK"]; pd_ = cross[cat]["PD"]
        for i in np.argsort(-(pk + pd_)):
            if pk[i] + pd_[i] < 0.01:
                continue
            ratio = (pd_[i] + 1e-9) / (pk[i] + 1e-9)
            log(f"{GROUP_ORDER[i]:22s} {pk[i]:9.2f} {pd_[i]:9.2f} {pd_[i]-pk[i]:9.2f} {ratio:7.2f}")

    log("\n############ Category A confluent, by RELATION CHANNEL (both drugs via same r) ############")
    log(f"{'channel':24s} {'PK/pair':>9s} {'PKcov':>7s} {'PD/pair':>9s} {'PDcov':>7s} {'PD/PK':>7s}")
    for ch in CHANNELS:
        pk = chan["PK"][ch]; pkc = chan_cov["PK"][ch] / len(samples["PK"])
        pdv = chan["PD"][ch]; pdc = chan_cov["PD"][ch] / len(samples["PD"])
        ratio = (pdv + 1e-9) / (pk + 1e-9)
        log(f"{ch:24s} {pk:9.2f} {pkc*100:6.0f}% {pdv:9.2f} {pdc*100:6.0f}% {ratio:7.2f}")

    # save
    save = {
        "groups": GROUP_ORDER,
        "category_total_per_pair": {c: cat_total[c] for c in ["A", "B", "C"]},
        "crosstab_kind": {c: {bk: cross[c][bk].tolist() for bk in ["PK", "PD"]} for c in ["A", "B", "C"]},
        "categoryA_channel": {bk: chan[bk] for bk in ["PK", "PD"]},
        "categoryA_channel_cov": {bk: {ch: chan_cov[bk][ch] / len(samples[bk]) for ch in CHANNELS} for bk in ["PK", "PD"]},
        "severed": {bk: severed[bk] / len(samples[bk]) for bk in ["PK", "PD"]},
    }
    (out_dir / "summary.json").write_text(json.dumps(save, indent=2))
    log(f"\n[save] {out_dir/'summary.json'}")
    log(f"[done] wall_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
