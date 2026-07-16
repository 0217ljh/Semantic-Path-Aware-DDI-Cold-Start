"""Full PK vs PD cross-table: 6 geometry cells x mediator type -> Excel.

Columns (horizontal): geometry category / cell
  A 共有   : (1,1)
  B 单边   : (1,2),(2,1),(1,3),(3,1)
  C 远端   : (2,2)
each split into PK / PD.

Rows (vertical): mediator type
  - sheet "by_kind"    : 11 node-kind groups (all merged-KG kinds).
  - sheet "A_by_channel": for confluent (1,1) only, relation channel
    (both drugs via same r) -> resolves enzyme/transporter/target.
  - sheet "cell_totals" : per-cell total mediators/pair + severed rate.

Values = mean mediators per pair (100 PK + 100 PD pairs, seed=42, undirected).

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_ddi_pkpd_full_crosstab.py --n 100 --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"

MAX_DEPTH = 3
BUDGET = 4
# (cell -> category)
CELLS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
CELL_CAT = {(1, 1): "A 共有", (1, 2): "B 单边", (2, 1): "B 单边",
            (1, 3): "B 单边", (3, 1): "B 单边", (2, 2): "C 远端"}

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
    "db:enzyme", "db:transporter", "db:carrier", "db:target", "db:pathway",
    "prime:drug_protein", "het:CbG", "het:CdG", "het:CuG",
    "het:CcSE", "prime:drug_effect",
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
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_ddi_pkpd_full_crosstab"))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    nodes = pd.read_parquet(NODES_PARQUET)
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)

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

    src = edges["src"].map(id2idx).to_numpy(); dst = edges["dst"].map(id2idx).to_numpy()
    rel = edges["relation"].to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64); dst = dst[keep].astype(np.int64); rel = rel[keep]
    nz = src != dst
    s2, d2 = src[nz], dst[nz]
    rows = np.concatenate([s2, d2]); cols = np.concatenate([d2, s2])
    A = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.data[:] = 1

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

    # tensor[cell][bucket] = (n_grp,) ; cell totals ; severed
    cell_kind = {c: {bk: np.zeros(n_grp) for bk in ["PK", "PD"]} for c in CELLS}
    cell_tot = {c: {bk: 0.0 for bk in ["PK", "PD"]} for c in CELLS}
    severed = {bk: 0 for bk in ["PK", "PD"]}

    for bk in ["PK", "PD"]:
        npairs = len(samples[bk])
        for (a, b) in samples[bk]:
            dA = get_dist(d2glob[a]); dB = get_dist(d2glob[b])
            mask_any = (dA >= 1) & (dB >= 1) & ((dA + dB) <= BUDGET)
            if mask_any.sum() == 0:
                severed[bk] += 1
            for (ca, cb) in CELLS:
                cm = (dA == ca) & (dB == cb)
                if not cm.any():
                    continue
                gc = node_grp[cm]; gc = gc[gc >= 0]
                np.add.at(cell_kind[(ca, cb)][bk], gc, 1.0)
                cell_tot[(ca, cb)][bk] += int(cm.sum())
        for c in CELLS:
            cell_kind[c][bk] /= npairs
            cell_tot[c][bk] /= npairs

    # category-A confluent by channel
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

    # ---------- build DataFrames ----------
    col_tuples = []
    for c in CELLS:
        for bk in ["PK", "PD"]:
            col_tuples.append((CELL_CAT[c], f"({c[0]},{c[1]})", bk))
    cols_mi = pd.MultiIndex.from_tuples(col_tuples, names=["category", "cell", "bucket"])

    data = np.zeros((n_grp, len(col_tuples)))
    for j, (cat, cell_s, bk) in enumerate(col_tuples):
        ca, cb = int(cell_s[1]), int(cell_s[3])
        data[:, j] = cell_kind[(ca, cb)][bk]
    df_kind = pd.DataFrame(data, index=GROUP_ORDER, columns=cols_mi)
    # total row
    tot_row = []
    for (cat, cell_s, bk) in col_tuples:
        ca, cb = int(cell_s[1]), int(cell_s[3])
        tot_row.append(cell_tot[(ca, cb)][bk])
    df_kind.loc["TOTAL/pair"] = tot_row
    df_kind = df_kind.round(2)

    # channel sheet (category A)
    rows_ch = []
    for ch in CHANNELS:
        rows_ch.append({
            "channel": ch,
            "PK_per_pair": round(chan["PK"][ch], 3), "PK_cov%": round(chan_cov["PK"][ch] / len(samples["PK"]) * 100, 1),
            "PD_per_pair": round(chan["PD"][ch], 3), "PD_cov%": round(chan_cov["PD"][ch] / len(samples["PD"]) * 100, 1),
            "PD/PK": round((chan["PD"][ch] + 1e-9) / (chan["PK"][ch] + 1e-9), 2),
        })
    df_chan = pd.DataFrame(rows_ch).set_index("channel")

    # cell totals sheet
    rows_ct = []
    for c in CELLS:
        rows_ct.append({"cell": f"({c[0]},{c[1]})", "category": CELL_CAT[c],
                        "PK_total/pair": round(cell_tot[c]["PK"], 1), "PD_total/pair": round(cell_tot[c]["PD"], 1)})
    df_ct = pd.DataFrame(rows_ct)
    df_ct.loc[len(df_ct)] = ["severed%", "-", round(severed["PK"] / len(samples["PK"]) * 100, 1),
                             round(severed["PD"] / len(samples["PD"]) * 100, 1)]

    # ---------- export ----------
    xlsx = out_dir / "pkpd_crosstab.xlsx"
    wrote_xlsx = False
    try:
        with pd.ExcelWriter(xlsx) as xw:
            df_kind.to_excel(xw, sheet_name="by_kind")
            df_chan.to_excel(xw, sheet_name="A_by_channel")
            df_ct.to_excel(xw, sheet_name="cell_totals", index=False)
        wrote_xlsx = True
        log(f"[save] {xlsx}")
    except Exception as e:  # noqa: BLE001
        log(f"[warn] xlsx export failed ({e}); CSV only")
    df_kind.to_csv(out_dir / "by_kind.csv")
    df_chan.to_csv(out_dir / "A_by_channel.csv")
    df_ct.to_csv(out_dir / "cell_totals.csv", index=False)
    log(f"[save] {out_dir/'by_kind.csv'} / A_by_channel.csv / cell_totals.csv")

    # ---------- print compact ----------
    pd.set_option("display.width", 200, "display.max_columns", 40)
    log("\n===== by_kind (mean mediators/pair) =====")
    log(df_kind.to_string())
    log("\n===== A_by_channel (confluent (1,1), both via same relation) =====")
    log(df_chan.to_string())
    log("\n===== cell_totals =====")
    log(df_ct.to_string(index=False))
    log(f"\n[done] xlsx={'yes' if wrote_xlsx else 'no'} wall_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
