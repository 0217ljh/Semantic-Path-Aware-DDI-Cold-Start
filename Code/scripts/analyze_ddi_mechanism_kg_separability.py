"""How distinguishable are DDI *mechanism* categories in the merged KG?

Each DDI sample carries a DrugBank mechanism category (`ddi_type`, 215 types,
templated text encoding pharmacokinetic vs pharmacodynamic mechanism + direction).
This script asks: does the merged-KG structure of a drug pair determine / separate
its mechanism? Tied to the C-MPNN framing "方式1" (each mechanism = a query
relation, different mechanisms should emerge different propagation paths).

Analyses:
  1. Mechanism inventory          -- frequency, long-tail mass.
  2. Per-relation typed shared-neighbour features  -- shared targets / enzymes /
       transporters / carriers / pathways / side-effects / gene-assoc /
       indications / contraindications / resemblance, per drug pair.
  3. Per-mechanism structural fingerprint  -- mean of each typed channel per
       mechanism (does each mechanism load on a distinct relational channel?).
  4. Structure -> mechanism predictability (headline 区分度)  -- multinomial
       logistic regression on the typed-feature vector over the top-K mechanisms;
       top-1 / top-5 accuracy + macro-F1 vs majority & random baselines.
  5. Per-mechanism one-vs-rest AUC  -- which mechanisms are structurally
       separable, which are not.
  6. Relational-WL mechanism collision  -- among WL-indistinguishable pair
       signatures, mechanism purity (the WL ceiling at mechanism resolution).
  7. Coarse PK / PD separability  -- can structure split the PK vs PD axis at all
       (informs whether a learned routing signal even exists in structure).

Operates on the merged KG parquet (drugbank + hetionet + primekg). Reuses the
relational-WL refinement from analyze_ddi_merged_kg_structure.py.

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_ddi_mechanism_kg_separability.py
  python Code/scripts/analyze_ddi_mechanism_kg_separability.py --top-k 30 --wl-rounds 3

Outputs (under --out, default Code/runs/analyze_ddi_mechanism_kg_separability/):
  summary.json                  all scalar stats
  mechanism_fingerprint.parquet per-mechanism mean structural channels
  per_mechanism_auc.parquet     one-vs-rest AUC + best channel per mechanism
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ddi_merged_kg_structure import (  # noqa: E402  (sibling reuse, no duplication)
    build_wl_edges,
    relational_wl,
)

NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"

# Mechanism-relevant relational channels (drug-incident). Shared neighbours along
# each are computed separately so we can see which channel each mechanism loads on.
CHANNELS = [
    "db:target", "db:enzyme", "db:transporter", "db:carrier", "db:pathway",
    "het:CcSE", "prime:drug_effect",
    "het:CdG", "het:CuG", "het:CbG", "prime:drug_protein",
    "prime:indication", "prime:contraindication",
    "het:CrC",
]


def log(m: str) -> None:
    print(m, flush=True)


def pkpd_bucket(ddi_type: str) -> str:
    """Coarse PK/PD axis from the templated mechanism text (keyword rule)."""
    t = ddi_type.lower()
    pk_kw = ("metabolism", "excretion", "serum concentration", "absorption",
             "protein binding", "clearance")
    if any(k in t for k in pk_kw):
        return "PK"
    pd_kw = ("risk or severity", "activities", "efficacy", "cns depression",
             "qtc", "hypertension", "hypotensive", "sedative", "adverse effects")
    if any(k in t for k in pd_kw):
        return "PD"
    return "other"


def build_channel_shared(edges, id2idx, n_nodes, drug_idx):
    """For each channel relation, build drug x node incidence and the dense
    1900x1900 shared-neighbour-count matrix shared[a,b]."""
    n_drug = len(drug_idx)
    glob2local = {int(g): i for i, g in enumerate(drug_idx)}
    drug_set = set(int(g) for g in drug_idx)

    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    rel = edges["relation"].to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    rel = rel[keep]

    shared = {}
    for ch in CHANNELS:
        m = rel == ch
        s, d = src[m], dst[m]
        # orient so the drug endpoint is the row, the entity endpoint the column
        a_src_drug = np.array([x in drug_set for x in s])
        rows, cols = [], []
        # drug as src
        rows.append(np.array([glob2local[x] for x in s[a_src_drug]], dtype=np.int64))
        cols.append(d[a_src_drug])
        # drug as dst
        a_dst_drug = np.array([x in drug_set for x in d]) if len(d) else np.array([], bool)
        if len(d):
            rows.append(np.array([glob2local[x] for x in d[a_dst_drug]], dtype=np.int64))
            cols.append(s[a_dst_drug])
        rr = np.concatenate(rows) if rows else np.array([], np.int64)
        cc = np.concatenate(cols) if cols else np.array([], np.int64)
        inc = sp.coo_matrix((np.ones(len(rr), np.float32), (rr, cc)),
                            shape=(n_drug, n_nodes)).tocsr()
        inc.data[:] = 1.0
        sh = np.asarray((inc @ inc.T).todense()).astype(np.int32)
        np.fill_diagonal(sh, 0)
        shared[ch] = sh
        log(f"[chan] {ch:22s} drug-incident edges={len(rr)}  shared nnz-pairs computed")
    return shared, glob2local


def build_common_distance(edges, id2idx, n_nodes, drug_idx):
    """Relation-agnostic common-neighbour count + shortest-distance proxy."""
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    s = src[keep].astype(np.int64)
    d = dst[keep].astype(np.int64)
    nz = s != d
    s, d = s[nz], d[nz]
    rows = np.concatenate([s, d]); cols = np.concatenate([d, s])
    A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(n_nodes, n_nodes)).tocsr()
    A.data[:] = 1.0; A.sum_duplicates(); A.data[:] = 1.0
    D = A[drug_idx, :].tocsr()
    Wco = np.asarray((D @ D.T).todense()).astype(np.int32); np.fill_diagonal(Wco, 0)
    M = (A @ D.T).tocsr()
    W3 = np.asarray((D @ M).todense()).astype(np.int32); np.fill_diagonal(W3, 0)
    Adr = np.asarray(A[np.ix_(drug_idx, drug_idx)].todense()).astype(np.int32)
    np.fill_diagonal(Adr, 0)
    dist = np.full(Wco.shape, 4, dtype=np.int8)
    dist[W3 > 0] = 3; dist[Wco > 0] = 2; dist[Adr > 0] = 1
    return Wco, dist, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=30, help="#most-frequent mechanisms for the predictability model")
    ap.add_argument("--wl-rounds", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=200_000, help="cap LR training rows for speed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_ddi_mechanism_kg_separability"))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[load] {NODES_PARQUET}")
    nodes = pd.read_parquet(NODES_PARQUET)
    log(f"[load] {EDGES_PARQUET}")
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)
    kinds = nodes["kind"].to_numpy()

    drug_db_ids = nodes["id"].to_numpy()[kinds == "Drug"]
    drug_idx = np.array([id2idx[d] for d in drug_db_ids], dtype=np.int64)
    n_drug = len(drug_idx)
    drug_db2local = {db: i for i, db in enumerate(drug_db_ids)}

    pos = pd.read_csv(DDI_POS_CSV)
    pa = pos["drug_a_id"].map(drug_db2local).to_numpy()
    pb = pos["drug_b_id"].map(drug_db2local).to_numpy()
    ok = (~pd.isna(pa)) & (~pd.isna(pb))
    pa = pa[ok].astype(np.int64); pb = pb[ok].astype(np.int64)
    mech = pos["ddi_type"].to_numpy()[ok]
    n_pos = len(pa)
    log(f"[pos] pairs={n_pos}  n_mechanisms={len(np.unique(mech))}")

    # ---- (1) mechanism inventory ----
    mech_vc = pd.Series(mech).value_counts()
    topk_names = mech_vc.head(args.top_k).index.tolist()
    topk_mass = float(mech_vc.head(args.top_k).sum() / n_pos)
    log(f"[mech] top-{args.top_k} cover {topk_mass*100:.1f}% of samples; "
        f"tail mechanisms (<100 samples)={int((mech_vc<100).sum())}")

    # ---- (2) features ----
    shared, glob2local = build_channel_shared(edges, id2idx, n_nodes, drug_idx)
    Wco, dist, A = build_common_distance(edges, id2idx, n_nodes, drug_idx)

    feat_names = [f"shared_{c}" for c in CHANNELS] + ["common_neighbors", "distance"]
    cols = [shared[c][pa, pb] for c in CHANNELS] + [Wco[pa, pb], dist[pa, pb]]
    X = np.stack(cols, axis=1).astype(np.float64)  # n_pos x n_feat
    # log1p on count channels (all but distance)
    Xt = X.copy()
    Xt[:, :-1] = np.log1p(Xt[:, :-1])

    # ---- (3) per-mechanism fingerprint (mean of each channel) ----
    fp = pd.DataFrame(X, columns=feat_names)
    fp["mechanism"] = mech
    fp_mean = fp.groupby("mechanism")[feat_names].mean()
    fp_mean["n"] = mech_vc.reindex(fp_mean.index).values
    fp_mean = fp_mean.sort_values("n", ascending=False)
    fp_mean.to_parquet(out_dir / "mechanism_fingerprint.parquet")
    log("\n================= PER-MECHANISM STRUCTURAL FINGERPRINT (top mechanisms) =================")
    show_ch = ["shared_db:target", "shared_db:enzyme", "shared_db:transporter",
               "shared_het:CcSE", "shared_prime:indication", "common_neighbors"]
    log(f"{'n':>7s} {'tgt':>6s} {'enz':>6s} {'trn':>6s} {'sideeff':>8s} {'indic':>6s} {'comNbr':>7s}  mechanism")
    for name in topk_names[:18]:
        row = fp_mean.loc[name]
        log(f"{int(row['n']):7d} {row['shared_db:target']:6.2f} {row['shared_db:enzyme']:6.2f} "
            f"{row['shared_db:transporter']:6.2f} {row['shared_het:CcSE']:8.2f} "
            f"{row['shared_prime:indication']:6.2f} {row['common_neighbors']:7.1f}  {name[:52]}")

    # ---- (4) structure -> mechanism predictability (top-K) ----
    in_topk = np.isin(mech, topk_names)
    Xk = Xt[in_topk]; yk_names = mech[in_topk]
    name2lab = {n: i for i, n in enumerate(topk_names)}
    yk = np.array([name2lab[n] for n in yk_names])
    Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.3, random_state=args.seed, stratify=yk)
    if len(Xtr) > args.max_train:
        sel = rng.choice(len(Xtr), args.max_train, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=300, C=1.0, multi_class="multinomial", n_jobs=-1)
    log(f"\n[clf] fit multinomial LR  train={len(Xtr)} test={len(Xte)} classes={len(topk_names)}")
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)
    pred = proba.argmax(1)
    labels_present = np.arange(len(topk_names))
    top1 = float((pred == yte).mean())
    top5 = float(top_k_accuracy_score(yte, proba, k=5, labels=labels_present))
    macro_f1 = float(f1_score(yte, pred, average="macro"))
    # baselines
    maj_lab = np.bincount(ytr).argmax()
    maj_top1 = float((yte == maj_lab).mean())
    rand_top1 = 1.0 / len(topk_names)
    log("\n================= STRUCTURE -> MECHANISM PREDICTABILITY (区分度 headline) =================")
    log(f"  top-1 accuracy     = {top1:.4f}   (majority={maj_top1:.4f}, random={rand_top1:.4f})")
    log(f"  top-5 accuracy     = {top5:.4f}")
    log(f"  macro-F1           = {macro_f1:.4f}")

    # ---- (5) per-mechanism one-vs-rest AUC ----
    ovr_rows = []
    for lab, name in enumerate(topk_names):
        y_bin = (yte == lab).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        auc = float(roc_auc_score(y_bin, proba[:, lab]))
        # best single channel for this mechanism (mean diff direction via |corr|)
        ovr_rows.append({"mechanism": name, "n": int(mech_vc[name]), "ovr_auc": auc})
    ovr = pd.DataFrame(ovr_rows).sort_values("ovr_auc", ascending=False)
    ovr.to_parquet(out_dir / "per_mechanism_auc.parquet")
    log("\n  most structurally-separable mechanisms (one-vs-rest AUC):")
    for _, r in ovr.head(6).iterrows():
        log(f"    AUC={r['ovr_auc']:.3f}  n={int(r['n']):6d}  {r['mechanism'][:60]}")
    log("  least structurally-separable mechanisms:")
    for _, r in ovr.tail(6).iterrows():
        log(f"    AUC={r['ovr_auc']:.3f}  n={int(r['n']):6d}  {r['mechanism'][:60]}")

    # ---- (6) relational-WL mechanism collision ----
    log("\n================= RELATIONAL-WL MECHANISM COLLISION =================")
    init_color = pd.factorize(kinds, sort=True)[0].astype(np.int64)
    u, r_, v, n_rel_total = build_wl_edges(edges, id2idx, n_nodes)
    hist = relational_wl(u, r_, v, init_color, n_nodes, n_rel_total, args.wl_rounds)
    drug_color = hist[-1][drug_idx]
    cA, cB = drug_color[pa], drug_color[pb]
    lo = np.minimum(cA, cB); hi = np.maximum(cA, cB)
    keys = np.stack([lo, hi], axis=1)
    _, sig = np.unique(keys, axis=0, return_inverse=True)
    dfm = pd.DataFrame({"sig": sig, "mech": mech})
    grp = dfm.groupby("sig")
    sig_size = grp.size()
    sig_nmech = grp["mech"].nunique()
    colliding = sig_size[sig_size >= 2].index
    n_pairs_on_colliding = int(sig_size[sig_size >= 2].sum())
    # mechanism purity among colliding signatures: dominant-mechanism share
    dom = grp["mech"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    pure_share_on_colliding = float(dom[colliding].mean()) if len(colliding) else float("nan")
    frac_pairs_mech_ambiguous = float((sig_nmech[sig].values > 1).mean())
    log(f"  distinct pair signatures           = {dfm['sig'].nunique()} / {n_pos} pairs")
    log(f"  pairs on a colliding signature(>=2)= {n_pairs_on_colliding} ({n_pairs_on_colliding/n_pos*100:.2f}%)")
    log(f"  pairs whose signature is mechanism-ambiguous = {frac_pairs_mech_ambiguous*100:.2f}%")
    log(f"  mean dominant-mechanism purity on colliding signatures = {pure_share_on_colliding:.3f}")

    # ---- (7) coarse PK/PD separability ----
    log("\n================= COARSE PK / PD SEPARABILITY =================")
    buck = np.array([pkpd_bucket(m) for m in mech])
    for b in ["PK", "PD", "other"]:
        log(f"  bucket {b:5s}: {int((buck==b).sum())} pairs ({(buck==b).mean()*100:.1f}%)")
    pkpd_auc = None
    sel = np.isin(buck, ["PK", "PD"])
    if sel.sum() > 1000:
        Xs = Xt[sel]; ys = (buck[sel] == "PK").astype(int)
        Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.3, random_state=args.seed, stratify=ys)
        sc = StandardScaler().fit(Xa)
        c2 = LogisticRegression(max_iter=300).fit(sc.transform(Xa), ya)
        pkpd_auc = float(roc_auc_score(yb, c2.predict_proba(sc.transform(Xb))[:, 1]))
        log(f"  PK-vs-PD structural AUC = {pkpd_auc:.4f}  (0.5 = structure cannot route PK/PD)")

    # ---- save ----
    summary = {
        "n_pos": n_pos,
        "n_mechanisms": int(len(np.unique(mech))),
        "topk": args.top_k,
        "topk_mass": topk_mass,
        "features": feat_names,
        "predictability": {
            "top1": top1, "top5": top5, "macro_f1": macro_f1,
            "majority_top1": maj_top1, "random_top1": rand_top1,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        },
        "wl_mechanism_collision": {
            "distinct_signatures": int(dfm["sig"].nunique()),
            "pairs_on_colliding_signature": n_pairs_on_colliding,
            "frac_pairs_mechanism_ambiguous": frac_pairs_mech_ambiguous,
            "mean_dominant_purity_on_colliding": pure_share_on_colliding,
        },
        "pkpd": {
            "pk_pairs": int((buck == "PK").sum()),
            "pd_pairs": int((buck == "PD").sum()),
            "other_pairs": int((buck == "other").sum()),
            "pk_vs_pd_auc": pkpd_auc,
        },
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n[save] {out_dir/'summary.json'}")
    log(f"[save] {out_dir/'mechanism_fingerprint.parquet'}")
    log(f"[save] {out_dir/'per_mechanism_auc.parquet'}")
    log(f"[done] wall_time={summary['wall_time_s']}s")


if __name__ == "__main__":
    sys.exit(main())
