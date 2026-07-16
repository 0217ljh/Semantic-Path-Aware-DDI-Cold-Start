"""Combined structural-feature ceiling for PD-vs-random binary classification.

Read-only. Merges all 1-hop / 2-hop / length-3-4 metapath features into one
vector per drug pair and reports the cross-validated AUROC of a simple
classifier (logistic + histogram gradient boosting). This estimates how much
the FULL structural/path signal can separate PD-positive pairs from random
drug pairs — the honest "basic binary-cls" ceiling.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_clf.py --n 800
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

DDI_EDGES = "Code/data/KG/drugbank/filtered/ddi_edges.csv"
PK_PD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
PROT = KIND_ORDER.index("protein_gene")
PATH = KIND_ORDER.index("pathway")
DIS = KIND_ORDER.index("disease")
BP = KIND_ORDER.index("biological_process")
SE = KIND_ORDER.index("side_effect")


def nbrs(kg, u):
    return kg.indices[kg.indptr[u]:kg.indptr[u + 1]]


def tnbrs(kg, u, t):
    nb = nbrs(kg, u)
    return nb[kg.type_id[nb] == t]


def feats(kg, a, b) -> dict:
    na = nbrs(kg, a)[~kg.is_drug[nbrs(kg, a)]]
    nb = nbrs(kg, b)[~kg.is_drug[nbrs(kg, b)]]
    sh = np.intersect1d(na, nb)
    d = kg.degree[sh]
    ty = kg.type_id[sh]
    f = {
        "n_shared": len(sh),
        "sh_prot": int((ty == PROT).sum()),
        "sh_se": int((ty == SE).sum()),
        "sh_dis": int((ty == DIS).sum()),
        "sh_maxdeg": int(d.max()) if len(d) else 0,
        "sh_hub1k": int((d >= 1000).sum()),
        "aa": float(np.sum(1.0 / np.log(np.clip(d, 2, None)))) if len(d) else 0.0,
    }
    wa, da = kg.neighborhood(a, 2)
    wb, db = kg.neighborhood(b, 2)
    wa = wa[(da >= 1) & ~kg.is_drug[wa]]
    wb = wb[(db >= 1) & ~kg.is_drug[wb]]
    s2 = np.intersect1d(wa, wb)
    t2 = kg.type_id[s2]
    f["n_sh2"] = len(s2)
    f["sh2_prot"] = int((t2 == PROT).sum())
    f["sh2_path"] = int((t2 == PATH).sum())
    f["sh2_dis"] = int((t2 == DIS).sum())
    PA, PB = tnbrs(kg, a, PROT), tnbrs(kg, b, PROT)
    PBs = set(PB.tolist())
    f["L3_ppi"] = sum(int(np.isin(tnbrs(kg, u, PROT), PB).sum()) for u in PA.tolist())
    for cn, ct in [("path", PATH), ("dis", DIS), ("bp", BP)]:
        WA: set = set()
        for u in PA.tolist():
            WA.update(tnbrs(kg, u, ct).tolist())
        WB: set = set()
        for v in PB.tolist():
            WB.update(tnbrs(kg, v, ct).tolist())
        conn = WA & WB
        f[f"L4_{cn}"] = len(conn)
        f[f"L4_{cn}_aa"] = sum(1.0 / math.log(max(int(kg.degree[w]), 2)) for w in conn)

    # ---- higher-order coherence features (beyond 1-WL) ----
    # (i) internal edges among the shared 1-hop mediator set: does the
    #     convergence set itself form a dense sub-cluster?
    int_edges = 0
    for m in sh.tolist():
        int_edges += int(np.isin(nbrs(kg, m), sh).sum())
    f["sh_int_edges"] = int_edges // 2
    # (ii) coherent convergence motif Phi_coh: u in P(A), v in P(B), u~v
    #      (interacting proteins), counting typed connectors W in N(u)&N(v).
    #      This is the triangle-closed (u,v,W) motif -> provably beyond 1-WL.
    CONN = [PATH, DIS, BP]
    coh = 0
    coh_aa = 0.0
    ppi_closed = 0
    for u in PA.tolist():
        Nu = nbrs(kg, u)
        vv = Nu[np.isin(Nu, PB)]            # v in P(B) with u~v (PPI edge)
        for v in vv.tolist():
            common = np.intersect1d(Nu, nbrs(kg, v), assume_unique=False)
            common = common[np.isin(kg.type_id[common], CONN)]
            if len(common):
                ppi_closed += 1
                coh += len(common)
                coh_aa += float(np.sum(
                    1.0 / np.log(np.clip(kg.degree[common], 2, None))))
    f["coh_tri"] = coh
    f["coh_tri_aa"] = coh_aa
    f["ppi_closed"] = ppi_closed
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    print("loading KG ...", flush=True)
    kg = MergedKG.from_parquet()
    pk = pd.read_csv(PK_PD)
    pdt = set(pk.loc[pk["pk_pd_label"] == "PD", "ddi_type"])
    ddi = pd.read_csv(DDI_EDGES)
    ddi = ddi[ddi["ddi_type"].isin(pdt)].copy()
    ddi["ai"] = ddi["drug_a_id"].map(kg.id_to_idx)
    ddi["bi"] = ddi["drug_b_id"].map(kg.id_to_idx)
    ddi = ddi.dropna(subset=["ai", "bi"]).astype({"ai": int, "bi": int})
    pool = sorted(set(ddi.ai) | set(ddi.bi))
    pos = set(map(frozenset, zip(ddi.ai, ddi.bi)))
    idx = rng.choice(len(ddi), size=min(args.n, len(ddi)), replace=False)
    pdp = list(zip(ddi.ai.to_numpy()[idx], ddi.bi.to_numpy()[idx]))
    rnd = []
    while len(rnd) < args.n:
        a, b = rng.choice(pool, 2, replace=False)
        if a != b and frozenset((a, b)) not in pos:
            rnd.append((int(a), int(b)))
    print(f"computing features for {2*args.n} pairs ...", flush=True)
    rows = [feats(kg, a, b) for a, b in pdp] + [feats(kg, a, b) for a, b in rnd]
    X = pd.DataFrame(rows).fillna(0.0)
    y = np.r_[np.ones(len(pdp)), np.zeros(len(rnd))]
    cols = list(X.columns)

    print("\n=== single-feature AUROC (sorted) ===")
    def au(s):
        o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        npos = int(y.sum()); return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))
    aus = sorted(((au(X[c].to_numpy().astype(float)), c) for c in cols), reverse=True)
    for a_, c in aus[:8]:
        print(f"  {c:<14} {max(a_,1-a_):.3f}")

    hi_cols = ["sh_int_edges", "coh_tri", "coh_tri_aa", "ppi_closed"]
    lo_cols = [c for c in cols if c not in hi_cols]
    print("\n=== combined CV AUROC (5-fold): first-order vs +higher-order ===")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08,
                                        max_iter=300, l2_regularization=1.0)
    for name, clf in [("logistic", lr), ("histGBDT", gb)]:
        s_lo = cross_val_score(clf, X[lo_cols].to_numpy(), y, cv=5, scoring="roc_auc")
        s_all = cross_val_score(clf, X.to_numpy(), y, cv=5, scoring="roc_auc")
        print(f"  {name:<10} first-order={s_lo.mean():.3f}  "
              f"+higher-order={s_all.mean():.3f}  "
              f"lift={s_all.mean()-s_lo.mean():+.3f}")

    gb.fit(X.to_numpy(), y)
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(gb, X.to_numpy(), y, n_repeats=5,
                                     random_state=0, scoring="roc_auc")
        print("\n=== top features (permutation importance) ===")
        for i in np.argsort(imp.importances_mean)[::-1][:8]:
            print(f"  {cols[i]:<14} {imp.importances_mean[i]:+.4f}")
    except Exception as e:
        print("perm-imp skipped:", e)


if __name__ == "__main__":
    main()
