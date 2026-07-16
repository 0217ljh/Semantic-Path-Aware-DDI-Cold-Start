"""Per-mechanism structural separability of PD DDI types.

Read-only. Tests whether the aggregate ~0.60 PD ceiling hides heterogeneity:
mechanistically specific PD types (QTc prolongation, serotonin syndrome,
bleeding, hyperkalemia ...) may converge on a specific shared protein/pathway
and be structurally separable, while diffuse types (CNS depression, adverse
effects, decreased efficacy) are structural noise.

One logistic model is trained (5-fold out-of-fold) on broad structural features
for PD-positive vs random pairs; per-ddi-type AUROC is then read off the OOF
scores. Also prints concrete example pairs (named drugs + named shared
mediators) for the best and worst separating types.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_by_mechanism.py
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

DDI = "Code/data/KG/drugbank/filtered/ddi_edges.csv"
PKPD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
PROT, PATH, DIS, BP, SE = (KIND_ORDER.index(k) for k in
                           ("protein_gene", "pathway", "disease",
                            "biological_process", "side_effect"))


def nb(kg, u):
    return kg.indices[kg.indptr[u]:kg.indptr[u + 1]]


def tnb(kg, u, t):
    x = nb(kg, u)
    return x[kg.type_id[x] == t]


def feats(kg, a, b):
    na = nb(kg, a)[~kg.is_drug[nb(kg, a)]]
    nbb = nb(kg, b)[~kg.is_drug[nb(kg, b)]]
    sh = np.intersect1d(na, nbb)
    d = kg.degree[sh]
    ty = kg.type_id[sh]
    f = {"sh_se": int((ty == SE).sum()), "sh_dis": int((ty == DIS).sum()),
         "sh_prot": int((ty == PROT).sum()),
         "aa": float(np.sum(1 / np.log(np.clip(d, 2, None)))) if len(d) else 0.0}
    wa, da = kg.neighborhood(a, 2); wb, db = kg.neighborhood(b, 2)
    wa = wa[(da >= 1) & ~kg.is_drug[wa]]; wb = wb[(db >= 1) & ~kg.is_drug[wb]]
    s2 = np.intersect1d(wa, wb); t2 = kg.type_id[s2]
    f["n_sh2"] = len(s2); f["sh2_prot"] = int((t2 == PROT).sum())
    f["sh2_path"] = int((t2 == PATH).sum()); f["sh2_dis"] = int((t2 == DIS).sum())
    PA, PB = tnb(kg, a, PROT), tnb(kg, b, PROT)
    for cn, ct in [("path", PATH), ("dis", DIS), ("bp", BP)]:
        WA = set(); WB = set()
        for u in PA.tolist():
            WA.update(tnb(kg, u, ct).tolist())
        for v in PB.tolist():
            WB.update(tnb(kg, v, ct).tolist())
        conn = WA & WB
        f[f"L4_{cn}_aa"] = sum(1 / math.log(max(int(kg.degree[w]), 2)) for w in conn)
    f["_sh"] = sh
    return f


def auroc(s, y):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")


def short(t):
    t = re.sub(r"\bcan be (increased|decreased)\b.*", "", t)
    t = t.replace("The risk or severity of", "").replace("The therapeutic efficacy of", "EFFICACY")
    t = t.replace("may increase the", "↑").replace("may decrease the", "↓")
    return t.strip()[:40]


def main():
    rng = np.random.default_rng(42)
    print("loading KG ...", flush=True)
    kg = MergedKG.from_parquet()
    names = pd.read_parquet(NODES, columns=["id", "name"]).set_index("id")["name"]
    pk = pd.read_csv(PKPD); pdt = set(pk[pk["pk_pd_label"] == "PD"]["ddi_type"])
    ddi = pd.read_csv(DDI); ddi = ddi[ddi["ddi_type"].isin(pdt)].copy()
    ddi["ai"] = ddi["drug_a_id"].map(kg.id_to_idx); ddi["bi"] = ddi["drug_b_id"].map(kg.id_to_idx)
    ddi = ddi.dropna(subset=["ai", "bi"]).astype({"ai": int, "bi": int})
    pool = sorted(set(ddi.ai) | set(ddi.bi))
    pos_all = set(map(frozenset, zip(ddi.ai, ddi.bi)))

    top = ddi["ddi_type"].value_counts()
    types = [t for t in top.index if top[t] >= 200][:18]
    M = 90
    rows, meta = [], []
    for t in types:
        sub = ddi[ddi["ddi_type"] == t]
        idx = rng.choice(len(sub), size=min(M, len(sub)), replace=False)
        for a, b in zip(sub.ai.to_numpy()[idx], sub.bi.to_numpy()[idx]):
            rows.append((int(a), int(b))); meta.append(t)
    nneg = 700
    neg = []
    while len(neg) < nneg:
        a, b = rng.choice(pool, 2, replace=False)
        if a != b and frozenset((a, b)) not in pos_all:
            neg.append((int(a), int(b)))
    allpairs = rows + neg
    lab = np.r_[np.ones(len(rows)), np.zeros(len(neg))]
    print(f"computing features for {len(allpairs)} pairs ({len(types)} PD types) ...", flush=True)
    F = [feats(kg, a, b) for a, b in allpairs]
    cols = [k for k in F[0] if not k.startswith("_")]
    X = pd.DataFrame([{k: f[k] for k in cols} for f in F]).fillna(0.0)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    oof = cross_val_predict(clf, X.to_numpy(), lab, cv=5, method="predict_proba")[:, 1]
    neg_scores = oof[len(rows):]

    print("\n=== per-mechanism structural separability (AUROC vs random pool) ===")
    res = []
    for t in types:
        m = np.array([mt == t for mt in meta])
        ps = oof[:len(rows)][m]
        s = np.r_[ps, neg_scores]; y = np.r_[np.ones(len(ps)), np.zeros(len(neg_scores))]
        res.append((auroc(s, y), len(ps), int(top[t]), t))
    for a, n, tot, t in sorted(res, reverse=True):
        print(f"  AUROC={a:.3f}  (n={n:>3}, total={tot:>6})  {short(t)}")

    # concrete examples for best & worst specific types
    ranked = sorted(res, reverse=True)
    def nm(idx):
        nid = kg.node_ids[idx]
        v = names.get(nid, "?")
        return str(v.iloc[0] if isinstance(v, pd.Series) else v)
    for tag, (a_, n_, tot_, t) in [("BEST", ranked[0]), ("WORST", ranked[-1])]:
        print(f"\n=== {tag} type: {short(t)}  (AUROC {a_:.3f}) — example pairs ===")
        ex = [(rows[i], F[i]) for i in range(len(rows)) if meta[i] == t][:3]
        for (a, b), f in ex:
            sh = f["_sh"]; d = kg.degree[sh]; order = np.argsort(-d)
            tops = [f"{nm(sh[j])}(d{kg.degree[sh[j]]})" for j in order[:5]]
            print(f"  {nm(a)} + {nm(b)}: {len(sh)} shared | top: {', '.join(tops) if tops else 'NONE'}")


if __name__ == "__main__":
    main()
