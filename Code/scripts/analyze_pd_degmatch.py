"""Decisive test: is there PD structural signal BEYOND drug popularity?

Read-only. Random-pair negatives are confounded — PD positives involve more
heavily-annotated (higher-degree) drugs, and drug degree alone already gives
AUROC ~0.58. This script rebuilds the negatives to be DEGREE-MATCHED to the
positives (each negative pair drawn from the same per-drug degree deciles), so
drug popularity is neutralised. Any remaining structural-model AUROC is the
genuine mechanism signal.

Reports: (a) drug-degree AUROC under matching (should collapse to ~0.50),
(b) structural-feature CV AUROC under matched vs random negatives.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_degmatch.py --n 800
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

DDI = "Code/data/KG/drugbank/filtered/ddi_edges.csv"
PKPD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
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
    d = kg.degree[sh]; ty = kg.type_id[sh]
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
    return f


def au(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = int(y.sum()); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))


def cv_auroc(rows, lab):
    X = pd.DataFrame(rows).fillna(0.0).to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X, lab, cv=5, scoring="roc_auc").mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args(); rng = np.random.default_rng(42)
    print("loading KG ...", flush=True)
    kg = MergedKG.from_parquet()
    pk = pd.read_csv(PKPD); pdt = set(pk[pk["pk_pd_label"] == "PD"]["ddi_type"])
    ddi = pd.read_csv(DDI); ddi = ddi[ddi["ddi_type"].isin(pdt)].copy()
    ddi["ai"] = ddi["drug_a_id"].map(kg.id_to_idx); ddi["bi"] = ddi["drug_b_id"].map(kg.id_to_idx)
    ddi = ddi.dropna(subset=["ai", "bi"]).astype({"ai": int, "bi": int})
    pool = np.array(sorted(set(ddi.ai) | set(ddi.bi)))
    posset = set(map(frozenset, zip(ddi.ai, ddi.bi)))
    deg = kg.degree

    # degree deciles over the drug pool, for matched negative sampling
    pool_deg = deg[pool]
    edges = np.quantile(pool_deg, np.linspace(0, 1, 11))
    edges[-1] += 1
    binof = {int(d): int(np.clip(np.digitize(deg[d], edges) - 1, 0, 9)) for d in pool}
    bins = {k: [] for k in range(10)}
    for d in pool.tolist():
        bins[binof[d]].append(d)

    idx = rng.choice(len(ddi), args.n, replace=False)
    pos = list(zip(ddi.ai.to_numpy()[idx], ddi.bi.to_numpy()[idx]))

    def match(d):
        c = bins[binof[int(d)]]
        return int(rng.choice(c))

    neg_match, neg_rand = [], []
    for a, b in pos:
        for _ in range(20):
            a2, b2 = match(a), match(b)
            if a2 != b2 and frozenset((a2, b2)) not in posset:
                neg_match.append((a2, b2)); break
        while True:
            a2, b2 = int(rng.choice(pool)), int(rng.choice(pool))
            if a2 != b2 and frozenset((a2, b2)) not in posset:
                neg_rand.append((a2, b2)); break

    y = np.r_[np.ones(len(pos)), np.zeros(len(pos))]
    # degree-AUROC sanity under each negative scheme
    for tag, neg in [("RANDOM neg", neg_rand), ("DEG-MATCHED neg", neg_match)]:
        allp = pos + neg
        ds = np.array([deg[a] + deg[b] for a, b in allp], float)
        print(f"\n[{tag}]  drug-degree-SUM AUROC = {au(ds, y):.3f}")
        rows = [feats(kg, a, b) for a, b in allp]
        print(f"[{tag}]  structural-feature CV AUROC = {cv_auroc(rows, y):.3f}")


if __name__ == "__main__":
    main()
