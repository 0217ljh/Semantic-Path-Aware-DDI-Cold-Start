"""idea2 v2 — L2 pure-PK typed-motif baseline on the real cold-start benchmark.

Tests the mechanism-typed thesis on the ACTUAL ddi_unified multi_cls benchmark (not the
scratchpad degree-matched probe): PK typed shared-motif features (shared enzyme/transporter/
target/carrier via kg_v15) should discriminate PK-type DDIs strongly and PD-type DDIs
weakly. Trains logreg + GBT (CPU) on cold-start S2 train, evaluates test AUROC/AUPRC
overall + bucketed by PK/PD mechanism (via ddi_type taxonomy). CPU-only.

Usage (WSL conda project_1, from project root):
    PYTHONPATH=Code/code-idea-2 python -u Code/code-idea-2/l2_pk_baseline.py --dataset drugbank_latest_full
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import kg_v15

TAX = "/mnt/c/Users/27911/AppData/Local/Temp/claude/D--My-Research/217a6adf-9de6-4f18-a70d-31133f2a82a3/scratchpad/ddi_type_taxonomy.parquet"
TYPEMAP = "Code/data/_cache/ddi_type_map.json"


def load_split(dataset: str, split: str = "S2", fold: str = "fold0"):
    base = f"Code/data/ddi_unified/binary_cls/{dataset}/inductive/{split}/{fold}"
    tr = pd.read_parquet(f"{base}/train.parquet")
    te = pd.read_parquet(f"{base}/test.parquet")
    return tr, te


def build_pair2pkpd(dataset: str, split: str, fold: str, idx2pkpd: dict):
    """(drug_a,drug_b)->PK/PD from the multi_cls typed pairs (all interacting)."""
    p2 = {}
    for sp in ("train", "test", "val"):
        fp = f"Code/data/ddi_unified/multi_cls/{dataset}/inductive/{split}/{fold}/{sp}.parquet"
        if not glob.glob(fp):
            continue
        m = pd.read_parquet(fp)
        for a, b, y in zip(m["drug_a_id"].astype(str), m["drug_b_id"].astype(str), m["y_cls"]):
            if y != 0:
                bucket = idx2pkpd.get(y)
                if bucket:
                    p2[(a, b)] = bucket; p2[(b, a)] = bucket
    return p2


def feat_matrix(df, adj, feat_cols):
    rows = []
    for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
        rows.append(kg_v15.pk_pair_feats(f"drug:{a}", f"drug:{b}", adj))
    F = pd.DataFrame(rows)[feat_cols].fillna(0.0).values
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="drugbank_latest_full")
    ap.add_argument("--split", default="S2")
    args = ap.parse_args()

    nd, ed = kg_v15.load_v15()
    adj, drugs = kg_v15.build_typed_drug_adj(nd, ed)
    feat_cols = [f"{p}_{s}" for s in ("target", "enzyme", "transporter", "carrier", "protein", "pclass")
                 for p in ("shared", "jacc")]

    tr, te = load_split(args.dataset, args.split)
    for df in (tr, te):
        df["a"] = "drug:" + df["drug_a_id"].astype(str)
        df["b"] = "drug:" + df["drug_b_id"].astype(str)
    inkg = lambda df: df[df["a"].isin(drugs) & df["b"].isin(drugs)].copy()
    tr, te = inkg(tr), inkg(te)
    ycol = "y_bin" if "y_bin" in tr.columns else ("y" if "y" in tr.columns else "label")
    ytr = tr[ycol].astype(int).values
    yte = te[ycol].astype(int).values
    print(f"[{args.dataset} {args.split}] train {len(tr)} (pos {ytr.sum()}) | test {len(te)} (pos {yte.sum()}) | label={ycol}")

    Xtr, Xte = feat_matrix(tr, adj, feat_cols), feat_matrix(te, adj, feat_cols)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    results = {}
    for name, clf, Xa, Xb in [("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"), Xtr_s, Xte_s),
                              ("gbt", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1), Xtr, Xte)]:
        clf.fit(Xa, ytr)
        p = clf.predict_proba(Xb)[:, 1]
        auroc, auprc = roc_auc_score(yte, p), average_precision_score(yte, p)
        results[name] = {"auroc": round(auroc, 4), "auprc": round(auprc, 4)}
        print(f"  {name}: test AUROC {auroc:.4f} AUPRC {auprc:.4f}")
        # bucket test positives by PK/PD (via multi_cls type join) — AUROC of bucket-pos vs all neg
        tax = pd.read_parquet(TAX)
        idx2pkpd = dict(zip(tax["idx"], tax["pk_pd"]))
        p2 = build_pair2pkpd(args.dataset, args.split, "fold0", idx2pkpd)
        te2 = te.copy()
        te2["pkpd"] = [p2.get((a, b), "UNK") for a, b in zip(te2["drug_a_id"].astype(str), te2["drug_b_id"].astype(str))]
        te2["score"] = p; te2["y"] = yte
        neg_mask = te2["y"] == 0
        for bucket in ["PK", "PD"]:
            m = (te2["pkpd"] == bucket) | neg_mask
            yb, sb = te2.loc[m, "y"].values, te2.loc[m, "score"].values
            if yb.sum() >= 20 and (yb == 0).sum() >= 20:
                ab = roc_auc_score(yb, sb)
                results[name][f"auroc_{bucket}"] = round(ab, 4)
                print(f"    {bucket}-bucket AUROC {ab:.4f} (pos {int(yb.sum())})")

    print("\nRESULT_JSON " + json.dumps({"dataset": args.dataset, "split": args.split,
                                         "n_test": len(te), "results": results}))


if __name__ == "__main__":
    main()
