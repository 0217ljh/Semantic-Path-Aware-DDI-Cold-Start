"""idea2 v2 — L4 go/no-go probe: does v15 MESO hierarchy add discriminative cold-start signal?

codex 019f1daa: lever A (renormalized layered KG -> GNN) is the only path to beat EmerGNN,
but it's "same GNN, better KG" and only defensible if v15 creates REACHABLE shared mediators
for UNSEEN-drug pairs that the flat micro KG lacks. Cheap CPU pre-check BEFORE any GPU retrain,
and it must be LABEL-AWARE (raw density inflates pos+neg equally via hubs).

Feature (codex):  x_k(a,b) = sum_{m in kind k, dist(a,m)<=2 and dist(b,m)<=2} 1/log(2+deg(m))
  drug->protein (dist1) ->meso (dist2), so shared meso m = meso_set(a,k) & meso_set(b,k),
  each downweighted by log KG-degree (kills hub mediators). k in {pathway,gobp,anatomy,disease,phenotype}.

Same L2 logreg, binary S2 (drugbank_latest_full = EmerGNN's ddi_full), train -> val:
  cond OLD  = micro typed features only (the established 0.695 feature set)
  cond ADD  = micro + v15 meso features
Decision:  dAUROC = AUROC(ADD) - AUROC(OLD) on S2 val.
  GO (pursue GNN KG-swap) iff dAUROC >= +0.015 AND AUROC(ADD) >= 0.72 ; else KILL lever A. CPU.

Usage: PYTHONPATH=Code/code-idea-2 python -u Code/code-idea-2/l4_v15_meso_probe.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import kg_v15

DATASET = "drugbank_latest_full"  # == EmerGNN --dataset ddi_full
MESO_KINDS = ["pathway", "gobp", "anatomy", "disease", "phenotype"]


def load_bin(split_dir, sp):
    d = pd.read_parquet(f"{split_dir}/{sp}.parquet")
    d["a"] = "drug:" + d["drug_a_id"].astype(str)
    d["b"] = "drug:" + d["drug_b_id"].astype(str)
    return d


def main():
    nd, ed = kg_v15.load_v15()
    adj, drugs = kg_v15.build_typed_drug_adj(nd, ed)
    p2 = kg_v15.build_protein_meso(ed)
    deg = pd.concat([ed["src"], ed["dst"]]).astype(str).value_counts().to_dict()
    w = lambda m: 1.0 / math.log(2 + deg.get(m, 0))

    base = f"Code/data/ddi_unified/binary_cls/{DATASET}/inductive/S2/fold0"
    tr, va = load_bin(base, "train"), load_bin(base, "val")
    inkg = lambda df: df[df["a"].isin(drugs) & df["b"].isin(drugs)].copy()
    tr, va = inkg(tr), inkg(va)
    ycol = "y_bin" if "y_bin" in tr.columns else ("y" if "y" in tr.columns else "label")
    print(f"[{DATASET} S2] train {len(tr)} (pos {int(tr[ycol].sum())}) | val {len(va)} (pos {int(va[ycol].sum())}) | label={ycol}")

    micro_cols = [f"{p}_{s}" for s in ("target", "enzyme", "transporter", "carrier", "protein", "pclass")
                  for p in ("shared", "jacc")]

    def meso_set(d, k):
        prots = adj["protein"].get(d, set())
        return set().union(*[p2[k].get(pp, set()) for pp in prots]) if prots else set()

    def feats(df):
        micro, meso = [], []
        # cache meso sets per drug
        cache = {}
        def ms(d, k):
            key = (d, k)
            if key not in cache:
                cache[key] = meso_set(d, k)
            return cache[key]
        for a, b in zip(df["a"], df["b"]):
            micro.append(kg_v15.pk_pair_feats(a, b, adj))
            row = {}
            for k in MESO_KINDS:
                sh = ms(a, k) & ms(b, k)
                row[f"v15_{k}"] = sum(w(m) for m in sh)
            meso.append(row)
        Xmi = pd.DataFrame(micro)[micro_cols].fillna(0.0).values
        Xme = pd.DataFrame(meso)[[f"v15_{k}" for k in MESO_KINDS]].fillna(0.0).values
        return Xmi, Xme

    Xmi_tr, Xme_tr = feats(tr)
    Xmi_va, Xme_va = feats(va)
    ytr, yva = tr[ycol].astype(int).values, va[ycol].astype(int).values

    def fit_score(Xtr, Xva):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        return roc_auc_score(yva, clf.predict_proba(sc.transform(Xva))[:, 1])

    auc_old = fit_score(Xmi_tr, Xmi_va)
    auc_add = fit_score(np.hstack([Xmi_tr, Xme_tr]), np.hstack([Xmi_va, Xme_va]))
    auc_meso_only = fit_score(Xme_tr, Xme_va)
    d = auc_add - auc_old
    print("\n== L4 v15-meso go/no-go (binary S2 val) ==")
    print(f"  OLD  (micro typed only)   AUROC {auc_old:.4f}")
    print(f"  ADD  (micro + v15 meso)   AUROC {auc_add:.4f}")
    print(f"  meso-only (5 feats)       AUROC {auc_meso_only:.4f}")
    print(f"  dAUROC (ADD - OLD)        {d:+.4f}")
    # per-meso standalone signal
    print("  per-meso standalone AUROC:")
    for j, k in enumerate(MESO_KINDS):
        s = np.r_[Xme_va[:, j]]
        try:
            print(f"    {k:<10} {roc_auc_score(yva, s):.3f}")
        except Exception:
            print(f"    {k:<10} n/a")
    go = (d >= 0.015) and (auc_add >= 0.72)
    print(f"\n  DECISION (linear): {'GO' if go else 'NO-GO'}"
          f"  (threshold: dAUROC>=+0.015 AND AUROC(ADD)>=0.72 | got d={d:+.4f}, add={auc_add:.4f})")

    # ---- codex 019f1db1 precommitted FINAL adjudicator: non-linear GBT, same gate ----
    def fit_score_gbt(Xtr, Xva):
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, l2_regularization=1.0)
        clf.fit(Xtr, ytr)
        return roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])

    g_old = fit_score_gbt(Xmi_tr, Xmi_va)
    g_add = fit_score_gbt(np.hstack([Xmi_tr, Xme_tr]), np.hstack([Xmi_va, Xme_va]))
    gd = g_add - g_old
    print("\n== GBT tie-break (non-linear, FINAL adjudicator; matches GNN non-linearity) ==")
    print(f"  GBT OLD  (micro only)     AUROC {g_old:.4f}")
    print(f"  GBT ADD  (micro + meso)   AUROC {g_add:.4f}")
    print(f"  GBT dAUROC (ADD - OLD)    {gd:+.4f}")
    gbt_go = (gd >= 0.015) and (g_add >= 0.72)
    print(f"\n  FINAL DECISION: {'GO -> greenlight GNN KG-swap (v15 meso is GNN-liftable)' if gbt_go else 'NO-GO -> KILL lever A as clear-margin-beat story; reframe idea2 as diagnosis + KG-resource contribution'}")
    print(f"           (gate: GBT dAUROC>=+0.015 AND GBT AUROC(ADD)>=0.72 | got d={gd:+.4f}, add={g_add:.4f})")


if __name__ == "__main__":
    main()
