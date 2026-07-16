"""idea2 v2 — L3 QTc kill-test: does hERG qt_liability beat the KG ceiling under CYP-match?

QTc-DDI (pos) vs CYP-matched non-QTc DDI (neg, both share >=1 CYP). Pair features (codex
019f1d52): max(qA,qB) PRIMARY (one QT-liable drug suffices), min & product secondary,
vs the ~0.60 KG-feature ceiling and shared-CYP. If max clears the ceiling clearly -> the
PD-liability story holds; else kill/revise. CPU.

Usage: PYTHONPATH=Code/code-idea-2 python -u Code/code-idea-2/liability/l3_qtc_killtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import kg_v15

TAX = "/mnt/c/Users/27911/AppData/Local/Temp/claude/D--My-Research/217a6adf-9de6-4f18-a70d-31133f2a82a3/scratchpad/ddi_type_taxonomy.parquet"
SCORES = "Code/code-idea-2/liability/qt_liability_scores.parquet"
RNG = np.random.RandomState(42)


def main():
    nd, ed = kg_v15.load_v15()
    adj, drugs = kg_v15.build_typed_drug_adj(nd, ed)
    p2 = kg_v15.build_protein_meso(ed)
    nm = dict(zip(nd["id"], nd["name"]))
    cyp = {i for i, n in nm.items() if str(i).startswith("prot:") and str(n).upper().startswith("CYP")}
    qs = pd.read_parquet(SCORES)
    q = dict(zip("drug:" + qs["drugbank_id"].astype(str), qs["qt_liability_score"]))

    tax = pd.read_parquet(TAX)
    qtc_idx = set(tax[tax["sub"] == "QTc/torsade"]["idx"])
    pos = pd.concat([pd.read_parquet(f"Code/data/ddi_unified/multi_cls/drugbank_ryu/inductive/S1/fold0/{sp}.parquet")
                     for sp in ("train", "val", "test")], ignore_index=True)
    pos = pos[pos["y_cls"] != 0].copy()
    pos["a"] = "drug:" + pos["drug_a_id"].astype(str); pos["b"] = "drug:" + pos["drug_b_id"].astype(str)
    pos = pos[pos["a"].isin(drugs) & pos["b"].isin(drugs) & pos["a"].isin(q) & pos["b"].isin(q)]

    def shared_cyp_ct(a, b):
        return len(adj["enzyme"].get(a, set()) & adj["enzyme"].get(b, set()) & cyp)
    qtc = [(r.a, r.b) for r in pos[pos["y_cls"].isin(qtc_idx)].itertuples() if shared_cyp_ct(r.a, r.b) >= 1]
    nonq = [(r.a, r.b) for r in pos[~pos["y_cls"].isin(qtc_idx)].itertuples() if shared_cyp_ct(r.a, r.b) >= 1]
    RNG.shuffle(nonq); nonq = nonq[:len(qtc) * 2]
    print(f"QTc-DDI(share CYP): {len(qtc)} | CYP-matched non-QTc: {len(nonq)}")

    def meso_gobp(a, b):
        sa = set().union(*[p2["gobp"].get(p, set()) for p in adj["protein"].get(a, set())]) if adj["protein"].get(a) else set()
        sb = set().union(*[p2["gobp"].get(p, set()) for p in adj["protein"].get(b, set())]) if adj["protein"].get(b) else set()
        return len(sa & sb)

    def feats(pairs):
        rows = []
        for a, b in pairs:
            qa, qb = q[a], q[b]
            rows.append({"liab_max": max(qa, qb), "liab_min": min(qa, qb), "liab_prod": qa * qb,
                         "kg_meso_gobp": meso_gobp(a, b), "shared_cyp_ct": shared_cyp_ct(a, b)})
        return pd.DataFrame(rows)
    Fp, Fn = feats(qtc), feats(nonq)
    y = np.r_[np.ones(len(Fp)), np.zeros(len(Fn))]
    print("\n== kill-test AUROC: QTc-DDI vs CYP-matched non-QTc (both share CYP) ==")
    print("   KG ceiling ~0.60 (best KG feature). Target: liab_max clears it.\n")
    for c in ["liab_max", "liab_min", "liab_prod", "kg_meso_gobp", "shared_cyp_ct"]:
        s = np.r_[Fp[c].values, Fn[c].values].astype(float)
        a = roc_auc_score(y, s)
        tag = " <- KG confound (want ~0.5)" if c == "shared_cyp_ct" else (" <- KG ceiling" if c == "kg_meso_gobp" else " *** LIABILITY")
        print(f"  {c:<16} AUROC {a:.3f}{tag}")
    # combined KG+liability (logreg)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = pd.concat([Fp, Fn], ignore_index=True).values
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=500).fit(Xs, y)
    print(f"\n  KG+liability combined (logreg 5-feat) AUROC {roc_auc_score(y, lr.predict_proba(Xs)[:,1]):.3f}")
    amax = roc_auc_score(y, np.r_[Fp['liab_max'].values, Fn['liab_max'].values])
    print(f"\n  VERDICT: liab_max {amax:.3f} vs KG ceiling ~0.60 -> "
          + ("PD-liability CLEARS ceiling, story holds" if amax >= 0.66 else
             ("marginal (0.62-0.66), needs stronger liability/router" if amax >= 0.62 else
              "does NOT clear ceiling -> revise PD-liability story")))


if __name__ == "__main__":
    main()
