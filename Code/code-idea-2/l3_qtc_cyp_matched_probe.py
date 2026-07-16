"""idea2 v2 — L3 necessity probe: QTc-DDI vs CYP-matched non-QTc DDI.

The QTc confound: QTc drugs share CYP3A4 (47%) but that is NOT the QTc cause (hERG is,
4.5% KG coverage). Kill-test necessity check: take QTc-DDI pairs (positive) vs
CYP-MATCHED negatives = non-QTc DDI pairs that ALSO share >=1 CYP enzyme (confound
matched, both interact, only the mechanism differs). If NO current KG feature (PK typed
+ meso + macro) separates them (AUROC ~0.5), the KG lacks the QTc-specific causal signal
=> the hERG/QT liability layer is necessary. If something separates them, that's a lead.
CPU-only.

Usage: PYTHONPATH=Code/code-idea-2 python -u Code/code-idea-2/l3_qtc_cyp_matched_probe.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import kg_v15

TAX = "/mnt/c/Users/27911/AppData/Local/Temp/claude/D--My-Research/217a6adf-9de6-4f18-a70d-31133f2a82a3/scratchpad/ddi_type_taxonomy.parquet"
RNG = np.random.RandomState(42)


def main():
    nd, ed = kg_v15.load_v15()
    adj, drugs = kg_v15.build_typed_drug_adj(nd, ed)
    p2 = kg_v15.build_protein_meso(ed)
    nm = dict(zip(nd["id"], nd["name"]))
    # CYP enzyme protein set (name starts with CYP)
    cyp = {i for i, n in nm.items() if str(i).startswith("prot:") and str(n).upper().startswith("CYP")}
    print(f"CYP enzyme nodes: {len(cyp)}")

    tax = pd.read_parquet(TAX)
    qtc_idx = set(tax[tax["sub"] == "QTc/torsade"]["idx"])
    # pooled typed DDI pairs (multi_cls, all interacting)
    pos = pd.concat([pd.read_parquet(f"Code/data/ddi_unified/multi_cls/drugbank_ryu/inductive/S1/fold0/{sp}.parquet")
                     for sp in ("train", "val", "test")], ignore_index=True)
    pos = pos[pos["y_cls"] != 0].copy()
    pos["a"] = "drug:" + pos["drug_a_id"].astype(str); pos["b"] = "drug:" + pos["drug_b_id"].astype(str)
    pos = pos[pos["a"].isin(drugs) & pos["b"].isin(drugs)]

    def shares_cyp(a, b):
        return bool((adj["enzyme"].get(a, set()) & cyp) & (adj["enzyme"].get(b, set()) & cyp)) or \
               bool(adj["enzyme"].get(a, set()) & adj["enzyme"].get(b, set()) & cyp)

    qtc = [(r.a, r.b) for r in pos[pos["y_cls"].isin(qtc_idx)].itertuples() if shares_cyp(r.a, r.b)]
    # CYP-matched negatives: non-QTc DDI pairs that also share a CYP
    nonqtc = pos[~pos["y_cls"].isin(qtc_idx)]
    cypneg = [(r.a, r.b) for r in nonqtc.itertuples() if shares_cyp(r.a, r.b)]
    RNG.shuffle(cypneg); cypneg = cypneg[:len(qtc) * 2]
    print(f"QTc-DDI pairs sharing CYP: {len(qtc)} | CYP-matched non-QTc negs: {len(cypneg)}")

    def feats(a, b):
        f = kg_v15.pk_pair_feats(a, b, adj)
        for key in ("pathway", "gobp", "anatomy", "disease", "phenotype"):
            sa = set().union(*[p2[key].get(p, set()) for p in adj["protein"].get(a, set())]) if adj["protein"].get(a) else set()
            sb = set().union(*[p2[key].get(p, set()) for p in adj["protein"].get(b, set())]) if adj["protein"].get(b) else set()
            f[f"meso_{key}"] = len(sa & sb)
        return f

    Xp = pd.DataFrame([feats(a, b) for a, b in qtc])
    Xn = pd.DataFrame([feats(a, b) for a, b in cypneg])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    cols = [c for c in Xp.columns if not c.startswith("jacc")]
    print("\n== single-feature AUROC: QTc-DDI vs CYP-matched non-QTc (both share CYP) ==")
    print("   (~0.5 = KG cannot separate QTc from the CYP confound -> hERG/QT liability layer NECESSARY)\n")
    aucs = {}
    for c in cols:
        s = np.r_[Xp[c].values, Xn[c].values].astype(float)
        try:
            aucs[c] = roc_auc_score(y, s)
        except Exception:
            aucs[c] = np.nan
    for c, a in sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        print(f"  {c:<20} {a:.3f}")
    best = max(abs(v - 0.5) for v in aucs.values())
    print(f"\n  max |AUROC-0.5| over all current KG features = {best:.3f}")
    print("  VERDICT:", "KG has NO QTc-specific signal under CYP-match -> liability layer NECESSARY"
          if best < 0.07 else "some KG feature separates QTc under CYP-match -> investigate before liability")


if __name__ == "__main__":
    main()
