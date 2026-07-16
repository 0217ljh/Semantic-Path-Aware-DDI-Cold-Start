"""idea2 v2 — L3 easier control: does qt_liability carry ANY pair-level signal?

codex 019f1d73 verdict: kill-test 0.515 is NOT a calibration artifact (AUROC is
rank-only; base-rate rescaling can't move it). Likely asymmetry (symmetric max throws
away victim+perpetrator structure; min/prod fell <0.5) + hostile type label. Before
retraining anything, run the CHEAP disambiguator on the EXISTING scores:

  QTc-DDI pairs (pos)  vs  degree-matched NON-INTERACTING random pairs (neg).

Negatives = permuted re-pairing of the SAME drugs that appear in the QTc positives
(holds per-drug degree/promiscuity marginal fixed), rejecting any known DDI edge.
Primary feature max(qA,qB). Interpretation (codex):
  liab_max >~0.60  -> signal alive, kill-test was misframed -> go DIRECTIONAL next.
  liab_max ~0.50   -> PD-liability line is DEAD -> pivot idea2 to PK-only strong story.
CPU.

Usage: PYTHONPATH=Code/code-idea-2 python -u Code/code-idea-2/liability/l3_qtc_easier_control.py
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
    qs = pd.read_parquet(SCORES)
    q = dict(zip("drug:" + qs["drugbank_id"].astype(str), qs["qt_liability_score"]))

    tax = pd.read_parquet(TAX)
    qtc_idx = set(tax[tax["sub"] == "QTc/torsade"]["idx"])
    allpos = pd.concat([pd.read_parquet(f"Code/data/ddi_unified/multi_cls/drugbank_ryu/inductive/S1/fold0/{sp}.parquet")
                        for sp in ("train", "val", "test")], ignore_index=True)
    allpos = allpos[allpos["y_cls"] != 0].copy()
    allpos["a"] = "drug:" + allpos["drug_a_id"].astype(str); allpos["b"] = "drug:" + allpos["drug_b_id"].astype(str)
    ok = lambda df: df[df["a"].isin(drugs) & df["b"].isin(drugs) & df["a"].isin(q) & df["b"].isin(q)]
    allpos = ok(allpos)
    # known interacting edge set (any type), undirected
    known = set(map(frozenset, zip(allpos["a"], allpos["b"])))

    qtc = [(r.a, r.b) for r in allpos[allpos["y_cls"].isin(qtc_idx)].itertuples()]
    print(f"QTc-DDI positives (in KG & scored): {len(qtc)}")

    # degree-matched negatives: permuted re-pairing of the SAME drug marginal as positives
    pool = [d for pair in qtc for d in pair]  # multiset preserves per-drug degree
    neg = []
    tries = 0
    target = len(qtc) * 2
    while len(neg) < target and tries < target * 50:
        a, b = RNG.choice(pool), RNG.choice(pool)
        tries += 1
        if a == b or frozenset((a, b)) in known:
            continue
        neg.append((a, b))
    print(f"non-interacting degree-matched negatives: {len(neg)} (from {len(set(pool))} distinct QTc drugs)")

    def col(pairs, fn):
        return np.array([fn(q[a], q[b]) for a, b in pairs], dtype=float)
    y = np.r_[np.ones(len(qtc)), np.zeros(len(neg))]
    feats = {"liab_max": max, "liab_min": min, "liab_prod": lambda x, y_: x * y_,
             "liab_mean": lambda x, y_: (x + y_) / 2}
    print("\n== easier control AUROC: QTc-DDI vs degree-matched NON-interacting ==")
    print("   (no CYP-confound control; just: does liability enrich QTc pairs AT ALL?)\n")
    for name, fn in feats.items():
        s = np.r_[col(qtc, fn), col(neg, fn)]
        tag = " *** PRIMARY" if name == "liab_max" else ""
        print(f"  {name:<11} AUROC {roc_auc_score(y, s):.3f}{tag}")
    amax = roc_auc_score(y, np.r_[col(qtc, max), col(neg, max)])
    print(f"\n  VERDICT: liab_max {amax:.3f} -> "
          + ("SIGNAL ALIVE (>0.60), kill-test misframed -> go DIRECTIONAL (victim+perpetrator)" if amax >= 0.60 else
             ("weak (0.55-0.60), liability marginal" if amax >= 0.55 else
              "DEAD (~0.5), even easier control fails -> PIVOT idea2 to PK-only strong story")))


if __name__ == "__main__":
    main()
