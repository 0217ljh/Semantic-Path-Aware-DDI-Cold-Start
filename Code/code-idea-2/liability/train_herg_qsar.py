"""idea2 v2 — L3 Step1: hERG-blockade QSAR liability predictor (label-blind).

Trains a structure-only QSAR on the PUBLIC TDC hERG assay (never sees DDI pairs/labels),
then scores every ddi_unified drug -> per-drug qt_liability_score. Reports benchmark<->hERG
train overlap (canonical SMILES) for the leakage audit codex required. CPU (RDKit + HistGBT).

Codex-locked (thread 019f1d52): (a) own QSAR on public hERG > pretrained > LLM; Morgan+RDKit
-> HistGBT/logreg, scaffold split; freeze ALL choices on external hERG only (no downstream
QTc selection = leakage); report drug overlap; label-blind = "DDI-label-blind external
pharmacology augmentation".

Usage: PYTHONPATH=Code/code-idea-2 python -u Code/code-idea-2/liability/train_herg_qsar.py
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

RDLogger.DisableLog("rdApp.*")
OUT = "Code/code-idea-2/liability/qt_liability_scores.parquet"
DESC = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumHAcceptors,
        Descriptors.NumHDonors, Descriptors.NumRotatableBonds, Descriptors.NumAromaticRings,
        Descriptors.FractionCSP3]


def featurize(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
    arr = np.zeros(1024, dtype=np.int8); DataStructs.ConvertToNumpyArray(fp, arr)
    return np.concatenate([arr.astype(np.float32), np.array([f(m) for f in DESC], dtype=np.float32)])


def canon(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None


def main():
    from tdc.single_pred import Tox
    data = Tox(name="hERG")
    sp = data.get_split(method="scaffold")
    tr, va, te = sp["train"], sp["valid"], sp["test"]
    print(f"TDC hERG: train {len(tr)} val {len(va)} test {len(te)} | pos-rate tr {tr['Y'].mean():.2f}")

    def build(df):
        X, y, sm = [], [], []
        for s, yy in zip(df["Drug"], df["Y"]):
            f = featurize(s)
            if f is not None:
                X.append(f); y.append(int(yy)); sm.append(s)
        return np.array(X), np.array(y), sm
    Xtr, ytr, smtr = build(pd.concat([tr, va]))
    Xte, yte, _ = build(te)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, l2_regularization=1.0)
    clf.fit(Xtr, ytr)
    pte = clf.predict_proba(Xte)[:, 1]
    print(f"hERG QSAR test: AUROC {roc_auc_score(yte, pte):.4f} AUPRC {average_precision_score(yte, pte):.4f}")
    herg_train_canon = set(c for c in (canon(s) for s in smtr) if c)

    # ---- score all ddi_unified drugs ----
    dfs = []
    for f in glob.glob("Code/data/ddi_unified/*/*/inductive/S1/drugs.parquet"):
        d = pd.read_parquet(f)
        if "drugbank_id" in d.columns and "smiles" in d.columns:
            dfs.append(d[["drugbank_id", "smiles"]])
    drugs = pd.concat(dfs, ignore_index=True).dropna(subset=["smiles"]).drop_duplicates("drugbank_id")
    rows, overlap = [], 0
    for db, s in zip(drugs["drugbank_id"], drugs["smiles"]):
        f = featurize(s)
        if f is None:
            continue
        score = float(clf.predict_proba(f.reshape(1, -1))[0, 1])
        c = canon(s)
        in_herg = c in herg_train_canon
        overlap += int(in_herg)
        rows.append({"drugbank_id": db, "qt_liability_score": round(score, 5), "in_herg_train": in_herg})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"scored ddi drugs: {len(out)} | overlap with hERG-train (canonical SMILES): {overlap} "
          f"({overlap/len(out)*100:.1f}%)")
    print(f"qt_liability_score: mean {out['qt_liability_score'].mean():.3f} "
          f"| >0.5: {int((out['qt_liability_score']>0.5).sum())} ({(out['qt_liability_score']>0.5).mean()*100:.0f}%)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
