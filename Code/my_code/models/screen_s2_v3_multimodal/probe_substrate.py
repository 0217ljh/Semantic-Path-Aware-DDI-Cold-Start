"""I6 — substrate signal probe (codex 019e6734 go/no-go).

Cold-start honest: train a SMALL logistic head on TRAIN (seen-drug) pairs only, eval on
test_s2. Compares per-drug feature substrates (Morgan vs LLM-text vs both), reports overall +
PK/PD subgroup AUROC, with shuffle + random controls. Go/No-Go: LLM lifts PD AUROC >= +0.03
over Morgan-only AND shuffle/random collapse to ~chance.

Pair feature is symmetric: [h_a*h_b, |h_a-h_b|, h_a+h_b].

Usage: python -u .../probe_substrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
MORGAN = ROOT / "Code/data/_cache/molecular_mu_morgan.npz"
LLM = ROOT / "Code/data/_cache/llm_pharma/llm_text_pubmedbert.npz"
SPLIT_DIR = ROOT / "Code/data/KG/drugbank/splits/seed42"
PKPD = ROOT / "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"

RNG = np.random.default_rng(42)


def _canon(a, b):
    return (a, b) if a <= b else (b, a)


def _pair_to_type():
    m = {}
    for pq in sorted(SPLIT_DIR.glob("*.parquet")):
        d = pd.read_parquet(pq)
        if "ddi_type" not in d.columns:
            continue
        for a, b, t in zip(d["drug_a_id"].astype(str), d["drug_b_id"].astype(str), d["ddi_type"]):
            m[_canon(a, b)] = t
    return m


def _load_feats():
    feats = {}
    mo = np.load(MORGAN, allow_pickle=True)
    feats["morgan"] = ({str(d): i for i, d in enumerate(mo["drug_ids"])}, mo["m"].astype(np.float32))
    if LLM.exists():
        ll = np.load(LLM, allow_pickle=True)
        feats["llm"] = ({str(d): i for i, d in enumerate(ll["drug_ids"])}, ll["emb"].astype(np.float32))
    return feats


def _drug_vec(did, row_map, mat):
    i = row_map.get(str(did))
    return mat[i] if i is not None else None


def _pair_matrix(pairs, which, feats, shuffle=False, random_dim=None):
    """Build symmetric pair features; returns (X, keep_mask)."""
    rows = []
    keep = []
    # optional shuffle: permute per-drug feature assignment
    maps = {}
    for name in which:
        row_map, mat = feats[name]
        if shuffle:
            ids = list(row_map)
            perm = RNG.permutation(len(ids))
            row_map = {ids[i]: row_map[ids[perm[i]]] for i in range(len(ids))}
        maps[name] = (row_map, mat)
    for a, b in zip(pairs["drug_a_id"].astype(str), pairs["drug_b_id"].astype(str)):
        parts = []
        ok = True
        for name in which:
            row_map, mat = maps[name]
            ha = _drug_vec(a, row_map, mat); hb = _drug_vec(b, row_map, mat)
            if ha is None or hb is None:
                ok = False; break
            parts.append(np.concatenate([ha * hb, np.abs(ha - hb), ha + hb]))
        if ok:
            rows.append(np.concatenate(parts)); keep.append(True)
        else:
            keep.append(False)
    keep = np.array(keep, dtype=bool)
    X = np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float32)
    if random_dim is not None and len(X):
        X = RNG.standard_normal((X.shape[0], random_dim)).astype(np.float32)
    return X, keep


def _auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main():
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    feats = _load_feats()
    has_llm = "llm" in feats
    print(f"[probe] feats: morgan + {'llm' if has_llm else 'NO-llm'}", flush=True)

    tr_pos = ds.splits.train[["drug_a_id", "drug_b_id"]]
    tr_neg = ds.get_train_negatives(0, regenerate=True)[["drug_a_id", "drug_b_id"]]
    te_pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    te_neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]

    p2t = _pair_to_type()
    lab = pd.read_csv(PKPD).set_index("ddi_type")["pk_pd_label"].to_dict()
    te_cls = np.array([lab.get(p2t.get(_canon(str(a), str(b))), "UNK")
                       for a, b in zip(te_pos["drug_a_id"], te_pos["drug_b_id"])])

    configs = [("morgan", ["morgan"], False, None)]
    if has_llm:
        configs += [("llm", ["llm"], False, None),
                    ("both", ["morgan", "llm"], False, None),
                    ("llm_SHUFFLE", ["llm"], True, None),
                    ("RANDOM768", ["llm"], False, 768 * 3)]

    print(f"{'config':14s} {'overall':>8s} {'PK':>8s} {'PD':>8s}  n_tr  n_te", flush=True)
    results = {}
    for name, which, shuf, rdim in configs:
        Xtr_p, kp = _pair_matrix(tr_pos, which, feats)
        Xtr_n, kn = _pair_matrix(tr_neg, which, feats)
        if len(Xtr_p) == 0 or len(Xtr_n) == 0:
            print(f"{name:14s} SKIP (no features)"); continue
        Xtr = np.vstack([Xtr_p, Xtr_n])
        ytr = np.concatenate([np.ones(len(Xtr_p)), np.zeros(len(Xtr_n))])
        sc = StandardScaler(with_mean=True).fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(Xtr), ytr)

        Xte_p, kp_te = _pair_matrix(te_pos, which, feats, shuffle=shuf, random_dim=rdim)
        Xte_n, kn_te = _pair_matrix(te_neg, which, feats, shuffle=shuf, random_dim=rdim)
        sp = clf.predict_proba(sc.transform(Xte_p))[:, 1]
        sn = clf.predict_proba(sc.transform(Xte_n))[:, 1]
        y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
        s = np.concatenate([sp, sn])
        overall = _auc(y, s)
        cls_kept = te_cls[kp_te]
        def sub(c):
            m = cls_kept == c
            if m.sum() == 0:
                return float("nan")
            yy = np.concatenate([np.ones(int(m.sum())), np.zeros(len(sn))])
            ss = np.concatenate([sp[m], sn])
            return _auc(yy, ss)
        pk, pd_ = sub("PK"), sub("PD")
        results[name] = (overall, pk, pd_)
        print(f"{name:14s} {overall:8.4f} {pk:8.4f} {pd_:8.4f}  {len(Xtr)} {len(sp)}", flush=True)

    if has_llm and "morgan" in results and "llm" in results:
        dpd = results["llm"][2] - results["morgan"][2]
        dboth = results["both"][2] - results["morgan"][2]
        print(f"\n[probe] PD lift: llm-vs-morgan={dpd:+.4f}  both-vs-morgan={dboth:+.4f} "
              f"(go bar >= +0.03)", flush=True)
        print(f"[probe] shuffle PD={results['llm_SHUFFLE'][2]:.4f} random PD={results['RANDOM768'][2]:.4f} "
              f"(controls should be ~0.5)", flush=True)


if __name__ == "__main__":
    main()
