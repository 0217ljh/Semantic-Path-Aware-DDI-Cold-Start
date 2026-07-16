"""Unified count-readout probe (auto-research redesign, codex-designed).

Tests whether the messy {attention-pool core + bolt-on count branch + two-stage bf}
can be REPLACED by ONE coherent design: a linear head over an exact, UNCAPPED,
relation-typed, specificity-weighted shared-mediator COUNT profile. If a plain
logistic head on these features beats naked AND by ~the missing +0.4pt with LOWER
variance, this probe IS the unified standalone adapter (no pool, no entity embed,
no W_T, no additive branch, no freeze-fit hack).

Feature vector per pair:
  * TYPE block (uncapped, reused from the cached struct_feats — s_tau is the EXACT
    pre-cap type_count): [s_tau(12) | log1p(s_tau)(12) | aa(12) | n_support | n_active]
  * RELATION-PAIR block (NEW, uncapped, computed here from the KG 1-hop relation
    buckets): for each symmetric rel-pair rho=sym(rel_a,rel_b) over the 11 buckets,
    count and Adamic-Adar weight of shared mediators BOTH drugs reach 1-hop via that
    relation (enzyme-enzyme=PK, target-target=PD live here). This is the relation-
    resolved signal the per-type counts lack and the pool normalized away.

Ablation ladder (codex): TypeCount -> +RelPair -> (+specificity already via aa).
Read-only; reuses the AND cache (type block + labels) + KG (relation buckets/degree).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
from my_code.models.spmn_v1.retrieval import MergedKG, N_REL_BUCKETS, N_TYPES  # noqa: E402

CACHE = ROOT / "Code/data/_cache"
THREE_SEED = ROOT / "Code/data/coldddi_legacy/800drug_3seed"
NAKED = {42: 0.7845, 43: 0.7471, 44: 0.7657}


def _type_block(struct):
    """Uncapped type block from cached struct: s_tau|log|aa|n_support|n_active."""
    nt = N_TYPES; dim = struct.shape[1]
    cols = list(range(0, 3 * nt)) + [dim - 2, dim - 1]
    return struct[:, cols]


def _rel_pair_block(kg, a_ids, b_ids):
    """(n, 2*C) uncapped 1-hop relation-pair count + AA, C=sym rel-pair cells."""
    nrel = N_REL_BUCKETS
    cells = [(i, j) for i in range(nrel) for j in range(i, nrel)]
    cidx = {(i, j): k for k, (i, j) in enumerate(cells)}
    C = len(cells)
    n = len(a_ids)
    cnt = np.zeros((n, C)); aa = np.zeros((n, C))
    deg = kg.degree
    for i in range(n):
        ai = kg.id_to_idx.get(str(a_ids[i])); bi = kg.id_to_idx.get(str(b_ids[i]))
        if ai is None or bi is None:
            continue
        ra = kg.drug_rel.get(ai, {}); rb = kg.drug_rel.get(bi, {})
        if not ra or not rb:
            continue
        common = ra.keys() & rb.keys()
        for m in common:
            if kg.is_drug[m]:
                continue
            lo, hi = sorted((ra[m], rb[m]))
            k = cidx[(lo, hi)]
            cnt[i, k] += 1.0
            aa[i, k] += 1.0 / np.log(max(deg[m], 2))
    return np.concatenate([cnt, aa], axis=1), cells


def _probe(Xtr, ytr, Xte, yte, c=1.0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=c)
    clf.fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    print("[count-probe] loading merged KG ...", flush=True)
    kg = MergedKG.from_parquet()

    res = {"TypeCount": [], "+RelPair": []}
    for seed in args.seeds:
        z = np.load(CACHE / f"spmn_v2_supports_and_seed{seed}_lmax3_kpt64_nmax400_cp0.npz")
        import pandas as pd
        fr_tr = pd.read_parquet(THREE_SEED / f"seed{seed}/train.parquet")
        fr_te = pd.read_parquet(THREE_SEED / f"seed{seed}/test_s2.parquet")
        tr_type = _type_block(z["train__struct"]); te_type = _type_block(z["test_s2__struct"])
        ytr = z["train__y"].astype(int); yte = z["test_s2__y"].astype(int)
        assert (fr_tr["label"].to_numpy().astype(int) == ytr).all(), "train order mismatch"
        assert (fr_te["label"].to_numpy().astype(int) == yte).all(), "test order mismatch"
        tr_rp, cells = _rel_pair_block(kg, fr_tr["drug_a_id"].to_numpy(),
                                       fr_tr["drug_b_id"].to_numpy())
        te_rp, _ = _rel_pair_block(kg, fr_te["drug_a_id"].to_numpy(),
                                   fr_te["drug_b_id"].to_numpy())
        a1 = _probe(tr_type, ytr, te_type, yte)
        a2 = _probe(np.concatenate([tr_type, tr_rp], 1), ytr,
                    np.concatenate([te_type, te_rp], 1), yte)
        res["TypeCount"].append(a1); res["+RelPair"].append(a2)
        print(f"[count-probe] seed{seed}: TypeCount={a1:.4f}  +RelPair={a2:.4f}  "
              f"(naked AND {NAKED[seed]:.4f})", flush=True)

    print(f"\n{'variant':<12} {'per-seed test AUC':>26} {'mean':>8} {'vs naked':>9}")
    nk = np.mean([NAKED[s] for s in args.seeds])
    for name, aucs in res.items():
        m = np.mean(aucs)
        print(f"{name:<12} {str([round(x,4) for x in aucs]):>26} {m:>8.4f} {m-nk:>+9.4f}")
    print(f"{'naked AND':<12} {str([NAKED[s] for s in args.seeds]):>26} {nk:>8.4f}")
    print("\nread: if +RelPair (linear, no pool/branch/bf) beats naked AND by ~+0.005-0.01 "
          "with low variance, this count-readout IS the unified adapter.")


if __name__ == "__main__":
    main()
