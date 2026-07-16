"""Are path cross-distance SKIP relations DDI-ASSOCIATED? Positive vs negative comparison
(read-only, no GPU). Complements the existence dump (analyze_spmn_v2_path_skip_relations.py).

Per pair, over up to --max-mid (2,2)-midpoints of the A-B-C-D-E chain (B in N(A)&N(C),
D in N(E)&N(C)), count chains that carry a non-adjacent KG skip edge (any, and B-D
specifically; generic co-annotation relations blacklisted). Then test whether positive DDI
pairs carry MORE such skip-chains than negatives, MATCHED on support size + degree
(stratified AUROC + within-stratum label permutation). This answers "are these cross-
distance connections related to DDI at all", not just "do they exist on positives".

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && CUDA_VISIBLE_DEVICES= \
    python Code/scripts/analyze_spmn_v2_path_skip_discrim.py --max-pairs 3000 --max-mid 8"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from run_spmn_v2_aware import _load_frames, _load_support_cache  # noqa: E402
from my_code.models.spmn_v1.retrieval import MergedKG  # noqa: E402
from analyze_spmn_v2_corridor_chords_crossarm import build_rel_adj  # noqa: E402

NONADJ = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]  # A-C,A-D,B-D,B-E,C-E (A-E excluded)
BD = (1, 3)


def strat_auroc(feat, y, strata):
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        ys = y[m]
        if ys.sum() < 1 or (1 - ys).sum() < 1:
            continue
        fs = feat[m]
        a = 0.5 if len(np.unique(fs)) < 2 else roc_auc_score(ys, fs)
        w = ys.sum() * (1 - ys).sum()
        num += w * a; den += w
    return num / den if den > 0 else float("nan")


def perm_p(feat, y, strata, obs, n=200, seed=0):
    rng = np.random.default_rng(seed); ge = 0
    for _ in range(n):
        yp = y.copy()
        for s in np.unique(strata):
            m = strata == s
            yp[m] = rng.permutation(yp[m])
        if strat_auroc(feat, yp, strata) >= obs:
            ge += 1
    return (ge + 1) / (n + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--max-mid", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kg = MergedKG.from_parquet()
    indptr, nbr, rel_s, rel_vocab, black_mask = build_rel_adj(kg)
    deg = kg.degree.astype(np.float64)

    def nbrs(x):
        return nbr[indptr[x]:indptr[x + 1]]

    def has_rel(a, b):
        s, e = indptr[a], indptr[a + 1]
        seg = nbr[s:e]; hit = np.where(seg == b)[0]
        return len(hit) > 0 and not black_mask[rel_s[s:e][hit[0]]]

    def low_deg_common(x, z):
        Nx = set(nbrs(x).tolist())
        cand = [w for w in nbrs(z).tolist() if w in Nx and w not in (x, z)]
        return min(cand, key=lambda w: deg[w]) if cand else None

    rows = []
    for s in (42, 43, 44):
        fr = _load_frames(s)["test_s2"].reset_index(drop=True)
        te = _load_support_cache("and", s, 3, 64, 400, False)["test_s2"]
        off = te["offsets"]; med = te["med"].astype(np.int64)
        da = te["da"].astype(np.int64); db = te["db"].astype(np.int64); y = te["y"].astype(int)
        for i in range(len(fr)):
            rows.append(dict(a=str(fr["drug_a_id"].iloc[i]), b=str(fr["drug_b_id"].iloc[i]),
                             y=int(y[i]), s0=int(off[i]), e0=int(off[i + 1]), med=med, da=da, db=db))
    rng = np.random.default_rng(args.seed)
    pos = [r for r in rows if r["y"] == 1]; neg = [r for r in rows if r["y"] == 0]
    k = min(args.max_pairs // 2, len(pos), len(neg))
    sel = [pos[j] for j in rng.choice(len(pos), k, replace=False)] + \
          [neg[j] for j in rng.choice(len(neg), k, replace=False)]
    print(f"[path-skip-discrim] sampled balanced={len(sel)} (k={k}/class)", flush=True)

    feats = []
    for n_done, r in enumerate(sel):
        A, E = r["a"], r["b"]
        rec = dict(y=r["y"], n_support=r["e0"] - r["s0"])
        if A in kg.id_to_idx and E in kg.id_to_idx and rec["n_support"] >= 2:
            ui, vi = kg.id_to_idx[A], kg.id_to_idx[E]
            s0, e0 = r["s0"], r["e0"]
            Sm = r["med"][s0:e0]
            rec["mean_log_deg"] = float(np.log1p(deg[Sm]).mean())
            mids = [int(Sm[j]) for j in range(e0 - s0) if r["da"][s0 + j] == 2 and r["db"][s0 + j] == 2]
            mids.sort(key=lambda c: deg[c])
            n_skip = n_bd = 0
            for C in mids[:args.max_mid]:
                B = low_deg_common(ui, C); D = low_deg_common(vi, C)
                if B is None or D is None or len({ui, B, C, D, vi}) < 5:
                    continue
                chain = [ui, B, C, D, vi]
                got = any(has_rel(chain[a_i], chain[b_i]) for a_i, b_i in NONADJ)
                if got:
                    n_skip += 1
                if has_rel(chain[BD[0]], chain[BD[1]]):
                    n_bd += 1
            rec.update(n_skip=n_skip, any_skip=int(n_skip > 0), n_bd=n_bd, any_bd=int(n_bd > 0))
        feats.append(rec)
        if (n_done + 1) % 500 == 0:
            print(f"  ... {n_done+1}/{len(sel)}", flush=True)

    df = pd.DataFrame(feats)
    dfc = df[df["n_support"] >= 2].dropna(subset=["mean_log_deg"]).copy()

    def sbin(n):
        return "2-4" if n <= 4 else "5-9" if n <= 9 else "10-19" if n <= 19 else "20-49" if n <= 49 else "50+"
    dfc["sbin"] = dfc["n_support"].map(sbin)
    y = dfc["y"].to_numpy()
    q = pd.qcut(dfc["mean_log_deg"], 4, labels=False, duplicates="drop").to_numpy()
    strata = np.array([f"{b}|{qq}" for b, qq in zip(dfc["sbin"], q)])

    print(f"\n[path-skip-discrim] pairs |S|>=2: {len(dfc)}")
    print(f"  any_skip rate: pos={dfc.loc[dfc.y==1,'any_skip'].mean():.3f}  neg={dfc.loc[dfc.y==0,'any_skip'].mean():.3f}")
    print(f"  any_bd   rate: pos={dfc.loc[dfc.y==1,'any_bd'].mean():.3f}  neg={dfc.loc[dfc.y==0,'any_bd'].mean():.3f}")
    print("\n=== DDI-ASSOCIATION (stratified AUROC: support-bin x mean_log_deg quartile) ===")
    for feat in ["any_skip", "n_skip", "any_bd", "n_bd"]:
        f = dfc[feat].to_numpy().astype(float)
        auc = strat_auroc(f, y, strata)
        p = perm_p(f, y, strata, auc, seed=args.seed)
        print(f"  {feat:10s} AUROC={auc:.4f}  perm_p={p:.4f}")
    print("  [ctrl] n_support unstratified AUROC={:.4f}".format(roc_auc_score(y, dfc['n_support'])))
    print("\n(if all skip-feature AUROC ~0.50 after matching => cross-distance skip relations are")
    print(" NOT DDI-associated beyond corridor size; existence on positives was not DDI-relevance.)")


if __name__ == "__main__":
    main()
