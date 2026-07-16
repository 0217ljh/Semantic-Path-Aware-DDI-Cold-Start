"""Do NON-SEQUENTIAL corridor "chords" exist in A->B shared-mediator corridors, do they
carry DDI signal beyond corridor density, and are they under-used by NBFNet? (read-only, no GPU)

codex-designed spec (thread 019f20f1). strict_chord = a support-INTERNAL undirected mediator
edge {m_i,m_j} whose position tuples p=(d_a,d_b) are NOT monotone-comparable (excludes edges
consistent with monotone shortest-path-DAG arm expansion; includes same-shell (2,2)-(2,2) and
cross-arm (1,2)-(2,1)). Three layers:
  1. EXISTENCE  (coverage / distribution / composition, |S|>=2)
  2. DISCRIMINATIVITY  (stratified AUROC on y, strata = support-bin x mean_log_deg quartile;
                        bootstrap CI + within-stratum label permutation)
  3. NBFNet UNDER-USE  (among positives, FN vs TP stratified AUROC, matched on corridor burden
                        + NBFNet confidence decile)

Uses the winning AND support cache (med=KG node idx, verified aligned to MergedKG) + a rebuilt
undirected KG adjacency + NBFNet v1_71 3-seed S2 predictions. Pools seeds 42/43/44.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && CUDA_VISIBLE_DEVICES= \
    python Code/scripts/analyze_spmn_v2_corridor_chords.py --max-pairs 4000"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("project root (with Code/data/KG) not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from run_spmn_v2_aware import _load_frames, _load_support_cache  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    DEFAULT_EDGES_PATH, KIND_ORDER, MergedKG,
)

SEEDS = (42, 43, 44)
L_MAX, K_PER_TYPE, N_MAX = 3, 64, 400
RES = ROOT / "Notes" / "Log" / "analysis"
NBF_GLOB = "Code/runs/*nbfnet_v1_71_3seed*seed{s}*/test_s2_scores.npz"


def _canon(a, b):
    return (a, b) if a <= b else (b, a)


def build_undirected_adj(kg: MergedKG):
    ed = pd.read_parquet(ROOT / DEFAULT_EDGES_PATH, columns=["src", "dst"])
    src = ed["src"].astype(str).map(kg.id_to_idx).to_numpy()
    dst = ed["dst"].astype(str).map(kg.id_to_idx).to_numpy()
    ok = ~(pd.isna(src) | pd.isna(dst))
    src = src[ok].astype(np.int64); dst = dst[ok].astype(np.int64)
    u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    keep = u != v
    u, v = u[keep], v[keep]
    order = np.argsort(u, kind="stable")
    u_s = u[order]; nbr = v[order].astype(np.int64)
    indptr = np.searchsorted(u_s, np.arange(kg.n_nodes + 1)).astype(np.int64)
    return indptr, nbr


def _dominates(pi, pj):
    return (pi[0] <= pj[0] and pi[1] <= pj[1]) and (pi[0] < pj[0] or pi[1] < pj[1])


def strat_auroc(feat, y, strata):
    """AUROC within each stratum, micro-aggregated by n_pos*n_neg."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        ys = y[m]
        if ys.sum() < 1 or (1 - ys).sum() < 1:
            continue
        fs = feat[m]
        if len(np.unique(fs)) < 2:
            a = 0.5
        else:
            a = roc_auc_score(ys, fs)
        w = ys.sum() * (1 - ys).sum()
        num += w * a; den += w
    return num / den if den > 0 else float("nan")


def bootstrap_ci(feat, y, strata, n=300, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y); vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(strat_auroc(feat[idx], y[idx], strata[idx]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p(feat, y, strata, obs, n=300, seed=0):
    rng = np.random.default_rng(seed)
    ge = 0
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
    ap.add_argument("--max-pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    print("[chords] loading KG + adjacency ...", flush=True)
    kg = MergedKG.from_parquet()
    indptr, nbr = build_undirected_adj(kg)
    deg = kg.degree.astype(np.float64)
    print(f"[chords] adjacency nnz={len(nbr):,}", flush=True)

    # gather pooled pairs
    rows = []  # dict per pair
    for s in SEEDS:
        fr = _load_frames(s)["test_s2"].reset_index(drop=True)
        te = _load_support_cache("and", s, L_MAX, K_PER_TYPE, N_MAX, False)["test_s2"]
        off = te["offsets"]; med = te["med"].astype(np.int64)
        da = te["da"].astype(np.int64); db = te["db"].astype(np.int64)
        typ = te["typ"].astype(np.int64); y = te["y"].astype(int)
        nbf = {}
        g = glob.glob(str(ROOT / NBF_GLOB.format(s=s)))
        if g:
            z = np.load(g[0], allow_pickle=True)
            nbf = {_canon(str(a), str(b)): float(sc)
                   for a, b, sc in zip(z["pair_a"], z["pair_b"], z["y_score"])}
        for i in range(len(fr)):
            rows.append(dict(seed=s, i=i,
                             a=str(fr["drug_a_id"].iloc[i]), b=str(fr["drug_b_id"].iloc[i]),
                             y=int(y[i]), s0=int(off[i]), e0=int(off[i + 1]),
                             med=med, da=da, db=db, typ=typ, nbf=nbf))
    # balanced sample
    rng = np.random.default_rng(args.seed)
    pos = [r for r in rows if r["y"] == 1]; neg = [r for r in rows if r["y"] == 0]
    k = min(args.max_pairs // 2, len(pos), len(neg))
    sel = [pos[j] for j in rng.choice(len(pos), k, replace=False)] + \
          [neg[j] for j in rng.choice(len(neg), k, replace=False)]
    print(f"[chords] pooled pairs={len(rows)}; sampled balanced={len(sel)} (k={k}/class)", flush=True)

    mask = np.zeros(kg.n_nodes, dtype=bool)
    feats = []
    rel_ctr: Counter = Counter(); tp_ctr: Counter = Counter()
    for n_done, r in enumerate(sel):
        s0, e0 = r["s0"], r["e0"]
        S = r["med"][s0:e0]
        nS = len(S)
        rec = dict(y=r["y"], n_support=nS, nbf=r["nbf"].get(_canon(r["a"], r["b"]), np.nan))
        if nS >= 2:
            pos_t = {int(m): (int(r["da"][s0 + j]), int(r["db"][s0 + j])) for j, m in enumerate(S)}
            typ_l = r["typ"][s0:e0]
            logd = np.log1p(deg[S])
            rec["mean_log_deg"] = float(logd.mean()); rec["max_log_deg"] = float(logd.max())
            rec["n_types"] = int(len(np.unique(typ_l)))
            mask[:] = False; mask[S] = True
            n_any = n_strict = 0; w_strict = 0.0; cross = same = 0
            strict_deg: Counter = Counter()
            for u in S.tolist():
                a0, a1 = indptr[u], indptr[u + 1]
                nb = nbr[a0:a1]
                sel_nb = nb[mask[nb] & (u < nb)]
                for w_ in sel_nb.tolist():
                    n_any += 1
                    pi, pj = pos_t[u], pos_t[w_]
                    if _dominates(pi, pj) or _dominates(pj, pi):
                        continue  # arm-monotone
                    n_strict += 1
                    w_strict += 1.0 / math.log1p(deg[u] + deg[w_])
                    strict_deg[u] += 1; strict_deg[w_] += 1
                    if pi == pj:
                        same += 1
                    elif np.sign(pi[0] - pj[0]) == -np.sign(pi[1] - pj[1]) and pi[0] != pj[0]:
                        cross += 1
                    ru, rw = int(kg.type_id[u]), int(kg.type_id[w_])
                    tp_ctr[tuple(sorted((KIND_ORDER[ru], KIND_ORDER[rw])))] += 1
            c2 = nS * (nS - 1) / 2
            rec.update(n_internal_any=n_any, n_strict=n_strict,
                       strict_any=int(n_strict > 0),
                       strict_density=n_strict / c2,
                       strict_wdensity=w_strict / c2,
                       max_strict_deg=(max(strict_deg.values()) if strict_deg else 0),
                       cross_frac=(cross / n_strict if n_strict else 0.0),
                       same_frac=(same / n_strict if n_strict else 0.0))
        feats.append(rec)
        if (n_done + 1) % 500 == 0:
            print(f"  ... {n_done+1}/{len(sel)}", flush=True)

    df = pd.DataFrame(feats)
    dfc = df[df["n_support"] >= 2].copy()
    print(f"\n[chords] pairs with |S|>=2: {len(dfc)}/{len(df)}")

    # ---------- 1. EXISTENCE ----------
    def supp_bin(n):
        return "2-4" if n <= 4 else "5-9" if n <= 9 else "10-19" if n <= 19 else "20-49" if n <= 49 else "50+"
    dfc["sbin"] = dfc["n_support"].map(supp_bin)
    existence = {
        "n_pairs_ge2": int(len(dfc)),
        "strict_any_frac_overall": float(dfc["strict_any"].mean()),
        "strict_any_frac_by_supp": {b: float(dfc.loc[dfc.sbin == b, "strict_any"].mean())
                                    for b in ["2-4", "5-9", "10-19", "20-49", "50+"] if (dfc.sbin == b).any()},
        "median_n_strict_when_pos": float(dfc.loc[dfc.strict_any == 1, "n_strict"].median())
        if (dfc.strict_any == 1).any() else 0.0,
        "n_strict_mean": float(dfc["n_strict"].mean()),
        "cross_frac_mean": float(dfc.loc[dfc.strict_any == 1, "cross_frac"].mean()) if (dfc.strict_any == 1).any() else 0.0,
        "same_frac_mean": float(dfc.loc[dfc.strict_any == 1, "same_frac"].mean()) if (dfc.strict_any == 1).any() else 0.0,
    }
    print("\n=== 1. EXISTENCE ===")
    print(json.dumps(existence, indent=2, ensure_ascii=False))
    print("top strict-chord type-pairs:", tp_ctr.most_common(10))

    # ---------- 2. DISCRIMINATIVITY ----------
    y = dfc["y"].to_numpy()
    q = pd.qcut(dfc["mean_log_deg"], 4, labels=False, duplicates="drop").to_numpy()
    strata = np.array([f"{b}|{qq}" for b, qq in zip(dfc["sbin"], q)])
    print("\n=== 2. DISCRIMINATIVITY (stratified AUROC: support-bin x mean_log_deg quartile) ===")
    disc = {}
    for feat in ["strict_any", "strict_density", "strict_wdensity"]:
        f = dfc[feat].to_numpy().astype(float)
        auc = strat_auroc(f, y, strata)
        lo, hi = bootstrap_ci(f, y, strata, seed=args.seed)
        p = perm_p(f, y, strata, auc, seed=args.seed)
        disc[feat] = dict(auroc=round(auc, 4), ci=[round(lo, 4), round(hi, 4)], perm_p=round(p, 4))
        print(f"  {feat:16s} AUROC={auc:.4f}  95%CI[{lo:.4f},{hi:.4f}]  perm_p={p:.4f}")
    # unmatched control (raw count) for contrast
    for feat in ["n_strict", "n_support"]:
        f = dfc[feat].to_numpy().astype(float)
        print(f"  [ctrl] {feat:12s} pooled AUROC={roc_auc_score(y,f):.4f} (unstratified)")

    # ---------- 3. NBFNet UNDER-USE ----------
    print("\n=== 3. NBFNet UNDER-USE (among positives: FN vs TP) ===")
    underuse = {}
    dpos = dfc[(dfc["y"] == 1) & dfc["nbf"].notna()].copy()
    if len(dpos) > 50:
        dpos["FN"] = (dpos["nbf"] < 0.5).astype(int)
        dec = pd.qcut(dpos["nbf"], 10, labels=False, duplicates="drop")
        st = np.array([f"{b}|{d}" for b, d in zip(dpos["sbin"], dec)])
        yfn = dpos["FN"].to_numpy()
        print(f"  positives with NBFNet score: {len(dpos)}  (FN={int(yfn.sum())} TP={int((1-yfn).sum())})")
        for feat in ["strict_any", "strict_density", "strict_wdensity"]:
            f = dpos[feat].to_numpy().astype(float)
            auc = strat_auroc(f, yfn, st)
            lo, hi = bootstrap_ci(f, yfn, st, seed=args.seed)
            underuse[feat] = dict(auroc=round(auc, 4), ci=[round(lo, 4), round(hi, 4)])
            print(f"  {feat:16s} FN-vs-TP AUROC={auc:.4f}  95%CI[{lo:.4f},{hi:.4f}]")
    else:
        print("  too few positives with NBFNet scores")

    # ---------- VERDICT ----------
    go_exist = (existence["strict_any_frac_overall"] >= 0.25 and
                existence["strict_any_frac_by_supp"].get("10-19", 0) >= 0.40 and
                existence["median_n_strict_when_pos"] >= 2)
    go_disc = any(d["auroc"] >= 0.56 and d["ci"][0] > 0.52 and d["perm_p"] < 0.01 for d in disc.values())
    go_under = any(u["auroc"] >= 0.58 and u["ci"][0] > 0.53 for u in underuse.values()) if underuse else False
    print("\n=== VERDICT ===")
    print(f"  existence GO: {go_exist}")
    print(f"  discriminativity GO: {go_disc}")
    print(f"  under-use GO: {go_under}  (needs NBFNet scores)")
    print(f"  OVERALL: {'GO (build chord/semantic-gate component)' if (go_exist and go_disc and go_under) else 'PARTIAL / NO-GO — see layers'}")

    out = dict(existence=existence, discriminativity=disc, underuse=underuse,
               go=dict(existence=go_exist, discriminativity=go_disc, underuse=go_under),
               config=dict(max_pairs=args.max_pairs, seeds=list(SEEDS), n_ge2=int(len(dfc))),
               top_typepairs=[[list(k), v] for k, v in tp_ctr.most_common(15)])
    (RES / "02__corridor_chords.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {RES / '02__corridor_chords.json'}")


if __name__ == "__main__":
    main()
