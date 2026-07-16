"""FINAL disambiguation for the corridor-chord hypothesis (codex thread 019f20fb).

The broad test (analyze_spmn_v2_corridor_chords.py) was NO-GO on discriminativity, BUT the
tested feature was 98.75% same-shell laterals flooded by the generic anatomy_protein_present
co-annotation relation -- so it did NOT directly test the rare CROSS-ARM chord the hypothesis
targeted. This script runs the one pre-registered refined test:

  * CROSS-ARM chords ONLY: support-internal mediator edges with STRICTLY INCOMPARABLE positions
    (d_i(a) < d_j(a) AND d_i(b) > d_j(b), or vice versa). Same-shell and monotone excluded.
  * RELATION BLACKLIST (frozen before results): drop generic co-annotation / membership /
    presence / ontology-hierarchy relations (non-mechanistic). Keep interaction/regulation/
    binding/causal/drug-action relations.
  * Discriminativity: stratified AUROC on y (support-bin x mean_log_deg quartile) + 300x
    bootstrap CI + within-stratum label permutation.

DECISION (codex): continue only if crossarm_any or crossarm_wdensity AUROC > 0.53 AND perm_p < 0.05.
Else declare full NO-GO and pivot.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && CUDA_VISIBLE_DEVICES= \
    python Code/scripts/analyze_spmn_v2_corridor_chords_crossarm.py --max-pairs 6000"
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
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from run_spmn_v2_aware import _load_frames, _load_support_cache  # noqa: E402
from my_code.models.spmn_v1.retrieval import DEFAULT_EDGES_PATH, MergedKG  # noqa: E402

SEEDS = (42, 43, 44)
L_MAX, K_PER_TYPE, N_MAX = 3, 64, 400
RES = ROOT / "Notes" / "Log" / "analysis"

# FROZEN blacklist: generic co-annotation / membership / presence / ontology-hierarchy
# relations (non-mechanistic). Everything else (interaction/regulation/binding/causal/
# drug-action) is KEPT.
BLACKLIST = {
    "prime:anatomy_protein_present", "prime:anatomy_protein_absent",
    "het:AeG", "het:AdG", "het:AuG",
    "het:GpBP", "het:GpMF", "het:GpCC", "het:GpPW",
    "prime:bioprocess_protein", "prime:molfunc_protein",
    "prime:cellcomp_protein", "prime:pathway_protein",
    "prime:bioprocess_bioprocess", "prime:disease_disease", "prime:phenotype_phenotype",
    "prime:anatomy_anatomy", "prime:molfunc_molfunc", "prime:cellcomp_cellcomp",
    "prime:disease_phenotype_positive", "prime:phenotype_protein",
}


def _canon(a, b):
    return (a, b) if a <= b else (b, a)


def build_rel_adj(kg: MergedKG):
    """Relation-aware symmetrised adjacency: per node, neighbor ids + relation string ids."""
    ed = pd.read_parquet(ROOT / DEFAULT_EDGES_PATH, columns=["src", "dst", "relation"])
    src = ed["src"].astype(str).map(kg.id_to_idx).to_numpy()
    dst = ed["dst"].astype(str).map(kg.id_to_idx).to_numpy()
    rel = ed["relation"].astype(str).to_numpy()
    ok = ~(pd.isna(src) | pd.isna(dst))
    src = src[ok].astype(np.int64); dst = dst[ok].astype(np.int64); rel = rel[ok]
    rel_vocab, rel_id = np.unique(rel, return_inverse=True)
    u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    r = np.concatenate([rel_id, rel_id]).astype(np.int32)
    keep = u != v
    u, v, r = u[keep], v[keep], r[keep]
    order = np.argsort(u, kind="stable")
    u_s = u[order]; nbr = v[order].astype(np.int64); rel_s = r[order]
    indptr = np.searchsorted(u_s, np.arange(kg.n_nodes + 1)).astype(np.int64)
    black_mask = np.array([rv in BLACKLIST for rv in rel_vocab], dtype=bool)
    return indptr, nbr, rel_s, rel_vocab, black_mask


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


def bootstrap_ci(feat, y, strata, n=300, seed=0):
    rng = np.random.default_rng(seed); N = len(y); vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(strat_auroc(feat[idx], y[idx], strata[idx]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p(feat, y, strata, obs, n=300, seed=0):
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
    ap.add_argument("--max-pairs", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    print("[crossarm] loading KG + relation-aware adjacency ...", flush=True)
    kg = MergedKG.from_parquet()
    indptr, nbr, rel_s, rel_vocab, black_mask = build_rel_adj(kg)
    deg = kg.degree.astype(np.float64)
    print(f"[crossarm] nnz={len(nbr):,}  relations={len(rel_vocab)}  blacklisted={int(black_mask.sum())}", flush=True)

    rows = []
    for s in SEEDS:
        fr = _load_frames(s)["test_s2"].reset_index(drop=True)
        te = _load_support_cache("and", s, L_MAX, K_PER_TYPE, N_MAX, False)["test_s2"]
        off = te["offsets"]; med = te["med"].astype(np.int64)
        da = te["da"].astype(np.int64); db = te["db"].astype(np.int64); y = te["y"].astype(int)
        for i in range(len(fr)):
            rows.append(dict(y=int(y[i]), s0=int(off[i]), e0=int(off[i + 1]),
                             med=med, da=da, db=db))
    rng = np.random.default_rng(args.seed)
    pos = [r for r in rows if r["y"] == 1]; neg = [r for r in rows if r["y"] == 0]
    k = min(args.max_pairs // 2, len(pos), len(neg))
    sel = [pos[j] for j in rng.choice(len(pos), k, replace=False)] + \
          [neg[j] for j in rng.choice(len(neg), k, replace=False)]
    print(f"[crossarm] sampled balanced={len(sel)} (k={k}/class)", flush=True)

    mask = np.zeros(kg.n_nodes, dtype=bool)
    feats = []
    rel_ctr: Counter = Counter()
    for n_done, r in enumerate(sel):
        s0, e0 = r["s0"], r["e0"]; S = r["med"][s0:e0]; nS = len(S)
        rec = dict(y=r["y"], n_support=nS)
        if nS >= 2:
            pos_t = {int(m): (int(r["da"][s0 + j]), int(r["db"][s0 + j])) for j, m in enumerate(S)}
            rec["mean_log_deg"] = float(np.log1p(deg[S]).mean())
            mask[:] = False; mask[S] = True
            cnt = 0; wsum = 0.0
            for u in S.tolist():
                a0, a1 = indptr[u], indptr[u + 1]
                nb = nbr[a0:a1]; rl = rel_s[a0:a1]
                sel_m = mask[nb] & (u < nb) & (~black_mask[rl])
                if not sel_m.any():
                    continue
                pu = pos_t[u]
                for w_, rid in zip(nb[sel_m].tolist(), rl[sel_m].tolist()):
                    pw = pos_t[w_]
                    # cross-arm = strictly incomparable (opposite ordering on the two arms)
                    if (pu[0] < pw[0] and pu[1] > pw[1]) or (pu[0] > pw[0] and pu[1] < pw[1]):
                        cnt += 1
                        wsum += 1.0 / math.log1p(deg[u] + deg[w_])
                        rel_ctr[rel_vocab[rid]] += 1
            c2 = nS * (nS - 1) / 2
            rec.update(crossarm_count=cnt, crossarm_any=int(cnt > 0),
                       crossarm_wdensity=wsum / c2)
        feats.append(rec)
        if (n_done + 1) % 1000 == 0:
            print(f"  ... {n_done+1}/{len(sel)}", flush=True)

    df = pd.DataFrame(feats)
    dfc = df[df["n_support"] >= 2].copy()

    def supp_bin(n):
        return "2-4" if n <= 4 else "5-9" if n <= 9 else "10-19" if n <= 19 else "20-49" if n <= 49 else "50+"
    dfc["sbin"] = dfc["n_support"].map(supp_bin)
    y = dfc["y"].to_numpy()
    q = pd.qcut(dfc["mean_log_deg"], 4, labels=False, duplicates="drop").to_numpy()
    strata = np.array([f"{b}|{qq}" for b, qq in zip(dfc["sbin"], q)])

    cov = float(dfc["crossarm_any"].mean())
    med_cnt = float(dfc.loc[dfc.crossarm_any == 1, "crossarm_count"].median()) if (dfc.crossarm_any == 1).any() else 0.0
    print(f"\n[crossarm] pairs |S|>=2: {len(dfc)}  cross-arm(filtered) coverage: {cov:.3f}  "
          f"median count when present: {med_cnt:.0f}")
    print("top kept cross-arm relations:", rel_ctr.most_common(12))

    print("\n=== CROSS-ARM (filtered) DISCRIMINATIVITY ===")
    out_disc = {}
    go = False
    for feat in ["crossarm_any", "crossarm_wdensity"]:
        f = dfc[feat].to_numpy().astype(float)
        auc = strat_auroc(f, y, strata)
        lo, hi = bootstrap_ci(f, y, strata, seed=args.seed)
        p = perm_p(f, y, strata, auc, seed=args.seed)
        out_disc[feat] = dict(auroc=round(auc, 4), ci=[round(lo, 4), round(hi, 4)], perm_p=round(p, 4))
        print(f"  {feat:18s} AUROC={auc:.4f}  95%CI[{lo:.4f},{hi:.4f}]  perm_p={p:.4f}")
        if auc > 0.53 and p < 0.05:
            go = True

    print("\n=== VERDICT (pre-registered) ===")
    if go:
        print("  cross-arm filtered chord IS discriminative (>0.53, p<0.05) -> hint survives; consider building.")
    else:
        print("  NULL -> FULL NO-GO: internal cross-arm mediator chords add no DDI signal beyond")
        print("  support size + degree in this KG/test. Pivot (do not build the chord/semantic-gate).")

    out = dict(coverage=cov, median_count_when_present=med_cnt,
               discriminativity=out_disc, go=go,
               blacklist=sorted(BLACKLIST),
               top_relations=[[k, v] for k, v in rel_ctr.most_common(20)],
               config=dict(max_pairs=args.max_pairs, seeds=list(SEEDS), n_ge2=int(len(dfc))))
    (RES / "03__corridor_chords_crossarm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {RES / '03__corridor_chords_crossarm.json'}")


if __name__ == "__main__":
    main()
