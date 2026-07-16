"""Cross-length probe: do PD drug pairs converge on the SAME effect node at
DIFFERENT hop-depths (off-diagonal (dA,dB))?

Motivation: a wave-source propagation DDI model with depth-resolved attention
can match A's effect node (at depth dA) to B's same effect node (at depth dB)
even when dA != dB. This probe quantifies — purely from KG structure — how much
"cross-length" convergence on effect-type mediators exists for PD vs PK vs
negatives. High off-diagonal mass / coverage-gap => cross-length matching has
real structural signal worth building; mostly on-diagonal => same-depth suffices.

NOTE (honest scope): this measures whether cross-length convergence EXISTS
structurally. It does NOT test whether RoPE's relative-depth term beats a
depth-blind attention — that is a model ablation, not a structural probe.

For PD / PK (sampled positives) and NEG (sampled seed42 test_s2 negatives), each
100 pairs (seed 42): BFS<=L from both drugs; collect shared mediators restricted
to effect-type node kinds; bin by (dA,dB); report:
  - (dA,dB) grid mean mediators/pair
  - on-diagonal (dA==dB) vs off-diagonal (dA!=dB) mass
  - coverage: any-depth / same-depth-only / OFF-ONLY gap (pairs whose ONLY
    shared effect mediator is off-diagonal = pure cross-length value)
Two effect definitions: "effect" (side-effect/phenotype) and "effect_plus"
(+ biological_process + disease, the "biological system" chain).

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_pd_crosslength_effect_probe.py --n 100 --seed 42 --L 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
sys.path.insert(0, str(ROOT))
NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"
NEG_PARQUET = ROOT / "data/KG/drugbank/splits/seed42/negatives/test_s2.parquet"

EFFECT_KINDS = ["Side Effect", "effect/phenotype", "Symptom"]
BIOSYS_KINDS = ["biological_process", "Biological Process", "disease", "Disease"]


def log(m: str) -> None:
    print(m, flush=True)


def pkpd_bucket(t: str) -> str:
    t = str(t).lower()
    if any(k in t for k in ("metabolism", "excretion", "serum concentration",
                            "absorption", "protein binding", "clearance")):
        return "PK"
    if any(k in t for k in ("risk or severity", "activities", "efficacy",
                            "cns depression", "qtc", "hypertension",
                            "hypotensive", "sedative", "adverse effects")):
        return "PD"
    return "other"


def bfs_depth(A_csr, source, n_nodes, max_depth):
    dist = np.full(n_nodes, -1, dtype=np.int16)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    for d in range(1, max_depth + 1):
        if len(frontier) == 0:
            break
        nbrs = np.unique(A_csr[frontier].indices)
        new = nbrs[dist[nbrs] == -1]
        if len(new) == 0:
            break
        dist[new] = d
        frontier = new
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_pd_crosslength_effect_probe"))
    args = ap.parse_args()
    t0 = time.time()
    L = args.L
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    nodes = pd.read_parquet(NODES_PARQUET)
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)
    kinds = nodes["kind"].to_numpy()
    drug_db = nodes["id"].to_numpy()[kinds == "Drug"]
    drug_set_db = set(drug_db.tolist())
    d2g = {db: id2idx[db] for db in drug_db}

    # effect-type masks (boolean over global node idx)
    eff_mask = np.isin(kinds, EFFECT_KINDS)
    effplus_mask = np.isin(kinds, EFFECT_KINDS + BIOSYS_KINDS)
    log(f"[probe] effect nodes={int(eff_mask.sum())}  effect_plus nodes={int(effplus_mask.sum())}")

    # undirected adjacency
    src = edges["src"].map(id2idx).to_numpy(); dst = edges["dst"].map(id2idx).to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    s = src[keep].astype(np.int64); d = dst[keep].astype(np.int64)
    nz = s != d
    s, d = s[nz], d[nz]
    rows = np.concatenate([s, d]); cols = np.concatenate([d, s])
    A = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.data[:] = 1

    # ---- samples ----
    pos = pd.read_csv(DDI_POS_CSV)
    pos = pos[pos["drug_a_id"].isin(drug_set_db) & pos["drug_b_id"].isin(drug_set_db)].reset_index(drop=True)
    bk = np.array([pkpd_bucket(t) for t in pos["ddi_type"].to_numpy()])
    samples = {}
    for b in ["PK", "PD"]:
        idx = np.where(bk == b)[0]
        sel = rng.choice(idx, size=min(args.n, len(idx)), replace=False)
        samples[b] = pos.iloc[sel][["drug_a_id", "drug_b_id"]].to_numpy()
    neg = pd.read_parquet(NEG_PARQUET)
    neg = neg[neg["drug_a_id"].isin(drug_set_db) & neg["drug_b_id"].isin(drug_set_db)].reset_index(drop=True)
    nsel = rng.choice(len(neg), size=min(args.n, len(neg)), replace=False)
    samples["NEG"] = neg.iloc[nsel][["drug_a_id", "drug_b_id"]].to_numpy()
    for b in samples:
        log(f"[sample] {b}: {len(samples[b])} pairs")

    dist_cache = {}

    def get_dist(g):
        if g not in dist_cache:
            dist_cache[g] = bfs_depth(A, g, n_nodes, L)
        return dist_cache[g]

    VERSIONS = {"effect": eff_mask, "effect_plus": effplus_mask}
    results = {}
    for vname, vmask in VERSIONS.items():
        for b, pairs in samples.items():
            grid = np.zeros((L, L), dtype=np.float64)   # [dA-1, dB-1] mean mediators/pair
            n_any = n_ondiag = n_offdiag = n_offonly = 0
            npairs = len(pairs)
            for (a_id, bb_id) in pairs:
                if a_id not in d2g or bb_id not in d2g:
                    continue
                dA = get_dist(d2g[a_id]); dB = get_dist(d2g[bb_id])
                shared = (dA >= 1) & (dB >= 1) & (dA <= L) & (dB <= L) & vmask
                if not shared.any():
                    continue
                aa = dA[shared]; bbv = dB[shared]
                for ca in range(1, L + 1):
                    for cb in range(1, L + 1):
                        grid[ca - 1, cb - 1] += int(np.sum((aa == ca) & (bbv == cb)))
                on = np.any(aa == bbv)
                off = np.any(aa != bbv)
                n_any += 1
                n_ondiag += int(on)
                n_offdiag += int(off)
                n_offonly += int(off and not on)
            grid /= npairs
            on_mass = float(np.trace(grid))
            off_mass = float(grid.sum() - on_mass)
            results[f"{vname}__{b}"] = {
                "n_pairs": npairs,
                "grid_mean_per_pair": grid.tolist(),
                "on_diag_mass_per_pair": on_mass,
                "off_diag_mass_per_pair": off_mass,
                "off_frac_of_mass": float(off_mass / (on_mass + off_mass)) if (on_mass + off_mass) > 0 else 0.0,
                "cov_any": n_any / npairs,
                "cov_same_depth": n_ondiag / npairs,
                "cov_offonly_GAP": n_offonly / npairs,
            }

    # ---- print ----
    for vname in VERSIONS:
        log(f"\n################ effect 定义 = {vname} ################")
        log(f"{'bucket':6s} {'cov_any':>8s} {'cov_same':>9s} {'OFF-ONLY gap':>13s} {'off_mass%':>10s}")
        for b in ["PD", "PK", "NEG"]:
            r = results[f"{vname}__{b}"]
            log(f"{b:6s} {r['cov_any']*100:7.1f}% {r['cov_same_depth']*100:8.1f}% "
                f"{r['cov_offonly_GAP']*100:12.1f}% {r['off_frac_of_mass']*100:9.1f}%")
        log("  (dA,dB) 均值 mediator/对 [行=dA 1..L, 列=dB 1..L], PD:")
        g = np.array(results[f"{vname}__PD"]["grid_mean_per_pair"])
        for ca in range(L):
            log("   " + " ".join(f"{g[ca, cb]:7.2f}" for cb in range(L)))

    summary = {"L": L, "n": args.n, "seed": args.seed,
               "decision_note": "off-only GAP = 仅靠跨深度(off-diagonal)才有共享effect的对占比; "
                                 "PD gap 显著大于PK且≥~15% => 跨长度匹配值得加",
               "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n[save] {out_dir/'summary.json'}")
    log(f"[done] wall_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
