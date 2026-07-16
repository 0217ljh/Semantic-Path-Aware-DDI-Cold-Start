"""Reach-vs-density diagnostic: does the receptive field need to ADAPT to a
drug pair's annotation density? (codex's "cheapest decisive test" for the
density-adaptive receptive field thesis, inspiration #1.)

The thesis (the GNN blind spot we claim to fill): a fixed-depth GNN reaches a
GLOBAL number of hops for every node. Annotation-RICH drugs already share
mediators in their immediate neighbourhood (short reach suffices); annotation-
SPARSE drugs have almost no near shared mediators, so they must reach FURTHER
(more hops) before ANY shared mechanism appears. A fixed receptive field cannot
do this without over-smoothing the rich drugs. Our retrieval+weighting decouples
reach from smoothing, so it CAN. But the empirical premise -- "sparse pairs gain
USEFUL (not hub) shared mediators as reach grows" -- is an inference, not a fact.
This script tests it offline, read-only, on the seed42 S2 test set.

For each test pair we BFS each drug to depth ``l_max`` on the symmetrised merged
KG (the SAME graph/distances ``build_pair_support`` uses, via ``kg.neighborhood``
-- no hub cap here so the reach curve is unbiased), intersect to get every shared
mediator with its ``(d_a, d_b, type, degree)``. A shared mediator is reachable at
box radius ("reach") ``l`` iff ``max(d_a, d_b) <= l`` -- exactly the AND/box
support at both-arm cap ``l`` (``l=2`` is our main config). We then ask, stratified
by pair annotation density:

  (A) reach curve     -- shared-mediator count at l=1..l_max (raw + Adamic-Adar
                         degree-discounted). Do sparse pairs start near-empty and
                         gain more (relatively) as l grows?
  (B) hubness         -- median KG degree of the NEW shell {max(d_a,d_b)==l}.
                         If the gain at high l is just high-degree hubs, the
                         degree rises sharply and the AA-weighted gain stalls.
  (C) pathway reach   -- (A) restricted to pathway mediators (the most variable,
                         highest-level mechanism type).
  (D) discriminative? -- AA-weighted reach split by label (pos vs neg). USEFUL
                         reach means positive pairs gain shared mechanism that
                         negatives do not; the pos-neg gap should grow with l
                         for sparse pairs if the extra reach is signal not noise.

Density per drug = #distinct non-drug 1-hop mediators (direct mechanism
annotations). Pair density = min of the two (the sparser arm is the bottleneck).
Pairs are binned into quartiles Q1 (sparsest) .. Q4 (richest).

Verdict the curves support inspiration #1 iff: sparse Q1 is near-empty at l=2,
its AA-weighted (not just raw) yield grows materially through l=3..4, the new
shell's degree does NOT explode (gain is specific not hub), AND the pos-neg gap
widens with l for Q1. If instead the high-l gain is raw-only / hub-degree /
label-blind, "sparse drugs reach further" collapses and the story must change.

Read-only. Writes a per-pair npz for follow-up and prints summary tables.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    KIND_ORDER, MergedKG,
)

THREE_SEED_DIR = ROOT / "Code/data/coldddi_legacy/800drug_3seed"
OUT_DIR = ROOT / "Code/runs/spmn_v2_standalone"
PATHWAY_TYPE = KIND_ORDER.index("pathway")
PROTEIN_TYPE = KIND_ORDER.index("protein_gene")


def _drug_reach(kg: MergedKG, idx: int, l_max: int):
    """Per-drug bounded BFS: non-drug mediators within ``l_max`` hops.

    Returns (nodes, dist) for mediators only (drug nodes dropped). deg1 (the
    annotation density) is #mediators at dist==1.
    """
    nodes, dist = kg.neighborhood(idx, l_max)
    keep = (~kg.is_drug[nodes]) & (dist >= 1)
    nodes, dist = nodes[keep], dist[keep]
    deg1 = int((dist == 1).sum())
    order = np.argsort(nodes, kind="stable")
    return nodes[order], dist[order].astype(np.int64), deg1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=4,
                    help="max box radius / BFS depth (reach l in 1..l_max)")
    ap.add_argument("--n-bins", type=int, default=4, help="density quantile bins")
    ap.add_argument("--hub-deg", type=int, default=1000,
                    help="degree above which a mediator counts as a hub")
    args = ap.parse_args()
    L = args.l_max
    levels = list(range(1, L + 1))

    frame = pd.read_parquet(THREE_SEED_DIR / f"seed{args.seed}" / "test_s2.parquet")
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    y = frame["label"].to_numpy().astype(int)

    print(f"[reach-density] loading merged KG ...", flush=True)
    t0 = time.time()
    kg = MergedKG.from_parquet()
    print(f"[reach-density] KG {kg.n_nodes} nodes, {len(frame)} test pairs, "
          f"l_max={L}  ({time.time()-t0:.1f}s)", flush=True)

    # Per-drug bounded neighbourhood (cached over unique drugs).
    uniq = sorted(set(a_ids.tolist()) | set(b_ids.tolist()))
    reach: dict[str, tuple] = {}
    deg1: dict[str, int] = {}
    t1 = time.time()
    for j, did in enumerate(uniq):
        idx = kg.id_to_idx.get(did)
        if idx is None:
            continue
        reach[did] = _drug_reach(kg, idx, L)
        deg1[did] = reach[did][2]
        if (j + 1) % 25 == 0:
            print(f"  bfs {j+1}/{len(uniq)} drugs ({time.time()-t1:.1f}s)", flush=True)
    print(f"[reach-density] per-drug BFS done ({time.time()-t1:.1f}s)", flush=True)

    n = len(frame)
    # per-pair, per-level accumulators
    cnt = np.zeros((n, L)); aa = np.zeros((n, L))
    pcnt = np.zeros((n, L)); paa = np.zeros((n, L))
    shell_deg_med = np.full((n, L), np.nan)
    hub_cnt = np.zeros((n, L))
    pair_density = np.full(n, -1.0)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        ra, rb = reach.get(a_ids[i]), reach.get(b_ids[i])
        if ra is None or rb is None or a_ids[i] == b_ids[i]:
            continue
        na, da_all, _ = ra
        nb, db_all, _ = rb
        common, ia, ib = np.intersect1d(na, nb, assume_unique=True,
                                        return_indices=True)
        valid[i] = True
        pair_density[i] = min(deg1[a_ids[i]], deg1[b_ids[i]])
        if common.size == 0:
            continue
        da = da_all[ia]; db = db_all[ib]
        mx = np.maximum(da, db)
        typ = kg.type_id[common]
        deg = kg.degree[common].astype(np.float64)
        aaw = 1.0 / np.log(np.clip(deg, 2.0, None))
        is_path = typ == PATHWAY_TYPE
        is_hub = deg > args.hub_deg
        for li, lev in enumerate(levels):
            m = mx <= lev
            cnt[i, li] = m.sum()
            aa[i, li] = aaw[m].sum()
            pcnt[i, li] = (m & is_path).sum()
            paa[i, li] = aaw[m & is_path].sum()
            hub_cnt[i, li] = (m & is_hub).sum()
            shell = mx == lev
            if shell.any():
                shell_deg_med[i, li] = np.median(deg[shell])

    v = valid
    dens = pair_density[v]
    yv = y[v]
    qs = np.quantile(dens, np.linspace(0, 1, args.n_bins + 1))
    binid = np.clip(np.digitize(dens, qs[1:-1]), 0, args.n_bins - 1)

    def col(arr):
        return arr[v]

    cntv, aav, pcntv, paav = col(cnt), col(aa), col(pcnt), col(paa)
    shellv, hubv = col(shell_deg_med), col(hub_cnt)

    print(f"\n[reach-density] {v.sum()} valid pairs (of {n}); pos rate "
          f"{yv.mean():.3f}; density min/med/max {dens.min():.0f}/"
          f"{np.median(dens):.0f}/{dens.max():.0f}")
    print(f"  density quartile edges (min 1-hop mediators): "
          f"{np.round(qs,1).tolist()}")

    lv_hdr = "  ".join(f"l={l}" for l in levels)

    print("\n[A] reach curve by density bin -- RAW shared-mediator count "
          "(mean per pair)")
    print(f"  {'bin':<10}{'n':>6}  {lv_hdr}   growth l2->lmax")
    for b in range(args.n_bins):
        m = binid == b
        means = cntv[m].mean(axis=0)
        g = means[-1] / means[1] if means[1] > 0 else float("nan")
        cells = "  ".join(f"{x:5.1f}" for x in means)
        tag = "sparse" if b == 0 else ("rich" if b == args.n_bins - 1 else "")
        print(f"  Q{b+1} {tag:<6}{int(m.sum()):>6}  {cells}   x{g:.2f}")

    print("\n[A'] reach curve by density bin -- ADAMIC-ADAR (hub-discounted) "
          "weight (mean per pair)")
    print(f"  {'bin':<10}{'n':>6}  {lv_hdr}   growth l2->lmax")
    for b in range(args.n_bins):
        m = binid == b
        means = aav[m].mean(axis=0)
        g = means[-1] / means[1] if means[1] > 0 else float("nan")
        cells = "  ".join(f"{x:5.2f}" for x in means)
        print(f"  Q{b+1} {'':<6}{int(m.sum()):>6}  {cells}   x{g:.2f}")

    print(f"\n[B] hubness -- median degree of the NEW shell {{max(d_a,d_b)==l}} "
          f"(mean over pairs); + #hub mediators (deg>{args.hub_deg})")
    print(f"  {'bin':<10}  shell-median-deg per l            hub-count per l")
    for b in range(args.n_bins):
        m = binid == b
        sd = np.nanmean(shellv[m], axis=0)
        hc = hubv[m].mean(axis=0)
        sds = " ".join(f"{x:6.0f}" for x in sd)
        hcs = " ".join(f"{x:5.1f}" for x in hc)
        print(f"  Q{b+1} {'':<6}  {sds}    {hcs}")

    print("\n[C] PATHWAY reach by density bin -- raw count / AA weight "
          "(mean per pair)")
    print(f"  {'bin':<10}  raw: {lv_hdr}      aa: {lv_hdr}")
    for b in range(args.n_bins):
        m = binid == b
        rc = pcntv[m].mean(axis=0)
        ac = paav[m].mean(axis=0)
        print(f"  Q{b+1} {'':<6}  {'  '.join(f'{x:5.2f}' for x in rc)}   "
              f"  {'  '.join(f'{x:5.2f}' for x in ac)}")

    print("\n[D] discriminative? AA-weight by label within each density bin "
          "(pos / neg / gap)")
    print(f"  {'bin':<10}  {'  '.join(f'l={l}(gap)' for l in levels)}")
    for b in range(args.n_bins):
        m = binid == b
        gaps = []
        for li in range(L):
            pp = aav[m & (yv == 1), li]
            nn = aav[m & (yv == 0), li]
            gaps.append(pp.mean() - nn.mean() if len(pp) and len(nn) else float("nan"))
        print(f"  Q{b+1} {'':<6}  {'  '.join(f'{g:+6.2f}' for g in gaps)}")

    out = OUT_DIR / f"reach_vs_density_seed{args.seed}_lmax{L}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, density=dens, binid=binid, y=yv, levels=np.array(levels),
             cnt=cntv, aa=aav, pcnt=pcntv, paa=paav,
             shell_deg_med=shellv, hub_cnt=hubv)
    print(f"\n[reach-density] per-pair arrays -> {out}")
    print("\nread: inspiration #1 holds iff sparse Q1 starts near-empty at l=2, "
          "its AA curve (not just raw) keeps growing through l=3..4, shell degree "
          "stays moderate (not hub), and the [D] pos-neg gap WIDENS with l for Q1.")


if __name__ == "__main__":
    main()
