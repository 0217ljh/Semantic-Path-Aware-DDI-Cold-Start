"""A2 acceptance test - per-seed L-hop neighborhood."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER))

from kg.store import build_kg_store                                    # noqa: E402
from kg.neighborhood import (build_neighborhoods_idx, neighborhood_of_idx,   # noqa: E402
                             neighborhoods)
from data.loader import load_rank_data                                # noqa: E402
from specs import TaskSpec                                            # noqa: E402

_fails: list[str] = []


def check(cond, msg):
    (_fails.append(msg) if not cond else None)
    print(f"  {'ok  ' if cond else 'FAIL'}: {msg}")


def brute_bfs(kg, seed, L):
    seen = {seed}; frontier = {seed}
    for _ in range(L):
        nxt = set()
        for u in frontier:
            nxt.update(kg.neighbors_idx(u).tolist())
        nxt -= seen; seen |= nxt; frontier = nxt
    seen.discard(seed)
    return seen


def main() -> int:
    kg = build_kg_store()
    rng = np.random.default_rng(0)

    print("== single-seed correctness (vs brute BFS) ==")
    seeds = rng.choice(kg.n_nodes, size=25, replace=False)
    ok_match = True; sorted_ok = True; noself = True
    for s in seeds:
        r, dd = neighborhood_of_idx(kg, int(s), L=2)
        check_dist = len(r) == len(dd) and (len(dd) == 0 or (dd.min() >= 1 and dd.max() <= 2))
        if not check_dist:
            sorted_ok = False
        if set(r.tolist()) != brute_bfs(kg, int(s), 2):
            ok_match = False
        if not np.array_equal(r, np.unique(r)):
            sorted_ok = False
        if int(s) in r.tolist():
            noself = False
    check(ok_match, "N_<=2(seed) matches brute-force BFS")
    check(sorted_ok, "result sorted + unique")
    check(noself, "seed excluded from its own neighborhood")

    print("\n== edge cases ==")
    check(len(neighborhood_of_idx(kg, int(seeds[0]), L=0)[0]) == 0, "L=0 -> empty")
    try:
        neighborhood_of_idx(kg, kg.n_nodes + 5, L=2); check(False, "out-of-range raises")
    except ValueError:
        check(True, "out-of-range seed raises ValueError")
    try:
        build_neighborhoods_idx(kg, [int(seeds[0])], L=2, directed=True); check(False, "directed raises")
    except NotImplementedError:
        check(True, "directed=True raises NotImplementedError")
    # arbitrary (non-drug) seed works
    nd = int(np.flatnonzero(~kg.drug_mask)[0])
    check(len(neighborhood_of_idx(kg, nd, L=1)[0]) >= 0, "arbitrary non-drug seed works")

    print("\n== batch CSR == single primitive ==")
    batch = build_neighborhoods_idx(kg, seeds.tolist(), L=2, rebuild=True)
    match = all(np.array_equal(batch.of_idx(int(s)), neighborhood_of_idx(kg, int(s), 2)[0])
                for s in seeds)
    check(match, "batch.of_idx(seed) == neighborhood_of_idx(seed)")
    check(len(batch.seed_idx) == len(set(seeds.tolist())), "batch dedups seeds")

    print("\n== M_uv selectivity sanity (real pairs, L=2) ==")
    d = load_rank_data("drugbank_latest_partial", TaskSpec.multiclass(165), "fold0")
    ppi = rng.choice(len(d.train_pairs), size=60, replace=False)
    med_sizes = []
    for i in ppi:
        a, b = kg.get_idx(d.train_pairs[i][0]), kg.get_idx(d.train_pairs[i][1])
        if a is None or b is None:
            continue
        Na = set(neighborhood_of_idx(kg, a, 2)[0].tolist())
        Nb = set(neighborhood_of_idx(kg, b, 2)[0].tolist())
        inter = np.array(sorted(Na & Nb), dtype=np.int32)
        med = inter[~kg.drug_mask[inter]] if len(inter) else inter    # \ D preview
        med_sizes.append(len(med))
    mm = int(np.median(med_sizes))
    check(100 <= mm <= 2500, f"median |N(u)∩N(v)\\D| in [100,2500] at L=2 (got {mm})")
    check(np.mean([s > 0 for s in med_sizes]) >= 0.85, "≥85% pairs have non-empty M_uv")

    print("\n== cache round-trip + string wrapper ==")
    b2 = build_neighborhoods_idx(kg, seeds.tolist(), L=2, rebuild=False)  # HIT
    check(np.array_equal(b2.reach_idx, batch.reach_idx), "cache reload matches")
    drug_ids = [kg.get_id(int(s)) for s in np.flatnonzero(kg.drug_mask)[:30]]
    w = neighborhoods(kg, drug_ids, L=2)
    check(len(w.seed_idx) == len(set(drug_ids)), "string wrapper maps drug_ids")

    print(f"\n{'='*50}\n{'ALL PASS' if not _fails else f'{len(_fails)} FAILURES'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
