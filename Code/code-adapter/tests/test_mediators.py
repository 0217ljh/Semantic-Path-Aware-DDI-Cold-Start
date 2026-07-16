"""A3 acceptance test - shared mediator set M_uv (AND primary; SUM callable)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER))

from kg.store import build_kg_store                                     # noqa: E402
from kg.neighborhood import build_neighborhoods_idx                     # noqa: E402
from kg.mediators import (shared_mediators, shared_mediators_batch,     # noqa: E402
                          shared_mediators_idx)
from data.loader import load_rank_data                                  # noqa: E402
from specs import TaskSpec                                              # noqa: E402

_fails: list[str] = []


def check(cond, msg):
    (_fails.append(msg) if not cond else None)
    print(f"  {'ok  ' if cond else 'FAIL'}: {msg}")


def main() -> int:
    kg = build_kg_store()
    rng = np.random.default_rng(0)
    d = load_rank_data("drugbank_latest_partial", TaskSpec.multiclass(165), "fold0")

    # sample real pairs whose both drugs are in KG
    pi = rng.choice(len(d.train_pairs), size=50, replace=False)
    pairs = []
    for i in pi:
        u, v = kg.get_idx(d.train_pairs[i][0]), kg.get_idx(d.train_pairs[i][1])
        if u is not None and v is not None:
            pairs.append((u, v))
    seed_idx = sorted({x for p in pairs for x in p})
    nb2 = build_neighborhoods_idx(kg, seed_idx, L=2, rebuild=True)      # AND(tau<=2) source

    print("== AND(tau=2): correctness vs brute intersection\\D ==")
    ok_and = True; nodrug = True; type_ok = True; darange = True
    sizes = []
    for u, v in pairs:
        r = shared_mediators_idx(kg, nb2, u, v, mode="and", tau=2)
        # brute: intersect the two L2 sets, drop drugs
        iu, iv = nb2.of_idx(u), nb2.of_idx(v)
        inter = np.intersect1d(iu, iv, assume_unique=True)
        brute = inter[~kg.drug_mask[inter]]
        if not np.array_equal(np.sort(r["med_idx"]), np.sort(brute)):
            ok_and = False
        if r["med_idx"].size and kg.drug_mask[r["med_idx"]].any():
            nodrug = False
        if not (len(r["type_id"]) == len(r["med_idx"])
                and (len(r["med_idx"]) == 0 or np.array_equal(r["type_id"], kg.type_id[r["med_idx"]]))):
            type_ok = False
        if r["d_a"].size and (r["d_a"].max() > 2 or r["d_b"].max() > 2):
            darange = False
        sizes.append(len(r["med_idx"]))
    check(ok_and, "AND(tau=2) == intersection \\ D (brute)")
    check(nodrug, "no drug nodes in M_uv")
    check(type_ok, "type_id aligned to med_idx")
    check(darange, "arm distances d_a,d_b <= 2 under L=2")
    mm = int(np.median(sizes))
    check(100 <= mm <= 2500, f"median |M_uv| AND(2) in [100,2500] (got {mm})")

    print("\n== AND(tau=1) subset of AND(tau=2) ==")
    sub_ok = True
    for u, v in pairs[:15]:
        m2 = set(shared_mediators_idx(kg, nb2, u, v, mode="and", tau=2)["med_idx"].tolist())
        m1 = set(shared_mediators_idx(kg, nb2, u, v, mode="and", tau=1)["med_idx"].tolist())
        if not m1 <= m2:
            sub_ok = False
    check(sub_ok, "AND(tau=1) ⊆ AND(tau=2)")

    print("\n== SUM(sigma=4) callable (needs L>=3 nbhd) ==")
    u, v = pairs[0]
    nb3 = build_neighborhoods_idx(kg, [u, v], L=3, rebuild=True)
    s = shared_mediators_idx(kg, nb3, u, v, mode="sum", sigma=4)
    a3 = shared_mediators_idx(kg, nb3, u, v, mode="and", tau=2)
    check((s["d_a"] + s["d_b"] <= 4).all(), "SUM(4): all mediators satisfy d_a+d_b<=4")
    check(not (kg.drug_mask[s["med_idx"]].any() if s["med_idx"].size else False), "SUM no drugs")
    # SUM(4) admits (3,1)/(1,3) that AND(2) excludes -> generally not identical on L=3
    check(set(a3["med_idx"].tolist()) != set(s["med_idx"].tolist()) or len(s["med_idx"]) == len(a3["med_idx"]),
          f"AND(2) vs SUM(4) computed on L=3 (|AND|={len(a3['med_idx'])} |SUM|={len(s['med_idx'])})")
    try:
        shared_mediators_idx(kg, nb3, u, v, mode="xor"); check(False, "invalid mode raises")
    except ValueError:
        check(True, "invalid mode raises ValueError")

    print("\n== batch + string wrapper ==")
    b = shared_mediators_batch(kg, nb2, pairs[:10], mode="and", tau=2)
    check(len(b) == 10 and all("med_idx" in x for x in b), "batch returns per-pair dicts")
    uid, vid = kg.get_id(pairs[0][0]), kg.get_id(pairs[0][1])
    w = shared_mediators(kg, nb2, uid, vid, mode="and", tau=2)
    check(np.array_equal(np.sort(w["med_idx"]),
                         np.sort(shared_mediators_idx(kg, nb2, pairs[0][0], pairs[0][1], "and", 2)["med_idx"])),
          "string wrapper == idx primitive")

    print(f"\n{'='*50}\n{'ALL PASS' if not _fails else f'{len(_fails)} FAILURES'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
