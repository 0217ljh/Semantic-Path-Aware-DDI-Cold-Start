"""Dump A-B-C-D-E chains + cross-distance skip relations for a BLIND mix of positive and
negative DDI pairs, for a semantic reading test: can reading the chain's semantics infer
DDI, and does it DISCRIMINATE pos from neg (not just rationalize DDI for everything)?
Labels printed as a hidden key at the very end. Read-only, no GPU.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && CUDA_VISIBLE_DEVICES= \
    python Code/scripts/analyze_spmn_v2_path_skip_mixed.py --n-per-class 6"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
from my_code.models.spmn_v1.retrieval import DEFAULT_NODES_PATH, MergedKG  # noqa: E402
from analyze_spmn_v2_corridor_chords_crossarm import build_rel_adj  # noqa: E402

NONADJ = [(0, 2, "A-C"), (0, 3, "A-D"), (1, 3, "B-D"), (1, 4, "B-E"), (2, 4, "C-E")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=6)
    ap.add_argument("--max-mid", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    kg = MergedKG.from_parquet()
    indptr, nbr, rel_s, rel_vocab, black_mask = build_rel_adj(kg)
    deg = kg.degree.astype(np.float64)
    nodes = pd.read_parquet(ROOT / DEFAULT_NODES_PATH)
    idx2name = ["?"] * kg.n_nodes; idx2kind = ["?"] * kg.n_nodes
    for _id, _kind, _name in zip(nodes["id"], nodes["kind"], nodes["name"]):
        j = kg.id_to_idx.get(str(_id))
        if j is not None:
            idx2name[j] = str(_name); idx2kind[j] = str(_kind)

    def nbrs(x):
        return nbr[indptr[x]:indptr[x + 1]]

    def rel_between(a, b):
        s, e = indptr[a], indptr[a + 1]
        seg = nbr[s:e]; hit = np.where(seg == b)[0]
        if not len(hit):
            return None
        rid = rel_s[s:e][hit[0]]
        return None if black_mask[rid] else rel_vocab[rid]

    def low_deg_common(x, z):
        Nx = set(nbrs(x).tolist())
        cand = [w for w in nbrs(z).tolist() if w in Nx and w not in (x, z)]
        return min(cand, key=lambda w: deg[w]) if cand else None

    fr = _load_frames(42)["test_s2"].reset_index(drop=True)
    te = _load_support_cache("and", 42, 3, 64, 400, False)["test_s2"]
    off = te["offsets"]; med = te["med"].astype(np.int64)
    da = te["da"].astype(np.int64); db = te["db"].astype(np.int64); y = te["y"].astype(int)

    def first_chain_with_skip(i):
        A, E = str(fr["drug_a_id"].iloc[i]), str(fr["drug_b_id"].iloc[i])
        if A not in kg.id_to_idx or E not in kg.id_to_idx:
            return None
        ui, vi = kg.id_to_idx[A], kg.id_to_idx[E]
        s0, e0 = int(off[i]), int(off[i + 1])
        mids = [int(med[s0 + j]) for j in range(e0 - s0) if da[s0 + j] == 2 and db[s0 + j] == 2]
        mids.sort(key=lambda c: deg[c])
        for C in mids[:args.max_mid]:
            B = low_deg_common(ui, C); D = low_deg_common(vi, C)
            if B is None or D is None or len({ui, B, C, D, vi}) < 5:
                continue
            chain = [ui, B, C, D, vi]
            skips = [(lab, rel_between(chain[a], chain[b]), chain[a], chain[b])
                     for a, b, lab in NONADJ if rel_between(chain[a], chain[b]) is not None]
            if skips:
                return (A, E, chain, skips)
        return None

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(fr))
    picks = []; got_p = got_n = 0
    for i in order:
        yy = int(y[i])
        if yy == 1 and got_p >= args.n_per_class:
            continue
        if yy == 0 and got_n >= args.n_per_class:
            continue
        r = first_chain_with_skip(int(i))
        if r is None:
            continue
        picks.append((yy, r))
        if yy == 1:
            got_p += 1
        else:
            got_n += 1
        if got_p >= args.n_per_class and got_n >= args.n_per_class:
            break

    rng.shuffle(picks)
    key = []
    for k, (yy, (A, E, chain, skips)) in enumerate(picks, 1):
        nm = [idx2name[x] for x in chain]
        kn = [idx2kind[x] for x in chain]
        print(f"\n=== CASE {k} ===  {A}  x  {E}")
        print(f"  chain (A-B-C-D-E):  {nm[0]} [{kn[0]}] - {nm[1]} [{kn[1]}] - {nm[2]} [{kn[2]}] "
              f"- {nm[3]} [{kn[3]}] - {nm[4]} [{kn[4]}]")
        for lab, r, na, nb_ in skips:
            print(f"  cross-distance {lab}:  {idx2name[na]}  ={r}=  {idx2name[nb_]}")
        key.append((k, yy))

    print("\n\n===== HIDDEN LABEL KEY (read AFTER making your semantic inferences) =====")
    print("  " + "  ".join(f"case{k}={'DDI' if yy else 'no-DDI'}" for k, yy in key))


if __name__ == "__main__":
    main()
