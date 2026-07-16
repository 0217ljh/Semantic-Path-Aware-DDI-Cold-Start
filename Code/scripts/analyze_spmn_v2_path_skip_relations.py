"""EXISTENCE probe (dumb, exploratory): on a single drug A -> B chain A-B-C-D-E
(arms l=2: B in N(A)&N(C), C a 2-hop-shared midpoint, D in N(E)&N(C)), do NON-ADJACENT
nodes on the chain (B-D, A-C, C-E, A-D, B-E, A-E) have a direct KG edge -- a "skip
relation" that a sequential path traversal would not use? Dump concrete examples with
node names + the skip relation for manual reading. Read-only, no GPU.

Generic co-annotation / ontology relations are blacklisted (reuse crossarm BLACKLIST) so
the skip edges surfaced are mechanistic. This only asks whether such chains EXIST.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && CUDA_VISIBLE_DEVICES= \
    python Code/scripts/analyze_spmn_v2_path_skip_relations.py --n-chains 20"
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
from analyze_spmn_v2_corridor_chords_crossarm import BLACKLIST, build_rel_adj  # noqa: E402

# non-adjacent node-index pairs on chain [A,B,C,D,E] = indices 0..4, |i-j|>=2
NONADJ = [(0, 2, "A-C"), (0, 3, "A-D"), (0, 4, "A-E"),
          (1, 3, "B-D"), (1, 4, "B-E"), (2, 4, "C-E")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains", type=int, default=20)
    ap.add_argument("--max-mid", type=int, default=8, help="(2,2) midpoints tried per pair")
    ap.add_argument("--seed", type=int, default=42)
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

    def low_deg_common(x, z, k=1):
        Nx = set(nbrs(x).tolist())
        cand = [w for w in nbrs(z).tolist() if w in Nx and w not in (x, z)]
        cand.sort(key=lambda w: deg[w])
        return cand[:k]

    fr = _load_frames(args.seed)["test_s2"].reset_index(drop=True)
    te = _load_support_cache("and", args.seed, 3, 64, 400, False)["test_s2"]
    off = te["offsets"]; med = te["med"].astype(np.int64)
    da = te["da"].astype(np.int64); db = te["db"].astype(np.int64); y = te["y"].astype(int)

    shown = 0
    n_pairs_scanned = 0
    n_pairs_with_chain = 0
    for i in range(len(fr)):
        if shown >= args.n_chains:
            break
        if int(y[i]) != 1:
            continue
        A, E = str(fr["drug_a_id"].iloc[i]), str(fr["drug_b_id"].iloc[i])
        if A not in kg.id_to_idx or E not in kg.id_to_idx:
            continue
        ui, vi = kg.id_to_idx[A], kg.id_to_idx[E]
        s0, e0 = int(off[i]), int(off[i + 1])
        mids = [int(med[s0 + j]) for j in range(e0 - s0)
                if da[s0 + j] == 2 and db[s0 + j] == 2]
        mids.sort(key=lambda c: deg[c])
        n_pairs_scanned += 1
        pair_had = False
        for C in mids[:args.max_mid]:
            Bs = low_deg_common(ui, C, k=1)
            Ds = low_deg_common(vi, C, k=1)
            if not Bs or not Ds:
                continue
            B, D = Bs[0], Ds[0]
            chain = [ui, B, C, D, vi]
            if len({ui, B, C, D, vi}) < 5:
                continue
            skips = []
            for a_i, b_i, lab in NONADJ:
                if lab == "A-E":
                    continue  # the DDI itself; skip
                r = rel_between(chain[a_i], chain[b_i])
                if r is not None:
                    skips.append((lab, r, chain[a_i], chain[b_i]))
            if not skips:
                continue
            pair_had = True
            shown += 1
            nm = [idx2name[x] for x in chain]
            print(f"\n[{shown}] {A}({nm[0]}) x {E}({nm[4]})")
            print(f"    chain:  {nm[0]} - {nm[1]} - {nm[2]} - {nm[3]} - {nm[4]}")
            print(f"    kinds:  A=Drug B={idx2kind[B]} C={idx2kind[C]} D={idx2kind[D]} E=Drug")
            for lab, r, na, nb_ in skips:
                print(f"    SKIP {lab}:  {idx2name[na]} ══[{r}]══ {idx2name[nb_]}")
            if shown >= args.n_chains:
                break
        if pair_had:
            n_pairs_with_chain += 1

    print(f"\n(scanned {n_pairs_scanned} positive pairs; {n_pairs_with_chain} had >=1 chain with a "
          f"non-adjacent skip edge; showed {shown} chains. l=2 arms, generic relations blacklisted.)")


if __name__ == "__main__":
    main()
