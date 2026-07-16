"""Show CONCRETE examples of corridor cross-arm "chords" and their jump-semantics
(read-only, no GPU). Answers: do such chains exist, and what does the non-sequential
semantic look like biomedically? (Complements the aggregate NO-GO in 02_03 verdict.)

For positive test_s2 pairs with >=1 cross-arm chord (incomparable positions, generic
co-annotation relations blacklisted), print: drug_a x drug_b, then the top chords
    m_i (name, type, pos) --[relation]-- m_j (name, type, pos)
ranked by inverse-degree weight (surface specific, non-hub chords first).

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && CUDA_VISIBLE_DEVICES= \
    python Code/scripts/analyze_spmn_v2_corridor_chord_examples.py --n-examples 12"
"""
from __future__ import annotations

import argparse
import math
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
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    DEFAULT_EDGES_PATH, DEFAULT_NODES_PATH, MergedKG,
)
from analyze_spmn_v2_corridor_chords_crossarm import BLACKLIST, build_rel_adj  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--top-chords", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    kg = MergedKG.from_parquet()
    indptr, nbr, rel_s, rel_vocab, black_mask = build_rel_adj(kg)
    deg = kg.degree.astype(np.float64)

    nodes = pd.read_parquet(ROOT / DEFAULT_NODES_PATH)
    idx2name = ["?"] * kg.n_nodes
    idx2kind = ["?"] * kg.n_nodes
    for _id, _kind, _name in zip(nodes["id"], nodes["kind"], nodes["name"]):
        j = kg.id_to_idx.get(str(_id))
        if j is not None:
            idx2name[j] = str(_name); idx2kind[j] = str(_kind)

    fr = _load_frames(args.seed)["test_s2"].reset_index(drop=True)
    te = _load_support_cache("and", args.seed, 3, 64, 400, False)["test_s2"]
    off = te["offsets"]; med = te["med"].astype(np.int64)
    da = te["da"].astype(np.int64); db = te["db"].astype(np.int64); y = te["y"].astype(int)

    mask = np.zeros(kg.n_nodes, dtype=bool)
    shown = 0
    for i in range(len(fr)):
        if shown >= args.n_examples:
            break
        if int(y[i]) != 1:
            continue
        s0, e0 = int(off[i]), int(off[i + 1])
        S = med[s0:e0]
        if len(S) < 2:
            continue
        pos_t = {int(m): (int(da[s0 + j]), int(db[s0 + j])) for j, m in enumerate(S)}
        mask[:] = False; mask[S] = True
        chords = []
        for u in S.tolist():
            a0, a1 = indptr[u], indptr[u + 1]
            nb = nbr[a0:a1]; rl = rel_s[a0:a1]
            sel_m = mask[nb] & (u < nb) & (~black_mask[rl])
            pu = pos_t[u]
            for w_, rid in zip(nb[sel_m].tolist(), rl[sel_m].tolist()):
                pw = pos_t[w_]
                if (pu[0] < pw[0] and pu[1] > pw[1]) or (pu[0] > pw[0] and pu[1] < pw[1]):
                    wgt = 1.0 / math.log1p(deg[u] + deg[w_])
                    chords.append((wgt, u, w_, rel_vocab[rid]))
        if not chords:
            continue
        chords.sort(reverse=True)
        a_name = idx2name[kg.id_to_idx.get(str(fr["drug_a_id"].iloc[i]), 0)]
        b_name = idx2name[kg.id_to_idx.get(str(fr["drug_b_id"].iloc[i]), 0)]
        shown += 1
        print(f"\n[{shown}] {fr['drug_a_id'].iloc[i]} ({a_name})  x  "
              f"{fr['drug_b_id'].iloc[i]} ({b_name})   |S|={len(S)}, "
              f"cross-arm chords={len(chords)}")
        for wgt, u, w_, rel in chords[:args.top_chords]:
            print(f"    {idx2name[u]:26.26s} [{idx2kind[u]:11.11s} pos{pos_t[u]}]  "
                  f"--{rel}--  {idx2name[w_]:26.26s} [{idx2kind[w_]:11.11s} pos{pos_t[w_]}]")

    print(f"\n(shown {shown} positive pairs with cross-arm chords; relations are the KG "
          f"relation label, generic co-annotation blacklisted)")


if __name__ == "__main__":
    main()
