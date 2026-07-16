"""Trace FULL drug A -> ... -> drug B chains that pass through a cross-arm "chord",
marking the jump-semantic position (read-only, no GPU). Arms <= length 2 (l_max=3).

For a chord (m_i @ pos(d_a_i,d_b_i)) -- m_j @ pos(d_a_j,d_b_j)) the concrete A->B walk is
  A --arm(d_a_i hops)--> m_i  ==JUMP[chord_rel]==  m_j --arm(d_b_j hops)--> B
The chord is the non-sequential "jump" linking the two arms. We reconstruct one shortest
arm on each side (length 1 = direct edge; length 2 = via a low-degree intermediate) and
print node names + edge relations, marking the JUMP.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && CUDA_VISIBLE_DEVICES= \
    python Code/scripts/analyze_spmn_v2_corridor_chain_trace.py"
"""
from __future__ import annotations

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

# example pairs to trace (from the chord examples)
TARGETS = [
    ("DB09389", "DB12015"),  # Norgestrel x Alpelisib  (AR - PIK3CA PPI)
    ("DB00749", "DB01028"),  # Etodolac x Methoxyflurane (GABRA3 - UGT2B7)
    ("DB01200", "DB01587"),  # Bromocriptine x Ketazolam (GABRG2 - DRD5)
    ("DB01087", "DB05294"),  # Primaquine x Vandetanib (CYP1A1 - long QT syndrome)
]


def main() -> None:
    kg = MergedKG.from_parquet()
    indptr, nbr, rel_s, rel_vocab, black_mask = build_rel_adj(kg)
    deg = kg.degree.astype(np.float64)
    nodes = pd.read_parquet(ROOT / DEFAULT_NODES_PATH)
    idx2name = ["?"] * kg.n_nodes
    for _id, _name in zip(nodes["id"], nodes["name"]):
        j = kg.id_to_idx.get(str(_id))
        if j is not None:
            idx2name[j] = str(_name)

    def rel_between(a, b):
        s, e = indptr[a], indptr[a + 1]
        seg = nbr[s:e]
        hit = np.where(seg == b)[0]
        return rel_vocab[rel_s[s:e][hit[0]]] if len(hit) else "?"

    def two_hop_mid(u, m):
        s, e = indptr[u], indptr[u + 1]
        Nu = set(nbr[s:e].tolist())
        s2, e2 = indptr[m], indptr[m + 1]
        cands = [x for x in nbr[s2:e2].tolist() if x in Nu and x not in (u, m)]
        if not cands:
            return None
        return min(cands, key=lambda x: deg[x])

    def arm(src, dst, d):
        """return list of (node, rel_from_prev) from src to dst; d = hops (1 or 2)."""
        if d <= 1:
            return [(dst, rel_between(src, dst))]
        mid = two_hop_mid(src, dst)
        if mid is None:
            return [(dst, "?(no 2-hop mid found)")]
        return [(mid, rel_between(src, mid)), (dst, rel_between(mid, dst))]

    fr = _load_frames(42)["test_s2"].reset_index(drop=True)
    te = _load_support_cache("and", 42, 3, 64, 400, False)["test_s2"]
    off = te["offsets"]; med = te["med"].astype(np.int64)
    da = te["da"].astype(np.int64); db = te["db"].astype(np.int64)
    key2i = {}
    for i in range(len(fr)):
        key2i[(str(fr["drug_a_id"].iloc[i]), str(fr["drug_b_id"].iloc[i]))] = i
        key2i[(str(fr["drug_b_id"].iloc[i]), str(fr["drug_a_id"].iloc[i]))] = i

    mask = np.zeros(kg.n_nodes, dtype=bool)
    for (A, B) in TARGETS:
        if (A, B) not in key2i:
            print(f"\n[{A} x {B}] not in seed42 test_s2, skipping"); continue
        i = key2i[(A, B)]
        s0, e0 = int(off[i]), int(off[i + 1])
        S = med[s0:e0]
        pos_t = {int(m): (int(da[s0 + j]), int(db[s0 + j])) for j, m in enumerate(S)}
        ui = kg.id_to_idx[A]; vi = kg.id_to_idx[B]
        mask[:] = False; mask[S] = True
        chords = []
        for u in S.tolist():
            a0, a1 = indptr[u], indptr[u + 1]
            nb = nbr[a0:a1]; rl = rel_s[a0:a1]
            selm = mask[nb] & (u < nb) & (~black_mask[rl])
            pu = pos_t[u]
            for w_, rid in zip(nb[selm].tolist(), rl[selm].tolist()):
                pw = pos_t[w_]
                if (pu[0] < pw[0] and pu[1] > pw[1]) or (pu[0] > pw[0] and pu[1] < pw[1]):
                    chords.append((1.0 / np.log1p(deg[u] + deg[w_]), u, w_, rel_vocab[rid]))
        chords.sort(reverse=True)
        print(f"\n============ {A} ({idx2name[ui]})  x  {B} ({idx2name[vi]})  "
              f"|S|={len(S)}, cross-arm chords={len(chords)} ============")
        for rank, (_, mi, mj, crel) in enumerate(chords[:3], 1):
            # orient so mi is the one closer to A (smaller d_a)
            if pos_t[mi][0] > pos_t[mj][0]:
                mi, mj = mj, mi
            pi, pj = pos_t[mi], pos_t[mj]
            armA = arm(ui, mi, pi[0])           # A -> m_i  (d_a_i hops)
            armB = arm(vi, mj, pj[1])           # B -> m_j  (d_b_j hops); print reversed
            # build printable chain: A -rel- [mid] -rel- m_i ==JUMP== m_j -rel- [mid] -rel- B
            left = f"{idx2name[ui]}"
            prev = ui
            for (nd, r) in armA:
                left += f"  --[{r}]-->  {idx2name[nd]}"
                prev = nd
            right = ""
            for (nd, r) in reversed(armB):
                right = f"{idx2name[nd]}  <--[{r}]--  " + right
            right += f"{idx2name[vi]}"
            print(f"\n  chord#{rank}:  {idx2name[mi]}(pos{pi}) =={crel}== {idx2name[mj]}(pos{pj})")
            print(f"    A: {left}")
            print(f"       {'':>{max(0,len(left)-2)}}  ==JUMP[{crel}]==>  (links to other arm)")
            print(f"    B: {idx2name[mj]}  ...  {right}")
            print(f"    => full walk:  {A}={idx2name[ui]} --> ...m_i={idx2name[mi]} "
                  f"=={crel}== m_j={idx2name[mj]}... --> {B}={idx2name[vi]}   (arms {pi[0]}|{pj[1]} hops)")


if __name__ == "__main__":
    main()
