"""E1a — Drug-pair shortest-path-length (SSPL) distribution on merged KG.

Supports i2 motivation: most drug pairs require ≥3 hops to connect, placing
them in the over-smoothing regime per Oono-Suzuki 2020.

Method: bidirectional BFS from each drug, max depth 5, masking direct DDI
edges (drug-drug). For each positive pair in train and test_s2, record min
hop count between drug_a and drug_b through non-drug intermediates.

To stay fast, we sample 2000 random pairs from each of train and test_s2.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")


PROJECT_ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
SPLITS = PROJECT_ROOT / "Code/data/KG/drugbank/splits/seed42"

SAMPLE_N = 2000
MAX_DEPTH = 5


def build_adj(edges: pd.DataFrame, drug_set: set) -> dict:
    """Build undirected adjacency map: id → set(neighbor_ids), excluding
    drug-drug edges."""
    adj = defaultdict(set)
    for s, d, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        if s in drug_set and d in drug_set:
            continue  # mask direct drug-drug
        adj[s].add(d)
        if not directed:
            adj[d].add(s)
        else:
            # Even directed edges should be traversable for "is there a path"
            # so we add reverse too (only for path-length analysis)
            adj[d].add(s)
    return adj


def bidir_bfs(src: str, dst: str, adj: dict, max_depth: int) -> int:
    """Return min hop count between src and dst, or -1 if not reachable within max_depth."""
    if src == dst:
        return 0
    front_s = {src: 0}
    front_d = {dst: 0}
    while front_s and front_d:
        if max(min(front_s.values()), min(front_d.values())) >= max_depth // 2 + max_depth % 2:
            # safety: don't expand too far
            pass
        # expand the smaller frontier
        if len(front_s) <= len(front_d):
            new_front = {}
            for node, depth in front_s.items():
                if depth >= max_depth:
                    continue
                for nb in adj.get(node, ()):
                    if nb in front_d:
                        return depth + 1 + front_d[nb]
                    if nb not in front_s:
                        new_front[nb] = depth + 1
            if not new_front:
                return -1
            front_s.update(new_front)
        else:
            new_front = {}
            for node, depth in front_d.items():
                if depth >= max_depth:
                    continue
                for nb in adj.get(node, ()):
                    if nb in front_s:
                        return depth + 1 + front_s[nb]
                    if nb not in front_d:
                        new_front[nb] = depth + 1
            if not new_front:
                return -1
            front_d.update(new_front)
        # stop if both fronts exceed half the max
        min_total = min(front_s.values()) + min(front_d.values())
        if min_total > max_depth:
            return -1
    return -1


def main() -> None:
    print("[E1a] Loading nodes / edges / splits ...")
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    print(f"[E1a] nodes={len(nodes)}  edges={len(edges)}  drugs={len(drug_set)}")

    print("[E1a] Building adjacency (drug-drug masked) ...")
    t0 = time.time()
    adj = build_adj(edges, drug_set)
    print(f"  built in {time.time()-t0:.1f}s; nodes-with-edges={len(adj)}")

    rng = np.random.default_rng(42)
    results = {}
    for split_name, fn in [("train", "train.parquet"), ("test_s2", "test_s2.parquet")]:
        df = pd.read_parquet(SPLITS / fn)[["drug_a_id", "drug_b_id"]]
        sample = df.sample(min(SAMPLE_N, len(df)), random_state=42).reset_index(drop=True)
        print(f"\n[E1a] computing SSPL for {len(sample)} sampled {split_name} pairs ...")
        sspls = []
        unreachable = 0
        t0 = time.time()
        for i, (a, b) in enumerate(zip(sample["drug_a_id"], sample["drug_b_id"])):
            d = bidir_bfs(a, b, adj, MAX_DEPTH)
            if d == -1:
                unreachable += 1
            else:
                sspls.append(d)
            if (i + 1) % 200 == 0:
                print(f"    progress {i+1}/{len(sample)} elapsed={time.time()-t0:.1f}s unreach={unreachable}")
        sspls = np.array(sspls)
        results[split_name] = {
            "n_sampled": int(len(sample)),
            "n_reachable": int(len(sspls)),
            "n_unreachable_within_5_hops": int(unreachable),
            "median": float(np.median(sspls)) if len(sspls) else None,
            "mean": float(sspls.mean()) if len(sspls) else None,
            "q25": float(np.quantile(sspls, 0.25)) if len(sspls) else None,
            "q75": float(np.quantile(sspls, 0.75)) if len(sspls) else None,
            "hist": {int(k): int(v) for k, v in zip(*np.unique(sspls, return_counts=True))} if len(sspls) else {},
        }
        print(f"  {split_name}: median SSPL = {results[split_name]['median']}  q25-q75 = [{results[split_name]['q25']}, {results[split_name]['q75']}]")
        print(f"  hist: {results[split_name]['hist']}  unreach within 5 hops: {unreachable}")

    # Summary
    print("\n[E1a i2 motivation verdict]")
    for s, v in results.items():
        if v["median"] is None:
            continue
        print(f"  {s}: median={v['median']}  fraction with SSPL ≥ 3: ", end="")
        n_total = v["n_reachable"]
        n_ge3 = sum(c for k, c in v["hist"].items() if k >= 3)
        print(f"{n_ge3/n_total*100:.1f}%")

    (OUT_DIR / "sspl_distribution.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[E1a] saved → sspl_distribution.json")


if __name__ == "__main__":
    main()
