"""Analyze whether |A_tau(a,b)| (the size of middle common neighbor set
restricted to anchor type tau) is discriminative between true DDI pairs
and random pairs, on the cold-start test_s2 split.

Outputs:
- |A_tau| size distribution for POS vs NEG pairs, per anchor group
- Single-feature AUC of |A_tau| size as a DDI predictor
- Per-anchor-degree analysis (does removing top-degree anchors help?)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KIND_GROUPS = {
    "protein_gene": ["Gene", "gene/protein", "Protein"],
    "side_effect": ["Side Effect", "effect/phenotype", "Symptom"],
    "disease": ["Disease", "disease"],
}
KIND_TO_GROUP = {k: g for g, kinds in KIND_GROUPS.items() for k in kinds}


def main() -> None:
    root = Path("/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start")
    nodes = pd.read_parquet(root / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    edges = pd.read_parquet(root / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    nodes["group"] = nodes["kind"].map(KIND_TO_GROUP).fillna("other")
    id_to_group = dict(zip(nodes["id"], nodes["group"]))
    drug_ids = set(nodes[nodes["kind"].isin(["drug", "Drug", "Compound"])]["id"].tolist())

    # Build drug -> {group: set of anchor nodes}
    nbrs: dict[str, dict[str, set[str]]] = {g: defaultdict(set) for g in ["protein_gene", "side_effect", "disease"]}
    for src, dst in zip(edges["src"].values, edges["dst"].values):
        s_drug = src in drug_ids
        d_drug = dst in drug_ids
        if s_drug ^ d_drug:
            drug = src if s_drug else dst
            other = dst if s_drug else src
            g = id_to_group.get(other)
            if g in nbrs:
                nbrs[g][drug].add(other)

    # Degree per anchor (how many drugs each anchor connects to)
    anchor_degree: dict[str, dict[str, int]] = {}
    for g, dn in nbrs.items():
        deg: dict[str, int] = defaultdict(int)
        for drug, anchors in dn.items():
            for m in anchors:
                deg[m] += 1
        anchor_degree[g] = dict(deg)

    test = pd.read_parquet(root / "Code/data/KG/drugbank/splits/seed42/test_s2.parquet")
    test_pos = test[["drug_a_id", "drug_b_id"]].head(5000).values
    drugs_in_test = list(set(test["drug_a_id"].tolist() + test["drug_b_id"].tolist()))
    print(f"test_s2 pos pairs sampled: {len(test_pos)}; unique drugs: {len(drugs_in_test)}", flush=True)

    rng = np.random.default_rng(42)
    pos_set = set(map(tuple, test_pos)) | {(b, a) for a, b in test_pos}
    neg_pairs: list[tuple[str, str]] = []
    while len(neg_pairs) < len(test_pos):
        a = rng.choice(drugs_in_test)
        b = rng.choice(drugs_in_test)
        if a != b and (a, b) not in pos_set:
            neg_pairs.append((a, b))

    def asize(a: str, b: str, group: str, exclude_hubs: int = 0) -> int:
        ng = nbrs[group]
        sa = ng.get(a, set())
        sb = ng.get(b, set())
        inter = sa & sb
        if exclude_hubs > 0:
            deg = anchor_degree[group]
            inter = {m for m in inter if deg.get(m, 0) < exclude_hubs}
        return len(inter)

    print("\n=== |A_tau(a,b)| distribution: POS vs NEG, sized 5000 each ===")
    print(f'{"group":12s} {"tag":4s}  med  mean  p10  p90  p99  max  nonzero')
    for g in ["protein_gene", "side_effect", "disease"]:
        sizes_pos = np.array([asize(a, b, g) for a, b in test_pos])
        sizes_neg = np.array([asize(a, b, g) for a, b in neg_pairs])
        for tag, s in [("pos", sizes_pos), ("neg", sizes_neg)]:
            print(
                f"{g:12s} {tag:4s}  "
                f"{np.median(s):>3.0f}  {s.mean():>4.1f}  "
                f"{np.quantile(s, 0.10):>3.0f}  {np.quantile(s, 0.90):>4.0f}  "
                f"{np.quantile(s, 0.99):>4.0f}  {s.max():>4.0f}  {int((s > 0).sum()):>5}/{len(s)}"
            )
        y = np.concatenate([np.ones_like(sizes_pos), np.zeros_like(sizes_neg)])
        s_all = np.concatenate([sizes_pos, sizes_neg]).astype(float)
        auc = roc_auc_score(y, s_all)
        print(f"  -> |A_{g}| size single-feature AUC: {auc:.4f}")
        print()

    print("\n=== After excluding top-degree (hub) anchors ===")
    print(f'{"group":12s} {"thr":4s}  pos_med  neg_med   AUC')
    for g in ["protein_gene", "side_effect", "disease"]:
        for thr in [None, 100, 50, 20, 10]:
            ex = thr if thr is not None else 0
            sizes_pos = np.array([asize(a, b, g, exclude_hubs=ex) for a, b in test_pos])
            sizes_neg = np.array([asize(a, b, g, exclude_hubs=ex) for a, b in neg_pairs])
            y = np.concatenate([np.ones_like(sizes_pos), np.zeros_like(sizes_neg)])
            s_all = np.concatenate([sizes_pos, sizes_neg]).astype(float)
            auc = roc_auc_score(y, s_all)
            thr_str = "none" if thr is None else f"<{thr}"
            print(f"{g:12s} {thr_str:4s}  {np.median(sizes_pos):>6.0f}  {np.median(sizes_neg):>6.0f}   {auc:.4f}")
        print()


if __name__ == "__main__":
    main()
