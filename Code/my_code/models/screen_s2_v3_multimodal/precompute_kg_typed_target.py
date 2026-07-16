"""E-frag prerequisite — RELATION-TYPED KG alignment target k_typed (codex 019e6747).

The plain k_u (mean of ALL non-DDI neighbors) makes molecular reconstruct what EmerGNN already
pools. Instead, build a TYPED target: bucket each drug's non-DDI 1-hop neighbors by node kind,
mean-pool PubMedBERT(name) WITHIN each bucket, concat -> a typed semantic KG view. The shared
molecular branch aligns to this; the residual branch stays KG-orthogonal.

Leakage-safe: merged KG is DDI-masked (same as k_u).

Output: Code/data/_cache/kg_typed_target_pubmedbert.npz
  drug_ids (N,) str | k_typed (N, B*768) float32 | buckets (B,) str | n_per_bucket (N,B) int
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
EMB_PT = PROJECT_ROOT / "Code/data/KG/_merged_kg/_cache/screen1_tag_init/d_name_only__pubmedbert.pt"
OUT = PROJECT_ROOT / "Code/data/_cache/kg_typed_target_pubmedbert.npz"

# node kind (lowercased substring) -> bucket index
BUCKETS = ["protein_gene", "pathway", "molfunc_bioproc", "disease", "phenotype",
           "side_effect", "anatomy", "other"]


def _bucket(kind: str) -> int:
    k = str(kind).lower()
    if "gene" in k or "protein" in k:
        return 0
    if "pathway" in k:
        return 1
    if "molecular_function" in k or "molecular function" in k or "biological" in k or "cellular" in k:
        return 2
    if "disease" in k:
        return 3
    if "phenotype" in k or "symptom" in k:
        return 4
    if "side effect" in k:
        return 5
    if "anatomy" in k:
        return 6
    return 7


def main() -> None:
    print("[ktyped] loading PubMedBERT + KG ...", flush=True)
    obj = torch.load(EMB_PT, map_location="cpu", weights_only=False)
    nid2row = {str(n): i for i, n in enumerate(obj["node_ids"])}
    emb = obj["embeddings"].numpy().astype(np.float32)
    D = emb.shape[1]
    B = len(BUCKETS)

    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    node_bucket = {nid: _bucket(k) for nid, k in id2kind.items()}

    print("[ktyped] collecting typed 1-hop neighbors ...", flush=True)
    nbrs: dict[str, list] = defaultdict(list)
    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        if src in drug_set and dst not in drug_set:
            nbrs[src].append(dst)
        if dst in drug_set and src not in drug_set and not directed:
            nbrs[dst].append(src)

    drug_ids = sorted(drug_set)
    k_typed = np.zeros((len(drug_ids), B * D), dtype=np.float32)
    n_per = np.zeros((len(drug_ids), B), dtype=np.int64)
    for i, d in enumerate(drug_ids):
        sums = np.zeros((B, D), dtype=np.float32)
        cnts = np.zeros(B, dtype=np.int64)
        for n in nbrs.get(d, ()):
            r = nid2row.get(n)
            if r is None:
                continue
            b = node_bucket.get(n, 7)
            sums[b] += emb[r]; cnts[b] += 1
        for b in range(B):
            if cnts[b] > 0:
                k_typed[i, b * D:(b + 1) * D] = sums[b] / cnts[b]
        n_per[i] = cnts
    cov = int((n_per.sum(axis=1) > 0).sum())
    print(f"[ktyped] drugs={len(drug_ids)} with_any_neighbor={cov} dim={B*D} "
          f"bucket_coverage={(n_per>0).mean(axis=0).round(2).tolist()}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, drug_ids=np.array(drug_ids), k_typed=k_typed,
             buckets=np.array(BUCKETS), n_per_bucket=n_per)
    print(f"[ktyped] saved -> {OUT} (k_typed {k_typed.shape})", flush=True)


if __name__ == "__main__":
    main()
