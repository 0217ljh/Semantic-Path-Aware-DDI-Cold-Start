"""Alignment target k_u — pooled NON-DDI KG-neighbor PubMedBERT embeddings per drug.

Edge-independent alignment (codex r22): k_u is the drug's KG-neighborhood semantic
summary, built ONLY from non-DDI biomedical neighbors (targets/enzymes/transporters/
pathways/proteins/etc.) — NEVER DDI edges, never meeting nodes, never test-leaking.

For each drug: collect its 1-hop non-drug KG neighbors (from the merged KG, which has
NO DDI edges — the merged KG is DDI-masked, verified in the i2 leakage audit), look up
their cached PubMedBERT(name) embeddings, mean-pool -> k_u (768d).

Output: Code/data/_cache/kg_neighbor_target_pubmedbert.npz
  drug_ids (list), k (N_drug x 768), n_neighbors (per drug)
This is the InfoNCE alignment TARGET; the molecular embedding will be projected to match it.
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
OUT = PROJECT_ROOT / "Code/data/_cache/kg_neighbor_target_pubmedbert.npz"


def main():
    print("[k_u] loading PubMedBERT embeddings + KG ...", flush=True)
    obj = torch.load(EMB_PT, map_location="cpu", weights_only=False)
    nid2row = {str(n): i for i, n in enumerate(obj["node_ids"])}
    emb = obj["embeddings"].numpy().astype(np.float32)  # [178029, 768]

    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)  # merged KG = NO DDI edges (DDI-masked, verified)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])

    # 1-hop non-drug neighbors per drug (these are non-DDI biomedical facts)
    print("[k_u] collecting non-drug 1-hop neighbors per drug ...", flush=True)
    nbrs: dict[str, set] = defaultdict(set)
    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        s_drug = src in drug_set; d_drug = dst in drug_set
        if s_drug and not d_drug:
            nbrs[src].add(dst)
        if d_drug and not s_drug and not directed:
            nbrs[dst].add(src)

    drug_ids = sorted(drug_set)
    K = np.zeros((len(drug_ids), 768), dtype=np.float32)
    ncount = np.zeros(len(drug_ids), dtype=np.int64)
    for i, d in enumerate(drug_ids):
        rows = [nid2row[n] for n in nbrs.get(d, ()) if n in nid2row]
        if rows:
            K[i] = emb[rows].mean(axis=0)
            ncount[i] = len(rows)
    n_with = int((ncount > 0).sum())
    print(f"[k_u] drugs={len(drug_ids)}, with >=1 KG neighbor={n_with}, "
          f"mean neighbors={ncount[ncount>0].mean():.1f}", flush=True)

    np.savez(OUT, drug_ids=np.array(drug_ids), k=K, n_neighbors=ncount)
    print(f"[k_u] saved -> {OUT}  (k shape {K.shape})", flush=True)


if __name__ == "__main__":
    main()
