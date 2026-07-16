"""Build a MULTI-HOT drug->neighbour relation index for the dedup KG.

Read-only on the KG; writes a new sidecar. The loader's `drug_rel`
({drug: {neighbour: ONE bucket}}, first-wins setdefault) is LOSSY on the dedup
KG: when a drug reaches the same (now-merged) node via several relations
(e.g. db:target bucket 0 AND het:CdG bucket 5), only one bucket survives. No
edges are lost — all relation rows are in the edge table — but the single-bucket
index can't represent them. This builds the faithful multi-hot version:

    for each distinct (drug, neighbour): the SET of REL_BUCKET buckets + the raw
    relation labels, over ALL drug-incident edges (both directions).

Output sidecar (co-located with the KG it indexes):
    <kg_dir>/drug_relation_multihot.parquet
columns: drug_id, neighbor_id, neighbor_kind, n_buckets, buckets (list[int]),
         bucket_mask (int bitfield), relations (list[str])

Does NOT modify retrieval.py / MergedKG (new artifact, new file only).

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_drug_relation_multihot.py
"""
from __future__ import annotations

import pandas as pd

from my_code.models.spmn_v1.retrieval import DRUG_KINDS, REL_BUCKET, REL_OTHER

KG_DIR = "Code/data/KG/_merged_kg_dedup_v2"
NODES = f"{KG_DIR}/nodes__dedup.parquet"
EDGES = f"{KG_DIR}/edges__dedup.parquet"
OUT = f"{KG_DIR}/drug_relation_multihot.parquet"


def main() -> None:
    nd = pd.read_parquet(NODES, columns=["id", "kind"])
    ed = pd.read_parquet(EDGES, columns=["src", "src_kind", "dst", "dst_kind", "relation"])
    kind_of = dict(zip(nd["id"], nd["kind"]))
    is_drug = {k for k in nd["kind"].unique() if k in DRUG_KINDS}

    # orient every drug-incident edge as (drug, neighbour); keep both directions
    s_is_drug = ed["src_kind"].isin(is_drug)
    d_is_drug = ed["dst_kind"].isin(is_drug)
    fwd = ed[s_is_drug][["src", "dst", "relation"]].rename(columns={"src": "drug", "dst": "nb"})
    bwd = ed[d_is_drug][["dst", "src", "relation"]].rename(columns={"dst": "drug", "src": "nb"})
    di = pd.concat([fwd, bwd], ignore_index=True).drop_duplicates(["drug", "nb", "relation"])
    di["bucket"] = di["relation"].map(lambda r: REL_BUCKET.get(r, REL_OTHER))

    g = di.groupby(["drug", "nb"]).agg(
        buckets=("bucket", lambda x: sorted(set(x))),
        relations=("relation", lambda x: sorted(set(x))),
    ).reset_index()
    g["n_buckets"] = g["buckets"].map(len)
    g["bucket_mask"] = g["buckets"].map(lambda bs: sum(1 << b for b in bs))
    g["neighbor_kind"] = g["nb"].map(kind_of)
    g = g.rename(columns={"drug": "drug_id", "nb": "neighbor_id"})
    g = g[["drug_id", "neighbor_id", "neighbor_kind", "n_buckets", "buckets",
           "bucket_mask", "relations"]]
    g.to_parquet(OUT, index=False)

    # ---- report ----
    print(f"distinct (drug, neighbour) pairs : {len(g)}")
    multi = g[g["n_buckets"] > 1]
    print(f"pairs with >1 bucket (drug_rel WOULD lose info): {len(multi)} "
          f"({len(multi)/len(g)*100:.1f}%)")
    # specifically: pairs that have db:target (bucket 0) AND a non-0 bucket -> the
    # exact loss case where the PK/PD target bucket could be dropped
    prot = g[g["neighbor_id"].astype(str).str.startswith("prot:")]
    has_target_plus = prot[prot["buckets"].map(lambda b: 0 in b and len(b) > 1)]
    print(f"drug->protein pairs with db:target(0) + another bucket: {len(has_target_plus)} "
          "(these are where the target bucket was being silently dropped)")
    print("\n--- example multi-bucket drug->protein pairs (now fully preserved) ---")
    for _, r in has_target_plus.head(6).iterrows():
        print(f"  {r['drug_id']} -> {r['neighbor_id']}  buckets={r['buckets']}  {r['relations']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
