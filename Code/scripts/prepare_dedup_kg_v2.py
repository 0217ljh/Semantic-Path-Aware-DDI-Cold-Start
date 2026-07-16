"""KG dedup pass v2: ontology-dedup GO/UBERON endpoints + merge 5 synonym relations.

Builds ON TOP of the protein-deduped KG (Code/data/KG/_merged_kg_dedup/, proteins
already collapsed to prot:<entrez>). Applies the 5 codex-approved cross-source
relation merges, each paired with deduping the OTHER endpoint node type by its
ontology id (reusing the PrimeKG node id as canonical, per codex):

  het:GpBP -> prime:bioprocess_protein   (BP nodes deduped by GO)
  het:GpMF -> prime:molfunc_protein       (MF nodes by GO)
  het:GpCC -> prime:cellcomp_protein      (CC nodes by GO)
  het:GiG  -> prime:protein_protein       (PPI; proteins already deduped; both undirected)
  het:AeG  -> prime:anatomy_protein_present (anatomy by UBERON; AeG endpoints SWAPPED
                                             so orientation matches prime: protein->anatomy)

Deferred (NOT in this build): DaG (disease DOID vs MONDO-cluster, not alignable),
GpPW (het pathway namespace unverified), drug->protein subtype multi-hot.

Invariants enforced (from codex review):
  * canonical endpoint node = the existing PrimeKG node id (no new go:/uberon: ids)
  * a node that ABSORBS a het partner gets source_kg="merged" (not stale single source)
  * an edge collapsed from >1 source KG gets source_kg="merged" (truthful provenance)
  * unmatched het-only / prime-only ontology terms kept untouched (no bad merges)
  * node table rebuilt from node2canon (no stale het rows left behind)
  * undirected edges order-normalized before dedup; self-loops dropped

Output (new dir, protein-dedup KG untouched): Code/data/KG/_merged_kg_dedup_v2/
Same column schema as the merged KG, so MergedKG.from_parquet works by repointing.

NOTE: this pass does NOT fix PD convergence fragmentation — the PD convergence node
types (disease, effect/phenotype) are exactly the non-alignable ones (cross-ontology
DOID/MONDO and UMLS/HPO), handled separately.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_dedup_kg_v2.py
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v2"

# het relation -> canonical (prime) relation
REL_RENAME = {
    "het:GpBP": "prime:bioprocess_protein",
    "het:GpMF": "prime:molfunc_protein",
    "het:GpCC": "prime:cellcomp_protein",
    "het:GiG": "prime:protein_protein",
    "het:AeG": "prime:anatomy_protein_present",
}
SWAP_RELATIONS = {"het:AeG"}  # endpoints swapped so orientation matches prime

# (het kind, prime kind, het-id GO/UBERON extractor, prime-id ontology extractor)
GO_RE = re.compile(r"GO:0*(\d+)")
UBERON_RE = re.compile(r"UBERON:0*(\d+)")


def build_endpoint_map(nd: pd.DataFrame) -> dict[str, str]:
    """het ontology node id -> canonical prime node id (same GO/UBERON), where it exists."""
    n2c: dict[str, str] = {}

    def align(het_kind, prime_kind, het_extract):
        het = nd[nd["kind"] == het_kind]
        prime = nd[nd["kind"] == prime_kind]
        prime_key2id = {str(i).split(":")[-1]: i for i in prime["id"]}  # ontology num -> prime id
        for nid in het["id"]:
            m = het_extract(str(nid))
            if not m:
                continue
            key = m.group(1)
            if key in prime_key2id:
                n2c[nid] = prime_key2id[key]
        return len(het), len(prime), sum(1 for nid in het["id"] if nid in n2c)

    stats = {}
    stats["BP"] = align("Biological Process", "biological_process", GO_RE.search)
    stats["MF"] = align("Molecular Function", "molecular_function", GO_RE.search)
    stats["CC"] = align("Cellular Component", "cellular_component", GO_RE.search)
    stats["anatomy"] = align("Anatomy", "anatomy", UBERON_RE.search)
    for k, (h, p, m) in stats.items():
        print(f"  endpoint {k:<8}: het {h:>6} / prime {p:>6} / aligned het->prime {m}")
    return n2c


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"input dedup KG: {len(nd)} nodes, {len(ed)} edges")

    # ---- endpoint node merge map (GO/UBERON) ----
    n2c = build_endpoint_map(nd)
    remapped_away = set(n2c)                    # het ontology nodes folded into prime
    absorbed = set(n2c.values())                # prime nodes that absorbed a het partner

    # ---- rebuild node table (drop folded het rows; mark absorbers merged) ----
    new_nodes = nd[~nd["id"].isin(remapped_away)].copy()
    new_nodes.loc[new_nodes["id"].isin(absorbed), "source_kg"] = "merged"
    assert new_nodes["id"].is_unique, "canonical node ids collided"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- GiG / protein_protein directedness invariant (codex) ----
    gig_dir = ed.loc[ed["relation"] == "het:GiG", "directed"]
    pp_dir = ed.loc[ed["relation"] == "prime:protein_protein", "directed"]
    assert (~gig_dir.astype(bool)).all() and (~pp_dir.astype(bool)).all(), \
        "GiG/protein_protein not all undirected — orientation unsafe to merge"

    # ---- AeG endpoint swap (orientation -> protein->anatomy, matching prime) ----
    aeg = ed["relation"].isin(SWAP_RELATIONS)
    s = ed["src"].to_numpy().copy()
    d = ed["dst"].to_numpy().copy()
    s_sw = np.where(aeg.to_numpy(), d, s)
    d_sw = np.where(aeg.to_numpy(), s, d)

    # ---- remap endpoints through n2c (GO/UBERON dedup) + rename relation ----
    n2c_get = lambda x: n2c.get(x, x)  # noqa: E731
    src = pd.Series(s_sw).map(n2c_get).to_numpy()
    dst = pd.Series(d_sw).map(n2c_get).to_numpy()
    relation = ed["relation"].map(lambda r: REL_RENAME.get(r, r)).to_numpy()
    directed = ed["directed"].to_numpy().astype(bool)
    skg = ed["source_kg"].astype(str).to_numpy()

    # endpoints must ALL exist in the rebuilt node table (no silent drops; v2
    # input has no dangling edges, so any miss is a remap bug -> hard fail)
    valid_nodes = set(kind_of)
    bad = ~(pd.Series(src).isin(valid_nodes).to_numpy() & pd.Series(dst).isin(valid_nodes).to_numpy())
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint (remap bug)"

    # ---- drop self-loops, order-normalize undirected ----
    loop = src != dst
    src, dst, relation, directed, skg = src[loop], dst[loop], relation[loop], directed[loop], skg[loop]
    und = ~directed
    u = np.where(und & (dst < src), dst, src)
    v = np.where(und & (dst < src), src, dst)

    # ---- dedup on (u,v,relation,directed); source_kg="merged" if multi-source ----
    df = pd.DataFrame({"u": u, "v": v, "r": relation, "dir": directed, "skg": skg})
    ucode, uuniq = pd.factorize(pd.concat([df["u"], df["v"]], ignore_index=True))
    nrow = len(df)
    uc = ucode[:nrow].astype(np.int64)
    vc = ucode[nrow:].astype(np.int64)
    rc, runiq = pd.factorize(df["r"])
    nn = len(uuniq)
    key = (uc * nn + vc) * (len(runiq) + 1) * 2 + rc.astype(np.int64) * 2 + df["dir"].to_numpy().astype(np.int64)
    df["key"] = key
    # which keys collapse >1 distinct source_kg
    ks = df[["key", "skg"]].drop_duplicates()
    multi = ks.groupby("key").size()
    multi_keys = set(multi.index[multi.values > 1])
    n_before = len(df)
    ded = df.drop_duplicates("key").copy()
    ded["source_kg"] = np.where(ded["key"].isin(multi_keys), "merged", ded["skg"])
    collapsed = n_before - len(ded)

    # ---- assemble output edges ----
    new_edges = pd.DataFrame({
        "src": ded["u"].values,
        "src_kind": [kind_of[x] for x in ded["u"].values],
        "dst": ded["v"].values,
        "dst_kind": [kind_of[x] for x in ded["v"].values],
        "relation": ded["r"].values,
        "source_kg": ded["source_kg"].values,
        "directed": ded["dir"].values,
    })

    # invariant: no het (pre-merge) relation labels survive — check BEFORE writing
    assert new_edges["relation"].isin(set(REL_RENAME)).sum() == 0, "het relations leaked into output"

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)

    # ---- report ----
    print(f"\n=== dedup v2 built ===")
    print(f"  nodes: {len(nd)} -> {len(new_nodes)}  (folded het ontology nodes: {len(remapped_away)}, absorbers->merged: {len(absorbed)})")
    print(f"  edges: {len(ed)} -> {len(new_edges)}  (collapsed by merge: {collapsed})")
    print(f"  merged-provenance edges: {int((new_edges['source_kg']=='merged').sum())}")
    print("\n  per merged-relation edge counts (after):")
    for canon in sorted(set(REL_RENAME.values())):
        before_h = int((ed["relation"].isin([k for k, vv in REL_RENAME.items() if vv == canon])).sum())
        before_p = int((ed["relation"] == canon).sum())
        after = int((new_edges["relation"] == canon).sum())
        print(f"    {canon:<32} het {before_h} + prime {before_p} = {before_h+before_p} -> {after}  (collapsed {before_h+before_p-after})")
    print(f"\n  wrote {OUT_DIR}/nodes__dedup.parquet")
    print(f"  wrote {OUT_DIR}/edges__dedup.parquet")


if __name__ == "__main__":
    main()
