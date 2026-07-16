"""exp3 C2 step 1 — compute mediator-set features per pair.

For each (drug_a, drug_b) pair:
- M2 = shared 2-hop intermediates: {m : drug_a—m AND m—drug_b in KG}
  (m's kind ∈ {Protein, Gene, Disease, Side Effect, Pathway, Phenotype, Anatomy,
   gene/protein, effect/phenotype, disease, anatomy, pathway, biological_process, ...})
- |M2| = mediator count
- H_kind: kind-distribution entropy (normalized by log K)
- H_rel: 2-hop relation-template entropy (template = (r1, r2))
- H_emb: k-means cluster entropy on mediator PubMedBERT embeddings (k=4)

Output: pair_features_c2.parquet (joined with previous pair_features.parquet)
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent
CACHE = ROOT / "Notes/Log/insight-discovery/exp2-0513-test-insight/_cache_features"

# Canonical kind taxonomy: collapse near-synonyms across source KGs
KIND_CANON = {
    "Drug": "Drug", "drug": "Drug",
    "Protein": "Protein", "gene/protein": "Protein",
    "Gene": "Gene",
    "Side Effect": "SideEffect", "effect/phenotype": "Phenotype",
    "Disease": "Disease", "disease": "Disease",
    "Pathway": "Pathway", "pathway": "Pathway",
    "Anatomy": "Anatomy", "anatomy": "Anatomy",
    "biological_process": "BioProcess", "Biological Process": "BioProcess",
    "molecular_function": "MolFunction", "Molecular Function": "MolFunction",
    "cellular_component": "CellComp",
    "Pharmacologic Class": "PharmClass",
    "Symptom": "Symptom",
}

# Mediator-relevant kinds (exclude Drug and rare cross-references)
MEDIATOR_KINDS = [
    "Protein", "Gene", "SideEffect", "Phenotype", "Disease",
    "Pathway", "Anatomy", "BioProcess", "MolFunction",
]


def entropy_norm(counts: list[int], K: int) -> float:
    """Normalized entropy: 0 if singleton, 1 if uniform over K bins."""
    total = sum(counts)
    if total <= 1 or K <= 1:
        return 0.0
    p = np.array(counts, dtype=np.float64) / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(H / np.log(K))


def main():
    t0 = time.time()
    print("[load] pair_features.parquet ...")
    df = pd.read_parquet(OUT_DIR / "pair_features.parquet")
    print(f"  pairs={len(df)}")

    print("[load] edges parquet ...")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    print(f"  edges={len(edges)} nodes={len(nodes)}")

    # Build node -> canonical kind map
    nodes["kind_canon"] = nodes["kind"].map(KIND_CANON).fillna("Other")
    id2kind = dict(zip(nodes["id"], nodes["kind_canon"]))

    # Build drug -> set(neighbors) and drug -> set((nbr, rel)) for relation-aware
    drug_set = set(df["drug_a_id"]) | set(df["drug_b_id"])
    print(f"[main] unique S2 drugs: {len(drug_set)}")

    # Drug-incident edges (both directions)
    e1 = edges[edges["src"].isin(drug_set)][["src", "dst", "relation"]].rename(columns={"src": "d", "dst": "m", "relation": "r"})
    e2 = edges[edges["dst"].isin(drug_set)][["dst", "src", "relation"]].rename(columns={"dst": "d", "src": "m", "relation": "r"})
    e1["dir"] = "out"
    e2["dir"] = "in"
    drug_edges = pd.concat([e1, e2], ignore_index=True)
    # Filter: m must be mediator-relevant kind, not Drug
    drug_edges["m_kind"] = drug_edges["m"].map(id2kind)
    drug_edges = drug_edges[drug_edges["m_kind"].isin(MEDIATOR_KINDS)].copy()
    print(f"[main] drug-mediator edges: {len(drug_edges)}")

    # Per drug, build: m -> list of (r, dir) tuples
    print("[main] building per-drug mediator maps ...")
    drug_to_mediators: dict[str, dict[str, list]] = {}
    for d, grp in drug_edges.groupby("d"):
        m_to_rels = {}
        for m, sub in grp.groupby("m"):
            m_to_rels[m] = list(zip(sub["r"], sub["dir"]))
        drug_to_mediators[d] = m_to_rels
    print(f"  drugs with mediator data: {len(drug_to_mediators)}")

    # Optional: mediator embedding (PubMedBERT [CLS]) for H_emb
    print("[main] loading PubMedBERT cache for mediators ...")
    X = np.load(CACHE / "pubmedbert_real.npz")["X"]
    id2idx = {nid: i for i, nid in enumerate(nodes["id"])}

    # Compute features pair-by-pair (vectorize-friendly only for kind; H_rel and H_emb are per-pair)
    M2_count = np.zeros(len(df), dtype=np.int32)
    H_kind = np.full(len(df), np.nan, dtype=np.float32)
    H_rel = np.full(len(df), np.nan, dtype=np.float32)
    H_emb = np.full(len(df), np.nan, dtype=np.float32)
    n_kinds_present = np.zeros(len(df), dtype=np.int32)

    try:
        from sklearn.cluster import KMeans
        sk_ok = True
    except ImportError:
        sk_ok = False
    print(f"[main] sklearn available: {sk_ok}")

    K_kind = len(MEDIATOR_KINDS)
    K_cluster = 4
    print("[main] iterating pairs (this is the slow part) ...")
    t_iter = time.time()
    for i, (a, b) in enumerate(zip(df["drug_a_id"], df["drug_b_id"])):
        ma = drug_to_mediators.get(a)
        mb = drug_to_mediators.get(b)
        if not ma or not mb:
            continue
        shared = set(ma.keys()) & set(mb.keys())
        if not shared:
            continue
        M2_count[i] = len(shared)

        # H_kind
        kinds = [id2kind.get(m) for m in shared]
        kinds = [k for k in kinds if k in MEDIATOR_KINDS]
        cnt = Counter(kinds)
        n_kinds_present[i] = len(cnt)
        H_kind[i] = entropy_norm(list(cnt.values()), K_kind)

        # H_rel: template = (r_a→m, r_b→m), use first relation each side
        templates = []
        for m in shared:
            ra = ma[m][0][0]  # first relation
            rb = mb[m][0][0]
            templates.append((ra, rb))
        tcnt = Counter(templates)
        # Use log(|template space|) as denom — use observed for normalization
        H_rel[i] = entropy_norm(list(tcnt.values()), max(len(tcnt), 2))

        # H_emb: kmeans on mediator embeddings if enough mediators
        if sk_ok and len(shared) >= K_cluster:
            idxs = [id2idx[m] for m in shared if m in id2idx]
            if len(idxs) >= K_cluster:
                emb = X[idxs]
                try:
                    km = KMeans(n_clusters=min(K_cluster, len(idxs)), n_init=3, random_state=0).fit(emb)
                    lcnt = Counter(km.labels_)
                    H_emb[i] = entropy_norm(list(lcnt.values()), K_cluster)
                except Exception:
                    pass

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t_iter
            eta = elapsed * (len(df) - i - 1) / (i + 1)
            print(f"  {i+1}/{len(df)} elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    df["M2_count"] = M2_count
    df["H_kind"] = H_kind
    df["H_rel"] = H_rel
    df["H_emb"] = H_emb
    df["n_kinds"] = n_kinds_present

    # Summary
    print("\n[summary] M2 stats:")
    print(f"  pairs with M2>0: {int((df['M2_count']>0).sum())} ({(df['M2_count']>0).mean()*100:.1f}%)")
    print(f"  M2_count quantiles: q50={df['M2_count'].quantile(0.5):.0f} q75={df['M2_count'].quantile(0.75):.0f} q90={df['M2_count'].quantile(0.9):.0f} max={df['M2_count'].max()}")
    print(f"  H_kind (where M2>1) mean={df.loc[df['M2_count']>1, 'H_kind'].mean():.3f}")
    print(f"  H_rel  (where M2>1) mean={df.loc[df['M2_count']>1, 'H_rel'].mean():.3f}")
    print(f"  H_emb  (where M2>3) mean={df.loc[df['M2_count']>=4, 'H_emb'].mean():.3f}")

    out = OUT_DIR / "pair_features_c2.parquet"
    df.to_parquet(out)
    print(f"\n[main] saved: {out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
