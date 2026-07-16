"""Precompute 22-dim shared-mediator (meeting-node) features for MNAH (v2 Stage 1).

Same protocol as E7 (08_meeting_node_lr.py):
- KG-ONLY graph (no DDI edges)
- 11 consolidated node kinds x {1-hop, 2-hop} shared-mediator counts
- log1p transform
- Pair canonicalization: sort (drug_a, drug_b) lexicographically

Computes features for all (drug_a, drug_b) pairs in:
- train.parquet (with paired epoch_0 negatives)
- val_s2.parquet (with negatives/val_s2.parquet)
- test_s2.parquet (with negatives/test_s2.parquet)

ALSO computes features over a CARTESIAN grid of all drug pairs that may appear
in epoch-regenerated training negatives. We cache by canonical pair (a,b) so
the trainer can look up arbitrary pairs at runtime.

Output: parquet at Code/data/_cache/meet_feat_{backbone_kg_source}_{seed}_kgonly_v1.parquet
  columns: drug_a_id, drug_b_id (canonical sorted), feat_0...feat_21
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
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
    raise FileNotFoundError(f"Project root not found from {cur}")


PROJECT_ROOT = _find_project_root()

NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
SPLITS = PROJECT_ROOT / "Code/data/KG/drugbank/splits/seed42"

# Same as E7 — 11 consolidated node kinds.
KIND_GROUPS = {
    "protein_gene": ["Gene", "gene/protein", "Protein"],
    "pathway": ["Pathway", "pathway"],
    "side_effect": ["Side Effect", "effect/phenotype", "Symptom"],
    "disease": ["Disease", "disease"],
    "anatomy": ["Anatomy", "anatomy"],
    "compound": ["Compound"],
    "biological_process": ["Biological Process", "biological_process"],
    "molecular_function": ["Molecular Function", "molecular_function"],
    "cellular_component": ["Cellular Component", "cellular_component"],
    "pharmacologic_class": ["Pharmacologic Class"],
    "exposure": ["exposure"],
}
KIND_ORDER = list(KIND_GROUPS.keys())
N_KINDS = len(KIND_ORDER)  # 11
N_FEAT = 2 * N_KINDS         # 22

KIND_TO_GROUP: dict[str, str] = {}
for grp, kinds in KIND_GROUPS.items():
    for k in kinds:
        KIND_TO_GROUP[k] = grp


def build_neighbor_sets(
    edges: pd.DataFrame, id2kind: dict, drug_set: set
) -> tuple[dict, dict]:
    """Return (n1, n2): drug_id -> {(node_id, group)} for 1-hop and 2-hop reach."""
    print("[mnah-pre] Building 1-hop neighbor sets ...", flush=True)
    n1: dict[str, set[tuple[str, str]]] = defaultdict(set)
    fwd: dict[str, set[str]] = defaultdict(set)
    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        fwd[src].add(dst)
        if not directed:
            fwd[dst].add(src)
        src_is_drug = src in drug_set
        dst_is_drug = dst in drug_set
        if src_is_drug and not dst_is_drug:
            n1[src].add((dst, KIND_TO_GROUP.get(id2kind.get(dst, ""), "other")))
        if dst_is_drug and not src_is_drug and not directed:
            n1[dst].add((src, KIND_TO_GROUP.get(id2kind.get(src, ""), "other")))
    print(f"[mnah-pre] drugs with >=1 1-hop non-drug neighbor: {sum(1 for d in drug_set if d in n1)}")

    print("[mnah-pre] Building 2-hop neighbor sets ...", flush=True)
    n2: dict[str, set[tuple[str, str]]] = defaultdict(set)
    t0 = time.time()
    for i, drug in enumerate(drug_set):
        for mid, _ in n1.get(drug, ()):
            for term in fwd.get(mid, ()):
                if term == drug or term in drug_set:
                    continue
                n2[drug].add((term, KIND_TO_GROUP.get(id2kind.get(term, ""), "other")))
        if (i + 1) % 1000 == 0:
            print(f"    progress: {i+1}/{len(drug_set)}  elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[mnah-pre] drugs with >=1 2-hop non-drug node: {sum(1 for d in drug_set if d in n2)}  time={time.time()-t0:.1f}s")
    return n1, n2


def featurize_pairs(pairs: list[tuple[str, str]], n1: dict, n2: dict) -> np.ndarray:
    """Compute log1p count features for (drug_a, drug_b) tuples.
    Assumes pairs are ALREADY canonicalized (a <= b lex).
    Returns X of shape (N, 22).
    """
    X = np.zeros((len(pairs), N_FEAT), dtype=np.float32)
    for i, (da, db) in enumerate(pairs):
        s1 = n1.get(da, set()) & n1.get(db, set())
        s2 = n2.get(da, set()) & n2.get(db, set())
        c1: dict[str, int] = defaultdict(int)
        c2: dict[str, int] = defaultdict(int)
        for _, g in s1:
            c1[g] += 1
        for _, g in s2:
            c2[g] += 1
        for j, grp in enumerate(KIND_ORDER):
            X[i, j] = np.log1p(c1[grp])
            X[i, N_KINDS + j] = np.log1p(c2[grp])
    return X


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Sort (a, b) lexicographically so (a,b) and (b,a) map to same cache row."""
    return (a, b) if a <= b else (b, a)


def main():
    out_dir = PROJECT_ROOT / "Code/data/_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "meet_feat_drugbank_seed42_kgonly_v1.parquet"

    if out_path.exists():
        print(f"[mnah-pre] cache EXISTS at {out_path}")
        df = pd.read_parquet(out_path)
        print(f"  shape: {df.shape}")
        print(f"  columns: {list(df.columns)[:5]} ... {list(df.columns)[-3:]}")
        print("  (delete the file to recompute)")
        return

    print("[mnah-pre] Loading nodes/edges/splits ...", flush=True)
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    print(f"[mnah-pre] nodes={len(nodes)}  edges={len(edges)}  drugs={len(drug_set)}")

    n1, n2 = build_neighbor_sets(edges, id2kind, drug_set)

    # Collect all unique canonical pairs we'll ever score:
    #   - train pos + train negatives (epoch_0)
    #   - val_s2 pos + neg
    #   - test_s2 pos + neg
    #   - Cartesian product of all drugs that appear in any split (for
    #     epoch-regenerated negatives at training time)
    all_drugs: set[str] = set()
    for sub in ["train.parquet", "val_s2.parquet", "test_s2.parquet"]:
        df = pd.read_parquet(SPLITS / sub)[["drug_a_id", "drug_b_id"]]
        all_drugs |= set(df["drug_a_id"].astype(str))
        all_drugs |= set(df["drug_b_id"].astype(str))
    for sub in ["negatives/val_s2.parquet", "negatives/test_s2.parquet"]:
        try:
            df = pd.read_parquet(SPLITS / sub)[["drug_a_id", "drug_b_id"]]
            all_drugs |= set(df["drug_a_id"].astype(str))
            all_drugs |= set(df["drug_b_id"].astype(str))
        except Exception as exc:
            print(f"[mnah-pre] skip {sub}: {exc}")
    # Train negatives can be epoch-regenerated; cache the actually-seen
    # epoch_0 set conservatively — others can be looked up at runtime
    # (we'll fall back to compute on miss; but cache the C(N,2) for the
    # 800-drug regime is feasible: 320k entries).
    try:
        df = pd.read_parquet(SPLITS / "train_negatives/epoch_0.parquet")[["drug_a_id", "drug_b_id"]]
        all_drugs |= set(df["drug_a_id"].astype(str))
        all_drugs |= set(df["drug_b_id"].astype(str))
    except Exception:
        pass

    all_drugs = sorted(all_drugs)
    n_drugs = len(all_drugs)
    print(f"[mnah-pre] |unique drugs across splits| = {n_drugs}")

    # Build full upper-triangular Cartesian (canonical pairs a<=b).
    print(f"[mnah-pre] generating C({n_drugs}, 2) + diag = {n_drugs*(n_drugs+1)//2} canonical pairs ...", flush=True)
    pairs: list[tuple[str, str]] = []
    for i in range(n_drugs):
        a = all_drugs[i]
        for j in range(i, n_drugs):
            b = all_drugs[j]
            pairs.append((a, b))
    print(f"[mnah-pre] total pairs: {len(pairs)}", flush=True)

    print("[mnah-pre] Featurizing all pairs (may take a few minutes) ...", flush=True)
    t0 = time.time()
    BATCH = 50000
    feats = []
    for s in range(0, len(pairs), BATCH):
        feats.append(featurize_pairs(pairs[s:s+BATCH], n1, n2))
        if (s // BATCH) % 5 == 0:
            print(f"  done {s}/{len(pairs)}  elapsed={time.time()-t0:.1f}s", flush=True)
    X = np.vstack(feats)
    print(f"[mnah-pre] featurization done. shape={X.shape}  time={time.time()-t0:.1f}s", flush=True)

    col_names = [f"f_1hop_{g}" for g in KIND_ORDER] + [f"f_2hop_{g}" for g in KIND_ORDER]
    df = pd.DataFrame(X, columns=col_names)
    df.insert(0, "drug_a_id", [p[0] for p in pairs])
    df.insert(1, "drug_b_id", [p[1] for p in pairs])

    print(f"[mnah-pre] writing -> {out_path}", flush=True)
    df.to_parquet(out_path, index=False)
    print(f"[mnah-pre] wrote {len(df)} rows, {len(col_names)} feature cols + 2 id cols")


if __name__ == "__main__":
    main()
