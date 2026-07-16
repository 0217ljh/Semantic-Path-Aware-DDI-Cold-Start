"""exp3 C4 step 1 — compute per-pair features on ColdDDI seed42 test_s2.

For each (drug_a, drug_b) pair (positives + negatives = 108,738 total):
- S_st: Tanimoto on Morgan fingerprint (radius=2, 1024 bits) — structure modality
- S_tx: cosine on PubMedBERT [CLS] of drug name — text modality (per-dim z-scored across drugs)
- S_kg: Jaccard of 1-hop neighbors in merged KG, excluding Drug nodes — KG modality
- per-pair NLL from E3 PubMedBERT-init GCN (cached predictions)

Output: pair_features.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")


ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent
CACHE = ROOT / "Notes/Log/insight-discovery/exp2-0513-test-insight/_cache_features"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_pairs_and_pred():
    """Load test_s2 positives + negatives (label 0/1) and model prediction."""
    pos = pd.read_parquet(ROOT / "Code/data/KG/drugbank/splits/seed42/test_s2.parquet")
    pos = pos[["drug_a_id", "drug_b_id"]].copy()
    pos["label"] = 1.0
    neg = pd.read_parquet(ROOT / "Code/data/KG/drugbank/splits/seed42/negatives/test_s2.parquet")
    neg = neg[["drug_a_id", "drug_b_id"]].copy()
    neg["label"] = 0.0
    df = pd.concat([pos, neg], ignore_index=True)
    print(f"[load] pairs={len(df)} (pos={len(pos)}, neg={len(neg)})")

    # Use E3 cached prediction (PubMedBERT + PCA proj, seed42)
    cache = np.load(CACHE / "pred_real_pubmedbert_pca_seed42.npz")
    pred_logit = cache["pred"]
    y = cache["y"]
    assert len(pred_logit) == len(df), f"pred {len(pred_logit)} vs pairs {len(df)}"
    # Verify y alignment
    diff = np.abs(y - df["label"].to_numpy()).sum()
    assert diff < 1, f"label mismatch sum={diff}"
    df["pred_logit"] = pred_logit.astype(np.float32)
    # NLL: -log p(y|x) ; sigmoid + safe clip
    p = 1.0 / (1.0 + np.exp(-pred_logit.clip(-30, 30)))
    eps = 1e-7
    df["pred_p"] = p.astype(np.float32)
    df["nll"] = -(df["label"] * np.log(p + eps) + (1 - df["label"]) * np.log(1 - p + eps)).astype(np.float32)
    print(f"[load] nll min={df['nll'].min():.4f} mean={df['nll'].mean():.4f} max={df['nll'].max():.4f}")
    return df


def load_drug_smiles() -> dict[str, str]:
    de = pd.read_csv(ROOT / "Code/data/KG/drugbank/enriched/drugs_enriched.csv", usecols=["drugbank_id", "smiles"])
    return dict(zip(de["drugbank_id"], de["smiles"]))


def compute_morgan_fps(drug_ids: list[str], smiles_map: dict[str, str], radius=2, nbits=1024) -> dict[str, np.ndarray]:
    """Return drug_id -> uint8 bit vector array."""
    out = {}
    skipped = []
    for d in drug_ids:
        smi = smiles_map.get(d)
        if not smi or not isinstance(smi, str):
            skipped.append(d)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped.append(d)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
        arr = np.zeros(nbits, dtype=np.uint8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[d] = arr
    print(f"[fp] computed Morgan FP for {len(out)} drugs, skipped {len(skipped)}")
    return out


def tanimoto_pairs(fp_map: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> np.ndarray:
    out = np.full(len(pairs), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        fa, fb = fp_map.get(a), fp_map.get(b)
        if fa is None or fb is None:
            continue
        inter = np.bitwise_and(fa, fb).sum()
        union = np.bitwise_or(fa, fb).sum()
        if union == 0:
            continue
        out[i] = inter / union
    return out


def load_pubmedbert_drug_embeddings(drug_ids_needed: set[str]) -> dict[str, np.ndarray]:
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    X = np.load(CACHE / "pubmedbert_real.npz")["X"]
    print(f"[bert] X={X.shape}, nodes={len(nodes)}")
    # restrict to Drug rows
    drug_rows = nodes[(nodes["kind"] == "Drug") & (nodes["id"].isin(drug_ids_needed))]
    print(f"[bert] drug rows matched: {len(drug_rows)}")
    emb = {}
    for _, row in drug_rows.iterrows():
        emb[row["id"]] = X[row.name].copy()  # row.name = original index
    # z-score per dim across drugs
    mat = np.stack(list(emb.values()), axis=0)
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True) + 1e-8
    for k in emb:
        emb[k] = ((emb[k] - mu[0]) / sd[0]).astype(np.float32)
    return emb


def cosine_pairs(emb: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> np.ndarray:
    out = np.full(len(pairs), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        va, vb = emb.get(a), emb.get(b)
        if va is None or vb is None:
            continue
        na = np.linalg.norm(va) + 1e-8
        nb = np.linalg.norm(vb) + 1e-8
        out[i] = float(np.dot(va, vb) / (na * nb))
    return out


def build_drug_neighbors(drug_ids_needed: set[str]) -> dict[str, set[str]]:
    """For each drug, gather 1-hop neighbor IDs (excluding Drug nodes), relation-blind."""
    print("[kg] loading edges ...")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    print(f"[kg] edges={len(edges)}")
    drug_set = set(drug_ids_needed)
    # both directions: drug as src or dst
    e1 = edges[edges["src"].isin(drug_set) & (edges["dst_kind"] != "Drug") & (edges["dst_kind"] != "drug")][["src", "dst"]]
    e1.columns = ["drug", "nbr"]
    e2 = edges[edges["dst"].isin(drug_set) & (edges["src_kind"] != "Drug") & (edges["src_kind"] != "drug")][["dst", "src"]]
    e2.columns = ["drug", "nbr"]
    e = pd.concat([e1, e2], ignore_index=True)
    print(f"[kg] drug-incident edges (excl drug-drug): {len(e)}")
    nbr_map = e.groupby("drug")["nbr"].agg(set).to_dict()
    print(f"[kg] drugs with neighbors: {len(nbr_map)}")
    sizes = np.array([len(v) for v in nbr_map.values()])
    print(f"[kg] nbr size: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}")
    return nbr_map


def jaccard_pairs(nbr_map: dict[str, set[str]], pairs: list[tuple[str, str]]) -> np.ndarray:
    out = np.full(len(pairs), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        na, nb = nbr_map.get(a), nbr_map.get(b)
        if na is None or nb is None:
            continue
        u = na | nb
        if not u:
            continue
        out[i] = len(na & nb) / len(u)
    return out


def main():
    t0 = time.time()
    df = load_pairs_and_pred()
    unique_drugs = sorted(set(df["drug_a_id"]) | set(df["drug_b_id"]))
    print(f"[main] unique drugs in S2: {len(unique_drugs)}")

    smiles_map = load_drug_smiles()
    fp_map = compute_morgan_fps(unique_drugs, smiles_map)

    pairs = list(zip(df["drug_a_id"], df["drug_b_id"]))
    print(f"[main] computing S_st (Tanimoto) ...")
    df["S_st"] = tanimoto_pairs(fp_map, pairs)

    bert_emb = load_pubmedbert_drug_embeddings(set(unique_drugs))
    print(f"[main] computing S_tx (PubMedBERT cosine, z-scored) ...")
    df["S_tx"] = cosine_pairs(bert_emb, pairs)

    nbr_map = build_drug_neighbors(set(unique_drugs))
    print(f"[main] computing S_kg (Jaccard 1-hop non-drug neighbors) ...")
    df["S_kg"] = jaccard_pairs(nbr_map, pairs)

    # Drug degree (for control)
    df["deg_a"] = df["drug_a_id"].map(lambda d: len(nbr_map.get(d, set())))
    df["deg_b"] = df["drug_b_id"].map(lambda d: len(nbr_map.get(d, set())))
    df["deg_min"] = df[["deg_a", "deg_b"]].min(axis=1)

    print("[main] summary:")
    for c in ["S_st", "S_tx", "S_kg", "nll"]:
        print(f"  {c}: nan={df[c].isna().sum()}, min={df[c].min():.4f}, mean={df[c].mean():.4f}, max={df[c].max():.4f}")

    out = OUT_DIR / "pair_features.parquet"
    df.to_parquet(out)
    print(f"[main] saved: {out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
