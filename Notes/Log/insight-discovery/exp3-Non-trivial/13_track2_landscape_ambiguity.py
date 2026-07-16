"""exp3 Track 2 — Landscape ambiguity (Candidate E from new codex thread).

Hypothesis: For cold-start S2 drug d, its position in the training-drug manifold
determines difficulty.
- closeness(d) = max similarity to any training drug (how "imitable" is d?)
- ambiguity(d) = how many training drugs are equally close (no unique twin?)

Paradox: pairs where BOTH drugs have high closeness AND high ambiguity should
fail (every cold drug "looks like" multiple training drugs that may have
conflicting DDI patterns) — even though "looks supported" by twin-availability.

Phase 1: compute per-drug closeness/ambiguity using S_tx (PubMedBERT) and S_st
(Morgan FP). Build pair-level aggregates. See marginal patterns and 2D heatmap.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_project_root():
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent
CACHE = ROOT / "Notes/Log/insight-discovery/exp2-0513-test-insight/_cache_features"


def main():
    t0 = time.time()
    print("[load] pair_features_c2.parquet ...")
    df = pd.read_parquet(OUT_DIR / "pair_features_c2.parquet")

    print("[load] training drug set ...")
    train_pairs = pd.read_parquet(ROOT / "Code/data/KG/drugbank/splits/seed42/train.parquet")
    train_drugs = sorted(set(train_pairs["drug_a_id"]) | set(train_pairs["drug_b_id"]))
    test_drugs = sorted(set(df["drug_a_id"]) | set(df["drug_b_id"]))
    print(f"  training drugs: {len(train_drugs)}; test S2 drugs: {len(test_drugs)}")
    overlap = set(train_drugs) & set(test_drugs)
    print(f"  overlap (should be 0 for true S2): {len(overlap)}")

    print("[load] nodes + PubMedBERT embeddings ...")
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    X_all = np.load(CACHE / "pubmedbert_real.npz")["X"]
    id2idx = dict(zip(nodes["id"], range(len(nodes))))
    # Get embedding for each test drug and each train drug
    train_drugs = [d for d in train_drugs if d in id2idx]
    test_drugs = [d for d in test_drugs if d in id2idx]
    print(f"  available train drugs: {len(train_drugs)}, test drugs: {len(test_drugs)}")

    E_train = np.stack([X_all[id2idx[d]] for d in train_drugs], axis=0)
    E_test = np.stack([X_all[id2idx[d]] for d in test_drugs], axis=0)
    # z-score per dim across all drugs (using union)
    E_all = np.vstack([E_train, E_test])
    mu = E_all.mean(0, keepdims=True)
    sd = E_all.std(0, keepdims=True) + 1e-8
    E_train_z = (E_train - mu) / sd
    E_test_z = (E_test - mu) / sd
    # Normalize for cosine
    def l2norm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    E_train_n = l2norm(E_train_z)
    E_test_n = l2norm(E_test_z)
    sim_tx = E_test_n @ E_train_n.T  # (n_test, n_train) cosine similarity
    print(f"  sim_tx shape: {sim_tx.shape}, mean={sim_tx.mean():.4f}, std={sim_tx.std():.4f}")

    # Also compute structure similarity (Morgan FP Tanimoto)
    print("[load] drug SMILES + Morgan FPs ...")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    de = pd.read_csv(ROOT / "Code/data/KG/drugbank/enriched/drugs_enriched.csv", usecols=["drugbank_id", "smiles"])
    smi = dict(zip(de["drugbank_id"], de["smiles"]))

    def compute_fp(d):
        s = smi.get(d)
        if not s: return None
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
        arr = np.zeros(1024, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    FP_train = []
    train_keep = []
    for d in train_drugs:
        fp = compute_fp(d)
        if fp is not None:
            FP_train.append(fp)
            train_keep.append(d)
    FP_train = np.stack(FP_train, axis=0).astype(np.uint8)
    print(f"  FP_train: {FP_train.shape}")
    FP_test = []
    test_keep = []
    for d in test_drugs:
        fp = compute_fp(d)
        if fp is not None:
            FP_test.append(fp)
            test_keep.append(d)
    FP_test = np.stack(FP_test, axis=0).astype(np.uint8)
    print(f"  FP_test: {FP_test.shape}")

    # Tanimoto: |A & B| / |A | B|
    # For batch: intersect = test @ train.T (popcount); union = test_pop + train_pop - intersect
    FP_train_pop = FP_train.sum(axis=1).astype(np.int32)
    FP_test_pop = FP_test.sum(axis=1).astype(np.int32)
    inter = FP_test.astype(np.int32) @ FP_train.T.astype(np.int32)
    union = FP_test_pop[:, None] + FP_train_pop[None, :] - inter
    sim_st = inter / np.maximum(union, 1)
    print(f"  sim_st shape: {sim_st.shape}, mean={sim_st.mean():.4f}, std={sim_st.std():.4f}")

    # Per cold drug, compute:
    #   closeness_tx = max sim_tx[i, :]  (top-1 cosine to training)
    #   ambiguity_tx = top1 - top2  (gap; small = no unique twin)
    #   closeness_st = max sim_st[i, :]
    #   ambiguity_st = top1 - top2
    def feat_drug(sim):
        # sim shape (n_test, n_train)
        top1 = np.partition(sim, -1, axis=1)[:, -1]
        top2 = np.partition(sim, -2, axis=1)[:, -2]
        return top1, top1 - top2
    close_tx, amb_tx = feat_drug(sim_tx)
    close_st, amb_st = feat_drug(sim_st)
    drug_feat_tx = pd.DataFrame({"drug": test_drugs, "close_tx": close_tx, "amb_tx": amb_tx})
    drug_feat_st = pd.DataFrame({"drug": test_keep, "close_st": close_st, "amb_st": amb_st})
    drug_feat = drug_feat_tx.merge(drug_feat_st, on="drug", how="outer")
    print(f"\n[drug landscape summary]")
    print(drug_feat.describe())

    # Build pair-level aggregates
    # pair_closeness = mean(close[a], close[b])
    # pair_ambiguity = mean(amb[a], amb[b])
    df = df.merge(drug_feat.rename(columns={"drug": "drug_a_id"}), on="drug_a_id", how="left", suffixes=("", "_a"))
    df = df.rename(columns={"close_tx": "close_tx_a", "amb_tx": "amb_tx_a", "close_st": "close_st_a", "amb_st": "amb_st_a"})
    df = df.merge(drug_feat.rename(columns={"drug": "drug_b_id"}), on="drug_b_id", how="left")
    df = df.rename(columns={"close_tx": "close_tx_b", "amb_tx": "amb_tx_b", "close_st": "close_st_b", "amb_st": "amb_st_b"})

    df["pair_close_tx"] = (df["close_tx_a"] + df["close_tx_b"]) / 2
    df["pair_amb_tx"] = (df["amb_tx_a"] + df["amb_tx_b"]) / 2
    df["pair_close_st"] = (df["close_st_a"] + df["close_st_b"]) / 2
    df["pair_amb_st"] = (df["amb_st_a"] + df["amb_st_b"]) / 2

    print("\n[pair-level closeness/ambiguity summary]")
    for c in ["pair_close_tx", "pair_amb_tx", "pair_close_st", "pair_amb_st"]:
        print(f"  {c}: nan={df[c].isna().sum()}, mean={df[c].mean():.4f}, q25={df[c].quantile(0.25):.4f}, q75={df[c].quantile(0.75):.4f}")

    # ===================
    # Phase 1 analysis: marginal patterns
    # ===================
    print("\n=== Marginal: closeness vs NLL ===")
    for modality in ["tx", "st"]:
        col = f"pair_close_{modality}"
        df[f"close_q_{modality}"] = pd.qcut(df[col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        tbl = df.groupby(f"close_q_{modality}", observed=True).agg(
            n=("nll", "count"),
            close_mean=(col, "mean"),
            nll_mean=("nll", "mean"),
            label_rate=("label", "mean"),
        )
        print(f"\n[{modality}] closeness quintile vs NLL")
        print(tbl)

    print("\n=== Marginal: ambiguity vs NLL ===")
    for modality in ["tx", "st"]:
        col = f"pair_amb_{modality}"
        df[f"amb_q_{modality}"] = pd.qcut(df[col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        tbl = df.groupby(f"amb_q_{modality}", observed=True).agg(
            n=("nll", "count"),
            amb_mean=(col, "mean"),
            nll_mean=("nll", "mean"),
            label_rate=("label", "mean"),
        )
        print(f"\n[{modality}] ambiguity quintile vs NLL (low amb = unique twin; high = many similar)")
        print(tbl)

    # ===================
    # Phase 2: 2D — closeness × ambiguity (paradox check)
    # ===================
    print("\n=== 2D: closeness × ambiguity quadrants (text modality) ===")
    df["hi_close_tx"] = df["pair_close_tx"] > df["pair_close_tx"].quantile(0.7)
    df["hi_amb_tx"] = df["pair_amb_tx"] > df["pair_amb_tx"].quantile(0.7)  # large amb = unique twin
    df["q2d_tx"] = np.where(df["hi_close_tx"] & df["hi_amb_tx"], "high-close, unique-twin",
                    np.where(df["hi_close_tx"] & ~df["hi_amb_tx"], "high-close, ambiguous (twin-flood)",
                             np.where(~df["hi_close_tx"] & df["hi_amb_tx"], "low-close, unique",
                                      "low-close, ambiguous")))
    print(df.groupby("q2d_tx").agg(
        n=("nll", "count"),
        nll=("nll", "mean"),
        label_rate=("label", "mean"),
    ))

    # Label-stratified
    print("\n[2D quadrant, label=1 only]")
    pos = df[df["label"] == 1]
    print(pos.groupby("q2d_tx").agg(n=("nll", "count"), nll=("nll", "mean")))
    print("[2D quadrant, label=0 only]")
    neg = df[df["label"] == 0]
    print(neg.groupby("q2d_tx").agg(n=("nll", "count"), nll=("nll", "mean")))

    # ===================
    # Phase 3: within high-closeness, is ambiguity the killer?
    # ===================
    print("\n=== Within high-closeness subset: ambiguity effect (text) ===")
    hi = df[df["hi_close_tx"]].copy()
    print(f"high-closeness n={len(hi)}, label_rate={hi['label'].mean():.3f}")
    for lbl in [1, 0]:
        sub = hi[hi["label"] == lbl].copy()
        if len(sub) < 200: continue
        sub["amb_dec"] = pd.qcut(sub["pair_amb_tx"], 10, labels=False, duplicates="drop")
        t = sub.groupby("amb_dec").agg(
            amb_mean=("pair_amb_tx", "mean"),
            close_mean=("pair_close_tx", "mean"),
            nll=("nll", "mean"),
            n=("nll", "count"),
        )
        print(f"\n[label={lbl}] high-close, decile by ambiguity (high amb = unique twin)")
        print(t)
        rho, p = spearmanr(sub["pair_amb_tx"], sub["nll"])
        print(f"  ρ(ambiguity, NLL): {rho:+.4f} p={p:.2e}")

    # Save
    df.to_parquet(OUT_DIR / "pair_features_landscape.parquet")
    print(f"\nsaved: pair_features_landscape.parquet  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
