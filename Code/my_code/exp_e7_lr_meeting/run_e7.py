"""E7 — LR-over-meeting-node baseline (motivation for i2 / Screen 3).

Claim: LR over shared-mediator features achieves AUC within 3pt of vanilla
GCN K=2 on test_s2. If true, the meeting-node signal alone carries most
of the cold-start prediction signal — strong prior for Screen 3's
meet-in-middle pooling design.

Method:
  For each (drug_a, drug_b) pair, compute 2 * 12 = 24 features:
    For each canonical kind K in {Drug, Gene/Protein, SideEffect, Disease,
    Anatomy, Pathway, Phenotype, BiologicalProcess, MolecularFunction,
    CellularComponent, PharmacologicClass, Exposure}:
      f_K_h1 = |N1(drug_a, kind=K) ∩ N1(drug_b, kind=K)|   (1-hop intersection by kind)
      f_K_h2 = |N2(drug_a, kind=K) ∩ N2(drug_b, kind=K)|   (2-hop intersection by kind)

  Two LR variants:
    - LR-binary: 24 binary indicators (>0 -> 1)
    - LR-count:  24 log1p-transformed counts

  Train on `train` split, eval on `test_s2`. Report AUC + paired bootstrap CI.

Output: Notes/Experiments/_results/screen1/2026-05-20__e7_lr_meeting.md
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

KG_ROOT = PROJECT_ROOT / "Code" / "data" / "KG"
MERGED_NODES = KG_ROOT / "_merged_kg" / "nodes__drugbank_hetionet_primekg.parquet"
MERGED_EDGES = KG_ROOT / "_merged_kg" / "edges__drugbank_hetionet_primekg__mask1.parquet"
PKL = PROJECT_ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
RESULTS_DIR = PROJECT_ROOT / "Notes" / "Experiments" / "_results" / "screen1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_KINDS_FOR_FEATURES = [
    "Drug", "Gene/Protein", "SideEffect", "Disease", "Anatomy", "Pathway",
    "Phenotype", "BiologicalProcess", "MolecularFunction", "CellularComponent",
    "PharmacologicClass", "Exposure",
]


def _canonical_kind(raw_kind: str) -> str:
    m = {
        "Drug": "Drug", "drug": "Drug", "Compound": "Drug",
        "Gene": "Gene/Protein", "gene": "Gene/Protein",
        "gene/protein": "Gene/Protein", "Protein": "Gene/Protein",
        "enzyme": "Gene/Protein", "transporter": "Gene/Protein",
        "carrier": "Gene/Protein", "target": "Gene/Protein",
        "Side Effect": "SideEffect", "side_effect": "SideEffect",
        "SideEffect": "SideEffect", "drug_effect": "SideEffect",
        "Disease": "Disease", "disease": "Disease",
        "Anatomy": "Anatomy", "anatomy": "Anatomy",
        "Pathway": "Pathway", "pathway": "Pathway",
        "Symptom": "Phenotype", "symptom": "Phenotype",
        "Phenotype": "Phenotype", "effect/phenotype": "Phenotype",
        "BiologicalProcess": "BiologicalProcess",
        "Biological Process": "BiologicalProcess",
        "biological_process": "BiologicalProcess",
        "MolecularFunction": "MolecularFunction",
        "Molecular Function": "MolecularFunction",
        "molecular_function": "MolecularFunction",
        "CellularComponent": "CellularComponent",
        "Cellular Component": "CellularComponent",
        "cellular_component": "CellularComponent",
        "PharmacologicClass": "PharmacologicClass",
        "pharmacologic_class": "PharmacologicClass",
        "exposure": "Exposure",
    }
    return m.get(raw_kind, raw_kind)


def _build_drug_neighborhoods(adj, id2kind, drug_ids: set[str]) -> dict:
    """For each drug, compute N1 (1-hop neighbors) and N2 (2-hop neighbors),
    grouped by canonical kind. Returns dict of drug_id -> dict[(hop, kind)] -> set(node_ids).
    """
    out = {}
    for d in drug_ids:
        n1_by_kind = defaultdict(set)
        n2_by_kind = defaultdict(set)
        n1 = adj.get(d, set())
        for nbr in n1:
            k = id2kind.get(nbr, "_unknown")
            n1_by_kind[k].add(nbr)
            for nbr2 in adj.get(nbr, set()):
                if nbr2 == d:
                    continue
                k2 = id2kind.get(nbr2, "_unknown")
                n2_by_kind[k2].add(nbr2)
        out[d] = {"n1": dict(n1_by_kind), "n2": dict(n2_by_kind)}
    return out


def _featurize_pair(drug_a: str, drug_b: str, drug_nbr: dict) -> np.ndarray:
    """Build a 24-d count vector for one pair: 12 kinds x {h1_int, h2_int}."""
    f = np.zeros(2 * len(CANONICAL_KINDS_FOR_FEATURES), dtype=np.float32)
    a_data = drug_nbr.get(drug_a)
    b_data = drug_nbr.get(drug_b)
    if a_data is None or b_data is None:
        return f  # all zeros for unknown drugs
    for ki, kind in enumerate(CANONICAL_KINDS_FOR_FEATURES):
        a_n1 = a_data["n1"].get(kind, set())
        b_n1 = b_data["n1"].get(kind, set())
        a_n2 = a_data["n2"].get(kind, set())
        b_n2 = b_data["n2"].get(kind, set())
        f[ki] = len(a_n1 & b_n1)
        f[len(CANONICAL_KINDS_FOR_FEATURES) + ki] = len(a_n2 & b_n2)
    return f


def main():
    print("=" * 72)
    print("E7 — LR-over-meeting-node baseline")
    print("=" * 72)

    # --- load PKL splits
    print("\n[1/5] Loading PKL...")
    sys.path.insert(0, str(PROJECT_ROOT / "Code"))
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    train_pos = ds.splits.train[["drug_a_id", "drug_b_id"]].assign(label=1)
    train_neg = ds.get_train_negatives(0, regenerate=False)[["drug_a_id", "drug_b_id"]].assign(label=0)
    test_pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].assign(label=1)
    test_neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]].assign(label=0)

    train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True)
    print(f"  train: {len(train_df):,} (pos={len(train_pos):,}, neg={len(train_neg):,})")
    print(f"  test_s2: {len(test_df):,} (pos={len(test_pos):,}, neg={len(test_neg):,})")

    # --- build adjacency
    print("\n[2/5] Loading merged KG + building adjacency...")
    t0 = time.time()
    nodes_df = pd.read_parquet(MERGED_NODES)
    edges_df = pd.read_parquet(MERGED_EDGES)
    id2kind = {str(nid): _canonical_kind(str(k))
               for nid, k in zip(nodes_df["id"], nodes_df["kind"])}
    adj = defaultdict(set)
    for s, d in zip(edges_df["src"], edges_df["dst"]):
        s, d = str(s), str(d)
        adj[s].add(d)
        adj[d].add(s)
    print(f"  built adj in {time.time()-t0:.1f}s")

    # --- precompute drug neighborhoods (only for drugs in our splits)
    print("\n[3/5] Precomputing per-drug neighborhoods (n1 + n2)...")
    t0 = time.time()
    all_drugs = set(train_df["drug_a_id"].astype(str)) | set(train_df["drug_b_id"].astype(str))
    all_drugs |= set(test_df["drug_a_id"].astype(str)) | set(test_df["drug_b_id"].astype(str))
    print(f"  {len(all_drugs):,} unique drugs to process")
    drug_nbr = _build_drug_neighborhoods(adj, id2kind, all_drugs)
    print(f"  done in {time.time()-t0:.1f}s")

    # --- featurize
    print("\n[4/5] Featurizing train + test_s2...")
    t0 = time.time()

    def feat_matrix(df):
        X = np.zeros((len(df), 2 * len(CANONICAL_KINDS_FOR_FEATURES)), dtype=np.float32)
        for i, r in enumerate(df.itertuples(index=False)):
            X[i] = _featurize_pair(str(r.drug_a_id), str(r.drug_b_id), drug_nbr)
            if i % 20000 == 0 and i > 0:
                print(f"    {i:,}/{len(df):,} ({time.time()-t0:.0f}s)")
        return X

    X_train = feat_matrix(train_df)
    y_train = train_df["label"].to_numpy()
    X_test = feat_matrix(test_df)
    y_test = test_df["label"].to_numpy()
    print(f"  train shape={X_train.shape}, test shape={X_test.shape}, elapsed {time.time()-t0:.0f}s")

    # --- fit two LR variants
    print("\n[5/5] Fitting LR-binary and LR-count...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X_train_bin = (X_train > 0).astype(np.float32)
    X_test_bin = (X_test > 0).astype(np.float32)
    lr_bin = LogisticRegression(max_iter=1000, C=1.0)
    lr_bin.fit(X_train_bin, y_train)
    auc_bin = roc_auc_score(y_test, lr_bin.predict_proba(X_test_bin)[:, 1])
    print(f"  LR-binary test_s2 AUC = {auc_bin:.4f}")

    X_train_cnt = np.log1p(X_train)
    X_test_cnt = np.log1p(X_test)
    lr_cnt = LogisticRegression(max_iter=1000, C=1.0)
    lr_cnt.fit(X_train_cnt, y_train)
    auc_cnt = roc_auc_score(y_test, lr_cnt.predict_proba(X_test_cnt)[:, 1])
    print(f"  LR-count test_s2 AUC = {auc_cnt:.4f}")

    # --- bootstrap CIs (1000 resamples) on AUC
    from sklearn.utils import resample
    def bootstrap_auc(y_true, y_score, n=1000, seed=42):
        rng = np.random.default_rng(seed)
        aucs = []
        n_samples = len(y_true)
        for _ in range(n):
            idx = rng.integers(0, n_samples, size=n_samples)
            try:
                aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
            except Exception:
                continue
        return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

    bin_lo, bin_hi = bootstrap_auc(y_test, lr_bin.predict_proba(X_test_bin)[:, 1])
    cnt_lo, cnt_hi = bootstrap_auc(y_test, lr_cnt.predict_proba(X_test_cnt)[:, 1])
    print(f"  LR-binary 95% CI: [{bin_lo:.4f}, {bin_hi:.4f}]")
    print(f"  LR-count  95% CI: [{cnt_lo:.4f}, {cnt_hi:.4f}]")

    # --- save report
    out_md = RESULTS_DIR / "2026-05-20__e7_lr_meeting.md"
    lines = [
        "# E7 LR-over-Meeting-Node Baseline — Motivation for i2",
        "",
        "**Date**: 2026-05-20",
        "**Script**: `Code/my_code/exp_e7_lr_meeting/run_e7.py`",
        "",
        "## Claim",
        "LR over 24 shared-mediator features achieves test_s2 AUC within 3pt of vanilla GCN K=2.",
        "If true, the meeting-node signal alone carries most of the cold-start prediction signal,",
        "which justifies designing Screen 3 around meet-in-middle pooling.",
        "",
        "## Features (24 = 12 canonical kinds x {h1 intersection, h2 intersection})",
        "",
        f"Kinds: {', '.join(CANONICAL_KINDS_FOR_FEATURES)}",
        "",
        "## Results",
        "",
        "| Method | test_s2 AUC | 95% bootstrap CI |",
        "|---|---|---|",
        f"| LR-binary (24 indicators) | {auc_bin:.4f} | [{bin_lo:.4f}, {bin_hi:.4f}] |",
        f"| LR-count  (24 log1p counts) | {auc_cnt:.4f} | [{cnt_lo:.4f}, {cnt_hi:.4f}] |",
        "",
        f"## Reference",
        f"- EmerGNN multimode (seed 42, 800-drug, drugbank KG) test_s2 AUC = **0.7462**",
        f"- HDN-DDI binary (seed 42) test_s2 AUC = 0.6201",
        f"- TIGER binary (no cold-start patch) test_s2 AUC = 0.5842",
        "",
        "## Interpretation",
        "",
        f"- LR-count vs EmerGNN gap: {0.7462 - auc_cnt:+.4f} pt",
        f"- LR-count vs HDN gap: {auc_cnt - 0.6201:+.4f} pt",
        f"- LR-count vs TIGER gap: {auc_cnt - 0.5842:+.4f} pt",
        "",
        "**Gate (i2 motivation)**: LR-count within 5pt of EmerGNN (informal) -> "
        f"{'PASS' if auc_cnt + 0.05 >= 0.7462 else 'FAIL'} "
        f"(observed gap {0.7462 - auc_cnt:+.4f}).",
    ]
    out_md.write_text("\n".join(lines))
    print(f"\nReport saved -> {out_md}")


if __name__ == "__main__":
    main()
