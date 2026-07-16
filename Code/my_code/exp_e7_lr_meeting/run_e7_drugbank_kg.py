"""E7 v2 — LR baseline with DRUGBANK-only KG access (matched to EmerGNN).

Per Codex round-5 caveat: original E7 used merged KG (178K nodes) for
feature computation, but EmerGNN multimode trains on drugbank KG only
(~10K nodes). This v2 recomputes the 24 mediator features using ONLY the
drugbank-source edges so the comparison is apples-to-apples.

Method changes from E7 v1:
  - Adjacency built from drugbank `filtered/*.csv` directly, NOT merged KG
  - Canonical kinds collapse {target, enzyme, transporter, carrier, pathway}
    to Gene/Protein and Pathway (consistent with E7 v1)
  - SideEffect, Disease, Anatomy, Phenotype features will be zero
    (drugbank doesn't have these) — that's intentional and shows what
    EmerGNN sees vs doesn't see
  - All other settings unchanged from E7 v1

Expected outcome:
  - If LR-count under drugbank-only KG STILL ≈ EmerGNN: architecture
    insight is real, Screen 3 motivation rock-solid
  - If LR drops to ~0.70: most of E7 v1's gain came from richer KG context,
    not shared-mediator structure. Reframe E7 finding as "merged KG features
    provide cold-start signal" not "meeting-node signal alone".
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
DRUGBANK_FILTERED = KG_ROOT / "drugbank" / "filtered"
PKL = PROJECT_ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
RESULTS_DIR = PROJECT_ROOT / "Notes" / "Experiments" / "_results" / "screen1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CANONICAL_KINDS_FOR_FEATURES = [
    "Drug", "Gene/Protein", "SideEffect", "Disease", "Anatomy", "Pathway",
    "Phenotype", "BiologicalProcess", "MolecularFunction", "CellularComponent",
    "PharmacologicClass", "Exposure",
]


def _build_drugbank_adj_and_kinds(drug_pool: set[str]):
    """Replicate the drugbank-only entity graph that EmerGNN sees.

    Entities: drugs + (target/enzyme/transporter/carrier/pathway) entities
    referenced by edges with at least one endpoint in drug_pool.

    Returns adj (dict node_id -> set), id2kind (canonical).
    """
    files = [
        ("drug_targets.csv",      "target_id",      "Gene/Protein"),
        ("drug_enzymes.csv",      "enzyme_id",      "Gene/Protein"),
        ("drug_transporters.csv", "transporter_id", "Gene/Protein"),
        ("drug_carriers.csv",     "carrier_id",     "Gene/Protein"),
        ("drug_pathways.csv",     "pathway_id",     "Pathway"),
    ]
    adj = defaultdict(set)
    id2kind = {d: "Drug" for d in drug_pool}
    for fn, id_col, kind in files:
        df = pd.read_csv(DRUGBANK_FILTERED / fn)
        df = df[df["drugbank_id"].isin(drug_pool)]
        for d, e in zip(df["drugbank_id"], df[id_col]):
            d, e = str(d), str(e)
            adj[d].add(e)
            adj[e].add(d)
            id2kind[e] = kind
    return adj, id2kind


def _build_drug_neighborhoods(adj, id2kind, drug_ids: set[str]):
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


def _featurize_pair(drug_a, drug_b, drug_nbr):
    f = np.zeros(2 * len(CANONICAL_KINDS_FOR_FEATURES), dtype=np.float32)
    a, b = drug_nbr.get(drug_a), drug_nbr.get(drug_b)
    if a is None or b is None:
        return f
    for ki, kind in enumerate(CANONICAL_KINDS_FOR_FEATURES):
        a_n1 = a["n1"].get(kind, set())
        b_n1 = b["n1"].get(kind, set())
        a_n2 = a["n2"].get(kind, set())
        b_n2 = b["n2"].get(kind, set())
        f[ki] = len(a_n1 & b_n1)
        f[len(CANONICAL_KINDS_FOR_FEATURES) + ki] = len(a_n2 & b_n2)
    return f


def main():
    print("=" * 72)
    print("E7 v2 — LR over meeting-node features (DRUGBANK-only KG, matched access)")
    print("=" * 72)

    sys.path.insert(0, str(PROJECT_ROOT / "Code"))
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    train_pos = ds.splits.train[["drug_a_id", "drug_b_id"]].assign(label=1)
    train_neg = ds.get_train_negatives(0, regenerate=False)[["drug_a_id", "drug_b_id"]].assign(label=0)
    test_pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].assign(label=1)
    test_neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]].assign(label=0)
    train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True)
    print(f"train={len(train_df):,} test_s2={len(test_df):,}")

    drug_pool = set(train_df["drug_a_id"]) | set(train_df["drug_b_id"]) | \
                set(test_df["drug_a_id"]) | set(test_df["drug_b_id"])
    drug_pool = {str(d) for d in drug_pool}
    print(f"Building drugbank-only adj from {DRUGBANK_FILTERED}...")
    t0 = time.time()
    adj, id2kind = _build_drugbank_adj_and_kinds(drug_pool)
    print(f"  adj: {len(adj):,} nodes, total edges={sum(len(v) for v in adj.values())//2:,}, "
          f"{time.time()-t0:.1f}s")

    drug_nbr = _build_drug_neighborhoods(adj, id2kind, drug_pool)

    def feat_matrix(df):
        X = np.zeros((len(df), 2 * len(CANONICAL_KINDS_FOR_FEATURES)), dtype=np.float32)
        for i, r in enumerate(df.itertuples(index=False)):
            X[i] = _featurize_pair(str(r.drug_a_id), str(r.drug_b_id), drug_nbr)
        return X

    X_train = feat_matrix(train_df); y_train = train_df["label"].to_numpy()
    X_test = feat_matrix(test_df);   y_test = test_df["label"].to_numpy()

    # Many features will be zero (no SideEffect/Disease etc. in drugbank KG)
    nonzero_feat = (X_train.sum(axis=0) > 0).sum()
    print(f"Non-zero feature columns: {nonzero_feat}/{X_train.shape[1]}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X_train_bin = (X_train > 0).astype(np.float32)
    X_test_bin = (X_test > 0).astype(np.float32)
    lr_bin = LogisticRegression(max_iter=1000)
    lr_bin.fit(X_train_bin, y_train)
    auc_bin = roc_auc_score(y_test, lr_bin.predict_proba(X_test_bin)[:, 1])

    X_train_cnt = np.log1p(X_train)
    X_test_cnt = np.log1p(X_test)
    lr_cnt = LogisticRegression(max_iter=1000)
    lr_cnt.fit(X_train_cnt, y_train)
    auc_cnt = roc_auc_score(y_test, lr_cnt.predict_proba(X_test_cnt)[:, 1])

    from sklearn.utils import resample
    def boot(y, s, n=1000, sd=42):
        rng = np.random.default_rng(sd)
        aucs = []
        for _ in range(n):
            idx = rng.integers(0, len(y), size=len(y))
            try:
                aucs.append(roc_auc_score(y[idx], s[idx]))
            except Exception:
                pass
        return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

    bin_lo, bin_hi = boot(y_test, lr_bin.predict_proba(X_test_bin)[:, 1])
    cnt_lo, cnt_hi = boot(y_test, lr_cnt.predict_proba(X_test_cnt)[:, 1])

    print()
    print(f"LR-binary (drugbank-only KG) AUC = {auc_bin:.4f}  CI [{bin_lo:.4f}, {bin_hi:.4f}]")
    print(f"LR-count  (drugbank-only KG) AUC = {auc_cnt:.4f}  CI [{cnt_lo:.4f}, {cnt_hi:.4f}]")
    print(f"vs EmerGNN multimode (drugbank-KG, seed 42): 0.7462")
    print(f"vs E7 v1 (merged KG): LR-binary=0.7384, LR-count=0.7573")

    # Save report
    out_md = RESULTS_DIR / "2026-05-20__e7_v2_drugbank_kg.md"
    lines = [
        "# E7 v2 — LR over Meeting-Node Features (DRUGBANK-only KG)",
        "",
        "**Date**: 2026-05-20",
        "**Motivation**: Codex round-5 caveat that E7 v1 used merged KG (178K nodes)",
        "while EmerGNN trains on drugbank KG (~10K nodes), creating possible",
        "KG access asymmetry. v2 recomputes with matched-access drugbank KG.",
        "",
        "## Results",
        "",
        "| Method | KG source | test_s2 AUC | 95% CI |",
        "|---|---|---|---|",
        f"| LR-binary | drugbank-only | {auc_bin:.4f} | [{bin_lo:.4f}, {bin_hi:.4f}] |",
        f"| LR-count | drugbank-only | {auc_cnt:.4f} | [{cnt_lo:.4f}, {cnt_hi:.4f}] |",
        f"| LR-count | merged (E7 v1) | 0.7573 | [0.7411, 0.7726] |",
        f"| LR-binary | merged (E7 v1) | 0.7384 | [0.7217, 0.7543] |",
        f"| EmerGNN multimode | drugbank-only | 0.7462 | — |",
        "",
        f"Non-zero feature columns: {nonzero_feat}/{X_train.shape[1]} "
        f"(SideEffect/Disease/Anatomy/etc. NOT in drugbank KG so zero by construction)",
        "",
        "## Interpretation",
        f"- Gap between LR-count drugbank-only and LR-count merged: {auc_cnt - 0.7573:+.4f}pt",
        f"- Gap between LR-count drugbank-only and EmerGNN multimode: {auc_cnt - 0.7462:+.4f}pt",
        "",
        "If LR-count drugbank-only stays close to EmerGNN: architecture insight",
        "is real, shared-mediator features capture most of EmerGNN's signal.",
        "If LR-count drops significantly: most of E7 v1's gain comes from richer",
        "merged-KG context (SE / Disease / Phenotype / etc. that drugbank lacks).",
    ]
    out_md.write_text("\n".join(lines))
    print(f"\nReport saved -> {out_md}")


if __name__ == "__main__":
    main()
