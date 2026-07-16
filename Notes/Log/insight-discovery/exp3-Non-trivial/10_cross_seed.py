"""exp3 step 7 — cross-seed verification (seed 43 + 44).

Re-run the C2 pipeline end-to-end on seed43 and seed44:
- Use same KG-derived features (M2, H_rel, log M2 control)
- Train fresh LR on each seed's train pairs
- Compute per-pair NLL on each seed's test_s2
- Check if U-shape phase transition replicates

For speed, use LR only (the cross-model amplifying signal). GCN cross-seed
is more expensive and not needed if LR replicates.
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

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
}
MEDIATOR_KINDS = ["Protein", "Gene", "SideEffect", "Phenotype", "Disease", "Pathway", "Anatomy", "BioProcess", "MolFunction"]

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
KIND_TO_GROUP = {k: g for g, ks in KIND_GROUPS.items() for k in ks}


def entropy_norm(counts, K):
    total = sum(counts)
    if total <= 1 or K <= 1:
        return 0.0
    p = np.array(counts, dtype=np.float64) / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(H / np.log(K))


def build_drug_mediators(edges, id2kind, drug_set):
    e1 = edges[edges["src"].isin(drug_set)][["src", "dst", "relation"]].rename(columns={"src": "d", "dst": "m", "relation": "r"})
    e2 = edges[edges["dst"].isin(drug_set)][["dst", "src", "relation"]].rename(columns={"dst": "d", "src": "m", "relation": "r"})
    de = pd.concat([e1, e2], ignore_index=True)
    de["m_kind"] = de["m"].map(id2kind)
    de = de[de["m_kind"].isin(MEDIATOR_KINDS)]
    drug_to_med = {}
    for d, grp in de.groupby("d"):
        drug_to_med[d] = {m: list(sub["r"]) for m, sub in grp.groupby("m")}
    return drug_to_med


def compute_H_rel_M2(df, drug_to_med):
    M2 = np.zeros(len(df), dtype=np.int32)
    Hr = np.full(len(df), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(df["drug_a_id"].values, df["drug_b_id"].values)):
        ma = drug_to_med.get(a)
        mb = drug_to_med.get(b)
        if not ma or not mb:
            continue
        shared = set(ma) & set(mb)
        if not shared:
            continue
        M2[i] = len(shared)
        if len(shared) > 1:
            templates = []
            for m in shared:
                ra = ma[m][0]
                rb = mb[m][0]
                templates.append((ra, rb))
            tc = Counter(templates)
            Hr[i] = entropy_norm(list(tc.values()), max(len(tc), 2))
    return M2, Hr


def build_count_features(edges, id2kind, drug_set):
    n1 = defaultdict(set)
    fwd = defaultdict(set)
    for src, dst, directed in zip(edges["src"].values, edges["dst"].values, edges["directed"].values):
        fwd[src].add(dst)
        if not directed:
            fwd[dst].add(src)
        sd = src in drug_set
        dd = dst in drug_set
        if sd and not dd:
            n1[src].add((dst, KIND_TO_GROUP.get(id2kind.get(dst, ""), "other")))
        if dd and not sd and not directed:
            n1[dst].add((src, KIND_TO_GROUP.get(id2kind.get(src, ""), "other")))
    n2 = defaultdict(set)
    for drug in drug_set:
        for mid, _ in n1.get(drug, ()):
            for term in fwd.get(mid, ()):
                if term == drug or term in drug_set:
                    continue
                n2[drug].add((term, KIND_TO_GROUP.get(id2kind.get(term, ""), "other")))
    return n1, n2


def featurize_pairs(df, n1, n2):
    n = len(df)
    n_g = len(KIND_ORDER)
    X = np.zeros((n, 2 * n_g), dtype=np.float32)
    for i, (da, db) in enumerate(zip(df["drug_a_id"].values, df["drug_b_id"].values)):
        s1 = n1.get(da, set()) & n1.get(db, set())
        s2 = n2.get(da, set()) & n2.get(db, set())
        c1 = defaultdict(int)
        c2 = defaultdict(int)
        for _, g in s1: c1[g] += 1
        for _, g in s2: c2[g] += 1
        for j, grp in enumerate(KIND_ORDER):
            X[i, j] = np.log1p(c1[grp])
            X[i, n_g + j] = np.log1p(c2[grp])
    return X


def load_split(pos_path, neg_path):
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]].copy()
    pos["label"] = 1
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
    neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)


def analyze_seed(seed: int, edges, nodes, id2kind, id2kc, drug_set, n1, n2):
    print(f"\n========== seed {seed} ==========")
    SP = ROOT / f"Code/data/KG/drugbank/splits/seed{seed}"
    train = load_split(SP / "train.parquet", SP / "train_negatives/epoch_0.parquet")
    test = load_split(SP / "test_s2.parquet", SP / "negatives/test_s2.parquet")
    print(f"  train n={len(train)}, test_s2 n={len(test)}, label balance test={test['label'].mean():.3f}")

    test_drugs = set(test["drug_a_id"]) | set(test["drug_b_id"])
    print(f"  test drugs: {len(test_drugs)}")

    drug_to_med = build_drug_mediators(edges, id2kc, test_drugs)
    M2, Hr = compute_H_rel_M2(test, drug_to_med)
    test["M2"] = M2
    test["H_rel"] = Hr
    print(f"  M2>1 pairs: {int((M2 > 1).sum())} ({(M2 > 1).mean() * 100:.1f}%)")

    # Train LR
    Xtr = featurize_pairs(train, n1, n2)
    ytr = train["label"].values
    lr_clf = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, ytr)

    Xte = featurize_pairs(test, n1, n2)
    p = lr_clf.predict_proba(Xte)[:, 1]
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    test["lr_pred_p"] = p.astype(np.float32)
    test["lr_nll"] = -(test["label"] * np.log(p) + (1 - test["label"]) * np.log(1 - p)).astype(np.float32)
    auc = roc_auc_score(test["label"], p)
    print(f"  LR test AUC: {auc:.4f}")

    # Slice Q3
    sub = test[test["M2"] > 1].copy()
    q67 = sub["M2"].quantile(0.67)
    q3 = sub[sub["M2"] > q67].copy()
    print(f"  Q3 (|M2|>{q67:.0f}) n={len(q3)}, label_rate={q3['label'].mean():.3f}")

    pos = q3[q3["label"] == 1].copy()
    neg = q3[q3["label"] == 0].copy()

    def residualize(s, col):
        lin = LinearRegression().fit(np.log(s["M2"].values).reshape(-1, 1), s[col].values)
        return s[col].values - lin.predict(np.log(s["M2"].values).reshape(-1, 1))
    pos["nll_resid"] = residualize(pos, "lr_nll")
    neg["nll_resid"] = residualize(neg, "lr_nll")
    pos["Hr_dec"] = pd.qcut(pos["H_rel"], 10, labels=False, duplicates="drop")
    neg["Hr_dec"] = pd.qcut(neg["H_rel"], 10, labels=False, duplicates="drop")

    pos_dec = pos.groupby("Hr_dec")["nll_resid"].mean()
    neg_dec = neg.groupby("Hr_dec")["nll_resid"].mean()
    print(f"  POS deciles: {pos_dec.tolist()}")
    print(f"  NEG deciles: {neg_dec.tolist()}")
    print(f"  POS U-shape (dec0>dec5 AND dec9>dec5): {(pos_dec.iloc[0] > pos_dec.iloc[5]) and (pos_dec.iloc[9] > pos_dec.iloc[5])}")
    print(f"  NEG inverted-U: {(neg_dec.iloc[5] > neg_dec.iloc[0]) and (neg_dec.iloc[5] > neg_dec.iloc[9])}")
    print(f"  POS decile 9-8 jump: {pos_dec.iloc[9] - pos_dec.iloc[8]:+.4f}")
    print(f"  POS decile 0-1 jump: {pos_dec.iloc[0] - pos_dec.iloc[1]:+.4f}")
    rho_pos, _ = spearmanr(pos["H_rel"], pos["lr_nll"])
    rho_neg, _ = spearmanr(neg["H_rel"], neg["lr_nll"])
    print(f"  ρ(H_rel, NLL | pos)={rho_pos:.4f}, ρ(H_rel, NLL | neg)={rho_neg:.4f}")
    return {
        "seed": seed,
        "auc": float(auc),
        "n_q3": int(len(q3)),
        "pos_decile_means": pos_dec.tolist(),
        "neg_decile_means": neg_dec.tolist(),
        "pos_U_holds": bool((pos_dec.iloc[0] > pos_dec.iloc[5]) and (pos_dec.iloc[9] > pos_dec.iloc[5])),
        "neg_inv_U_holds": bool((neg_dec.iloc[5] > neg_dec.iloc[0]) and (neg_dec.iloc[5] > neg_dec.iloc[9])),
        "pos_89_jump": float(pos_dec.iloc[9] - pos_dec.iloc[8]),
        "pos_01_jump": float(pos_dec.iloc[0] - pos_dec.iloc[1]),
        "rho_pos": float(rho_pos),
        "rho_neg": float(rho_neg),
    }


def main():
    t0 = time.time()
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    nodes["kind_canon"] = nodes["kind"].map(KIND_CANON).fillna("Other")
    id2kc = dict(zip(nodes["id"], nodes["kind_canon"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    n1, n2 = build_count_features(edges, id2kind, drug_set)
    print(f"loaded: nodes={len(nodes)} edges={len(edges)} drugs={len(drug_set)}")

    results = []
    for seed in [42, 43, 44]:
        r = analyze_seed(seed, edges, nodes, id2kind, id2kc, drug_set, n1, n2)
        results.append(r)

    print("\n========== CROSS-SEED SUMMARY ==========")
    for r in results:
        print(f"\nseed {r['seed']}: AUC={r['auc']:.4f}, n_Q3={r['n_q3']}")
        print(f"  POS U-shape: {r['pos_U_holds']}, POS dec9-8 jump: {r['pos_89_jump']:+.4f}")
        print(f"  NEG inv-U: {r['neg_inv_U_holds']}, ρ(pos)={r['rho_pos']:.4f}")

    import json
    with open(OUT_DIR / "cross_seed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {OUT_DIR/'cross_seed_results.json'}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
