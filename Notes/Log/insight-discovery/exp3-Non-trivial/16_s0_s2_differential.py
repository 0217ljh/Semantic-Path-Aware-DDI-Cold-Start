"""exp3 step 16 — S0 vs S2 differential failure map.

Codex pivot: instead of finding 'a phenomenon in S2', find what's structurally
different about S2 failure mode compared to S0.

Approach:
1. Build same features (H_rel, consensus_vote, consensus_var, M2, etc) on S0
2. Pool S0+S2 in one dataframe with split indicator
3. Fit: NLL ~ split + feature + split*feature + label interactions
4. Identify which feature × split interaction is significant (= cold-start specific)

If the interaction terms are flat across features (no feature × split significance),
then NO feature we have separates cold-start from warm-start failure mode — the
"cold-start specificity" of any finding is just AUC degradation, not pattern shift.
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


def _find_project_root():
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent
CACHE = ROOT / "Notes/Log/insight-discovery/exp2-0513-test-insight/_cache_features"

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
    if total <= 1 or K <= 1: return 0.0
    p = np.array(counts) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(K))


def main():
    t0 = time.time()
    SPLITS = ROOT / "Code/data/KG/drugbank/splits/seed42"

    print("[load] all data ...")
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    X_all = np.load(CACHE / "pubmedbert_real.npz")["X"]
    id2idx = dict(zip(nodes["id"], range(len(nodes))))
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    nodes["kind_canon"] = nodes["kind"].map(KIND_CANON).fillna("Other")
    id2kc = dict(zip(nodes["id"], nodes["kind_canon"]))

    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])

    train_pairs = pd.read_parquet(SPLITS / "train.parquet")
    train_drugs = sorted([d for d in (set(train_pairs["drug_a_id"]) | set(train_pairs["drug_b_id"])) if d in id2idx])

    # Train DDI lookup
    train_pair_set = set()
    for a, b in zip(train_pairs["drug_a_id"], train_pairs["drug_b_id"]):
        if a in id2idx and b in id2idx:
            train_pair_set.add((a, b) if a <= b else (b, a))

    def load_test_split(label_name):
        pos = pd.read_parquet(SPLITS / f"test_{label_name}.parquet")[["drug_a_id", "drug_b_id"]].copy()
        pos["label"] = 1.0
        neg = pd.read_parquet(SPLITS / f"negatives/test_{label_name}.parquet")[["drug_a_id", "drug_b_id"]].copy()
        neg["label"] = 0.0
        df_ = pd.concat([pos, neg], ignore_index=True)
        df_["split"] = label_name
        return df_

    s0 = load_test_split("s0")
    s2 = load_test_split("s2")
    print(f"  S0: {len(s0)}, S2: {len(s2)}")

    # Drug-mediator map for all relevant drugs
    rel_drugs = set(s0["drug_a_id"]) | set(s0["drug_b_id"]) | set(s2["drug_a_id"]) | set(s2["drug_b_id"])
    e1 = edges[edges["src"].isin(rel_drugs)][["src", "dst", "relation"]].rename(columns={"src": "d", "dst": "m", "relation": "r"})
    e2 = edges[edges["dst"].isin(rel_drugs)][["dst", "src", "relation"]].rename(columns={"dst": "d", "src": "m", "relation": "r"})
    de = pd.concat([e1, e2], ignore_index=True)
    de["m_kind"] = de["m"].map(id2kc)
    de = de[de["m_kind"].isin(MEDIATOR_KINDS)]
    drug_to_med = {}
    for d, grp in de.groupby("d"):
        drug_to_med[d] = {m: list(sub["r"]) for m, sub in grp.groupby("m")}

    # Compute M2 + H_rel for each split
    def compute_H_rel_M2(df_):
        M2 = np.zeros(len(df_), dtype=np.int32)
        Hr = np.full(len(df_), np.nan, dtype=np.float32)
        for i, (a, b) in enumerate(zip(df_["drug_a_id"].values, df_["drug_b_id"].values)):
            ma = drug_to_med.get(a)
            mb = drug_to_med.get(b)
            if not ma or not mb: continue
            shared = set(ma) & set(mb)
            if not shared: continue
            M2[i] = len(shared)
            if len(shared) > 1:
                templates = [(ma[m][0], mb[m][0]) for m in shared]
                tc = Counter(templates)
                Hr[i] = entropy_norm(list(tc.values()), max(len(tc), 2))
        df_["M2"] = M2
        df_["H_rel"] = Hr
        return df_

    s0 = compute_H_rel_M2(s0)
    s2 = compute_H_rel_M2(s2)
    print(f"  S0 M2>1 pairs: {(s0['M2']>1).sum()}, S2 M2>1: {(s2['M2']>1).sum()}")

    # Compute neighbor consensus for each split
    print("[compute] neighbor consensus K=10 ...")
    K = 10
    E_train = np.stack([X_all[id2idx[d]] for d in train_drugs], axis=0)
    mu = E_train.mean(0, keepdims=True); sd = E_train.std(0, keepdims=True) + 1e-8
    E_train = (E_train - mu) / sd
    def l2norm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    E_train_n = l2norm(E_train)

    def top_k_neighbors_for(drugs):
        E = np.stack([X_all[id2idx[d]] for d in drugs], axis=0)
        E = (E - mu) / sd
        E = l2norm(E)
        sim = E @ E_train_n.T
        top_idx = np.argpartition(-sim, K, axis=1)[:, :K]
        for i in range(len(top_idx)):
            s_i = sim[i, top_idx[i]]
            order = np.argsort(-s_i)
            top_idx[i] = top_idx[i][order]
        return {d: [train_drugs[j] for j in top_idx[i]] for i, d in enumerate(drugs)}

    all_drugs = list(set(s0["drug_a_id"]) | set(s0["drug_b_id"]) | set(s2["drug_a_id"]) | set(s2["drug_b_id"]))
    drug_to_topk = top_k_neighbors_for(all_drugs)

    def compute_consensus(df_):
        vote = np.full(len(df_), np.nan, dtype=np.float32)
        var = np.full(len(df_), np.nan, dtype=np.float32)
        for i, (a, b) in enumerate(zip(df_["drug_a_id"].values, df_["drug_b_id"].values)):
            na = drug_to_topk.get(a)
            nb = drug_to_topk.get(b)
            if not na or not nb: continue
            total = 0
            hits = 0
            for x in na:
                for y in nb:
                    if x == y: continue
                    key = (x, y) if x <= y else (y, x)
                    if key in train_pair_set:
                        hits += 1
                    total += 1
            if total == 0: continue
            vote[i] = hits / total
            var[i] = vote[i] * (1 - vote[i])
        df_["consensus_vote"] = vote
        df_["consensus_var"] = var
        return df_

    s0 = compute_consensus(s0)
    s2 = compute_consensus(s2)

    # Train LR on training pairs (consistent feature design from before)
    n1 = defaultdict(set); fwd = defaultdict(set)
    for src, dst, directed in zip(edges["src"].values, edges["dst"].values, edges["directed"].values):
        fwd[src].add(dst)
        if not directed: fwd[dst].add(src)
        sd_ = src in drug_set; dd_ = dst in drug_set
        if sd_ and not dd_: n1[src].add((dst, KIND_TO_GROUP.get(id2kind.get(dst, ""), "other")))
        if dd_ and not sd_ and not directed: n1[dst].add((src, KIND_TO_GROUP.get(id2kind.get(src, ""), "other")))
    n2 = defaultdict(set)
    for drug in drug_set:
        for mid, _ in n1.get(drug, ()):
            for term in fwd.get(mid, ()):
                if term == drug or term in drug_set: continue
                n2[drug].add((term, KIND_TO_GROUP.get(id2kind.get(term, ""), "other")))

    def featurize(df_):
        n = len(df_); n_g = len(KIND_ORDER)
        X = np.zeros((n, 2 * n_g), dtype=np.float32)
        for i, (da, db) in enumerate(zip(df_["drug_a_id"].values, df_["drug_b_id"].values)):
            s1 = n1.get(da, set()) & n1.get(db, set())
            s2_set = n2.get(da, set()) & n2.get(db, set())
            c1 = defaultdict(int); c2 = defaultdict(int)
            for _, g in s1: c1[g] += 1
            for _, g in s2_set: c2[g] += 1
            for j, grp in enumerate(KIND_ORDER):
                X[i, j] = np.log1p(c1[grp])
                X[i, n_g + j] = np.log1p(c2[grp])
        return X

    train_pos_lr = pd.read_parquet(SPLITS / "train.parquet")[["drug_a_id", "drug_b_id"]]
    train_pos_lr["label"] = 1
    train_neg_lr = pd.read_parquet(SPLITS / "train_negatives/epoch_0.parquet")[["drug_a_id", "drug_b_id"]]
    train_neg_lr["label"] = 0
    train_lr = pd.concat([train_pos_lr, train_neg_lr], ignore_index=True)
    print("[fit] LR on training ...")
    Xtr = featurize(train_lr)
    ytr = train_lr["label"].values
    lr_clf = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, ytr)

    for split_df, name in [(s0, "S0"), (s2, "S2")]:
        Xte = featurize(split_df)
        p = lr_clf.predict_proba(Xte)[:, 1]
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        split_df["lr_pred_p"] = p
        split_df["nll"] = -(split_df["label"] * np.log(p) + (1 - split_df["label"]) * np.log(1 - p))
        auc = roc_auc_score(split_df["label"], p)
        print(f"  {name} LR AUC: {auc:.4f}, n={len(split_df)}")

    # Pool & analyze
    pool = pd.concat([s0, s2], ignore_index=True)
    pool = pool[(pool["M2"] > 1) & pool["consensus_vote"].notna() & pool["H_rel"].notna()].copy()
    pool["logM2"] = np.log(pool["M2"])
    pool["is_S2"] = (pool["split"] == "s2").astype(int)
    print(f"\nPooled (M2>1, consensus available): {len(pool)} (S0={(pool['split']=='s0').sum()}, S2={(pool['split']=='s2').sum()})")

    # Restrict to Q3 within each split for fair comparison
    pool_q3 = []
    for nm, sub in pool.groupby("split"):
        q67 = sub["M2"].quantile(0.67)
        pool_q3.append(sub[sub["M2"] > q67])
    pool_q3 = pd.concat(pool_q3, ignore_index=True)
    print(f"Q3-within-split pool: {len(pool_q3)} (S0={(pool_q3['split']=='s0').sum()}, S2={(pool_q3['split']=='s2').sum()})")

    # Interaction regression
    print("\n=== Interaction model: NLL ~ ... + feature × split ===")
    from itertools import product
    for feature in ["H_rel", "consensus_vote", "consensus_var"]:
        # Standardize feature within Q3
        pool_q3[f"{feature}_z"] = (pool_q3[feature] - pool_q3[feature].mean()) / pool_q3[feature].std()
        # Centered split indicator (S2 = +1, S0 = -1)
        # Model: NLL ~ label + logM2 + feature_z + is_S2 + feature_z*is_S2 + label*feature_z + label*is_S2 + label*feature_z*is_S2
        X = np.column_stack([
            pool_q3["label"].values,
            pool_q3["logM2"].values,
            pool_q3[f"{feature}_z"].values,
            pool_q3["is_S2"].values,
            pool_q3[f"{feature}_z"].values * pool_q3["is_S2"].values,
            pool_q3["label"].values * pool_q3[f"{feature}_z"].values,
            pool_q3["label"].values * pool_q3["is_S2"].values,
            pool_q3["label"].values * pool_q3[f"{feature}_z"].values * pool_q3["is_S2"].values,
        ])
        y = pool_q3["nll"].values
        lr_ = LinearRegression().fit(X, y)
        coefs = dict(zip([
            "label", "logM2", f"{feature}_z", "is_S2",
            f"{feature}_z*is_S2",
            f"label*{feature}_z",
            "label*is_S2",
            f"label*{feature}_z*is_S2",
        ], lr_.coef_))
        print(f"\nFeature: {feature}")
        for k, v in coefs.items():
            tag = "  ← INTERACTION" if "is_S2" in k and feature in k else ""
            print(f"  {k:35s}: {v:+.5f}{tag}")
        print(f"  R²: {lr_.score(X, y):.4f}")

    # Decile-level comparison: S0 vs S2 deciles by consensus_var (instead of H_rel)
    print("\n=== Consensus_var decile by split (POS only, Q3, residualized on logM2) ===")
    pool_q3_pos = pool_q3[pool_q3["label"] == 1].copy()
    for nm, sub in pool_q3_pos.groupby("split"):
        sub = sub.dropna(subset=["consensus_var"]).copy()
        if len(sub) < 500: continue
        lr_lin = LinearRegression().fit(np.log(sub["M2"].values).reshape(-1, 1), sub["nll"].values)
        sub["resid"] = sub["nll"].values - lr_lin.predict(np.log(sub["M2"].values).reshape(-1, 1))
        sub["dec"] = pd.qcut(sub["consensus_var"], 10, labels=False, duplicates="drop")
        t = sub.groupby("dec").agg(NLL_resid=("resid", "mean"), vote=("consensus_vote", "mean"))
        print(f"\n  {nm} (n={len(sub)}):")
        print(t)
        rho, _ = spearmanr(sub["consensus_var"], sub["nll"])
        print(f"  ρ(consensus_var, NLL | pos, {nm}) = {rho:.4f}")

    # SAme for neg
    print("\n=== Consensus_var decile by split (NEG only, Q3, residualized on logM2) ===")
    pool_q3_neg = pool_q3[pool_q3["label"] == 0].copy()
    for nm, sub in pool_q3_neg.groupby("split"):
        sub = sub.dropna(subset=["consensus_var"]).copy()
        if len(sub) < 200: continue
        lr_lin = LinearRegression().fit(np.log(sub["M2"].values).reshape(-1, 1), sub["nll"].values)
        sub["resid"] = sub["nll"].values - lr_lin.predict(np.log(sub["M2"].values).reshape(-1, 1))
        sub["dec"] = pd.qcut(sub["consensus_var"], 10, labels=False, duplicates="drop")
        t = sub.groupby("dec").agg(NLL_resid=("resid", "mean"), vote=("consensus_vote", "mean"))
        print(f"\n  {nm} (n={len(sub)}):")
        print(t)
        rho, _ = spearmanr(sub["consensus_var"], sub["nll"])
        print(f"  ρ(consensus_var, NLL | neg, {nm}) = {rho:.4f}")

    print(f"\nelapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
