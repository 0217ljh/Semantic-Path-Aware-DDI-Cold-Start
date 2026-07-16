"""exp3 step 15 — partial correlation + S0 warm-start contrast.

Two questions:
(1) Are H_rel and neighbor-consensus the SAME phenomenon or distinct?
    Regression: nll ~ H_rel + consensus_vote + consensus_var + label + interactions
(2) Is the H_rel U-shape cold-start specific?
    Build same features on test_S0 (warm-start) using LR predictions trained
    on train.parquet. Check if U-shape attenuates.
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


# ======================================
# Part 1: partial correlation
# ======================================
def part1_partial():
    print("="*60)
    print("PART 1: H_rel vs neighbor-consensus — same or distinct?")
    print("="*60)
    df = pd.read_parquet(OUT_DIR / "pair_features_consensus.parquet")
    df = df[df["M2_count"] > 1].copy()
    q67 = df["M2_count"].quantile(0.67)
    q3 = df[df["M2_count"] > q67].copy()
    print(f"Q3 n={len(q3)}")

    # Raw correlations
    print("\n[Raw pairwise correlations within Q3]")
    cols = ["H_rel", "pair_consensus_vote", "pair_consensus_var", "pair_close_tx", "M2_count", "nll", "label"]
    available = [c for c in cols if c in q3.columns and q3[c].notna().any()]
    sub = q3[available].dropna()
    print(sub.corr().round(3))

    # Variance decomposition
    print("\n[OLS: nll ~ H_rel + consensus_vote + consensus_var + label + interactions]")
    sub2 = q3.dropna(subset=["H_rel", "pair_consensus_vote", "pair_consensus_var"]).copy()
    sub2["logM2"] = np.log(sub2["M2_count"])
    sub2["H_rel_x_label"] = sub2["H_rel"] * sub2["label"]
    sub2["var_x_label"] = sub2["pair_consensus_var"] * sub2["label"]
    sub2["vote_x_label"] = sub2["pair_consensus_vote"] * sub2["label"]

    # Hierarchical: just label, then add H_rel, then add consensus
    from sklearn.linear_model import LinearRegression
    y = sub2["nll"].values
    base = sub2[["label", "logM2"]].values
    r2_base = LinearRegression().fit(base, y).score(base, y)
    print(f"  R²(base: label + logM2) = {r2_base:.4f}")

    plus_hrel = sub2[["label", "logM2", "H_rel", "H_rel_x_label"]].values
    r2_hrel = LinearRegression().fit(plus_hrel, y).score(plus_hrel, y)
    print(f"  R²(+H_rel + H_rel*label) = {r2_hrel:.4f}, ΔR² = {r2_hrel - r2_base:+.4f}")

    plus_cons = sub2[["label", "logM2", "pair_consensus_vote", "pair_consensus_var", "vote_x_label", "var_x_label"]].values
    r2_cons = LinearRegression().fit(plus_cons, y).score(plus_cons, y)
    print(f"  R²(+consensus + cons*label) = {r2_cons:.4f}, ΔR² = {r2_cons - r2_base:+.4f}")

    full = sub2[["label", "logM2", "H_rel", "H_rel_x_label", "pair_consensus_vote", "pair_consensus_var", "vote_x_label", "var_x_label"]].values
    r2_full = LinearRegression().fit(full, y).score(full, y)
    print(f"  R²(full: all together) = {r2_full:.4f}")
    print(f"  ΔR²(H_rel | consensus) = {r2_full - r2_cons:+.4f}  ← H_rel's unique contribution AFTER consensus")
    print(f"  ΔR²(consensus | H_rel) = {r2_full - r2_hrel:+.4f}  ← consensus's unique contribution AFTER H_rel")

    # Visual: H_rel decile within consensus_var stratified
    pos = q3[q3["label"] == 1].copy()
    pos = pos.dropna(subset=["pair_consensus_var", "H_rel"])
    pos["cons_tertile"] = pd.qcut(pos["pair_consensus_var"], 3, labels=["lowVar", "midVar", "highVar"], duplicates="drop")
    print("\n[POS within Q3: NLL by H_rel decile × consensus_var tertile]")
    for ct, sub in pos.groupby("cons_tertile", observed=True):
        if len(sub) < 200: continue
        sub = sub.copy()
        sub["Hr_dec"] = pd.qcut(sub["H_rel"], 10, labels=False, duplicates="drop")
        t = sub.groupby("Hr_dec").agg(NLL=("nll", "mean"), n=("nll", "count"))
        print(f"\n  Consensus tertile: {ct}, n={len(sub)}")
        # Show key deciles only
        for d in [0, 1, 5, 8, 9]:
            if d in t.index:
                print(f"    dec{d}: NLL={t.loc[d, 'NLL']:.4f} n={t.loc[d, 'n']}")


# ======================================
# Part 2: S0 contrast
# ======================================
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


def part2_s0_contrast():
    print("\n" + "="*60)
    print("PART 2: S0 warm-start contrast")
    print("="*60)
    SPLITS = ROOT / "Code/data/KG/drugbank/splits/seed42"
    # Check S0
    s0_pos = pd.read_parquet(SPLITS / "test_s0.parquet")[["drug_a_id", "drug_b_id"]].copy()
    s0_pos["label"] = 1.0
    s0_neg_path = SPLITS / "negatives/test_s0.parquet"
    if not s0_neg_path.exists():
        print(f"  S0 negatives not found at {s0_neg_path}")
        return
    s0_neg = pd.read_parquet(s0_neg_path)[["drug_a_id", "drug_b_id"]].copy()
    s0_neg["label"] = 0.0
    s0 = pd.concat([s0_pos, s0_neg], ignore_index=True)
    print(f"  S0 test pairs: {len(s0)} (pos={len(s0_pos)}, neg={len(s0_neg)})")

    s0_drugs = set(s0["drug_a_id"]) | set(s0["drug_b_id"])
    print(f"  S0 unique drugs: {len(s0_drugs)}")

    # Build edges + mediator features for S0 pairs
    print("[S0] loading edges/nodes ...")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    nodes["kind_canon"] = nodes["kind"].map(KIND_CANON).fillna("Other")
    id2kc = dict(zip(nodes["id"], nodes["kind_canon"]))

    # Drug-mediator map for S0 drugs
    e1 = edges[edges["src"].isin(s0_drugs)][["src", "dst", "relation"]].rename(columns={"src": "d", "dst": "m", "relation": "r"})
    e2 = edges[edges["dst"].isin(s0_drugs)][["dst", "src", "relation"]].rename(columns={"dst": "d", "src": "m", "relation": "r"})
    de = pd.concat([e1, e2], ignore_index=True)
    de["m_kind"] = de["m"].map(id2kc)
    de = de[de["m_kind"].isin(MEDIATOR_KINDS)]
    drug_to_med = {}
    for d, grp in de.groupby("d"):
        drug_to_med[d] = {m: list(sub["r"]) for m, sub in grp.groupby("m")}

    # Compute M2_count and H_rel for S0
    M2 = np.zeros(len(s0), dtype=np.int32)
    Hr = np.full(len(s0), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(s0["drug_a_id"].values, s0["drug_b_id"].values)):
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
    s0["M2"] = M2
    s0["H_rel"] = Hr
    print(f"  S0 pairs with M2>1: {int((M2>1).sum())}")

    # Train LR on train.parquet pairs using E7 22-feature design
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
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
                if term == drug or term in drug_set: continue
                n2[drug].add((term, KIND_TO_GROUP.get(id2kind.get(term, ""), "other")))

    def featurize(df_):
        n = len(df_)
        n_g = len(KIND_ORDER)
        X = np.zeros((n, 2 * n_g), dtype=np.float32)
        for i, (da, db) in enumerate(zip(df_["drug_a_id"].values, df_["drug_b_id"].values)):
            s1 = n1.get(da, set()) & n1.get(db, set())
            s2 = n2.get(da, set()) & n2.get(db, set())
            c1 = defaultdict(int); c2 = defaultdict(int)
            for _, g in s1: c1[g] += 1
            for _, g in s2: c2[g] += 1
            for j, grp in enumerate(KIND_ORDER):
                X[i, j] = np.log1p(c1[grp])
                X[i, n_g + j] = np.log1p(c2[grp])
        return X

    train_pos = pd.read_parquet(SPLITS / "train.parquet")[["drug_a_id", "drug_b_id"]]
    train_pos["label"] = 1
    train_neg = pd.read_parquet(SPLITS / "train_negatives/epoch_0.parquet")[["drug_a_id", "drug_b_id"]]
    train_neg["label"] = 0
    train = pd.concat([train_pos, train_neg], ignore_index=True)
    print(f"[S0] training LR ...")
    Xtr = featurize(train)
    ytr = train["label"].values
    lr_clf = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, ytr)

    Xte = featurize(s0)
    p = lr_clf.predict_proba(Xte)[:, 1]
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    s0["lr_pred_p"] = p
    s0["lr_nll"] = -(s0["label"] * np.log(p) + (1 - s0["label"]) * np.log(1 - p))
    auc_s0 = roc_auc_score(s0["label"], p)
    print(f"  S0 LR AUC: {auc_s0:.4f}")

    # Same Q3 analysis
    s0_q = s0[s0["M2"] > 1].copy()
    q67_s0 = s0_q["M2"].quantile(0.67)
    s0_q3 = s0_q[s0_q["M2"] > q67_s0].copy()
    print(f"  S0 Q3 n={len(s0_q3)}, |M2|>{q67_s0:.0f}, label_rate={s0_q3['label'].mean():.3f}")

    s0_pos = s0_q3[s0_q3["label"] == 1].copy()
    s0_neg = s0_q3[s0_q3["label"] == 0].copy()

    if len(s0_pos) < 500:
        print("  S0 pos too small")
        return
    lr_lin_p = LinearRegression().fit(np.log(s0_pos["M2"].values).reshape(-1, 1), s0_pos["lr_nll"].values)
    s0_pos["resid"] = s0_pos["lr_nll"].values - lr_lin_p.predict(np.log(s0_pos["M2"].values).reshape(-1, 1))
    s0_pos["Hr_dec"] = pd.qcut(s0_pos["H_rel"], 10, labels=False, duplicates="drop")
    p_dec_s0 = s0_pos.groupby("Hr_dec")["resid"].mean()

    lr_lin_n = LinearRegression().fit(np.log(s0_neg["M2"].values).reshape(-1, 1), s0_neg["lr_nll"].values)
    s0_neg["resid"] = s0_neg["lr_nll"].values - lr_lin_n.predict(np.log(s0_neg["M2"].values).reshape(-1, 1))
    s0_neg["Hr_dec"] = pd.qcut(s0_neg["H_rel"], 10, labels=False, duplicates="drop")
    n_dec_s0 = s0_neg.groupby("Hr_dec")["resid"].mean()

    print(f"\n[S0 POS deciles by H_rel]: {p_dec_s0.tolist()}")
    print(f"[S0 NEG deciles by H_rel]: {n_dec_s0.tolist()}")
    print(f"  POS U-shape: {(p_dec_s0.iloc[0] > p_dec_s0.iloc[5]) and (p_dec_s0.iloc[9] > p_dec_s0.iloc[5])}")
    print(f"  POS dec9-8 jump: {p_dec_s0.iloc[9] - p_dec_s0.iloc[8]:+.4f}")
    print(f"  POS dec0-1 jump: {p_dec_s0.iloc[0] - p_dec_s0.iloc[1]:+.4f}")
    print(f"  NEG inv-U: {(n_dec_s0.iloc[5] > n_dec_s0.iloc[0]) and (n_dec_s0.iloc[5] > n_dec_s0.iloc[9])}")

    rho_pos_s0, _ = spearmanr(s0_pos["H_rel"], s0_pos["lr_nll"])
    print(f"  ρ(H_rel, NLL | pos, S0) = {rho_pos_s0:.4f}")

    # Compare to S2 (load from previous run)
    print("\n[S2 reference]: seed42 Q3 LR jump_89=+0.429 (from script 08), pos U-shape ✓")
    print(f"[S0 contrast]: jump_89={p_dec_s0.iloc[9] - p_dec_s0.iloc[8]:+.4f}")


if __name__ == "__main__":
    part1_partial()
    part2_s0_contrast()
