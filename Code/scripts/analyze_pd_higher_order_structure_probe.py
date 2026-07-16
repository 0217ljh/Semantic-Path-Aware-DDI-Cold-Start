from __future__ import annotations

"""Pair-level higher-order STRUCTURAL probe for pharmacodynamic (PD) DDIs (S2 cold-start).

Research question. For PD DDIs in cold-start (S2), can pair-level higher-order
graph structure (path multiplicity, degree-corrected common-neighbors, and
especially subgraph/motif density of the shared neighborhood) separate
positives from negatives BETTER than a simple walk-count baseline? This tests
whether structure ABOVE NBFNet's relational-1-WL ceiling carries PD signal.

No semantics, no molecular, no node identity. Pure topology only. NO model
training. Hand-crafted pair features plus cheap classifiers (LogisticRegression,
HistGradientBoosting). We report TEST AUROC and AUPRC.

Feature families (nested).
  F_count   = len-2 walks (common neighbors), len-3 walks, len-4 walks.
  F_degcorr = F_count plus Adamic-Adar and Resource-Allocation.
  F_motif   = F_degcorr plus subgraph/motif density of the shared neighborhood.

Reuses the fast dense 1900x1900 pair-matrix machinery from
analyze_ddi_merged_kg_structure.py and the EXACT S2 split, PD bucket, cap, and
seed from analyze_pd_effect_identity_probe.py so results are comparable.

Run (from project root, via WSL conda env per project rules):
  python Code/scripts/analyze_pd_higher_order_structure_probe.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
MAX_POS_TRAIN = 8000  # cap per class (train AND test), same as identity probe
S_CAP = 1000  # subsample cap on the shared-neighbor set for motif density

ROOT = "/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start"
SPLIT = os.path.join(ROOT, "Code/data/KG/drugbank/splits/seed42")
KG = os.path.join(ROOT, "Code/data/KG/_merged_kg")
RUN_DIR = os.path.join(ROOT, "Code/runs/2026-06-08__pd_higher_order_structure_probe")
os.makedirs(RUN_DIR, exist_ok=True)

# PD buckets and pd_mask copied EXACTLY from analyze_pd_effect_identity_probe.py
PD_BUCKETS = (
    "risk or severity",
    "activities",
    "efficacy",
    "cns depression",
    "qtc",
    "hypertension",
    "hypotensive",
    "sedative",
    "adverse effects",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pd_mask(ddi_type: pd.Series) -> pd.Series:
    sl = ddi_type.str.lower()
    m = pd.Series(False, index=ddi_type.index)
    for b in PD_BUCKETS:
        m = m | sl.str.contains(b, regex=False, na=False)
    return m


# --------------------------------------------------------------------------- #
# KG load + symmetric binary self-loop-free adjacency over all merged-KG nodes
# (machinery adapted from analyze_ddi_merged_kg_structure.py load_kg / build_adjacency)
# --------------------------------------------------------------------------- #
def load_kg():
    nodes = pd.read_parquet(os.path.join(KG, "nodes__drugbank_hetionet_primekg.parquet"))
    edges = pd.read_parquet(
        os.path.join(KG, "edges__drugbank_hetionet_primekg__mask1.parquet"),
        columns=["src", "dst"],
    )
    node_ids = nodes["id"].to_numpy()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)
    log(f"nodes={nodes.shape} edges={edges.shape}")
    return nodes, node_ids, id2idx, n_nodes, edges


def build_adjacency(edges, id2idx, n_nodes):
    """Symmetric, binary, self-loop-free, relation-agnostic adjacency CSR A."""
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    nz = src != dst
    src, dst = src[nz], dst[nz]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    log(f"adjacency symmetric binary nnz={A.nnz}")
    return A


# --------------------------------------------------------------------------- #
# Pair feature matrices (only ~1900 drugs -> dense 1900x1900 is cheap)
# --------------------------------------------------------------------------- #
def build_count_matrices(A, drug_idx):
    """F_count matrices: Wco (len-2), W3 (len-3), W4 (len-4).

    Returns (D, deg_node, Wco, W3, W4, w4_skipped).
    """
    D = A[drug_idx, :].tocsr()  # n_drug x n_nodes binary
    deg_node = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)  # node KG degree

    log("[count] len-2 walks Wco = D @ D.T")
    Wco = np.asarray((D @ D.T).todense()).astype(np.float64)
    np.fill_diagonal(Wco, 0.0)

    log("[count] len-3 walks W3 = D @ (A @ D.T)")
    M = (A @ D.T).tocsr()  # n_nodes x n_drug
    W3 = np.asarray((D @ M).todense()).astype(np.float64)
    np.fill_diagonal(W3, 0.0)

    # len-4 walks: M2 = D @ A (sparse n_drug x n_nodes), W4 = M2 @ M2.T dense
    w4_skipped = False
    try:
        log("[count] len-4 walks W4 = (D @ A) @ (D @ A).T")
        M2 = (D @ A).tocsr()
        W4 = np.asarray((M2 @ M2.T).todense()).astype(np.float64)
        np.fill_diagonal(W4, 0.0)
    except MemoryError:
        log("[count] W4 MemoryError -> SKIPPING W4 (logged)")
        W4 = None
        w4_skipped = True
    return D, deg_node, Wco, W3, W4, w4_skipped


def build_degcorr_matrices(D, deg_node):
    """Degree-corrected common-neighbor matrices: Adamic-Adar, Resource-Allocation.

    AA = D_aa @ D.T  with node-column z scaled by 1/log(deg_node[z]) (deg>1 else 0).
    RA = D_ra @ D.T  with node-column z scaled by 1/deg_node[z]      (deg>=1 else 0).
    """
    n_nodes = D.shape[1]

    aa_w = np.zeros(n_nodes, dtype=np.float64)
    ok_aa = deg_node > 1.0
    aa_w[ok_aa] = 1.0 / np.log(deg_node[ok_aa])
    D_aa = D @ sp.diags(aa_w)
    log("[degcorr] Adamic-Adar AA = D_aa @ D.T")
    AA = np.asarray((D_aa @ D.T).todense()).astype(np.float64)
    np.fill_diagonal(AA, 0.0)

    ra_w = np.zeros(n_nodes, dtype=np.float64)
    ok_ra = deg_node >= 1.0
    ra_w[ok_ra] = 1.0 / deg_node[ok_ra]
    D_ra = D @ sp.diags(ra_w)
    log("[degcorr] Resource-Allocation RA = D_ra @ D.T")
    RA = np.asarray((D_ra @ D.T).todense()).astype(np.float64)
    np.fill_diagonal(RA, 0.0)

    return AA, RA


def compute_motif_features(pairs_la, pairs_lb, drug_idx, A, rng):
    """Per-pair subgraph/motif density of the shared neighborhood of (A, B).

    For each pair let S = common neighbors of A and B (intersection of their
    KG-neighbor index sets). Compute |S|, edges within S, density of S, and a
    4-cycle proxy (== edges within S). If |S| > S_CAP we subsample S with the
    seeded rng and compute density over the subsample, logging how many pairs
    were capped.

    Returns (feat_array Nx4, n_capped).
    """
    n = len(pairs_la)
    out = np.zeros((n, 4), dtype=np.float64)  # n_common, n_edges_within_S, density_S, n_4cycle_proxy
    n_capped = 0
    # global indices of drug rows -> their KG neighbor index sets via A
    A_csr = A.tocsr()
    g_a = drug_idx[pairs_la]
    g_b = drug_idx[pairs_lb]
    t_start = time.time()
    for i in range(n):
        ia = A_csr.indices[A_csr.indptr[g_a[i]]:A_csr.indptr[g_a[i] + 1]]
        ib = A_csr.indices[A_csr.indptr[g_b[i]]:A_csr.indptr[g_b[i] + 1]]
        S = np.intersect1d(ia, ib, assume_unique=False)
        n_common = S.size
        if n_common < 2:
            out[i] = (float(n_common), 0.0, 0.0, 0.0)
            continue
        if S.size > S_CAP:
            S = rng.choice(S, size=S_CAP, replace=False)
            S.sort()
            n_capped += 1
        sub = A_csr[S][:, S]
        n_edges_within = sub.nnz / 2.0  # symmetric, self-loop-free
        denom = S.size * (S.size - 1) / 2.0
        density = n_edges_within / denom if denom > 0 else 0.0
        out[i] = (float(n_common), float(n_edges_within), float(density), float(n_edges_within))
        if (i + 1) % 5000 == 0:
            log(f"  [motif] {i + 1}/{n} pairs  capped={n_capped}  elapsed={time.time() - t_start:.1f}s")
    return out, n_capped


# --------------------------------------------------------------------------- #
# Pair assembly: EXACT S2 split / PD bucket / cap / seed from identity probe
# --------------------------------------------------------------------------- #
def assemble_pairs(node_ids, drug_global_idx):
    kg_drug_set = set(node_ids[i] for i in drug_global_idx)

    tr = pd.read_parquet(os.path.join(SPLIT, "train.parquet"))
    tr_neg = pd.read_parquet(os.path.join(SPLIT, "train_negatives/epoch_0.parquet"))
    te = pd.read_parquet(os.path.join(SPLIT, "test_s2.parquet"))
    te_neg = pd.read_parquet(os.path.join(SPLIT, "negatives/test_s2.parquet"))

    rng = np.random.default_rng(SEED)

    def in_kg(df):
        return df[df["drug_a_id"].isin(kg_drug_set) & df["drug_b_id"].isin(kg_drug_set)]

    tr_pos = in_kg(tr[pd_mask(tr["ddi_type"])][["drug_a_id", "drug_b_id"]]).reset_index(drop=True)
    if len(tr_pos) > MAX_POS_TRAIN:
        idx = rng.choice(len(tr_pos), size=MAX_POS_TRAIN, replace=False)
        tr_pos = tr_pos.iloc[idx].reset_index(drop=True)
    tr_neg = in_kg(tr_neg[["drug_a_id", "drug_b_id"]]).reset_index(drop=True)
    n_neg = min(len(tr_pos), len(tr_neg))
    idx = rng.choice(len(tr_neg), size=n_neg, replace=False)
    tr_neg = tr_neg.iloc[idx].reset_index(drop=True)

    te_pos = in_kg(te[pd_mask(te["ddi_type"])][["drug_a_id", "drug_b_id"]]).reset_index(drop=True)
    if len(te_pos) > MAX_POS_TRAIN:
        te_pos = te_pos.iloc[rng.choice(len(te_pos), size=MAX_POS_TRAIN, replace=False)].reset_index(drop=True)
    te_neg = in_kg(te_neg[["drug_a_id", "drug_b_id"]]).reset_index(drop=True)
    if len(te_neg) > MAX_POS_TRAIN:
        te_neg = te_neg.iloc[rng.choice(len(te_neg), size=MAX_POS_TRAIN, replace=False)].reset_index(drop=True)

    log(f"TRAIN pos={len(tr_pos)} neg={len(tr_neg)} | TEST pos={len(te_pos)} neg={len(te_neg)}")
    train_pairs = pd.concat([tr_pos.assign(y=1), tr_neg.assign(y=0)], ignore_index=True)
    test_pairs = pd.concat([te_pos.assign(y=1), te_neg.assign(y=0)], ignore_index=True)
    return train_pairs, test_pairs, rng


def to_local(pairs, db2local):
    la = pairs["drug_a_id"].map(db2local).to_numpy().astype(np.int64)
    lb = pairs["drug_b_id"].map(db2local).to_numpy().astype(np.int64)
    y = pairs["y"].to_numpy().astype(np.int64)
    return la, lb, y


# --------------------------------------------------------------------------- #
# Classifier eval helpers
# --------------------------------------------------------------------------- #
def eval_lr(Xtr, ytr, Xte, yte):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    clf = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000)
    clf.fit(Xtr_s, ytr)
    p = clf.predict_proba(Xte_s)[:, 1]
    return float(roc_auc_score(yte, p)), float(average_precision_score(yte, p))


def eval_gbdt(Xtr, ytr, Xte, yte):
    clf = HistGradientBoostingClassifier(random_state=SEED)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auroc = float(roc_auc_score(yte, p))
    auprc = float(average_precision_score(yte, p))
    return clf, auroc, auprc


def main():
    t0 = time.time()
    nodes, node_ids, id2idx, n_nodes, edges = load_kg()

    drug_mask = (nodes["kind"] == "Drug").to_numpy()
    drug_db_ids = node_ids[drug_mask]
    drug_global_idx = np.array([id2idx[d] for d in drug_db_ids], dtype=np.int64)
    n_drug = len(drug_global_idx)
    db2local = {db: i for i, db in enumerate(drug_db_ids)}
    log(f"n_drug(Drug nodes)={n_drug}")

    A = build_adjacency(edges, id2idx, n_nodes)

    # ---- pair-level dense matrices (computed once) ----
    D, deg_node, Wco, W3, W4, w4_skipped = build_count_matrices(A, drug_global_idx)
    AA, RA = build_degcorr_matrices(D, deg_node)

    # ---- assemble pairs (exact split/cap/seed) ----
    train_pairs, test_pairs, rng = assemble_pairs(node_ids, drug_global_idx)
    tr_la, tr_lb, y_tr = to_local(train_pairs, db2local)
    te_la, te_lb, y_te = to_local(test_pairs, db2local)

    # ---- index count + degcorr features per pair ----
    def index_pairs(la, lb):
        cols = {
            "walks_len2_common_neighbors": Wco[la, lb],
            "walks_len3": W3[la, lb],
        }
        if not w4_skipped:
            cols["walks_len4"] = W4[la, lb]
        deg_cols = {
            "adamic_adar": AA[la, lb],
            "resource_allocation": RA[la, lb],
        }
        return cols, deg_cols

    tr_count, tr_deg = index_pairs(tr_la, tr_lb)
    te_count, te_deg = index_pairs(te_la, te_lb)

    # ---- motif features per pair ----
    log("[motif] computing TRAIN motif features")
    tr_motif_arr, tr_capped = compute_motif_features(tr_la, tr_lb, drug_global_idx, A, rng)
    log("[motif] computing TEST motif features")
    te_motif_arr, te_capped = compute_motif_features(te_la, te_lb, drug_global_idx, A, rng)
    n_capped = int(tr_capped + te_capped)
    log(f"[motif] |S|>{S_CAP} cap hit on {n_capped} pairs (train={tr_capped} test={te_capped})")
    motif_names = ["motif_n_common", "motif_n_edges_within_S", "motif_density_S", "motif_n_4cycle_proxy"]

    # ---- assemble feature families (nested, named) ----
    count_names = list(tr_count.keys())
    degcorr_names = count_names + list(tr_deg.keys())
    motif_family_names = degcorr_names + motif_names

    def stack(cols_dict, extra_dict=None, motif_arr=None, names=None):
        order = list(cols_dict.keys())
        mats = [cols_dict[k].reshape(-1, 1) for k in order]
        if extra_dict is not None:
            for k in extra_dict:
                mats.append(extra_dict[k].reshape(-1, 1))
        if motif_arr is not None:
            mats.append(motif_arr)
        return np.hstack(mats).astype(np.float64)

    Xtr_count = stack(tr_count)
    Xte_count = stack(te_count)
    Xtr_degcorr = stack(tr_count, tr_deg)
    Xte_degcorr = stack(te_count, te_deg)
    Xtr_motif = stack(tr_count, tr_deg, tr_motif_arr)
    Xte_motif = stack(te_count, te_deg, te_motif_arr)

    log(f"feature dims: count={Xtr_count.shape[1]} degcorr={Xtr_degcorr.shape[1]} motif={Xtr_motif.shape[1]}")

    # ---- fit + eval each family x classifier ----
    results = {}
    gbdt_models = {}
    family_specs = [
        ("F_count", Xtr_count, Xte_count, count_names),
        ("F_degcorr", Xtr_degcorr, Xte_degcorr, degcorr_names),
        ("F_motif", Xtr_motif, Xte_motif, motif_family_names),
    ]
    for fam, Xtr, Xte, fnames in family_specs:
        lr_auroc, lr_auprc = eval_lr(Xtr, y_tr, Xte, y_te)
        gb, gb_auroc, gb_auprc = eval_gbdt(Xtr, y_tr, Xte, y_te)
        gbdt_models[fam] = (gb, Xte, fnames)
        results[fam] = {
            "LR": {"AUROC": lr_auroc, "AUPRC": lr_auprc},
            "GBDT": {"AUROC": gb_auroc, "AUPRC": gb_auprc},
            "best_AUROC": max(lr_auroc, gb_auroc),
            "best_classifier": "LR" if lr_auroc >= gb_auroc else "GBDT",
        }
        log(f"{fam}: LR AUROC={lr_auroc:.4f} AUPRC={lr_auprc:.4f} | "
            f"GBDT AUROC={gb_auroc:.4f} AUPRC={gb_auprc:.4f}")

    # ---- headline gains (best-classifier AUROC per family) ----
    best_count = results["F_count"]["best_AUROC"]
    best_degcorr = results["F_degcorr"]["best_AUROC"]
    best_motif = results["F_motif"]["best_AUROC"]
    gain_motif = best_motif - best_count
    gain_degcorr = best_degcorr - best_count
    log(f"best AUROC: F_count={best_count:.4f} F_degcorr={best_degcorr:.4f} F_motif={best_motif:.4f}")
    log(f"gain F_degcorr-F_count={gain_degcorr:+.4f}  F_motif-F_count={gain_motif:+.4f}")

    # ---- feature importance: permutation importance on F_motif GBDT ----
    gb_motif, Xte_motif_fi, motif_fnames = gbdt_models["F_motif"]
    log("[importance] permutation importance on F_motif GBDT (TEST)")
    perm = permutation_importance(
        gb_motif, Xte_motif_fi, y_te, scoring="roc_auc",
        n_repeats=10, random_state=SEED, n_jobs=1,
    )
    imp_rows = []
    for j, fname in enumerate(motif_fnames):
        imp_rows.append({
            "feature": fname,
            "perm_importance_mean": float(perm.importances_mean[j]),
            "perm_importance_std": float(perm.importances_std[j]),
        })
    imp_df = pd.DataFrame(imp_rows).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    imp_csv = os.path.join(RUN_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_csv, index=False)
    log(f"wrote {imp_csv}")
    top_feat = imp_df.iloc[0]["feature"]

    # ---- verdict ----
    if gain_motif >= 0.03 and best_motif >= 0.68:
        verdict = ("motif-separates: higher-order/motif structure separates PD pos/neg "
                   "beyond walk-counts -> worth building a subgraph-GNN / SEAL / k-WL model")
    elif gain_motif < 0.01 and gain_degcorr < 0.01:
        verdict = ("tapped: even motif/subgraph structure is exhausted -> pure structure is "
                   "tapped, remaining PD signal must come from molecular/text")
    else:
        verdict = ("borderline: gain is between thresholds, report honestly "
                   "(neither a clear motif win nor a clear tapped result)")
    log(f"VERDICT: {verdict}")

    # ---- reference numbers (cited as context, NOT recomputed) ----
    reference = {
        "real_nbfnet_v1_7_s2_pd_auroc": 0.727,
        "prior_handcrafted_shared_effect_pd_auroc_range": "0.60-0.64",
        "note": "reference numbers cited from project context, not recomputed here",
    }

    out = {
        "config": {
            "seed": SEED, "max_pos_train": MAX_POS_TRAIN, "s_cap": S_CAP,
            "pd_buckets": list(PD_BUCKETS),
            "w4_skipped": w4_skipped,
            "adjacency": "symmetric binary self-loop-free relation-agnostic over all merged-KG nodes",
        },
        "sizes": {
            "n_drug": int(n_drug),
            "n_train_pos": int((y_tr == 1).sum()), "n_train_neg": int((y_tr == 0).sum()),
            "n_test_pos": int((y_te == 1).sum()), "n_test_neg": int((y_te == 0).sum()),
        },
        "feature_families": {
            "F_count": count_names,
            "F_degcorr": degcorr_names,
            "F_motif": motif_family_names,
        },
        "results": results,
        "headline": {
            "best_auroc_F_count": best_count,
            "best_auroc_F_degcorr": best_degcorr,
            "best_auroc_F_motif": best_motif,
            "gain_F_degcorr_minus_F_count": gain_degcorr,
            "gain_F_motif_minus_F_count": gain_motif,
        },
        "motif_cap": {
            "s_cap": S_CAP,
            "n_pairs_capped_total": n_capped,
            "n_pairs_capped_train": int(tr_capped),
            "n_pairs_capped_test": int(te_capped),
        },
        "top_importance_feature_F_motif_gbdt": top_feat,
        "feature_importance_csv": imp_csv,
        "verdict": verdict,
        "reference_numbers_context": reference,
        "elapsed_sec": time.time() - t0,
    }
    metrics_path = os.path.join(RUN_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {metrics_path}")
    log(f"DONE in {time.time() - t0:.1f}s")

    print("\n=== FEATURE IMPORTANCE (F_motif GBDT, permutation, TEST AUROC) ===")
    print(imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
