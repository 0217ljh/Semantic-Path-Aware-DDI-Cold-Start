from __future__ import annotations

"""Global Structural Proximity (GSP) probe for pharmacodynamic (PD) DDIs (S2 cold-start).

Research question. Do GSP features (Personalized-PageRank / random-walk-with-restart
proximity, diffusion distance, truncated Katz, and typed length-2 metapath counts)
separate PD positives from negatives BETTER than the Local Structural Proximity (LSP)
features we already proved dead (common neighbours / walk counts / Adamic-Adar /
Resource-Allocation, which top out around 0.66 AUROC)? This tests the one structural
axis we have NOT yet tested.

Literature motivation. Mao et al. "Revisiting Link Prediction: A Data Perspective",
ICLR 2024 (arXiv:2310.00793) proves GSP (Katz / PPR / SimRank) recovers
link-prediction performance precisely when LSP is deficient, which is exactly our PD
regime (PD connects through high-degree hubs and is LSP-deficient).

NO-MODEL probe. Hand-crafted pair features plus cheap classifiers
(LogisticRegression, HistGradientBoosting). We report TEST AUROC and AUPRC. Pure
topology, no molecular, no text, no node identity.

Feature families.
  F_lsp     = [walks_len2, walks_len3, adamic_adar, resource_allocation]  (LSP reference)
  F_ppr     = PPR/RWR proximity (raw RWR + sym-degnorm operator), diffusion distance,
              PPR inner product, directional max/min, for both operators.
  F_katz    = truncated Katz drug-drug path sums for several beta values.
  F_typed   = typed length-2 metapath counts per node-kind bucket (TYPE_BUCKETS).
  F_gsp_all = F_ppr + F_katz + F_typed.

Reuses the EXACT S2 split, PD bucket, cap (8000/class), seed (42), symmetric binary
adjacency, drug-index machinery, and LR+GBDT eval from
analyze_pd_higher_order_structure_probe.py so results are directly comparable. Reuses
TYPE_BUCKETS from analyze_ddi_merged_kg_structure.py for typed metapath features.

Run (from project root, via WSL conda env per project rules):
  python Code/scripts/analyze_pd_global_proximity_probe.py
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
MAX_POS_TRAIN = 8000  # cap per class (train AND test), same as twin probe

# PPR / RWR hyper-parameters
PPR_ALPHA = 0.85  # walk continuation prob (restart = 1 - alpha = 0.15)
PPR_MAX_ITERS = 20
PPR_TOL = 1e-6  # L1 convergence delta threshold
PPR_CHUNK = 0  # 0 = no chunking; otherwise chunk drug-source columns this many at a time

# Katz hyper-parameters
KATZ_BETAS = (0.1, 0.05, 0.01)
KATZ_L = 4  # path length truncation (sum l = 2..L)

ROOT = "/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start"
SPLIT = os.path.join(ROOT, "Code/data/KG/drugbank/splits/seed42")
KG = os.path.join(ROOT, "Code/data/KG/_merged_kg")
RUN_DIR = os.path.join(ROOT, "Code/runs/2026-06-08__pd_global_proximity_probe")
os.makedirs(RUN_DIR, exist_ok=True)

# PD buckets and pd_mask copied EXACTLY from the twin probe
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

# Node-kind -> bucket map reused from analyze_ddi_merged_kg_structure.py (lines 57-66).
TYPE_BUCKETS = {
    "gene_protein": {"gene/protein", "Gene", "Protein"},
    "side_effect": {"Side Effect"},
    "pheno_effect": {"effect/phenotype", "Symptom"},
    "disease": {"disease", "Disease"},
    "pathway": {"pathway", "Pathway"},
    "drug": {"drug", "Drug", "Compound"},
    "biological_process": {"biological_process", "Biological Process"},
    "anatomy": {"anatomy", "Anatomy"},
}

# Buckets used for the typed metapath family (mechanism-bearing biological kinds).
TYPED_FAMILY_BUCKETS = (
    "gene_protein",
    "side_effect",
    "pheno_effect",
    "disease",
    "pathway",
    "anatomy",
    "biological_process",
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
# KG load + symmetric binary self-loop-free adjacency (machinery from twin probe)
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


def build_node_bucket(nodes):
    """Per-node bucket id over TYPE_BUCKETS (-1 if not in any bucket)."""
    kinds = nodes["kind"].to_numpy()
    kind2bucket = {}
    for b, ks in TYPE_BUCKETS.items():
        for k in ks:
            kind2bucket[k] = b
    bucket_names = list(TYPE_BUCKETS.keys())
    bucket2id = {b: i for i, b in enumerate(bucket_names)}
    node_bucket = np.array(
        [bucket2id.get(kind2bucket.get(k, "_other"), -1) for k in kinds], dtype=np.int32
    )
    return node_bucket, bucket_names


# --------------------------------------------------------------------------- #
# F_lsp matrices (reference, copied from the twin probe so the baseline is identical)
# --------------------------------------------------------------------------- #
def build_lsp_matrices(A, drug_idx):
    """LSP reference matrices: Wco (len-2), W3 (len-3), AA (Adamic-Adar), RA (RA)."""
    D = A[drug_idx, :].tocsr()  # n_drug x n_nodes binary
    deg_node = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)  # node KG degree

    log("[lsp] len-2 walks Wco = D @ D.T")
    Wco = np.asarray((D @ D.T).todense()).astype(np.float64)
    np.fill_diagonal(Wco, 0.0)

    log("[lsp] len-3 walks W3 = D @ (A @ D.T)")
    M = (A @ D.T).tocsr()
    W3 = np.asarray((D @ M).todense()).astype(np.float64)
    np.fill_diagonal(W3, 0.0)

    n_nodes = D.shape[1]
    aa_w = np.zeros(n_nodes, dtype=np.float64)
    ok_aa = deg_node > 1.0
    aa_w[ok_aa] = 1.0 / np.log(deg_node[ok_aa])
    D_aa = D @ sp.diags(aa_w)
    log("[lsp] Adamic-Adar AA = D_aa @ D.T")
    AA = np.asarray((D_aa @ D.T).todense()).astype(np.float64)
    np.fill_diagonal(AA, 0.0)

    ra_w = np.zeros(n_nodes, dtype=np.float64)
    ok_ra = deg_node >= 1.0
    ra_w[ok_ra] = 1.0 / deg_node[ok_ra]
    D_ra = D @ sp.diags(ra_w)
    log("[lsp] Resource-Allocation RA = D_ra @ D.T")
    RA = np.asarray((D_ra @ D.T).todense()).astype(np.float64)
    np.fill_diagonal(RA, 0.0)

    return D, deg_node, Wco, W3, AA, RA


# --------------------------------------------------------------------------- #
# F_ppr: Personalized PageRank / RWR landing vectors by power iteration
# --------------------------------------------------------------------------- #
def build_rw_operator(A, deg_node, mode):
    """Random-walk transition operator applied as Ahat @ P.

    mode == 'rw'  : row-normalized random walk Ahat_rw = D^{-1} A
                    (so column j of A is scaled by 1/deg of the SOURCE row -> we
                     implement Ahat_rw @ P as Dinv @ (A @ P)).
    mode == 'sym' : symmetric-normalized Ahat_sym = D^{-1/2} A D^{-1/2}.
    """
    n = A.shape[0]
    deg_safe = deg_node.copy()
    deg_safe[deg_safe == 0.0] = 1.0
    if mode == "rw":
        dinv = sp.diags((1.0 / deg_safe).astype(np.float32))
        return dinv @ A  # D^{-1} A (CSR)
    if mode == "sym":
        dinv_sqrt = sp.diags((1.0 / np.sqrt(deg_safe)).astype(np.float32))
        return dinv_sqrt @ A @ dinv_sqrt
    raise ValueError(mode)


def compute_ppr(A, deg_node, drug_idx, mode):
    """PPR / RWR landing matrix P (n_nodes x n_drug) for all drug sources.

    Iteration  P = (1 - alpha) * E + alpha * (Ahat @ P), with E[:, j] the one-hot
    teleport for drug-source j and alpha = PPR_ALPHA. Optionally chunk the drug
    sources if PPR_CHUNK > 0. Returns (P float32, n_iters, final_delta, chunked_flag).
    """
    n_nodes = A.shape[0]
    n_drug = len(drug_idx)
    Ahat = build_rw_operator(A, deg_node, mode).tocsr().astype(np.float32)

    chunked = PPR_CHUNK > 0 and PPR_CHUNK < n_drug
    P = np.zeros((n_nodes, n_drug), dtype=np.float32)
    last_delta = float("nan")
    last_iters = 0

    if not chunked:
        E = np.zeros((n_nodes, n_drug), dtype=np.float32)
        E[drug_idx, np.arange(n_drug)] = 1.0
        P = E.copy()
        for it in range(PPR_MAX_ITERS):
            P_new = (1.0 - PPR_ALPHA) * E + PPR_ALPHA * (Ahat @ P)
            delta = float(np.abs(P_new - P).sum() / n_drug)  # mean L1 per source
            P = P_new
            last_iters = it + 1
            last_delta = delta
            if (it + 1) % 5 == 0 or it == 0:
                log(f"[ppr-{mode}] iter {it + 1}/{PPR_MAX_ITERS} mean_L1_delta={delta:.3e}")
            if delta < PPR_TOL:
                log(f"[ppr-{mode}] converged at iter {it + 1} delta={delta:.3e}")
                break
    else:
        log(f"[ppr-{mode}] CHUNKING drug sources, chunk={PPR_CHUNK}")
        col0 = 0
        deltas = []
        iters_seen = []
        while col0 < n_drug:
            col1 = min(col0 + PPR_CHUNK, n_drug)
            idx_chunk = drug_idx[col0:col1]
            m = col1 - col0
            E = np.zeros((n_nodes, m), dtype=np.float32)
            E[idx_chunk, np.arange(m)] = 1.0
            Pc = E.copy()
            d = float("nan")
            ni = 0
            for it in range(PPR_MAX_ITERS):
                Pc_new = (1.0 - PPR_ALPHA) * E + PPR_ALPHA * (Ahat @ Pc)
                d = float(np.abs(Pc_new - Pc).sum() / m)
                Pc = Pc_new
                ni = it + 1
                if d < PPR_TOL:
                    break
            P[:, col0:col1] = Pc
            deltas.append(d)
            iters_seen.append(ni)
            log(f"[ppr-{mode}] chunk [{col0}:{col1}] iters={ni} delta={d:.3e}")
            col0 = col1
        last_delta = float(np.max(deltas))
        last_iters = int(np.max(iters_seen))

    return P, last_iters, last_delta, chunked


def ppr_pair_features(P, drug_idx):
    """From a landing matrix P (n_nodes x n_drug) build drug-drug proximity blocks.

    Returns dict of 1900x1900 dense matrices:
      P_dd            = P[drug_idx, :]  (rows = drug nodes, cols = drug sources)
      sym             = P_dd + P_dd.T   (symmetrized directional proximity)
      pmax, pmin      = elementwise max/min of the two directions
      gram            = P[:, :].T @ P[:, :]  (full-vector inner product, 1900x1900)
      diff_dist       = sqrt( g[u,u] + g[v,v] - 2 g[u,v] )  (diffusion distance)
    """
    P_dd = P[drug_idx, :].astype(np.float64)  # 1900 x 1900
    sym = P_dd + P_dd.T
    pmax = np.maximum(P_dd, P_dd.T)
    pmin = np.minimum(P_dd, P_dd.T)

    # full-vector Gram over the drug-source columns of P (cheap: 1900x1900)
    Pcols = P[:, :].astype(np.float64)  # n_nodes x n_drug
    gram = Pcols.T @ Pcols  # 1900 x 1900 inner products
    diag = np.diag(gram)
    d2 = diag[:, None] + diag[None, :] - 2.0 * gram
    np.clip(d2, 0.0, None, out=d2)
    diff_dist = np.sqrt(d2)
    return {"sym": sym, "pmax": pmax, "pmin": pmin, "gram": gram, "diff_dist": diff_dist}


# --------------------------------------------------------------------------- #
# F_katz: truncated Katz drug-drug path sums  K = sum_{l=2..L} beta^l (A^l)_dd
# --------------------------------------------------------------------------- #
def build_katz_matrices(A, drug_idx, betas, L):
    """One drug-drug Katz matrix per beta. Uses sparse drug-row propagation.

    Walk-length-l drug-drug counts: start from D (n_drug x n_nodes), repeatedly
    multiply by A. K = sum_{l=2..L} beta^l * (D A^{l-1})[:, drug_idx], accumulated
    in dense 1900x1900 (cheap). Returns {beta: K (1900x1900 float64)}.
    """
    D = A[drug_idx, :].tocsr().astype(np.float64)  # n_drug x n_nodes
    # frontier F_l = D @ A^{l-1} (n_drug x n_nodes). l=1 -> F = D.
    F = D
    katz = {b: np.zeros((len(drug_idx), len(drug_idx)), dtype=np.float64) for b in betas}
    for l in range(2, L + 1):
        F = (F @ A).tocsr()  # now F = D @ A^{l-1}, length-l walk endpoints
        block = np.asarray(F[:, drug_idx].todense()).astype(np.float64)  # 1900x1900
        np.fill_diagonal(block, 0.0)
        for b in betas:
            katz[b] += (b ** l) * block
        log(f"[katz] accumulated length-{l} walks (nnz frontier={F.nnz})")
    return katz


# --------------------------------------------------------------------------- #
# F_typed: typed length-2 metapath counts per node-kind bucket
# --------------------------------------------------------------------------- #
def build_typed_matrices(A, drug_idx, node_bucket, bucket_names, buckets):
    """Typed common-neighbour counts Db @ Db.T per bucket (1900x1900 each)."""
    D = A[drug_idx, :].tocsr()
    typed = {}
    for bname in buckets:
        bid = bucket_names.index(bname)
        cols = np.where(node_bucket == bid)[0]
        Db = D[:, cols]
        mat = np.asarray((Db @ Db.T).todense()).astype(np.float64)
        np.fill_diagonal(mat, 0.0)
        typed[bname] = mat
        log(f"[typed] {bname} (|cols|={len(cols)})")
    return typed


# --------------------------------------------------------------------------- #
# Pair assembly: EXACT S2 split / PD bucket / cap / seed from the twin probe
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
# Classifier eval helpers (identical to the twin probe)
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

    node_bucket, bucket_names = build_node_bucket(nodes)
    A = build_adjacency(edges, id2idx, n_nodes)

    # ============================ FEATURE MATRICES ============================ #
    # F_lsp (reference)
    D, deg_node, Wco, W3, AA, RA = build_lsp_matrices(A, drug_global_idx)

    # F_ppr: two operators, landing matrices + pair blocks
    ppr_meta = {}
    log("[ppr] computing RWR landing matrix (operator=rw)")
    P_rw, it_rw, dl_rw, ch_rw = compute_ppr(A, deg_node, drug_global_idx, mode="rw")
    blk_rw = ppr_pair_features(P_rw, drug_global_idx)
    ppr_meta["rw"] = {"iters": it_rw, "final_delta": dl_rw, "chunked": ch_rw}
    del P_rw

    log("[ppr] computing sym-degnorm landing matrix (operator=sym)")
    P_sym, it_sym, dl_sym, ch_sym = compute_ppr(A, deg_node, drug_global_idx, mode="sym")
    blk_sym = ppr_pair_features(P_sym, drug_global_idx)
    ppr_meta["sym"] = {"iters": it_sym, "final_delta": dl_sym, "chunked": ch_sym}
    del P_sym

    # F_katz
    katz = build_katz_matrices(A, drug_global_idx, KATZ_BETAS, KATZ_L)

    # F_typed
    typed = build_typed_matrices(A, drug_global_idx, node_bucket, bucket_names, TYPED_FAMILY_BUCKETS)

    # ============================ PAIR ASSEMBLY ============================ #
    train_pairs, test_pairs, rng = assemble_pairs(node_ids, drug_global_idx)
    tr_la, tr_lb, y_tr = to_local(train_pairs, db2local)
    te_la, te_lb, y_te = to_local(test_pairs, db2local)

    # ---- per-pair feature builders for each family ----
    def lsp_cols(la, lb):
        return {
            "walks_len2": Wco[la, lb],
            "walks_len3": W3[la, lb],
            "adamic_adar": AA[la, lb],
            "resource_allocation": RA[la, lb],
        }

    def ppr_cols(la, lb):
        return {
            "ppr_rw_sym": blk_rw["sym"][la, lb],
            "ppr_rw_max": blk_rw["pmax"][la, lb],
            "ppr_rw_min": blk_rw["pmin"][la, lb],
            "ppr_rw_inner": blk_rw["gram"][la, lb],
            "ppr_rw_diffdist": blk_rw["diff_dist"][la, lb],
            "ppr_sym_sym": blk_sym["sym"][la, lb],
            "ppr_sym_max": blk_sym["pmax"][la, lb],
            "ppr_sym_min": blk_sym["pmin"][la, lb],
            "ppr_sym_inner": blk_sym["gram"][la, lb],
            "ppr_sym_diffdist": blk_sym["diff_dist"][la, lb],
        }

    def katz_cols(la, lb):
        cols = {}
        for b in KATZ_BETAS:
            key = f"katz_beta{str(b).replace('.', 'p')}"
            cols[key] = katz[b][la, lb]
        return cols

    def typed_cols(la, lb):
        return {f"typed_{bname}": typed[bname][la, lb] for bname in TYPED_FAMILY_BUCKETS}

    fam_builders = {
        "F_lsp": lsp_cols,
        "F_ppr": ppr_cols,
        "F_katz": katz_cols,
        "F_typed": typed_cols,
    }

    def gsp_all_cols(la, lb):
        d = {}
        d.update(ppr_cols(la, lb))
        d.update(katz_cols(la, lb))
        d.update(typed_cols(la, lb))
        return d

    fam_builders["F_gsp_all"] = gsp_all_cols

    def stack(cols_dict):
        order = list(cols_dict.keys())
        mats = [np.asarray(cols_dict[k], dtype=np.float64).reshape(-1, 1) for k in order]
        return np.hstack(mats), order

    # ---- fit + eval each family x classifier ----
    results = {}
    feature_names = {}
    gbdt_models = {}
    for fam, builder in fam_builders.items():
        Xtr, names = stack(builder(tr_la, tr_lb))
        Xte, _ = stack(builder(te_la, te_lb))
        feature_names[fam] = names
        lr_auroc, lr_auprc = eval_lr(Xtr, y_tr, Xte, y_te)
        gb, gb_auroc, gb_auprc = eval_gbdt(Xtr, y_tr, Xte, y_te)
        gbdt_models[fam] = (gb, Xte, names)
        results[fam] = {
            "LR": {"AUROC": lr_auroc, "AUPRC": lr_auprc},
            "GBDT": {"AUROC": gb_auroc, "AUPRC": gb_auprc},
            "best_AUROC": max(lr_auroc, gb_auroc),
            "best_classifier": "LR" if lr_auroc >= gb_auroc else "GBDT",
            "n_features": Xtr.shape[1],
        }
        log(f"{fam}: LR AUROC={lr_auroc:.4f} AUPRC={lr_auprc:.4f} | "
            f"GBDT AUROC={gb_auroc:.4f} AUPRC={gb_auprc:.4f}  (d={Xtr.shape[1]})")

    # ---- headline gains over F_lsp (best-classifier AUROC per family) ----
    best = {fam: results[fam]["best_AUROC"] for fam in results}
    base = best["F_lsp"]
    gains = {
        "gain_F_ppr_minus_F_lsp": best["F_ppr"] - base,
        "gain_F_katz_minus_F_lsp": best["F_katz"] - base,
        "gain_F_typed_minus_F_lsp": best["F_typed"] - base,
        "gain_F_gsp_all_minus_F_lsp": best["F_gsp_all"] - base,
    }
    log(f"best AUROC: " + " ".join(f"{k}={v:.4f}" for k, v in best.items()))
    log("gains over F_lsp: " + " ".join(f"{k}={v:+.4f}" for k, v in gains.items()))

    # ---- feature importance: permutation importance on F_gsp_all GBDT ----
    gb_gsp, Xte_gsp, gsp_names = gbdt_models["F_gsp_all"]
    log("[importance] permutation importance on F_gsp_all GBDT (TEST)")
    perm = permutation_importance(
        gb_gsp, Xte_gsp, y_te, scoring="roc_auc",
        n_repeats=10, random_state=SEED, n_jobs=1,
    )
    imp_rows = []
    for j, fname in enumerate(gsp_names):
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
    gsp_gain = max(gains["gain_F_ppr_minus_F_lsp"], gains["gain_F_gsp_all_minus_F_lsp"])
    gsp_best = max(best["F_ppr"], best["F_gsp_all"])
    all_gains = list(gains.values())
    if gsp_gain >= 0.03 and gsp_best >= 0.70:
        verdict = ("gsp-is-the-lever: global structural proximity (PPR/GSP) separates PD "
                   "pos/neg materially beyond local proximity and approaches 0.70+ -> PD "
                   "needs GLOBAL not local structure (publishable structural lever)")
    elif max(all_gains) < 0.01:
        verdict = ("fundamental-deficiency: even GSP (PPR/Katz/typed-metapath) fails to beat "
                   "the LSP baseline -> the PD structural deficiency is FUNDAMENTAL across both "
                   "LSP and GSP. Pure topology is exhausted, remaining signal must come from "
                   "molecular/text (now with ICLR-2024-backed evidence the GSP escape hatch fails)")
    else:
        verdict = ("borderline: GSP gain over LSP is between thresholds, report honestly "
                   "(neither a clear global-structure win nor a clean fundamental-deficiency result)")
    log(f"VERDICT: {verdict}")

    # ---- reference numbers (cited as context, NOT recomputed) ----
    reference = {
        "real_nbfnet_v1_7_s2_pd_auroc": 0.727,
        "prior_lsp_handcrafted_pd_auroc_range": "0.60-0.66",
        "literature": "Mao et al., Revisiting Link Prediction: A Data Perspective, ICLR 2024, arXiv:2310.00793",
        "note": "reference numbers cited from project context, not recomputed here",
    }

    out = {
        "config": {
            "seed": SEED, "max_pos_train": MAX_POS_TRAIN,
            "pd_buckets": list(PD_BUCKETS),
            "ppr_alpha": PPR_ALPHA, "ppr_max_iters": PPR_MAX_ITERS, "ppr_tol": PPR_TOL,
            "ppr_chunk": PPR_CHUNK,
            "katz_betas": list(KATZ_BETAS), "katz_L": KATZ_L,
            "typed_family_buckets": list(TYPED_FAMILY_BUCKETS),
            "adjacency": "symmetric binary self-loop-free relation-agnostic over all merged-KG nodes",
        },
        "sizes": {
            "n_nodes": int(n_nodes),
            "n_drug": int(n_drug),
            "n_train_pos": int((y_tr == 1).sum()), "n_train_neg": int((y_tr == 0).sum()),
            "n_test_pos": int((y_te == 1).sum()), "n_test_neg": int((y_te == 0).sum()),
        },
        "feature_families": feature_names,
        "ppr_convergence": ppr_meta,
        "results": results,
        "headline": {
            "best_auroc_per_family": best,
            **gains,
        },
        "top_importance_feature_F_gsp_all_gbdt": top_feat,
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

    print("\n=== FEATURE IMPORTANCE (F_gsp_all GBDT, permutation, TEST AUROC) ===")
    print(imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
