from __future__ import annotations

"""STRUCTURAL-PRIOR probe at BFS depth L=3 for cold-start PD DDI.

Question: at L=3, where the discriminative shared-effect signal is buried under
high-degree HUB nodes that almost every drug pair reaches, can a purely-structural
PRIOR / inductive weighting on the shared-effect identity features RECOVER
pos-vs-negative signal that raw 0/1 identity cannot?

NO model training. Hand-crafted shared-effect features at L=3 + cheap classifiers
(LogisticRegression, HistGradientBoosting). Each prior variant re-weights the
identity multi-hot over a train-only effect-node vocabulary using a label-free
structural prior (IDF / degree / top-k specificity / PMI).

Headline = AUROC(variant) - AUROC(F0=count-only) on TEST  (the recovered gain).

Machinery (build_kg, bfs_effect_depths_batch, data assembly, per-pair shared-effect
intersection) is copied from the working L=1 script
Code/scripts/analyze_pd_effect_identity_probe.py.  Only L is changed to 3 and the
feature-construction / variant logic is new.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
L = 3  # BFS depth limit (L=3 = the buried-under-hubs regime this probe targets)
MINFREQ = 20  # min distinct TRAIN pairs for a node to enter vocab V
MAX_POS_TRAIN = 8000  # cap per class (train AND test)
TOPK_VOCAB = 3000  # cap identity vocab to top-K most-frequent shared effect nodes
TOPK_SPECIFIC = 30  # topk_specific variant: keep K most-specific shared nodes per pair
GBDT_MAX_FEAT = 1500  # cap multihot cols for the dense GBDT matrices (per spec)

ROOT = "/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start"
SPLIT = os.path.join(ROOT, "Code/data/KG/drugbank/splits/seed42")
KG = os.path.join(ROOT, "Code/data/KG/_merged_kg")
RUN_DIR = os.path.join(ROOT, "Code/runs/2026-06-07__pd_effect_identity_priors")
os.makedirs(RUN_DIR, exist_ok=True)

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
EFFECT_KINDS = {"Side Effect", "effect/phenotype", "Symptom", "disease", "Disease"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pd_mask(ddi_type: pd.Series) -> pd.Series:
    sl = ddi_type.str.lower()
    m = pd.Series(False, index=ddi_type.index)
    for b in PD_BUCKETS:
        m = m | sl.str.contains(b, regex=False, na=False)
    return m


# ----------------------------------------------------------------------------
# Step 0 - load KG, build undirected adjacency over ALL nodes  (copied from L=1)
# ----------------------------------------------------------------------------
def build_kg():
    nodes = pd.read_parquet(os.path.join(KG, "nodes__drugbank_hetionet_primekg.parquet"))
    edges = pd.read_parquet(
        os.path.join(KG, "edges__drugbank_hetionet_primekg__mask1.parquet"),
        columns=["src", "dst", "relation"],
    )
    log(f"nodes={nodes.shape} edges={edges.shape}")

    node_ids = nodes["id"].to_numpy()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    drug_mask = (nodes["kind"] == "Drug").to_numpy()
    effect_mask = nodes["kind"].isin(EFFECT_KINDS).to_numpy()
    drug_idx = set(np.where(drug_mask)[0].tolist())
    effect_idx = np.zeros(n, dtype=bool)
    effect_idx[np.where(effect_mask)[0]] = True
    log(f"#drug nodes={len(drug_idx)} #effect-type nodes={int(effect_mask.sum())}")

    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    keep = ~(pd.isna(src) | pd.isna(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    s2 = np.concatenate([src, dst])
    d2 = np.concatenate([dst, src])
    nz = s2 != d2
    s2, d2 = s2[nz], d2[nz]
    data = np.ones(len(s2), dtype=np.int8)
    A = sp.csr_matrix((data, (s2, d2)), shape=(n, n))
    A.data[:] = 1
    A = A.tocsr()
    log(f"adjacency nnz(dir)={A.nnz}")
    return nodes, node_ids, id2idx, A, drug_idx, effect_idx


def bfs_effect_depths_batch(A: sp.csr_matrix, sources: list[int],
                            effect_idx: np.ndarray, L: int):
    """Fast per-source depth-limited frontier-expansion BFS (copied from L=1).

    Uses A[frontier].indices (NOT sparse SpMV Ab @ frontier, which OOMs on hubs).
    Returns: dict {drug_node_idx -> {effect_node_idx -> min_depth<=L}}.
    """
    n = A.shape[0]
    out: dict = {}
    for s in sources:
        dist = np.full(n, -1, dtype=np.int16)
        dist[s] = 0
        frontier = np.array([s], dtype=np.int64)
        for depth in range(1, L + 1):
            if frontier.size == 0:
                break
            nbrs = np.unique(A[frontier].indices)
            new = nbrs[dist[nbrs] == -1]
            if new.size == 0:
                break
            dist[new] = depth
            frontier = new
        reached_eff = np.where((dist >= 1) & effect_idx)[0]
        out[s] = {int(e): int(dist[e]) for e in reached_eff.tolist()}
    return out


def main():
    t0 = time.time()
    nodes, node_ids, id2idx, A, drug_idx, effect_idx = build_kg()
    kg_drug_set = set(node_ids[i] for i in drug_idx)
    n_drugs = len(drug_idx)

    # ------------------------------------------------------------------
    # Step A - assemble pairs (copied from L=1)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step B - BFS cache for all drug nodes that appear (L=3)
    # ------------------------------------------------------------------
    drugs_needed = set(train_pairs["drug_a_id"]) | set(train_pairs["drug_b_id"]) \
        | set(test_pairs["drug_a_id"]) | set(test_pairs["drug_b_id"])
    drugs_needed = [d for d in drugs_needed if d in id2idx]
    log(f"caching BFS depth<={L} for {len(drugs_needed)} drug nodes")
    cache = {}
    BATCH = 100
    for start in range(0, len(drugs_needed), BATCH):
        chunk = drugs_needed[start:start + BATCH]
        srcs = [id2idx[d] for d in chunk]
        res = bfs_effect_depths_batch(A, srcs, effect_idx, L)
        for d, s in zip(chunk, srcs):
            dd = res[s]
            if dd:
                eidx = np.fromiter(dd.keys(), dtype=np.int64, count=len(dd))
                dep = np.fromiter(dd.values(), dtype=np.int8, count=len(dd))
                order = np.argsort(eidx)
                cache[d] = (eidx[order], dep[order])
            else:
                cache[d] = (np.empty(0, np.int64), np.empty(0, np.int8))
        log(f"  bfs {min(start+BATCH, len(drugs_needed))}/{len(drugs_needed)}  "
            f"elapsed={time.time()-t0:.1f}s")
    log(f"BFS cache done in {time.time()-t0:.1f}s")

    EMPTY = (np.empty(0, np.int64), np.empty(0, np.int8))

    # ------------------------------------------------------------------
    # drugreach[X] = #of the cached KG drugs (~all 1900 drug-pool drugs that
    # appear) that reach effect node X within L=3.  This is the HUB measure.
    # Build it from the FULL drug-pool BFS cache, not only assembled-pair drugs,
    # so the prior is a genuine corpus-level structural statistic.
    # We cache ALL drug-pool nodes (the 1900-drug KG drugs) to make drugreach honest.
    # ------------------------------------------------------------------
    all_drug_nodes = sorted(drug_idx)
    log(f"caching BFS depth<={L} for ALL {len(all_drug_nodes)} KG drug-pool nodes "
        f"(for drugreach/p priors)")
    drugreach = np.zeros(A.shape[0], dtype=np.int64)
    for start in range(0, len(all_drug_nodes), BATCH):
        chunk = all_drug_nodes[start:start + BATCH]
        res = bfs_effect_depths_batch(A, chunk, effect_idx, L)
        for s in chunk:
            dd = res[s]
            if dd:
                eidx = np.fromiter(dd.keys(), dtype=np.int64, count=len(dd))
                drugreach[eidx] += 1
        if (start // BATCH) % 5 == 0:
            log(f"  drugreach bfs {min(start+BATCH, len(all_drug_nodes))}/"
                f"{len(all_drug_nodes)}  elapsed={time.time()-t0:.1f}s")
    log(f"drugreach cache done in {time.time()-t0:.1f}s "
        f"(max drugreach={int(drugreach.max())}, n_drugs={n_drugs})")

    # ------------------------------------------------------------------
    # per-pair shared-effect computation (copied from L=1)
    # ------------------------------------------------------------------
    def shared_arr(a, b):
        ea, da_ = cache.get(a, EMPTY)
        eb, db_ = cache.get(b, EMPTY)
        if len(ea) == 0 or len(eb) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.int8)
        pos = np.searchsorted(eb, ea)
        pos_clip = np.clip(pos, 0, len(eb) - 1)
        match = eb[pos_clip] == ea
        sh_e = ea[match]
        sh_dA = da_[match]
        sh_dB = db_[pos_clip[match]]
        return sh_e, sh_dA, sh_dB

    def compute(pairs):
        n = len(pairs)
        F0 = np.zeros((n, 4), dtype=np.float64)
        sh_list = []  # per-pair shared effect global-idx arrays
        A_ids = pairs["drug_a_id"].to_numpy()
        B_ids = pairs["drug_b_id"].to_numpy()
        for i in range(n):
            sh_e, dA, dB = shared_arr(A_ids[i], B_ids[i])
            sh_list.append(sh_e)
            total = len(sh_e)
            if total:
                c_d1 = int(np.count_nonzero(np.minimum(dA, dB) <= 1))
                c_d2 = int(np.count_nonzero(np.minimum(dA, dB) <= 2))
            else:
                c_d1 = c_d2 = 0
            # F0 = [total #shared, #shared min-depth<=1, #shared min-depth<=2, total]
            F0[i] = (total, c_d1, c_d2, total)
        return sh_list, F0

    log("computing shared effects + F0 for TRAIN (L=3)")
    tr_shared, tr_F0 = compute(train_pairs)
    log("computing shared effects + F0 for TEST (L=3)")
    te_shared, te_F0 = compute(test_pairs)

    # ------------------------------------------------------------------
    # Vocab V from TRAIN ONLY: effect nodes shared by >= MINFREQ distinct
    # train pairs, top-TOPK_VOCAB by train-pair frequency (copied from L=1)
    # ------------------------------------------------------------------
    all_tr = np.concatenate(tr_shared) if tr_shared else np.empty(0, np.int64)
    uniq, counts = np.unique(all_tr, return_counts=True)
    V = uniq[counts >= MINFREQ]
    vc = counts[counts >= MINFREQ]
    if len(V) > TOPK_VOCAB:
        keep = np.argsort(-vc)[:TOPK_VOCAB]
        V = V[keep]
        vc = vc[keep]
    order_v = np.argsort(V)
    V = V[order_v]
    vc = vc[order_v]
    cnt = dict(zip(uniq.tolist(), counts.tolist()))  # train-pair freq per node
    v2col = np.full(A.shape[0], -1, dtype=np.int64)
    v2col[V] = np.arange(len(V))
    N_train = len(train_pairs)
    log(f"|V| (effect nodes shared by >= {MINFREQ} distinct train pairs, "
        f"capped {TOPK_VOCAB}) = {len(V)}")

    # ------------------------------------------------------------------
    # Per-vocab-node prior statistics
    # ------------------------------------------------------------------
    deg = np.asarray((A > 0).sum(axis=1)).ravel()  # undirected node degree
    # per-vocab-column aligned arrays (index j in [0,|V|))
    V_drugreach = drugreach[V].astype(np.float64)      # HUB measure
    V_deg = deg[V].astype(np.float64)
    V_cooc = vc.astype(np.float64)                     # #train pairs sharing X
    V_p = V_drugreach / float(n_drugs)                 # reach probability

    # prior weight vectors over V (length |V|)
    w_raw = np.ones(len(V), dtype=np.float64)
    w_idf = np.log(float(n_drugs) / np.maximum(V_drugreach, 1.0))
    w_degnorm = 1.0 / np.sqrt(np.maximum(V_deg, 1.0))
    w_pmi = np.maximum(
        0.0,
        np.log((V_cooc / float(N_train)) / (V_p ** 2 + 1e-9)),
    )

    # ------------------------------------------------------------------
    # Build base sparse multihot (rows=pairs, cols=V, value=1 if shares)
    # plus per-(row) drugreach so topk_specific can pick most-specific nodes.
    # We keep both the col index and the drugreach value per nnz.
    # ------------------------------------------------------------------
    def build_nnz(sh_list):
        rows, cols = [], []
        for i, sh_e in enumerate(sh_list):
            if len(sh_e) == 0:
                continue
            j = v2col[sh_e]
            keep = j >= 0
            if keep.any():
                jj = j[keep]
                rows.append(np.full(len(jj), i, dtype=np.int64))
                cols.append(jj)
        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
        else:
            rows = np.empty(0, np.int64)
            cols = np.empty(0, np.int64)
        return rows, cols

    tr_rows, tr_cols = build_nnz(tr_shared)
    te_rows, te_cols = build_nnz(te_shared)

    def weighted_mh(rows, cols, n_rows, wvec):
        vals = wvec[cols]
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, len(V)))

    def topk_specific_mh(rows, cols, n_rows):
        """Per row, keep only the K cols with SMALLEST drugreach (most specific);
        value 1 for those. Implemented by grouping nnz per row."""
        order = np.argsort(rows, kind="stable")
        r_s = rows[order]
        c_s = cols[order]
        out_r, out_c = [], []
        i = 0
        m = len(r_s)
        while i < m:
            j = i
            while j < m and r_s[j] == r_s[i]:
                j += 1
            cc = c_s[i:j]
            if len(cc) > TOPK_SPECIFIC:
                dr = V_drugreach[cc]
                sel = np.argpartition(dr, TOPK_SPECIFIC)[:TOPK_SPECIFIC]
                cc = cc[sel]
            out_r.append(np.full(len(cc), r_s[i], dtype=np.int64))
            out_c.append(cc)
            i = j
        if out_r:
            orr = np.concatenate(out_r)
            occ = np.concatenate(out_c)
        else:
            orr = np.empty(0, np.int64)
            occ = np.empty(0, np.int64)
        return sp.csr_matrix((np.ones(len(orr)), (orr, occ)), shape=(n_rows, len(V)))

    variants = {
        "raw": w_raw,
        "IDF": w_idf,
        "degnorm": w_degnorm,
        "PMI": w_pmi,
    }
    tr_MH = {k: weighted_mh(tr_rows, tr_cols, len(train_pairs), w) for k, w in variants.items()}
    te_MH = {k: weighted_mh(te_rows, te_cols, len(test_pairs), w) for k, w in variants.items()}
    tr_MH["topk_specific"] = topk_specific_mh(tr_rows, tr_cols, len(train_pairs))
    te_MH["topk_specific"] = topk_specific_mh(te_rows, te_cols, len(test_pairs))

    y_tr = train_pairs["y"].to_numpy()
    y_te = test_pairs["y"].to_numpy()

    # ------------------------------------------------------------------
    # Standardize the F0 count block (for LR). GBDT uses raw F0.
    # ------------------------------------------------------------------
    sc = StandardScaler()
    tr_F0s = sc.fit_transform(tr_F0)
    te_F0s = sc.transform(te_F0)

    # GBDT multihot column cap (top-GBDT_MAX_FEAT by train freq)
    if len(V) > GBDT_MAX_FEAT:
        sel_cols = np.argsort(-vc)[:GBDT_MAX_FEAT]
        sel_cols.sort()
        gbdt_note = (f"GBDT variants used top {GBDT_MAX_FEAT}/{len(V)} vocab cols "
                     f"by train freq (dense-matrix guard).")
    else:
        sel_cols = np.arange(len(V))
        gbdt_note = ""
    if gbdt_note:
        log(gbdt_note)

    results = {}
    models_lr = {}

    # ---- F0 baseline (no identity) ----
    lr0 = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000)
    lr0.fit(tr_F0s, y_tr)
    p = lr0.predict_proba(te_F0s)[:, 1]
    results["F0-LR"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))

    gb0 = HistGradientBoostingClassifier(random_state=SEED)
    gb0.fit(tr_F0, y_tr)
    p = gb0.predict_proba(te_F0)[:, 1]
    results["F0-GBDT"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))

    # ---- each identity variant ----
    VARIANT_ORDER = ["raw", "IDF", "degnorm", "topk_specific", "PMI"]
    for name in VARIANT_ORDER:
        trM = tr_MH[name]
        teM = te_MH[name]
        # LR: standardized count block ++ weighted multihot (sparse)
        tr_X = sp.hstack([sp.csr_matrix(tr_F0s), trM], format="csr")
        te_X = sp.hstack([sp.csr_matrix(te_F0s), teM], format="csr")
        lr = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000)
        lr.fit(tr_X, y_tr)
        p = lr.predict_proba(te_X)[:, 1]
        results[f"{name}-LR"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))
        models_lr[name] = lr

        # GBDT: dense F0 ++ capped multihot
        tr_d = np.hstack([tr_F0, trM[:, sel_cols].toarray()])
        te_d = np.hstack([te_F0, teM[:, sel_cols].toarray()])
        gb = HistGradientBoostingClassifier(random_state=SEED)
        gb.fit(tr_d, y_tr)
        p = gb.predict_proba(te_d)[:, 1]
        results[f"{name}-GBDT"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))

    for k, (au, ap) in results.items():
        log(f"{k}: AUROC={au:.4f} AUPRC={ap:.4f}")

    # ------------------------------------------------------------------
    # gains vs F0 (per classifier) + best variant (better of LR/GBDT)
    # ------------------------------------------------------------------
    f0_lr = results["F0-LR"][0]
    f0_gb = results["F0-GBDT"][0]
    gains = {}
    for name in VARIANT_ORDER:
        g_lr = results[f"{name}-LR"][0] - f0_lr
        g_gb = results[f"{name}-GBDT"][0] - f0_gb
        gains[name] = {"LR": g_lr, "GBDT": g_gb, "best": max(g_lr, g_gb)}
        log(f"GAIN {name}: LR={g_lr:+.4f} GBDT={g_gb:+.4f} best={max(g_lr, g_gb):+.4f}")

    best_variant = max(VARIANT_ORDER, key=lambda n: gains[n]["best"])
    best_gain = gains[best_variant]["best"]
    recover = best_gain >= 0.03
    log(f"BEST variant = {best_variant} (best gain={best_gain:+.4f})  "
        f"recover>=0.03? {recover}")

    # ------------------------------------------------------------------
    # interpretability: top positive-weight effect nodes for best variant (LR)
    # ------------------------------------------------------------------
    name_map = nodes.set_index("id")["name"].to_dict()
    kind_map = nodes.set_index("id")["kind"].to_dict()
    lr_best = models_lr[best_variant]
    coef = lr_best.coef_.ravel()
    id_coef = coef[4:]  # first 4 are count feats
    rows = []
    for j, v in enumerate(V):
        v = int(v)
        nid = node_ids[v]
        rows.append({
            "node_id": nid,
            "name": name_map.get(nid, ""),
            "kind": kind_map.get(nid, ""),
            "degree": int(deg[v]),
            "drugreach": int(drugreach[v]),
            "train_pair_freq": int(cnt.get(v, 0)),
            "lr_weight": float(id_coef[j]),
        })
    wdf = pd.DataFrame(rows).sort_values("lr_weight", ascending=False)
    wdf.to_csv(os.path.join(RUN_DIR, f"effect_node_weights__{best_variant}.csv"), index=False)
    top_pos = wdf.head(15)
    top_neg = wdf.tail(10)

    # ------------------------------------------------------------------
    # persist metrics
    # ------------------------------------------------------------------
    out = {
        "config": {
            "seed": SEED, "L": L, "minfreq": MINFREQ, "max_pos_train": MAX_POS_TRAIN,
            "topk_vocab": TOPK_VOCAB, "topk_specific": TOPK_SPECIFIC,
            "gbdt_max_feat": GBDT_MAX_FEAT,
            "pd_buckets": list(PD_BUCKETS), "effect_kinds": sorted(EFFECT_KINDS),
        },
        "sizes": {
            "n_train_pos": int((y_tr == 1).sum()), "n_train_neg": int((y_tr == 0).sum()),
            "n_test_pos": int((y_te == 1).sum()), "n_test_neg": int((y_te == 0).sum()),
            "vocab_V": len(V), "n_drugs": n_drugs,
            "max_drugreach": int(drugreach.max()),
        },
        "results": {k: {"AUROC": v[0], "AUPRC": v[1]} for k, v in results.items()},
        "gains_vs_f0": gains,
        "best_variant": best_variant, "best_gain": best_gain,
        "recover_verdict": bool(recover),
        "gbdt_note": gbdt_note,
        "top_positive": top_pos.to_dict("records"),
        "top_negative": top_neg.to_dict("records"),
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {os.path.join(RUN_DIR, 'metrics.json')}")
    log(f"DONE in {time.time()-t0:.1f}s")

    print(f"\n=== BEST VARIANT = {best_variant} (best gain {best_gain:+.4f}) ===")
    print(f"\n=== TOP POSITIVE EFFECT NODES ({best_variant}-LR) ===")
    print(top_pos[["name", "kind", "degree", "drugreach", "train_pair_freq", "lr_weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
