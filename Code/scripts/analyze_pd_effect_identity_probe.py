from __future__ import annotations

"""Go/no-go STRUCTURAL probe: does the IDENTITY of shared effect KG nodes add
pos-vs-negative discriminability for pharmacodynamic (PD) DDIs BEYOND the COUNT
of shared effects?

No model training. Hand-crafted BFS depth<=3 shared-effect features + cheap
classifiers (LogisticRegression, HistGradientBoosting). Upper-bound test via
one-hot identity over a train-only effect-node vocabulary.

Headline = AUROC(F1 = count + identity) - AUROC(F0 = count) on TEST.
"""

import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
L = 1  # BFS depth limit (L=1 = direct shared effects, the clean identity test)
MINFREQ = 20  # min distinct TRAIN pairs for a node to enter vocab V
MAX_POS_TRAIN = 8000  # cap per class (train AND test) to keep feature matrices in RAM
TOPK_VOCAB = 3000  # cap identity vocab to the top-K most-frequent shared effect nodes

ROOT = "/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start"
SPLIT = os.path.join(ROOT, "Code/data/KG/drugbank/splits/seed42")
KG = os.path.join(ROOT, "Code/data/KG/_merged_kg")
RUN_DIR = os.path.join(ROOT, "Code/runs/2026-06-07__pd_effect_identity_probe")
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
# Step 0 - load KG, build undirected adjacency over ALL nodes
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

    # map edge endpoints to indices, drop edges with unknown endpoints
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    keep = ~(pd.isna(src) | pd.isna(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    # symmetrize, drop self-loops
    s2 = np.concatenate([src, dst])
    d2 = np.concatenate([dst, src])
    nz = s2 != d2
    s2, d2 = s2[nz], d2[nz]
    data = np.ones(len(s2), dtype=np.int8)
    A = sp.csr_matrix((data, (s2, d2)), shape=(n, n))
    A.data[:] = 1  # binary
    A = A.tocsr()
    log(f"adjacency nnz(dir)={A.nnz}")
    return nodes, node_ids, id2idx, A, drug_idx, effect_idx


def bfs_effect_depths_batch(A: sp.csr_matrix, sources: list[int],
                            effect_idx: np.ndarray, L: int):
    """Vectorized multi-source BFS via sparse boolean SpMV.

    sources : list of node indices (drug nodes).
    Returns: dict {drug_node_idx -> {effect_node_idx -> min_depth<=L}}.

    We represent the set of nodes reached so far per source as a boolean
    (n_nodes x n_sources) CSC matrix. One BFS layer = A @ frontier (reach
    neighbours), then subtract already-visited. min-depth is the first layer
    at which a node becomes reached.
    """
    # Fast per-source depth-limited frontier-expansion BFS (replaces the slow
    # batched SpMV, whose dense frontier exploded on hubs over the 178k-node KG).
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

    # ------------------------------------------------------------------
    # Step A - assemble pairs
    # ------------------------------------------------------------------
    tr = pd.read_parquet(os.path.join(SPLIT, "train.parquet"))
    tr_neg = pd.read_parquet(os.path.join(SPLIT, "train_negatives/epoch_0.parquet"))
    te = pd.read_parquet(os.path.join(SPLIT, "test_s2.parquet"))
    te_neg = pd.read_parquet(os.path.join(SPLIT, "negatives/test_s2.parquet"))

    rng = np.random.default_rng(SEED)

    def in_kg(df):
        return df[df["drug_a_id"].isin(kg_drug_set) & df["drug_b_id"].isin(kg_drug_set)]

    # TRAIN positives: PD bucket, subsample to MAX_POS_TRAIN
    tr_pos = in_kg(tr[pd_mask(tr["ddi_type"])][["drug_a_id", "drug_b_id"]]).reset_index(drop=True)
    if len(tr_pos) > MAX_POS_TRAIN:
        idx = rng.choice(len(tr_pos), size=MAX_POS_TRAIN, replace=False)
        tr_pos = tr_pos.iloc[idx].reset_index(drop=True)
    # TRAIN negatives: equal number, sampled
    tr_neg = in_kg(tr_neg[["drug_a_id", "drug_b_id"]]).reset_index(drop=True)
    n_neg = min(len(tr_pos), len(tr_neg))
    idx = rng.choice(len(tr_neg), size=n_neg, replace=False)
    tr_neg = tr_neg.iloc[idx].reset_index(drop=True)

    # TEST positives: PD bucket, all; TEST negatives: all
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
    # Step B - BFS cache for all drug nodes that appear
    # ------------------------------------------------------------------
    drugs_needed = set(train_pairs["drug_a_id"]) | set(train_pairs["drug_b_id"]) \
        | set(test_pairs["drug_a_id"]) | set(test_pairs["drug_b_id"])
    drugs_needed = [d for d in drugs_needed if d in id2idx]
    log(f"caching BFS depth<={L} for {len(drugs_needed)} drug nodes (batched SpMV)")
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
    # per-pair shared-effect computation (vectorized intersection of sorted arrays)
    # returns shared effect global indices + dA + dB per pair
    # ------------------------------------------------------------------
    def shared_arr(a, b):
        ea, da_ = cache.get(a, EMPTY)
        eb, db_ = cache.get(b, EMPTY)
        if len(ea) == 0 or len(eb) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.int8)
        # positions of ea entries inside eb (both sorted)
        pos = np.searchsorted(eb, ea)
        pos_clip = np.clip(pos, 0, len(eb) - 1)
        match = eb[pos_clip] == ea
        sh_e = ea[match]
        sh_dA = da_[match]
        sh_dB = db_[pos_clip[match]]
        return sh_e, sh_dA, sh_dB

    # compute shared per pair, accumulate F0 + global effect frequency
    def compute(pairs):
        n = len(pairs)
        F0 = np.zeros((n, 4), dtype=np.float64)
        sh_list = []  # list of shared effect global-idx arrays per pair
        A_ids = pairs["drug_a_id"].to_numpy()
        B_ids = pairs["drug_b_id"].to_numpy()
        for i in range(n):
            sh_e, dA, dB = shared_arr(A_ids[i], B_ids[i])
            sh_list.append(sh_e)
            total = len(sh_e)
            if total:
                c_1hop = int(np.count_nonzero((dA <= 1) & (dB <= 1)))
                c_min2 = int(np.count_nonzero(np.minimum(dA, dB) <= 2))
            else:
                c_1hop = c_min2 = 0
            F0[i] = (total, c_1hop, c_min2, total)
        return sh_list, F0

    log("computing shared effects + F0 for TRAIN")
    tr_shared, tr_F0 = compute(train_pairs)
    log("computing shared effects + F0 for TEST")
    te_shared, te_F0 = compute(test_pairs)

    # ------------------------------------------------------------------
    # Vocab V from TRAIN ONLY: effect nodes shared by >= MINFREQ distinct train pairs
    # ------------------------------------------------------------------
    all_tr = np.concatenate(tr_shared) if tr_shared else np.empty(0, np.int64)
    uniq, counts = np.unique(all_tr, return_counts=True)
    V = uniq[counts >= MINFREQ]
    vc = counts[counts >= MINFREQ]
    if len(V) > TOPK_VOCAB:
        keep = np.argsort(-vc)[:TOPK_VOCAB]
        V = V[keep]
    V.sort()
    cnt = dict(zip(uniq.tolist(), counts.tolist()))
    v2col = np.full(A.shape[0], -1, dtype=np.int64)
    v2col[V] = np.arange(len(V))
    log(f"|V| (effect nodes shared by >= {MINFREQ} distinct train pairs) = {len(V)}")

    def multihot(sh_list):
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
        M = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(len(sh_list), len(V)),
        )
        return M

    tr_MH = multihot(tr_shared)
    te_MH = multihot(te_shared)

    # F1 = F0 (count) horizontally stacked with identity multi-hot
    tr_F1 = sp.hstack([sp.csr_matrix(tr_F0), tr_MH], format="csr")
    te_F1 = sp.hstack([sp.csr_matrix(te_F0), te_MH], format="csr")

    y_tr = train_pairs["y"].to_numpy()
    y_te = test_pairs["y"].to_numpy()

    # ------------------------------------------------------------------
    # Step C - classifiers + eval
    # ------------------------------------------------------------------
    results = {}

    # F0-LR (standardized dense count feats)
    sc = StandardScaler()
    tr_F0s = sc.fit_transform(tr_F0)
    te_F0s = sc.transform(te_F0)
    lr0 = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000)
    lr0.fit(tr_F0s, y_tr)
    p = lr0.predict_proba(te_F0s)[:, 1]
    results["F0-LR"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))

    # F1-LR (sparse: count feats unscaled + multihot). Standardize only count cols.
    # build standardized-count + raw-multihot sparse matrix
    tr_F1_lr = sp.hstack([sp.csr_matrix(tr_F0s), tr_MH], format="csr")
    te_F1_lr = sp.hstack([sp.csr_matrix(te_F0s), te_MH], format="csr")
    lr1 = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000)
    lr1.fit(tr_F1_lr, y_tr)
    p = lr1.predict_proba(te_F1_lr)[:, 1]
    results["F1-LR"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))

    # F0-GBDT
    gb0 = HistGradientBoostingClassifier(random_state=SEED)
    gb0.fit(tr_F0, y_tr)
    p = gb0.predict_proba(te_F0)[:, 1]
    results["F0-GBDT"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))

    # F1-GBDT (dense; HistGBDT needs dense). Guard against huge |V| OOM:
    # if |V| large, keep the top GBDT_MAX_FEAT vocab cols by train frequency.
    GBDT_MAX_FEAT = 4000
    gbdt_note = ""
    if len(V) > GBDT_MAX_FEAT:
        v_freq = np.array([cnt.get(int(v), 0) for v in V])
        sel_cols = np.argsort(-v_freq)[:GBDT_MAX_FEAT]
        sel_cols.sort()
        tr_MH_g = tr_MH[:, sel_cols]
        te_MH_g = te_MH[:, sel_cols]
        gbdt_note = (f"F1-GBDT used top {GBDT_MAX_FEAT}/{len(V)} vocab cols "
                     f"by train freq (dense-matrix OOM guard).")
        log(gbdt_note)
    else:
        tr_MH_g, te_MH_g = tr_MH, te_MH
    tr_F1d = np.hstack([tr_F0, tr_MH_g.toarray()])
    te_F1d = np.hstack([te_F0, te_MH_g.toarray()])
    gb1 = HistGradientBoostingClassifier(random_state=SEED)
    gb1.fit(tr_F1d, y_tr)
    p = gb1.predict_proba(te_F1d)[:, 1]
    results["F1-GBDT"] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))

    for k, (au, ap) in results.items():
        log(f"{k}: AUROC={au:.4f} AUPRC={ap:.4f}")

    gain_lr = results["F1-LR"][0] - results["F0-LR"][0]
    gain_gb = results["F1-GBDT"][0] - results["F0-GBDT"][0]
    log(f"AUROC gain (F1-F0): LR={gain_lr:+.4f}  GBDT={gain_gb:+.4f}")

    # ------------------------------------------------------------------
    # Step D - interpretability (F1 LR weights over identity cols)
    # ------------------------------------------------------------------
    # node degree (undirected) for context
    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    coef = lr1.coef_.ravel()
    id_coef = coef[4:]  # first 4 are count feats
    name_map = nodes.set_index("id")["name"].to_dict()
    kind_map = nodes.set_index("id")["kind"].to_dict()
    rows = []
    for j, v in enumerate(V):
        v = int(v)
        nid = node_ids[v]
        rows.append({
            "node_id": nid,
            "name": name_map.get(nid, ""),
            "kind": kind_map.get(nid, ""),
            "degree": int(deg[v]),
            "train_pair_freq": int(cnt.get(v, 0)),
            "lr_weight": float(id_coef[j]),
        })
    wdf = pd.DataFrame(rows).sort_values("lr_weight", ascending=False)
    wdf.to_csv(os.path.join(RUN_DIR, "effect_node_weights.csv"), index=False)
    top_pos = wdf.head(20)
    top_neg = wdf.tail(10)

    # ------------------------------------------------------------------
    # persist metrics
    # ------------------------------------------------------------------
    out = {
        "config": {"seed": SEED, "L": L, "minfreq": MINFREQ, "max_pos_train": MAX_POS_TRAIN,
                   "pd_buckets": list(PD_BUCKETS), "effect_kinds": sorted(EFFECT_KINDS)},
        "sizes": {
            "n_train_pos": int((y_tr == 1).sum()), "n_train_neg": int((y_tr == 0).sum()),
            "n_test_pos": int((y_te == 1).sum()), "n_test_neg": int((y_te == 0).sum()),
            "vocab_V": len(V),
        },
        "results": {k: {"AUROC": v[0], "AUPRC": v[1]} for k, v in results.items()},
        "auroc_gain": {"LR": gain_lr, "GBDT": gain_gb},
        "gbdt_note": gbdt_note,
        "top_positive": top_pos.to_dict("records"),
        "top_negative": top_neg.to_dict("records"),
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {os.path.join(RUN_DIR, 'metrics.json')}")
    log(f"DONE in {time.time()-t0:.1f}s")

    print("\n=== TOP POSITIVE EFFECT NODES ===")
    print(top_pos[["name", "kind", "degree", "train_pair_freq", "lr_weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
