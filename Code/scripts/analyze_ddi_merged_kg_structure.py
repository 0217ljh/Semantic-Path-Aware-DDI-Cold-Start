"""Structural analysis of all DDI samples inside the *merged* KG.

Motivated by the conditional-message-passing (C-MPNN / NBFNet / ULTRA) framing:
DDI is a pairwise, query-conditioned link-prediction task whose only signal in
the merged KG flows through the non-DDI neighbourhood (targets, enzymes, genes,
pathways, side effects, ...). This script characterises, for every existing
positive DDI pair (and a verified negative pool):

  1. Pair-level path evidence    -- the quantities C-MPNN actually propagates:
       direct-edge flag, #common neighbours (length-2 walks), typed shared
       neighbours (genes/proteins, side effects, pathways, diseases),
       length-3 walk counts, shortest-path distance (<=3), connectivity rate.
  2. Relational-WL collision      -- the centrepiece of the framing: run
       relational colour-refinement (relational 1-WL, the expressivity ceiling
       of C-MPNN) over the whole merged KG, then measure how often DDI pairs are
       WL-indistinguishable yet carry different labels (different mechanism, or
       positive vs negative). This is where the relational-WL ceiling bites.
  3. Pos-vs-neg contrast          -- the same features for positives vs the
       negative pool, plus the AUC of each trivial single-feature predictor
       (is there a learnable structural signal at all?).

Operates on the merged KG parquet (drugbank + hetionet + primekg, ~7.1M edges).
This is NOT the legacy drugbank-only KG used by analyze_nbfnet_kg_structure.py.

Run (from project root, via WSL conda env per project rules):
  python Code/scripts/analyze_ddi_merged_kg_structure.py
  python Code/scripts/analyze_ddi_merged_kg_structure.py --wl-rounds 3 --seed 42

Outputs (under --out, default Code/runs/analyze_ddi_merged_kg_structure/):
  summary.json          all scalar stats (verified, machine-readable)
  pos_features.parquet  per-positive-pair structural features
  neg_features.parquet  per-negative-pair structural features
  wl_collision.json     relational-WL collision tables per round
And prints a full text report to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]  # -> Code/

NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"

# Normalise the heterogeneous (multi-source, mixed-case) node kinds into a few
# biologically meaningful buckets for typed-common-neighbour features.
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

U64 = np.uint64


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer -- order-independent, low-collision uint64 hash."""
    x = x.astype(U64, copy=False)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
        x = x ^ (x >> U64(31))
    return x


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_kg():
    log(f"[load] nodes  {NODES_PARQUET}")
    nodes = pd.read_parquet(NODES_PARQUET)
    log(f"[load] edges  {EDGES_PARQUET}")
    edges = pd.read_parquet(EDGES_PARQUET)

    node_ids = nodes["id"].to_numpy()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)
    kinds = nodes["kind"].to_numpy()

    # bucket id per node (for typed common neighbours)
    kind2bucket = {}
    for b, ks in TYPE_BUCKETS.items():
        for k in ks:
            kind2bucket[k] = b
    bucket_names = list(TYPE_BUCKETS.keys())
    bucket2id = {b: i for i, b in enumerate(bucket_names)}
    node_bucket = np.array(
        [bucket2id.get(kind2bucket.get(k, "_other"), -1) for k in kinds], dtype=np.int32
    )

    log(f"[load] n_nodes={n_nodes}  n_edges={len(edges)}")
    return nodes, edges, id2idx, n_nodes, kinds, node_bucket, bucket_names


def build_adjacency(edges, id2idx, n_nodes):
    """Symmetric, binary, self-loop-free adjacency over all merged-KG nodes
    (relation-agnostic) for path counting / connectivity."""
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    keep = (~np.isnan(src)) & (~np.isnan(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    # drop self loops
    nz = src != dst
    src, dst = src[nz], dst[nz]
    # symmetric
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.data[:] = 1.0  # binarise (collapse parallel relations)
    A.sum_duplicates()
    A.data[:] = 1.0
    log(f"[adj] symmetric binary A: nnz={A.nnz}")
    return A


# --------------------------------------------------------------------------- #
# Pair feature matrices (only ~1900 drugs -> dense 1900x1900 is cheap)
# --------------------------------------------------------------------------- #
def build_pair_matrices(A, drug_idx, node_bucket, bucket_names):
    n_drug = len(drug_idx)
    D = A[drug_idx, :].tocsr()  # n_drug x n_nodes, binary
    deg = np.asarray(D.sum(axis=1)).ravel().astype(np.int64)

    # direct drug-drug edge (any non-DDI relation, e.g. het:CrC resemblance)
    Adr = np.asarray(A[np.ix_(drug_idx, drug_idx)].todense()).astype(np.int64)
    np.fill_diagonal(Adr, 0)

    # length-2 walks == #common neighbours
    log("[pair] common neighbours (D @ D.T)")
    Wco = np.asarray((D @ D.T).todense()).astype(np.int64)
    np.fill_diagonal(Wco, 0)

    # length-3 walks  (D @ A @ D.T)
    log("[pair] length-3 walks (D @ A @ D.T)")
    M = (A @ D.T).tocsr()  # n_nodes x n_drug
    W3 = np.asarray((D @ M).todense()).astype(np.int64)
    np.fill_diagonal(W3, 0)

    # typed common neighbours for key biological buckets
    typed = {}
    for bname in ["gene_protein", "side_effect", "pathway", "disease"]:
        bid = bucket_names.index(bname)
        cols = np.where(node_bucket == bid)[0]
        Db = D[:, cols]
        typed[bname] = np.asarray((Db @ Db.T).todense()).astype(np.int64)
        np.fill_diagonal(typed[bname], 0)
        log(f"[pair] typed common neighbours: {bname} (|cols|={len(cols)})")

    return D, deg, Adr, Wco, W3, typed


def shortest_distance(adr, wco, w3):
    """Distance proxy from walk matrices: 1 direct, 2 common-nbr, 3 length-3,
    else >=4 (coded as 4). Vectorised over arrays."""
    dist = np.full(adr.shape, 4, dtype=np.int8)
    dist[w3 > 0] = 3
    dist[wco > 0] = 2
    dist[adr > 0] = 1
    return dist


# --------------------------------------------------------------------------- #
# Relational WL colour refinement over the whole merged KG
# --------------------------------------------------------------------------- #
def build_wl_edges(edges, id2idx, n_nodes):
    """Directed (u, rel_id, v) message edges. Directed KG edges get a distinct
    inverse relation id; undirected edges share one id in both directions."""
    rel_codes, _ = pd.factorize(edges["relation"], sort=True)
    rel_codes = rel_codes.astype(np.int64)
    n_rel = int(rel_codes.max()) + 1

    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    directed = edges["directed"].to_numpy().astype(bool)
    keep = (~np.isnan(src)) & (~np.isnan(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    rel = rel_codes[keep]
    directed = directed[keep]

    # forward: v is reached from u via rel  -> message edge (u=src, rel, v=dst)
    u_f, r_f, v_f = src, rel, dst
    # backward: for directed edges use inverse relation id (rel + n_rel),
    #           for undirected reuse same relation id (symmetric semantics).
    r_b = np.where(directed, rel + n_rel, rel)
    u_b, v_b = dst, src

    u = np.concatenate([u_f, u_b])
    r = np.concatenate([r_f, r_b])
    v = np.concatenate([v_f, v_b])
    n_rel_total = int(r.max()) + 1
    log(f"[wl] message edges={len(u)}  base_rel={n_rel}  total_rel(+inv)={n_rel_total}")
    return u, r, v, n_rel_total


def relational_wl(u, r, v, init_color, n_nodes, n_rel_total, rounds):
    """Order-invariant hashing variant of relational 1-WL.

    new_color[x] = hash( old_color[x], SUM_{(x,r,y)} hash(r, old_color[y]) )
    Returns list of colour arrays (length n_nodes, int64 small ids) per round,
    index 0 == initial colouring.
    """
    color = init_color.astype(np.int64).copy()
    history = [color.copy()]
    R = U64(n_rel_total + 1)
    for it in range(rounds):
        cu = color.astype(U64)
        msg = _mix64(cu[v] * R + r.astype(U64))  # per-edge hashed message
        agg = np.zeros(n_nodes, dtype=U64)
        np.add.at(agg, u, msg)  # order-invariant aggregation
        newraw = _mix64(_mix64(color.astype(U64)) + agg)
        _, color = np.unique(newraw, return_inverse=True)
        color = color.astype(np.int64)
        history.append(color.copy())
        log(f"[wl] round {it + 1}: distinct colours over all nodes = {color.max() + 1}")
    return history


def pair_signature(color_a, color_b):
    """Unordered pair colour signature -> compact int id (per round)."""
    lo = np.minimum(color_a, color_b).astype(np.int64)
    hi = np.maximum(color_a, color_b).astype(np.int64)
    key = lo * np.int64(1_000_003) + hi  # collisions impossible if max colour < 1e6
    # safety: if hi can exceed key base, fall back to unique of stacked
    if hi.max() >= 1_000_003:
        keys = np.stack([lo, hi], axis=1)
        _, key = np.unique(keys, axis=0, return_inverse=True)
    return key


# --------------------------------------------------------------------------- #
# Negative pool
# --------------------------------------------------------------------------- #
def load_negative_pool(seed, drug_db2local, pos_key_set, n_drug):
    base = ROOT / f"data/KG/drugbank/splits/seed{seed}/negatives"
    if not base.is_dir():
        log(f"[neg] WARNING negatives dir missing: {base}")
        return np.empty(0, np.int64), np.empty(0, np.int64)
    frames = []
    for f in sorted(base.glob("*.parquet")):
        frames.append(pd.read_parquet(f)[["drug_a_id", "drug_b_id"]])
    neg = pd.concat(frames, ignore_index=True)
    a = neg["drug_a_id"].map(drug_db2local).to_numpy()
    b = neg["drug_b_id"].map(drug_db2local).to_numpy()
    ok = (~pd.isna(a)) & (~pd.isna(b))
    a = a[ok].astype(np.int64)
    b = b[ok].astype(np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo.astype(np.int64) * n_drug + hi
    # dedup + drop any that collide with a positive pair
    _, uniq = np.unique(key, return_index=True)
    a, b, key = a[uniq], b[uniq], key[uniq]
    mask = np.array([k not in pos_key_set for k in key])
    log(f"[neg] pool raw={len(neg)} mapped_unique={len(uniq)} after_pos_filter={mask.sum()}")
    return a[mask], b[mask]


# --------------------------------------------------------------------------- #
# Feature gather + contrast
# --------------------------------------------------------------------------- #
def gather_features(la, lb, Adr, Wco, W3, typed, dist):
    feats = {
        "direct_edge": Adr[la, lb],
        "common_neighbors": Wco[la, lb],
        "walks_len3": W3[la, lb],
        "distance": dist[la, lb],
    }
    for bname, mat in typed.items():
        feats[f"shared_{bname}"] = mat[la, lb]
    return feats


def describe(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "frac_gt0": float((arr > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="which split's negatives to use as the negative pool")
    ap.add_argument("--wl-rounds", type=int, default=3)
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_ddi_merged_kg_structure"))
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges, id2idx, n_nodes, kinds, node_bucket, bucket_names = load_kg()

    # ---- drugs ----
    drug_mask = nodes["kind"].to_numpy() == "Drug"
    drug_db_ids = nodes["id"].to_numpy()[drug_mask]
    drug_idx = np.array([id2idx[d] for d in drug_db_ids], dtype=np.int64)
    n_drug = len(drug_idx)
    drug_db2local = {db: i for i, db in enumerate(drug_db_ids)}  # DBxxxx -> 0..n_drug-1
    local_of_global = {int(g): i for i, g in enumerate(drug_idx)}
    log(f"[drug] n_drug(Drug nodes)={n_drug}")

    # ---- positives ----
    pos = pd.read_csv(DDI_POS_CSV)
    pa = pos["drug_a_id"].map(drug_db2local).to_numpy()
    pb = pos["drug_b_id"].map(drug_db2local).to_numpy()
    ok = (~pd.isna(pa)) & (~pd.isna(pb))
    n_drop = int((~ok).sum())
    pa = pa[ok].astype(np.int64)
    pb = pb[ok].astype(np.int64)
    ddi_type = pos["ddi_type"].to_numpy()[ok]
    log(f"[pos] positive DDI pairs={len(pa)} (dropped {n_drop} with a non-Drug-node drug)")

    pos_lo = np.minimum(pa, pb)
    pos_hi = np.maximum(pa, pb)
    pos_key = pos_lo.astype(np.int64) * n_drug + pos_hi
    pos_key_set = set(pos_key.tolist())

    # ---- negatives ----
    na, nb = load_negative_pool(args.seed, drug_db2local, pos_key_set, n_drug)

    # ---- pair feature matrices ----
    A = build_adjacency(edges, id2idx, n_nodes)
    D, deg, Adr, Wco, W3, typed = build_pair_matrices(A, drug_idx, node_bucket, bucket_names)
    dist = shortest_distance(Adr, Wco, W3)

    # ---- gather features ----
    pos_feats = gather_features(pa, pb, Adr, Wco, W3, typed, dist)
    neg_feats = gather_features(na, nb, Adr, Wco, W3, typed, dist) if len(na) else {}

    # ---- drug KG-degree summary ----
    deg_summary = {
        "mean": float(deg.mean()), "median": float(np.median(deg)),
        "min": int(deg.min()), "max": int(deg.max()),
        "frac_zero_degree": float((deg == 0).mean()),
    }
    log(f"[deg] drug KG-degree mean={deg_summary['mean']:.1f} "
        f"median={deg_summary['median']:.0f} min={deg_summary['min']} max={deg_summary['max']}")

    # ======================= PRINT: path evidence ======================= #
    log("\n================= PAIR-LEVEL PATH EVIDENCE (positives) =================")
    dist_pos = pos_feats["distance"]
    for d in [1, 2, 3, 4]:
        frac = float((dist_pos == d).mean())
        lbl = ">=4 / disconnected" if d == 4 else str(d)
        log(f"  shortest distance == {lbl:>20s}: {frac * 100:6.2f}%")
    log(f"  connectivity (distance <= 3)        : {float((dist_pos <= 3).mean()) * 100:6.2f}%")
    for fname in ["direct_edge", "common_neighbors", "walks_len3",
                  "shared_gene_protein", "shared_side_effect", "shared_pathway", "shared_disease"]:
        s = describe(pos_feats[fname])
        log(f"  {fname:22s}: mean={s['mean']:9.2f} median={s['median']:7.1f} "
            f"p90={s['p90']:9.1f} %>0={s['frac_gt0'] * 100:5.1f}")

    # ======================= PRINT: pos vs neg ======================= #
    contrast = {}
    if neg_feats:
        log("\n================= POS vs NEG CONTRAST + trivial-predictor AUC =================")
        for fname in ["direct_edge", "common_neighbors", "walks_len3", "distance",
                      "shared_gene_protein", "shared_side_effect", "shared_pathway", "shared_disease"]:
            sp_ = describe(pos_feats[fname])
            sn_ = describe(neg_feats[fname])
            y = np.concatenate([np.ones(len(pos_feats[fname])), np.zeros(len(neg_feats[fname]))])
            score = np.concatenate([pos_feats[fname], neg_feats[fname]]).astype(np.float64)
            if fname == "distance":
                score = -score  # closer == more likely positive
            auc = float(roc_auc_score(y, score))
            contrast[fname] = {"pos": sp_, "neg": sn_, "auc": auc}
            log(f"  {fname:22s}: POS mean={sp_['mean']:8.2f} %>0={sp_['frac_gt0'] * 100:5.1f} | "
                f"NEG mean={sn_['mean']:8.2f} %>0={sn_['frac_gt0'] * 100:5.1f} | AUC={auc:.4f}")

    # ======================= Relational WL collision ======================= #
    log("\n================= RELATIONAL-WL COLLISION DIAGNOSTIC =================")
    init_color = pd.factorize(kinds, sort=True)[0].astype(np.int64)
    u, r, v, n_rel_total = build_wl_edges(edges, id2idx, n_nodes)
    hist = relational_wl(u, r, v, init_color, n_nodes, n_rel_total, args.wl_rounds)

    wl_report = {}
    for rnd, color in enumerate(hist):
        drug_color = color[drug_idx]  # length n_drug
        n_drug_colors = int(len(np.unique(drug_color)))
        # twin drugs: drugs sharing their colour with >=1 other drug
        _, inv, cnt = np.unique(drug_color, return_inverse=True, return_counts=True)
        frac_twin_drugs = float((cnt[inv] > 1).mean())

        # pair signatures
        pa_c, pb_c = drug_color[pa], drug_color[pb]
        pos_sig = pair_signature(pa_c, pb_c)

        # (a) mechanism ambiguity: signature mapping to >1 distinct ddi_type
        df = pd.DataFrame({"sig": pos_sig, "ty": ddi_type})
        n_types_per_sig = df.groupby("sig")["ty"].nunique()
        ambiguous_sigs = set(n_types_per_sig[n_types_per_sig > 1].index.tolist())
        frac_pos_mech_ambiguous = float(np.isin(pos_sig, list(ambiguous_sigs)).mean()) if ambiguous_sigs else 0.0
        n_distinct_pos_sig = int(df["sig"].nunique())

        rep = {
            "n_drug_colors": n_drug_colors,
            "frac_twin_drugs": frac_twin_drugs,
            "n_distinct_pos_signatures": n_distinct_pos_sig,
            "frac_pos_mechanism_ambiguous": frac_pos_mech_ambiguous,
        }

        # (b) pos/neg indistinguishability: positive signatures also seen in negatives
        if len(na):
            na_c, nb_c = drug_color[na], drug_color[nb]
            neg_sig = pair_signature(na_c, nb_c)
            # signatures must be comparable across pos/neg -> rebuild a shared map
            both_lo = np.concatenate([np.minimum(pa_c, pb_c), np.minimum(na_c, nb_c)])
            both_hi = np.concatenate([np.maximum(pa_c, pb_c), np.maximum(na_c, nb_c)])
            keys = np.stack([both_lo, both_hi], axis=1)
            _, shared = np.unique(keys, axis=0, return_inverse=True)
            ps = shared[: len(pa_c)]
            ns = shared[len(pa_c):]
            neg_sig_set = set(ns.tolist())
            pos_sig_set = set(ps.tolist())
            frac_pos_collide_neg = float(np.isin(ps, list(neg_sig_set)).mean())
            frac_neg_collide_pos = float(np.isin(ns, list(pos_sig_set)).mean())
            rep["frac_pos_signature_also_in_neg"] = frac_pos_collide_neg
            rep["frac_neg_signature_also_in_pos"] = frac_neg_collide_pos

        wl_report[f"round_{rnd}"] = rep
        log(f"  [round {rnd}] drug_colors={n_drug_colors}/{n_drug} "
            f"twin_drugs={frac_twin_drugs * 100:5.1f}%  "
            f"pos_sig={n_distinct_pos_sig}  mech_ambig={frac_pos_mech_ambiguous * 100:5.1f}%"
            + (f"  pos_sig_in_neg={rep.get('frac_pos_signature_also_in_neg', float('nan')) * 100:5.1f}%"
               if len(na) else ""))

    # ======================= Save artifacts ======================= #
    summary = {
        "kg": {"n_nodes": int(n_nodes), "n_edges": int(len(edges)), "adj_nnz": int(A.nnz)},
        "n_drug": int(n_drug),
        "n_pos_pairs": int(len(pa)),
        "n_neg_pairs": int(len(na)),
        "neg_seed": args.seed,
        "drug_degree": deg_summary,
        "pos_path_evidence": {
            "distance_hist": {str(d): float((dist_pos == d).mean()) for d in [1, 2, 3, 4]},
            "connectivity_le3": float((dist_pos <= 3).mean()),
            **{k: describe(v) for k, v in pos_feats.items() if k != "distance"},
        },
        "pos_vs_neg_contrast": contrast,
        "relational_wl": wl_report,
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n[save] {out_dir / 'summary.json'}")

    pos_df = pd.DataFrame({"a_local": pa, "b_local": pb, "ddi_type": ddi_type, **pos_feats})
    pos_df.to_parquet(out_dir / "pos_features.parquet")
    log(f"[save] {out_dir / 'pos_features.parquet'}  ({len(pos_df)} rows)")
    if neg_feats:
        neg_df = pd.DataFrame({"a_local": na, "b_local": nb, **neg_feats})
        neg_df.to_parquet(out_dir / "neg_features.parquet")
        log(f"[save] {out_dir / 'neg_features.parquet'}  ({len(neg_df)} rows)")
    (out_dir / "wl_collision.json").write_text(json.dumps(wl_report, indent=2))
    log(f"[save] {out_dir / 'wl_collision.json'}")

    log(f"\n[done] wall_time={summary['wall_time_s']}s")


if __name__ == "__main__":
    sys.exit(main())
