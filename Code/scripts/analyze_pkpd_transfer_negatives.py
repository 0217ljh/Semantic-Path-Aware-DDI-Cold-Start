"""Are the (generated) test negatives structurally more PK-like or PD-like?

Compares the confluent (1,1) relation-channel signature of the test_s2 negatives
against PK-positive and PD-positive signatures. If the negatives' profile is
closer to PD-positives, then PD-pos vs neg separation is intrinsically harder
(both diffuse) -> part of why PD AUC is low; if closer to PK-positives, the
opposite.

Signature = per-channel mean # of confluent (1,1) mediators shared via the SAME
relation r (db:enzyme/transporter/target ... side_effect/contraindication ...).

Run (from project root, via WSL conda env project_1) AFTER prepare:
  python Code/scripts/analyze_pkpd_transfer_negatives.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
sys.path.insert(0, str(ROOT))
NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
EXP_DIR = ROOT / "experiments/pkpd_transfer"

CHANNELS = [
    "db:enzyme", "db:transporter", "db:carrier", "db:target",
    "prime:drug_protein", "het:CbG", "het:CdG", "het:CuG",
    "het:CcSE", "prime:drug_effect", "prime:contraindication",
    "prime:indication", "het:CrC",
]


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nodes = pd.read_parquet(NODES_PARQUET)
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)
    drug_db = nodes["id"].to_numpy()[nodes["kind"].to_numpy() == "Drug"]
    drug_idx = np.array([id2idx[d] for d in drug_db])
    g2l = {int(g): i for i, g in enumerate(drug_idx)}
    d2g = {db: id2idx[db] for db in drug_db}
    dset = set(int(g) for g in drug_idx)

    src = edges["src"].map(id2idx).to_numpy(); dst = edges["dst"].map(id2idx).to_numpy()
    rel = edges["relation"].to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64); dst = dst[keep].astype(np.int64); rel = rel[keep]
    inc = {}
    for ch in CHANNELS:
        m = rel == ch; s, d = src[m], dst[m]
        sd = np.array([x in dset for x in s]); dd = np.array([x in dset for x in d]) if len(d) else np.array([], bool)
        rr = np.concatenate([np.array([g2l[x] for x in s[sd]], dtype=np.int64),
                             np.array([g2l[x] for x in d[dd]], dtype=np.int64)])
        cc = np.concatenate([d[sd], s[dd]])
        M = sp.coo_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(len(drug_idx), n_nodes)).tocsr()
        M.data[:] = 1
        inc[ch] = M

    def signature(pairs):
        """mean per-channel confluent shared count over pairs."""
        vec = np.zeros(len(CHANNELS))
        used = 0
        for a, b in zip(pairs["drug_a_id"].astype(str), pairs["drug_b_id"].astype(str)):
            if a not in d2g or b not in d2g:
                continue
            la, lb = g2l[d2g[a]], g2l[d2g[b]]
            for j, ch in enumerate(CHANNELS):
                vec[j] += int(inc[ch][la].multiply(inc[ch][lb]).sum())
            used += 1
        return vec / max(used, 1), used

    # positives from prepared CSVs; negatives from dataset
    pk = pd.read_csv(EXP_DIR / "data/test_s2_PK.csv")
    pd_ = pd.read_csv(EXP_DIR / "data/test_s2_PD.csv")
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(ROOT / f"data/coldddi_legacy/800drug/seed{args.seed}.pkl"))
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]

    sig_pk, n_pk = signature(pk)
    sig_pd, n_pd = signature(pd_)
    sig_neg, n_neg = signature(neg)

    log(f"[neg-analysis] n_pk={n_pk} n_pd={n_pd} n_neg={n_neg}")
    def _pick(m_pk, m_pd, lower_better):
        """Robust verdict: handle NaN and exact ties explicitly (no silent default)."""
        if np.isnan(m_pk) or np.isnan(m_pd):
            return "undetermined(NaN)"
        if m_pk == m_pd:
            return "tie"
        if lower_better:
            return "PK" if m_pk < m_pd else "PD"
        return "PK" if m_pk > m_pd else "PD"

    log(f"\n{'channel':22s} {'PK_pos':>9s} {'PD_pos':>9s} {'NEG':>9s}  closer_to")
    for j, ch in enumerate(CHANNELS):
        closer = _pick(abs(sig_neg[j] - sig_pk[j]), abs(sig_neg[j] - sig_pd[j]), lower_better=True)
        log(f"{ch:22s} {sig_pk[j]:9.3f} {sig_pd[j]:9.3f} {sig_neg[j]:9.3f}  {closer}")

    def cos(u, v):
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")

    def l1(u, v):
        return float(np.abs(u - v).sum())

    cos_pk, cos_pd = cos(sig_neg, sig_pk), cos(sig_neg, sig_pd)
    l1_pk, l1_pd = l1(sig_neg, sig_pk), l1(sig_neg, sig_pd)
    log("\n[overall] NEG profile vs PK / PD positives:")
    log(f"  cosine(NEG, PK_pos) = {cos_pk:.4f}   cosine(NEG, PD_pos) = {cos_pd:.4f}")
    log(f"  L1(NEG, PK_pos)     = {l1_pk:.3f}     L1(NEG, PD_pos)     = {l1_pd:.3f}")
    verdict_cos = _pick(cos_pk, cos_pd, lower_better=False)
    verdict_l1 = _pick(l1_pk, l1_pd, lower_better=True)
    log(f"  => by cosine NEG is closer to {verdict_cos}; by L1 to {verdict_l1}")

    out = {
        "n": {"pk": n_pk, "pd": n_pd, "neg": n_neg},
        "channels": CHANNELS,
        "sig_pk": sig_pk.tolist(), "sig_pd": sig_pd.tolist(), "sig_neg": sig_neg.tolist(),
        "cosine_neg_pk": cos(sig_neg, sig_pk), "cosine_neg_pd": cos(sig_neg, sig_pd),
        "l1_neg_pk": l1(sig_neg, sig_pk), "l1_neg_pd": l1(sig_neg, sig_pd),
    }
    (EXP_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "runs" / "negatives_signature.json").write_text(json.dumps(out, indent=2))
    log(f"\n[save] {EXP_DIR/'runs'/'negatives_signature.json'}")


if __name__ == "__main__":
    sys.exit(main())
