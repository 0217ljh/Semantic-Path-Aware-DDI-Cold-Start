"""Test: do drug->mediator RELATION features add KG-only signal (AND support)?

For each pair's support mediators, build a relation-bucket histogram of the
a->m edge relation (db:target / enzyme / gene-regulation / side-effect / ...;
REL_2HOP if the mediator is not 1-hop). Append to the struct features and
compare GBDT test AUC with / without. If relation adds nothing KG-only, its
value (like co-path) is realized only via molecular alignment.

Run: python Code/scripts/analyze_spmn_v1_relation.py --seed 42 --train-sample 30000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import MergedKG  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
EDGES = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"

REL_BUCKET = {
    "db:target": 0, "prime:drug_protein": 0,
    "db:enzyme": 1, "db:transporter": 2, "db:carrier": 3, "db:pathway": 4,
    "het:CdG": 5, "het:CuG": 5, "het:CbG": 5,            # gene regulation
    "het:CcSE": 6, "prime:drug_effect": 6,               # side effect / effect
    "prime:indication": 7, "prime:off-label use": 7,
    "prime:contraindication": 8,
}
N_BUCKET = 11           # 0..8 named + 9 other + 10 = REL_2HOP
REL_2HOP = 10


def _binary_frame(pos, neg):
    p = pos[["drug_a_id", "drug_b_id"]].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--train-sample", type=int, default=30000)
    args = ap.parse_args()

    z = np.load(ROOT / "Code/data/_cache" /
                f"spmn_v1_phase2_supports_v3and_seed{args.seed}_lmax{args.l_max}.npz")
    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    frames = {"train": _binary_frame(ds.splits.train, ds.get_train_negatives()),
              "test_s2": _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))}

    kg = MergedKG.from_parquet()
    print("[rel] building drug->neighbor relation lookup ...", flush=True)
    e = pd.read_parquet(EDGES, columns=["src", "dst", "relation"])
    # bucket each relation
    buck = e["relation"].map(lambda r: REL_BUCKET.get(r, 9)).to_numpy()
    src = e["src"].astype(str).map(kg.id_to_idx).to_numpy()
    dst = e["dst"].astype(str).map(kg.id_to_idx).to_numpy()
    drug_rel: dict[int, dict[int, int]] = {}
    for s, d, b in zip(src, dst, buck):
        if s != s or d != d:  # NaN
            continue
        s, d = int(s), int(d)
        if kg.is_drug[s]:
            drug_rel.setdefault(s, {}).setdefault(d, int(b))
        if kg.is_drug[d]:
            drug_rel.setdefault(d, {}).setdefault(s, int(b))

    def rel_hist(frame, split, sample=None):
        n = len(frame)
        idx = np.arange(n) if sample is None else \
            np.random.default_rng(args.seed).permutation(n)[:sample]
        med, off = z[f"{split}__med"], z[f"{split}__offsets"]
        struct = z[f"{split}__struct"][idx]
        y = z[f"{split}__y"][idx]
        a_ids = frame["drug_a_id"].astype(str).to_numpy()
        b_ids = frame["drug_b_id"].astype(str).to_numpy()
        H = np.zeros((len(idx), 2 * N_BUCKET), dtype=np.float32)
        t0 = time.time()
        for r, p in enumerate(idx):
            a = kg.id_to_idx.get(a_ids[p]); b = kg.id_to_idx.get(b_ids[p])
            meds = med[off[p]:off[p + 1]]
            ra = drug_rel.get(a, {}); rb = drug_rel.get(b, {})
            for m in meds.tolist():
                H[r, ra.get(m, REL_2HOP)] += 1
                H[r, N_BUCKET + rb.get(m, REL_2HOP)] += 1
        print(f"  [{split}] {len(idx)} pairs {time.time()-t0:.0f}s", flush=True)
        return struct, np.log1p(H), y

    xtr, htr, ytr = rel_hist(frames["train"], "train", args.train_sample)
    xte, hte, yte = rel_hist(frames["test_s2"], "test_s2")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    def fit(Xtr, Xte, tag):
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        c = HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                           learning_rate=0.1, random_state=args.seed)
        c.fit((Xtr - mu) / sd, ytr)
        p = c.predict_proba((Xte - mu) / sd)[:, 1]
        print(f"  {tag:32s} dim={Xtr.shape[1]:3d}  test AUC={roc_auc_score(yte, p):.4f}")

    print("=== relation-feature test (AND support, GBDT) ===")
    fit(xtr, xte, "struct only")
    fit(np.concatenate([xtr, htr], 1), np.concatenate([xte, hte], 1),
        "struct + relation-histogram")
    fit(htr, hte, "relation-histogram only")


if __name__ == "__main__":
    main()
