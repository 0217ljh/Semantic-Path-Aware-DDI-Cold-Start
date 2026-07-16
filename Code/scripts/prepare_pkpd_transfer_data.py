"""Prepare PK/PD-bucketed pair lists for the NBFNet transfer experiments.

The 800-drug legacy seed42 dataset splits carry no ddi_type, so each (a,b)
positive pair is bucketed PK / PD / other via the 1900-drug ddi_edges.csv
(unordered key -> ddi_type -> keyword bucket). Dumps filtered pair CSVs that
run_nbfnet_pkpd_transfer.py consumes.

Outputs (Code/experiments/pkpd_transfer/data/):
  {split}_ALL.csv / {split}_PK.csv / {split}_PD.csv   for split in
  {train, val_s2, test_s2}; columns drug_a_id,drug_b_id,ddi_type,bucket.

Run (from project root, via WSL conda env project_1):
  python Code/scripts/prepare_pkpd_transfer_data.py --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
sys.path.insert(0, str(ROOT))
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"
EXP_DIR = ROOT / "experiments/pkpd_transfer"
DATA_DIR = EXP_DIR / "data"


def pkpd_bucket(t) -> str:
    t = str(t).lower()
    if any(k in t for k in ("metabolism", "excretion", "serum concentration",
                            "absorption", "protein binding", "clearance")):
        return "PK"
    if any(k in t for k in ("risk or severity", "activities", "efficacy",
                            "cns depression", "qtc", "hypertension",
                            "hypotensive", "sedative", "adverse effects")):
        return "PD"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    from data_utils import PairDataset
    pkl = ROOT / f"data/coldddi_legacy/800drug/seed{args.seed}.pkl"
    ds = PairDataset.from_pkl(str(pkl))
    print(f"[prep] dataset: {pkl}", flush=True)

    # unordered key -> ddi_type
    pos = pd.read_csv(DDI_POS_CSV)
    key2type = {}
    for x, y, t in zip(pos["drug_a_id"].astype(str), pos["drug_b_id"].astype(str),
                       pos["ddi_type"].astype(str)):
        key2type[frozenset((x, y))] = t

    for split in ["train", "val_s2", "test_s2"]:
        df = getattr(ds.splits, split)[["drug_a_id", "drug_b_id"]].copy()
        df["drug_a_id"] = df["drug_a_id"].astype(str)
        df["drug_b_id"] = df["drug_b_id"].astype(str)
        # restrict to label==1 positives if a label column exists
        raw = getattr(ds.splits, split)
        if "label" in raw.columns:
            df = df[raw["label"].to_numpy() == 1].reset_index(drop=True)
        types = [key2type.get(frozenset((a, b))) for a, b in
                 zip(df["drug_a_id"], df["drug_b_id"])]
        df["ddi_type"] = types
        df["bucket"] = [pkpd_bucket(t) if t is not None else "unmapped" for t in types]

        n = len(df)
        vc = df["bucket"].value_counts().to_dict()
        print(f"[prep] {split}: n={n}  {vc}", flush=True)

        df.to_csv(DATA_DIR / f"{split}_ALL.csv", index=False)
        for bk in ["PK", "PD"]:
            sub = df[df["bucket"] == bk].reset_index(drop=True)
            sub.to_csv(DATA_DIR / f"{split}_{bk}.csv", index=False)
            print(f"        -> {split}_{bk}.csv  ({len(sub)} pairs)", flush=True)

    print(f"\n[prep] done. data in {DATA_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
