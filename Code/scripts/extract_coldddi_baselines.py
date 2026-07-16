"""Extract baseline binary cls metrics from ColdDDI/Code-Released/baseline/Output.

Each baseline has Baseline.csv with test-auc_roc, test-auc_prc, test-f1_score
for splits s0/s1/s2, run on seeds 42/43/44 against our 1900-drug dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("/mnt/d/My-Research/03-Projects/ColdDDI/Code-Released/baseline/Output")
    methods = [
        "EmerGNN",
        "MKG-FENN",
        "MKG-FENN-NEW",
        "TIGER",
        "HDN-DDI",
        "HDN-DDI-NEW",
        "SSI-DDI",
        "DSN-DDI",
        "DeepDDI",
        "TextDDI",
    ]

    print("Per-seed binary cls results on 1900-drug data (subdir=my):")
    print(f"{'Method':14s}  Seed  Split   AUROC   AUPRC      F1")
    all_records = []
    for m in methods:
        for seed in [42, 43, 44]:
            csv = root / m / "my" / f"seed{seed}" / "Baseline.csv"
            if not csv.is_file():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            for _, r in df.iterrows():
                split = r["split"]
                auc = r.get("test-auc_roc")
                auprc = r.get("test-auc_prc")
                f1 = r.get("test-f1_score")
                if pd.isna(auc):
                    continue
                print(f"{m:14s}  {seed}    {split:4s}  {auc:.4f}  {auprc:.4f}  {f1:.4f}")
                all_records.append(
                    {
                        "method": m,
                        "seed": seed,
                        "split": split,
                        "auc": auc,
                        "auprc": auprc,
                        "f1": f1,
                    }
                )

    print()
    print("=== S2 multi-seed aggregate (mean +/- std) ===")
    df = pd.DataFrame(all_records)
    s2 = df[df["split"] == "s2"]
    print(f"{'Method':14s}  n   AUROC                AUPRC                F1")
    for m, grp in s2.groupby("method"):
        n = len(grp)
        auc_m = grp.auc.mean()
        auc_s = grp.auc.std() if n > 1 else 0.0
        auprc_m = grp.auprc.mean()
        auprc_s = grp.auprc.std() if n > 1 else 0.0
        f1_m = grp.f1.mean()
        f1_s = grp.f1.std() if n > 1 else 0.0
        print(
            f"{m:14s}  {n}   {auc_m:.4f}+-{auc_s:.4f}    "
            f"{auprc_m:.4f}+-{auprc_s:.4f}    {f1_m:.4f}+-{f1_s:.4f}"
        )

    print()
    print("=== S0/S1/S2 aggregate by method (mean only, for full comparison) ===")
    for m, grp in df.groupby("method"):
        for split in ["s0", "s1", "s2"]:
            sub = grp[grp["split"] == split]
            if len(sub) == 0:
                continue
            print(
                f"{m:14s}  {split:4s}  n={len(sub)}  "
                f"AUC={sub.auc.mean():.4f}  AUPRC={sub.auprc.mean():.4f}  F1={sub.f1.mean():.4f}"
            )
        print()


if __name__ == "__main__":
    main()
