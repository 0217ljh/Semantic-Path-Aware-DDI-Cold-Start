"""Generate 800-drug cold-start S2 splits for seeds 42/43/44 (multi-seed).

The legacy 800-drug bundle ships only seed42 as a pickle. For multi-seed
robustness we regenerate the SAME split protocol for seeds 42/43/44 using the
canonical `data_utils.splits.build_splits`, reading the positive edge set out of
the legacy seed42 pickle.

Protocol (matched to legacy seed42 by inspection):
    drug_ratio = 1.25   ->  |G1|=640 / |G2|=160  (matches legacy)
    val_ratio  = 0.5    ->  val_s2 / test_s2 split  (matches legacy 1918 / 1919)
seed42 under this protocol reproduces the legacy val_s2 / test_s2 EXACTLY. Since
these are S2-only experiments (we never evaluate S0/S1), train uses the FULL
G1xG1 positive set: the unused S0 val/test holdout is merged back into train
(~59.7k positives, larger than legacy 53.7k), maximising training data. The S2
eval sets (val_s2 / test_s2, G2xG2) stay held out and identical across seeds.

Writes per-seed binary frames (positives label=1 + negatives label=0) to
    Code/data/coldddi_legacy/800drug_3seed/seed{N}/{train,val_s2,test_s2}.parquet
plus ddi_type carried through (for the later multi-class task). Read-only on the
source pickle; only writes new parquet.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

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
from data_utils.negatives import build_static_negatives, build_train_negatives  # noqa: E402
from data_utils.splits import build_splits  # noqa: E402

LEGACY_PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
OUT_DIR = ROOT / "Code/data/coldddi_legacy/800drug_3seed"
DRUG_RATIO = 1.25   # matches legacy |G1|=640 / |G2|=160
VAL_RATIO = 0.5     # matches legacy val_s2=1918 / test_s2=1919
SEEDS = (42, 43, 44)


def _binframe(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ("drug_a_id", "drug_b_id", "ddi_type") if c in pos.columns]
    p = pos[keep].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def main() -> None:
    print(f"[prepare] reading positive edges from {LEGACY_PKL.name}", flush=True)
    ds = PairDataset.from_pkl(str(LEGACY_PKL))
    edges = ds.edges.reset_index(drop=True)
    print(f"[prepare] {len(edges)} positive pairs; legacy seed42 ref: "
          f"val_s2={len(ds.splits.val_s2)} test_s2={len(ds.splits.test_s2)} "
          f"|G1|={len(ds.splits.g1_drugs)} |G2|={len(ds.splits.g2_drugs)}", flush=True)

    for seed in SEEDS:
        folds = build_splits(edges, seed=seed, drug_ratio=DRUG_RATIO, val_ratio=VAL_RATIO)
        sneg = build_static_negatives(folds, base_seed=seed)
        # S2-only experiments never evaluate S0/S1, so the S0 val/test holdout is
        # wasted training data. Merge ALL G1xG1 positives back into train (full
        # both-seen pool); negatives are resampled for the enlarged train. val_s2
        # / test_s2 (our eval, G2xG2) stay held out and unchanged.
        train_pos = pd.concat([folds.train, folds.val_s0, folds.test_s0],
                              ignore_index=True)
        tneg = build_train_negatives(replace(folds, train=train_pos),
                                     base_seed=seed, epoch=0, n_per_pos=1)
        frames = {
            "train": _binframe(train_pos, tneg),
            "val_s2": _binframe(folds.val_s2, sneg["val_s2"]),
            "test_s2": _binframe(folds.test_s2, sneg["test_s2"]),
        }
        out = OUT_DIR / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        for name, fr in frames.items():
            fr.to_parquet(out / f"{name}.parquet", index=False)
        print(f"[prepare] seed{seed}: |G1|={len(folds.g1_drugs)} |G2|={len(folds.g2_drugs)} "
              f"train={len(frames['train'])} val_s2={len(frames['val_s2'])} "
              f"test_s2={len(frames['test_s2'])} -> {out}", flush=True)

    print(f"[prepare] done. frames under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
