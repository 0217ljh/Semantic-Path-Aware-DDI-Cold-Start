"""Materialize the TWOSIDES multi-label side-effect corpus into the unified layout.

TWOSIDES (Tatonetti et al. 2012) as preprocessed by EmerGNN: 604 drugs, 200 UMLS
side-effect labels (multihot), 1:1 positive:negative where a NEGATIVE corrupts ONE
endpoint of its paired positive (same tail + same relation set). See
`Notes/Log/twosides_source_format.md` for the verified source/format. Three dataset_ids:
  - twosides_warm_ml : warm / double-known (S0). Source ships ONE official split; per
    the unified 3-fold protocol we re-split the S0 pair pool into 3 SEEDED folds
    (transductive — pairs split, drug pool shared). pos+its mate neg stay in one fold.
  - twosides_s1_ml   : one-drug-unseen (S1), official folds _1/_12/_123 -> fold0/1/2.
  - twosides_s2_ml   : both-drugs-unseen / cold-start (S2), folds _1/_12/_123.

Per codex review (2026-06-29): cold-start drug validation uses an ASYMMETRIC
seen-set — train side = EVERY train-row drug (pos AND neg), matching EmerGNN's
train_ent (load_data.py:75-77 adds x,y for every train line, so a drug seen only as a
corrupted-negative endpoint still counts as seen); eval side = POSITIVES ONLY
(eval negatives are endpoint-corruptions and would falsely trip the S1/S2 disjointness
check). drug_split is built with the same asymmetric rule. Pairs keep source order
(pair_format="ordered_source"); the pos<->neg corruption pairing is preserved in a
sidecar `<split>_pair_links.parquet` without touching the fixed multilabel row schema.
Read-only on the source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import json
import pickle

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
from data_utils import unified  # noqa: E402

SRC = ROOT / "Code/reproductions/EmerGNN/_Original-Dataset/TWOSIDES/data"
NEC = SRC / "necessary"
OUT = ROOT / "Code/data/ddi_unified"
N_LABELS = 200
MORGAN_DIM = 1024
FOLD_DIRS = ("1", "12", "123", "1234", "12345")   # 5 official CV folds -> fold0..4
WARM_SEED = 20260629                     # S0 self 3-fold (transductive)
KG_REF = {"scope": "dataset", "source": str(SRC.relative_to(ROOT)),
          "drug_node_key": "drug_id"}


def _read_ddi(path: Path) -> tuple[list, list]:
    """Parse one *_ddi.txt -> (pos_rows, neg_rows) in FILE ORDER.

    Each row = (drug_a:str, drug_b:str, y_label_ids:list[int]). EmerGNN pairs the
    k-th positive with the k-th negative (file order) -> we keep that index pairing.
    """
    pos, neg = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y, z, w = line.split("\t")
            labels = [i for i, v in enumerate(map(int, z.split(","))) if v == 1]
            row = (str(int(x)), str(int(y)), labels)
            (pos if int(w) == 1 else neg).append(row)
    return pos, neg


def _frame(pos_rows: list, neg_rows: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a split row-frame (file order: positives then negatives) + its pair_links
    sidecar. group_id = index within class; mate links k-th pos <-> k-th neg."""
    if len(pos_rows) != len(neg_rows):
        raise ValueError(f"pos/neg not 1:1 ({len(pos_rows)} vs {len(neg_rows)})")
    recs, links = [], []
    for g, (a, b, labels) in enumerate(pos_rows):
        recs.append((a, b, 1, labels, g))
    for g, (a, b, labels) in enumerate(neg_rows):
        recs.append((a, b, 0, labels, g))
    df = pd.DataFrame(recs, columns=["drug_a_id", "drug_b_id", "is_positive",
                                     "y_label_ids", "group_id"])
    df.insert(0, "pair_id", np.arange(len(df), dtype=np.int64))
    pos_id = df[df.is_positive == 1].set_index("group_id")["pair_id"]
    neg_id = df[df.is_positive == 0].set_index("group_id")["pair_id"]
    for g in pos_id.index:
        links.append((int(pos_id[g]), int(neg_id[g]), int(g)))   # pos<->neg
    pair_links = pd.DataFrame(links, columns=["pos_pair_id", "neg_pair_id", "group_id"])
    rows = df[["pair_id", "drug_a_id", "drug_b_id", "is_positive", "y_label_ids"]]
    return rows, pair_links


def _positives(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.is_positive == 1][["drug_a_id", "drug_b_id"]]


def _load_drug_tables() -> tuple[dict, pd.DataFrame]:
    """id2drug + id2drug_feat -> per-entity {cid, db, smiles, morgan}. Returns the
    raw dict and a builder for drugs.parquet given a used-id set."""
    i2d = json.load(open(NEC / "id2drug__official.json"))
    feat = pickle.load(open(NEC / "id2drug_feat__official.pkl", "rb"))
    info = {}
    for eid, d in i2d.items():
        m = feat.get(eid, {}).get("Morgan")
        info[str(eid)] = {
            "cid": d.get("cid"), "db": d.get("db"), "smiles": d.get("smiles"),
            "morgan": (np.asarray(m, dtype=np.float64).tolist() if m is not None else None),
        }
    return info, feat  # feat returned only to keep ref alive (unused downstream)


def _drugs_frame(used: set[str], info: dict) -> pd.DataFrame:
    miss = used - set(info)
    if miss:
        raise ValueError(f"[twosides] {len(miss)} used drugs lack id2drug entry: "
                         f"{sorted(miss)[:5]}")
    ids = sorted(used, key=int)
    # drugbank_id stays NA when the source has no DrugBank mapping (db is null) —
    # do NOT backfill the entity id, that would fake a DrugBank identifier (codex).
    return unified.make_drugs_frame(
        ids,
        smiles=[info[i]["smiles"] for i in ids],
        source_drug_id=[info[i]["cid"] for i in ids],
        drugbank_id=[(info[i]["db"] if info[i]["db"] else pd.NA) for i in ids],
        morgan_fp_1024=[info[i]["morgan"] for i in ids],
        smiles_source="emergnn_id2drug__official.json")


def _label_vocab() -> pd.DataFrame:
    r2i = json.load(open(NEC / "relation2id__official.json"))
    names = [str(r2i[str(i)]) for i in range(N_LABELS)]   # ids 0..199 = side-effects
    return pd.DataFrame({"label_idx_global": np.arange(N_LABELS, dtype=np.int64),
                         "label_name": names})


def _ds_dir(split_type: str) -> Path:
    """On-disk leaf dir for a twosides setting (multilabel) in the unified layout:
    multi_label_cls/twoside/<regime>/<split_code>/ ."""
    return unified.layout_dir(OUT, "twosides", "multilabel", split_type)


def _write_pair_links(split_type: str, links: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Sidecar: <fold>/<split>_pair_links.parquet (codex: preserve pos/neg corruption
    pairing without extending the fixed row schema). Folds are direct children of the leaf."""
    base = _ds_dir(split_type)
    for sid, per in links.items():
        for split, pl in per.items():
            pl.to_parquet(base / sid / f"{split}_pair_links.parquet", index=False)


def _materialize(dataset_id: str, split_type: str,
                 split_frames: dict, pair_links: dict, info: dict,
                 label_vocab: pd.DataFrame) -> None:
    """Validate (positives-only for cold) + write one dataset_id + pair_links sidecar."""
    drug_splits = None
    if split_type.startswith("cold"):
        drug_splits = {}
        for sid, fr in split_frames.items():
            # "train seen" = EVERY train-row drug (pos AND neg), matching EmerGNN's
            # train_ent (load_data.py:75-77 adds x,y for every train line) -> a drug
            # seen only as a corrupted-negative endpoint still counts as seen by the
            # model. Eval side uses POSITIVES only (codex: eval negatives are
            # endpoint-corruptions and would falsely trip the disjointness check).
            tr = fr["train"][["drug_a_id", "drug_b_id"]]
            va, te = _positives(fr["val"]), _positives(fr["test"])
            evals = [("val", va), ("test", te)]
            if split_type == "cold_s2":
                unified.check_cold_s2(tr, evals)
            else:
                unified.check_cold_s1(tr, evals)
            drug_splits[sid] = unified.build_drug_split(tr, evals)

    used = unified.drug_universe(*[df for fr in split_frames.values() for df in fr.values()])
    drugs = _drugs_frame(used, info)
    meta = unified.base_meta("twoside", "twosides", "multilabel", split_type,
                             sorted(split_frames), kg=KG_REF,
                             smiles_source="emergnn_id2drug__official.json",
                             n_labels=N_LABELS, morgan_dim=MORGAN_DIM)
    unified.write_dataset(_ds_dir(split_type), meta, drugs, label_vocab,
                          split_frames, drug_splits)
    _write_pair_links(split_type, pair_links)
    n_pos = sum(int((fr["train"].is_positive == 1).sum()) for fr in split_frames.values())
    print(f"[twosides] done -> {dataset_id}  sids={sorted(split_frames)} "
          f"drugs={len(drugs)} train_pos(sum over sids)={n_pos}", flush=True)


def _official(setting: str) -> None:
    """S1/S2: official folds _1/_12/_123 -> fold0/1/2."""
    info, _ = _load_drug_tables()
    lv = _label_vocab()
    split_frames, pair_links = {}, {}
    for k, fd in enumerate(FOLD_DIRS):
        sf, pl = {}, {}
        for split, fname in (("train", "train"), ("val", "valid"), ("test", "test")):
            pos, neg = _read_ddi(SRC / f"{setting}_{fd}" / f"{fname}_ddi.txt")
            sf[split], pl[split] = _frame(pos, neg)
        split_frames[f"fold{k}"] = sf
        pair_links[f"fold{k}"] = pl
    st = "cold_s2" if setting == "S2" else "cold_s1"
    _materialize(f"twosides_{'s2' if setting == 'S2' else 's1'}_ml", st,
                 split_frames, pair_links, info, lv)


def _warm() -> None:
    """S0: re-split the pooled S0 pair groups into 3 SEEDED folds (transductive)."""
    info, _ = _load_drug_tables()
    lv = _label_vocab()
    # pool all S0 (pos, mate-neg) groups across the official train/valid/test files
    groups = []  # each = (pos_row, neg_row)
    for fname in ("train", "valid", "test"):
        pos, neg = _read_ddi(SRC / "S0" / f"{fname}_ddi.txt")
        if len(pos) != len(neg):
            raise ValueError(f"S0/{fname}: pos/neg not 1:1")
        groups.extend(zip(pos, neg))
    rng = np.random.default_rng(WARM_SEED)
    order = rng.permutation(len(groups))
    fold_of = np.array_split(order, 5)              # 5 disjoint test folds
    split_frames, pair_links = {}, {}
    for k in range(5):
        test_idx = set(int(i) for i in fold_of[k])
        rest = [i for i in range(len(groups)) if i not in test_idx]
        rest_perm = rng.permutation(rest)
        n_val = max(1, int(round(0.125 * len(rest_perm))))
        val_idx = set(int(i) for i in rest_perm[:n_val])
        train_idx = [i for i in rest if i not in val_idx]
        parts = {"train": train_idx, "val": sorted(val_idx), "test": sorted(test_idx)}
        sf, pl = {}, {}
        for split, idxs in parts.items():
            pos = [groups[i][0] for i in idxs]
            neg = [groups[i][1] for i in idxs]
            sf[split], pl[split] = _frame(pos, neg)
        split_frames[f"fold{k}"] = sf
        pair_links[f"fold{k}"] = pl
    _materialize("twosides_warm_ml", "transductive", split_frames, pair_links, info, lv)


def main() -> None:
    _warm()
    _official("S1")
    _official("S2")


if __name__ == "__main__":
    main()
