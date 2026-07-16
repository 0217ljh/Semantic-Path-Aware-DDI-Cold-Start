"""Phase D prerequisite — build pair -> ddi_type index map for joint binary + 215-class task.

The legacy seed42 pkl positives don't carry ddi_type; the RELEASE seed42 split parquets do
(verified earlier - same approach as analyze_v2_pkpd.py). For Phase D's multi-class auxiliary
head we need (canonical-pair) -> ddi_type, then -> class index in [0..K-1].

Output: Code/data/_cache/ddi_type_map.json
  {
    "type_to_idx": {ddi_type_string: int},
    "idx_to_type": [...K...],
    "pair_to_idx": {"DBxxx|DByyy": int}   # canonical sorted pair
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
SPLIT_DIR = ROOT / "Code/data/KG/drugbank/splits/seed42"
OUT = ROOT / "Code/data/_cache/ddi_type_map.json"


def _canon(a: str, b: str) -> str:
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def main() -> None:
    pair_to_type: dict[str, str] = {}
    for pq in sorted(SPLIT_DIR.glob("*.parquet")):
        d = pd.read_parquet(pq)
        if "ddi_type" not in d.columns:
            continue
        for a, b, t in zip(d["drug_a_id"].astype(str),
                            d["drug_b_id"].astype(str), d["ddi_type"]):
            t = str(t).strip()
            if not t:
                continue
            pair_to_type[_canon(a, b)] = t

    types = sorted(set(pair_to_type.values()))
    type_to_idx = {t: i for i, t in enumerate(types)}
    pair_to_idx = {p: type_to_idx[t] for p, t in pair_to_type.items()}
    K = len(types)
    print(f"[ddi_type_map] pairs={len(pair_to_idx)}  num_ddi_types={K}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "type_to_idx": type_to_idx,
        "idx_to_type": types,
        "pair_to_idx": pair_to_idx,
    }))
    print(f"[ddi_type_map] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
