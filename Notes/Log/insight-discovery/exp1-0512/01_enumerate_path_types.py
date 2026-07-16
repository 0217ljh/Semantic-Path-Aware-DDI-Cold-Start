"""Enumerate all 2-hop path types that appear between drug pairs in our
cold-start S2 evaluation set, using the merged KG.

We compare:
  - B-class positives: all positive pairs in S2 test that are PK-B or PD-B
  - Random negatives: S2 test negatives (cold-start paired negatives)

Output:
  - Console report of path types sorted by frequency
  - per_pair feature CSV with: drug_a, drug_b, label, pkpd_4way, total_paths,
    n_paths_by_signature_top_N ...
"""
from __future__ import annotations
import sys
from pathlib import Path
# Ensure Code/ is importable (this file is 4 levels deep from project root)
_HERE = Path(__file__).resolve()
for _p in [_HERE, *_HERE.parents]:
    if (_p / "Code" / "my_code").is_dir():
        sys.path.insert(0, str(_p / "Code"))
        break

import pandas as pd
from collections import Counter

# Auto-encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from my_code.kg_lib.loader import build_merged_kg
from my_code.paths.extractor import (
    build_neighbor_index, enumerate_2hop_paths,
    summarize_path_types, signature_to_str,
)
from my_code.datasets.ddi_ds import load_binary_split, load_multiclass_split

KG_OUT = Path(__file__).parent

print("=" * 80)
print("Step 1: Load merged KG")
print("=" * 80)
out = build_merged_kg()
edges = out["edges"]
print(f"Edges: {len(edges):,}  Cache hit: {out['cache_hit']}")

print("\nBuilding drug-neighbor index (one-time)...")
import time
t = time.time()
index = build_neighbor_index(edges)
print(f"  done in {time.time()-t:.1f}s. Drugs with neighbors: {len(index['drug_to_neighbors']):,}")

# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 2: Load cold-start S2 test split (binary)")
print("=" * 80)
test = load_binary_split(42, "test_s2", include_negatives=True)
pos = test[test["label"] == 1].copy()
neg = test[test["label"] == 0].copy()
print(f"Positives: {len(pos):,}   Negatives: {len(neg):,}")

# Annotate positives with 4-way PK/PD
from pathlib import Path as _P
import pandas as _pd
_root = None
for _p in [_HERE, *_HERE.parents]:
    if (_p / "Code" / "my_code").is_dir():
        _root = _p
        break
pkpd_csv = _pd.read_csv(_root / "Code" / "data" / "KG" /
                        "drugbank" / "enriched" / "ddi_pk_pd_labels.csv")

def _4way(row):
    label = row["pk_pd_label"]
    if pd.isna(label) or label == "Mixed":
        return label
    if label == "PK":
        kws = str(row.get("matched_pk_keywords", "") or "").lower()
        return "PK-B" if ("metabolism" in kws or "serum concentration" in kws) else "PK-A"
    pkws = str(row.get("matched_pd_keywords", "") or "").lower()
    return "PD-B" if ("risk" in pkws or "adverse" in pkws or "severity" in pkws) else "PD-A"

pkpd_csv["pkpd_4way"] = pkpd_csv.apply(_4way, axis=1)
labels_map = dict(zip(pkpd_csv["ddi_type"], pkpd_csv["pkpd_4way"]))
pos["pkpd_4way"] = pos["ddi_type"].map(labels_map)
print(f"\nPositive S2 by pkpd_4way:")
print(pos["pkpd_4way"].value_counts().to_string())

# Subset: sample 500 from each of PK-B / PD-B / NEG (manageable + statistical)
SEED = 42
N_EACH = 500
pkb = pos[pos["pkpd_4way"] == "PK-B"].sample(n=min(N_EACH, sum(pos["pkpd_4way"]=="PK-B")),
                                              random_state=SEED)
pdb = pos[pos["pkpd_4way"] == "PD-B"].sample(n=min(N_EACH, sum(pos["pkpd_4way"]=="PD-B")),
                                              random_state=SEED)
neg_s = neg.sample(n=N_EACH * 2, random_state=SEED)  # 1000 negs to match 500+500 positives
print(f"\nSubsample for enumeration: {len(pkb)} PK-B + {len(pdb)} PD-B + {len(neg_s)} NEG")

# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 3: Enumerate path types for each subset")
print("=" * 80)

def pairs_of(df):
    return list(zip(df["drug_a_id"], df["drug_b_id"]))

print("Running on PK-B...")
s_pkb = summarize_path_types(pairs_of(pkb), index)
print(f"  in_kg={s_pkb['pairs_in_kg']}/{len(pkb)}, with_paths={s_pkb['pairs_with_paths']}")
print("Running on PD-B...")
s_pdb = summarize_path_types(pairs_of(pdb), index)
print(f"  in_kg={s_pdb['pairs_in_kg']}/{len(pdb)}, with_paths={s_pdb['pairs_with_paths']}")
print("Running on NEG...")
s_neg = summarize_path_types(pairs_of(neg_s), index)
print(f"  in_kg={s_neg['pairs_in_kg']}/{len(neg_s)}, with_paths={s_neg['pairs_with_paths']}")

# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Step 4: Path-type frequency report")
print("=" * 80)

all_sigs = set()
for s in (s_pkb, s_pdb, s_neg):
    all_sigs.update(s["type_counter"].keys())
print(f"Total unique path-type signatures across all 3 groups: {len(all_sigs):,}")

# Rank signatures by combined frequency across all groups
combined = Counter()
for s in (s_pkb, s_pdb, s_neg):
    for sig, c in s["type_counter"].items():
        combined[sig] += c

# Print top 50 signatures with counts per group
def norm(s_counter: Counter, sig, denom):
    return f"{s_counter.get(sig, 0):>6}  ({100*s_counter.get(sig,0)/max(denom,1):>4.1f}%)"

denom_pkb = sum(s_pkb["type_counter"].values()) or 1
denom_pdb = sum(s_pdb["type_counter"].values()) or 1
denom_neg = sum(s_neg["type_counter"].values()) or 1

print(f"\n{'#':<4}{'Path-type signature':<88}{'PK-B':<14}{'PD-B':<14}{'NEG':<14}")
print("-" * 132)
top_sigs = [sig for sig, _ in combined.most_common()]
for i, sig in enumerate(top_sigs):
    if i >= 60:
        break
    s = signature_to_str(sig)
    print(f"{i+1:<4}{s[:86]:<88}"
          f"{norm(s_pkb['type_counter'], sig, denom_pkb)}  "
          f"{norm(s_pdb['type_counter'], sig, denom_pdb)}  "
          f"{norm(s_neg['type_counter'], sig, denom_neg)}")

print(f"\nDenominators (total paths summed across all pairs):")
print(f"  PK-B: {denom_pkb:,}    PD-B: {denom_pdb:,}    NEG: {denom_neg:,}")

# Save per-pair feature table
import json
def per_pair_features(df_subset, summary, group_label):
    rows = []
    for _, r in df_subset.iterrows():
        a, b = r["drug_a_id"], r["drug_b_id"]
        n_paths = summary["per_pair_total"].get((a, b), 0)
        rows.append(dict(
            group=group_label,
            drug_a_id=a, drug_b_id=b,
            ddi_type=r.get("ddi_type", ""),
            pkpd_4way=r.get("pkpd_4way", ""),
            label=r.get("label", -1),
            n_paths=n_paths,
        ))
    return rows

per_pair = []
per_pair += per_pair_features(pkb, s_pkb, "PK-B")
per_pair += per_pair_features(pdb, s_pdb, "PD-B")
per_pair += per_pair_features(neg_s, s_neg, "NEG")
pd.DataFrame(per_pair).to_parquet(KG_OUT / "per_pair_path_counts.parquet", index=False)
print(f"\nSaved: per_pair_path_counts.parquet ({len(per_pair)} rows)")

# Save full path-type breakdown to JSON
breakdown = {
    "PK-B": [(signature_to_str(s), int(c)) for s, c in s_pkb["type_counter"].most_common()],
    "PD-B": [(signature_to_str(s), int(c)) for s, c in s_pdb["type_counter"].most_common()],
    "NEG":  [(signature_to_str(s), int(c)) for s, c in s_neg["type_counter"].most_common()],
}
with open(KG_OUT / "path_type_breakdown.json", "w", encoding="utf-8") as f:
    json.dump(breakdown, f, indent=2, ensure_ascii=False)
print(f"Saved: path_type_breakdown.json")

print(f"\nDone.")
