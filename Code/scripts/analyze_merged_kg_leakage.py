"""Leakage audit of the merged KG for cold-start S2 DDI.

Checks whether the eval KG (base merged KG, drug-incident) contains a DIRECT
edge between the two drugs of a test/val pair — which would let NBFNet read the
answer in 1 hop. Reports the rate for POSITIVES vs NEGATIVES and the relation
breakdown, separately for drug-drug edges. Read-only; no training.

Run:
  python Code/scripts/analyze_merged_kg_leakage.py
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.emergnn.kg_builder_merged import build_kg_from_merged_parquet
from data_utils import PairDataset

MERGED = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"

ds = PairDataset.from_pkl(str(ROOT / "data/coldddi_legacy/800drug/seed42.pkl"))
kg = ds.kg
drug_ids = set()
for _n, df in ds.splits.items():
    drug_ids.update(df["drug_a_id"].astype(str)); drug_ids.update(df["drug_b_id"].astype(str))
if hasattr(kg, "drug_ids"):
    drug_ids.update(kg.drug_ids)
art = build_kg_from_merged_parquet(MERGED, sorted(drug_ids), verbose=False)
e2i = art["entity2id"]; trip = np.asarray(art["triplets"], dtype=np.int64)
n_drugs = len(art["drug_ids"]); id2rel = {v: k for k, v in art["rel2id"].items()}
print(f"n_drugs={n_drugs} n_ent={art['n_ent']} n_rel={art['n_rel']} n_edges={len(trip)}")

# Direct edge lookup between any two nodes: {(min,max): set(rel)}
pair_rel = defaultdict(set)
drugdrug_pair_rel = defaultdict(set)
for h, t, r in trip:
    key = (int(min(h, t)), int(max(h, t)))
    pair_rel[key].add(int(r))
    if h < n_drugs and t < n_drugs:           # both endpoints are drugs
        drugdrug_pair_rel[key].add(int(r))
print(f"distinct drug-drug pairs with a direct edge: {len(drugdrug_pair_rel)}")
dd_rel_counts = defaultdict(int)
for s in drugdrug_pair_rel.values():
    for r in s:
        dd_rel_counts[r] += 1
print("drug-drug relation breakdown:")
for r, c in sorted(dd_rel_counts.items(), key=lambda x: -x[1]):
    print(f"  rel {r} ({id2rel.get(r,'?')}): {c} drug-drug pairs")

def audit(split):
    pos = getattr(ds.splits, split)[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives(split)[["drug_a_id", "drug_b_id"]]
    def rates(dfp):
        n = 0; direct_any = 0; direct_dd = 0; rels = defaultdict(int)
        for a, b in zip(dfp["drug_a_id"].astype(str), dfp["drug_b_id"].astype(str)):
            if a not in e2i or b not in e2i:
                continue
            n += 1
            key = (min(e2i[a], e2i[b]), max(e2i[a], e2i[b]))
            if key in pair_rel:
                direct_any += 1
            if key in drugdrug_pair_rel:
                direct_dd += 1
                for r in drugdrug_pair_rel[key]:
                    rels[r] += 1
        return n, direct_any, direct_dd, rels
    for label, dfp in [("POS", pos), ("NEG", neg)]:
        n, da, dd, rels = rates(dfp)
        rel_str = ", ".join(f"{id2rel.get(r,'?')}={c}" for r, c in sorted(rels.items(), key=lambda x: -x[1]))
        print(f"[{split} {label}] n={n}  direct_any_edge={da} ({100*da/max(n,1):.1f}%)  "
              f"direct_drug-drug={dd} ({100*dd/max(n,1):.1f}%)  rels: {rel_str}")

for split in ["test_s2", "val_s2", "test_s0"]:
    audit(split)
