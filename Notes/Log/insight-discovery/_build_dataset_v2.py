"""Extend the eval set: keep 100 PK-B + 100 PD-B, add 400 negative pairs.

Negatives = random pairs (a, b) from the 1900-drug set, where the canonical
(unordered) pair is NOT in ddi_edges. This gives a clean "no documented DDI"
contrast set for B-vs-negative analysis.
"""
import pandas as pd, json, sys, random
from itertools import combinations
sys.stdout.reconfigure(encoding='utf-8')

KG = r"D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Code/data/KG"
OUT = r"D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes/Log/insight-discovery"

# Existing 200-pair set
pos200 = pd.read_parquet(f"{OUT}/eval_200_PKB_PDB.parquet")
print(f"Loaded 200 positives: {pos200['pkpd_4way'].value_counts().to_dict()}")

# All known DDI pairs (canonical, sorted)
ddi_edges = pd.read_csv(f"{KG}/drugbank/filtered/ddi_edges.csv")
ddi_pairs = set()
for a, b in zip(ddi_edges['drug_a_id'], ddi_edges['drug_b_id']):
    ddi_pairs.add(tuple(sorted([a, b])))
print(f"All DDI pairs in DrugBank-1900: {len(ddi_pairs):,}")

# Drug pool
drugs_df = pd.read_csv(f"{KG}/drugbank/filtered/drugs.csv")
drug_ids = drugs_df['drugbank_id'].tolist()
id2name = json.load(open(f"{KG}/drugbank/filtered/id2name.json"))
print(f"Drug pool: {len(drug_ids)}")

# Sample 400 negative pairs (canonical, not in DDI set, no self-pair)
random.seed(42)
neg_pairs = set()
TARGET = 400
n_attempts = 0
while len(neg_pairs) < TARGET and n_attempts < 100000:
    a, b = random.sample(drug_ids, 2)
    key = tuple(sorted([a, b]))
    if key not in ddi_pairs and key not in neg_pairs:
        neg_pairs.add(key)
    n_attempts += 1
print(f"Sampled {len(neg_pairs)} negative pairs after {n_attempts} attempts")

# Convert negatives to rows matching the schema
neg_rows = []
for i, (a, b) in enumerate(sorted(neg_pairs)):
    neg_rows.append({
        'pair_id': f"NEG-{i+1:03d}",
        'pkpd_4way': 'NEG',
        'drug_a_id': a,
        'drug_a_name': id2name.get(a, ''),
        'drug_b_id': b,
        'drug_b_name': id2name.get(b, ''),
        'ddi_type': '',
        'description': '',
        'ddinter_mechanism': '',
    })
neg_df = pd.DataFrame(neg_rows)

# Combine with positives
full = pd.concat([pos200, neg_df], ignore_index=True)
print(f"\nCombined dataset: {len(full)} rows")
print(full['pkpd_4way'].value_counts())

# Save parquet + Excel (xlsx)
out_parquet = f"{OUT}/eval_600_PKB_PDB_NEG.parquet"
out_xlsx    = f"{OUT}/eval_600_PKB_PDB_NEG.xlsx"
full.to_parquet(out_parquet, index=False)

# Excel writer — wrap long text columns
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
    # Sheet 1: all 600
    full.to_excel(w, sheet_name='all_600', index=False)
    # Sheet 2: PK-B only
    full[full['pkpd_4way']=='PK-B'].to_excel(w, sheet_name='PK-B (100)', index=False)
    # Sheet 3: PD-B only
    full[full['pkpd_4way']=='PD-B'].to_excel(w, sheet_name='PD-B (100)', index=False)
    # Sheet 4: NEG only
    full[full['pkpd_4way']=='NEG'].to_excel(w, sheet_name='NEG (400)', index=False)

print(f"\nSaved: {out_parquet}")
print(f"Saved: {out_xlsx}  (4 sheets: all_600 / PK-B / PD-B / NEG)")

# === Sanity check on negatives ===
print(f"\n=== Negative pool sanity ===")
neg_drug_a = set(neg_df['drug_a_id'])
neg_drug_b = set(neg_df['drug_b_id'])
neg_all = neg_drug_a | neg_drug_b
print(f"  Unique drugs in negatives: {len(neg_all)} / 1900")
print(f"  All negative pairs canonical (no duplicates): {len(set(tuple(sorted([r['drug_a_id'], r['drug_b_id']])) for _, r in neg_df.iterrows())) == 400}")
print(f"  All negatives NOT in DDI: {all(tuple(sorted([r['drug_a_id'], r['drug_b_id']])) not in ddi_pairs for _, r in neg_df.iterrows())}")
