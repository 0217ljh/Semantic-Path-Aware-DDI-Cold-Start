"""Export path types in a CSV format optimized for manual labeling.

Output columns:
    rank              — sorted by total count desc
    length            — # hops
    node_kinds        — readable kind sequence (Drug → X → Y → ...)
    end_kind          — what the path ends at (semantic target)
    relations         — relation labels along the path (compact)
    count_total       — across all 200 pairs
    count_PKB         — count among PK-B pairs
    count_PDB         — count among PD-B pairs
    diff_PKB_vs_PDB   — count_PKB - count_PDB
    layer_top         — most frequent L3 layer that triggered this path
    example_drug      — sample drug
    example_term      — sample L3 entity term in DDInter text
    example_target    — concrete KG node name reached
    example_path      — full named path
    manual_label      — *** YOU FILL THIS ***
                        suggested values:
                          PK-useful / PD-useful / noise / ambiguous / structural
    notes             — *** YOU FILL THIS *** free-form
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter, defaultdict
_HERE = Path(__file__).resolve()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

OUT = _HERE.parent

# Load aggregated templates + per-pair records
templates = pd.read_parquet(OUT / "path_type_templates.parquet")
records = pd.read_parquet(OUT / "ddinter_anchored_paths.parquet")
print(f"Templates: {len(templates)}  records: {len(records)}")

# Build per-template breakdown by class and by layer
# Signature key = (node_kinds string, relations string) since tuples don't survive parquet
records["sig_kinds_str"] = records["sig_kinds"].apply(lambda t: " → ".join(t))
records["sig_rels_str"] = records["sig_rels"].apply(lambda t: " | ".join("/".join(r) for r in t))
records["sig_key"] = records.apply(lambda r: (r["sig_kinds_str"], r["sig_rels_str"]), axis=1)

# Compute counts per signature
pkb_counts = defaultdict(int)
pdb_counts = defaultdict(int)
layer_counts = defaultdict(Counter)
for _, r in records.iterrows():
    key = (r["sig_kinds_str"], r["sig_rels_str"])
    if r["class_label"] == "PK-B":
        pkb_counts[key] += 1
    elif r["class_label"] == "PD-B":
        pdb_counts[key] += 1
    layer_counts[key][r["layer"]] += 1

# Now enrich templates
rows = []
for idx, t in templates.iterrows():
    key = (t["node_kinds"], t["relations"])
    examples = json.loads(t["examples"]) if t["examples"] else []
    ex = examples[0] if examples else {}
    path_names = ex.get("path_names", [])
    example_path_str = " → ".join(path_names) if path_names else ""
    end_kind = t["node_kinds"].split(" → ")[-1] if t["node_kinds"] else ""

    layer_dist = layer_counts.get(key, Counter())
    top_layer = layer_dist.most_common(1)[0][0] if layer_dist else ""

    rows.append({
        "rank": idx + 1,
        "length": int(t["length"]),
        "node_kinds": t["node_kinds"],
        "end_kind": end_kind,
        "relations": t["relations"],
        "count_total": int(t["count"]),
        "count_PKB": pkb_counts.get(key, 0),
        "count_PDB": pdb_counts.get(key, 0),
        "diff_PKB_vs_PDB": pkb_counts.get(key, 0) - pdb_counts.get(key, 0),
        "layer_top": top_layer,
        "example_drug": ex.get("drug_name", ""),
        "example_term": ex.get("term", ""),
        "example_target": ex.get("kg_node_name", ""),
        "example_path": example_path_str,
        "manual_label": "",
        "notes": "",
    })

out_df = pd.DataFrame(rows).sort_values("count_total", ascending=False).reset_index(drop=True)
out_df["rank"] = range(1, len(out_df) + 1)

# Save as CSV (excel-friendly: utf-8-sig BOM so Excel renders Chinese/special chars correctly)
csv_path = OUT / "path_types_for_labeling.csv"
xlsx_path = OUT / "path_types_for_labeling.xlsx"
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
out_df.to_excel(xlsx_path, index=False)

print(f"\nSaved:")
print(f"  {csv_path}  ({len(out_df)} rows)")
print(f"  {xlsx_path}")
print(f"\nColumns:")
for c in out_df.columns:
    print(f"  - {c}")

print(f"\nFirst 10 rows preview:")
preview_cols = ["rank", "length", "node_kinds", "count_total", "count_PKB",
                "count_PDB", "diff_PKB_vs_PDB", "layer_top"]
print(out_df[preview_cols].head(10).to_string(index=False))

print(f"\nDistribution by length:")
print(out_df["length"].value_counts().sort_index().to_string())

print(f"\nDistribution by end_kind (top 15):")
print(out_df["end_kind"].value_counts().head(15).to_string())

print(f"\n=== Suggested labels for `manual_label` column ===")
print("  PK-useful  — useful for predicting PK-B mechanism")
print("  PD-useful  — useful for predicting PD-B mechanism")
print("  both       — useful for both")
print("  noise      — likely uninformative / artifact")
print("  ambiguous  — unsure, need to see examples")
print("  structural — captures graph-distance only, no mechanism content")
