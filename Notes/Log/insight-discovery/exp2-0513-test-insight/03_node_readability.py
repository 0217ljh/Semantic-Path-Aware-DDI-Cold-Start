"""E1c — Node-name readability statistics on the merged KG.

Feasibility check for i4 (node-name biomedical text semantics as cold-start
prior). For each node, classify the `name` field as one of:
  - "readable"  : contains word-like content (≥ one lowercase token of len ≥3)
                  AND not a pure-ID pattern
  - "id_only"   : empty/null OR matches a typed-ID / numeric-ID pattern

Output (saved next to this script):
  - node_readability.csv       : per-(kind × source_kg) tabulation
  - node_readability_sample.csv: 20 random readable + 20 random id_only samples
                                 per kind, for spot-check
  - node_readability.json      : summary stats (overall % readable per kind)

Run:
  PYTHONPATH=Code/my_code python Notes/Log/insight-discovery/exp2-0513-test-insight/03_node_readability.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Locate project root robustly
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")


PROJECT_ROOT = _find_project_root()
KG_NODES = (
    PROJECT_ROOT
    / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
)
OUT_DIR = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------
ID_PATTERNS = [
    r"^[A-Z]+:[\w\-.:]+$",        # typed IDs: GO:0006915, UBERON:0000178, CHEBI:1234
    r"^DB\d{4,}$",                # DrugBank IDs
    r"^[A-Z]?\d+$",               # numeric IDs (possibly prefixed with one letter)
    r"^[A-Z]\d{4,}[A-Z]?\d*$",    # MESH-like: D012345, C0123456
    r"^[A-Z]{2,5}\d{4,}$",        # HGNC:1234 stripped, etc.
    r"^[a-fA-F0-9]{8,}$",         # UUIDs / hex hashes
    r"^[\d.\-]+$",                # purely numeric/punct
]
ID_REGEX = re.compile("|".join(ID_PATTERNS))

# Per codex round-1 sanity: short biomedical gene symbols like IL6, TP53,
# NF-kB, 5-HT2A should be READABLE since PubMedBERT meaningfully encodes them.
# We only treat names as id_only if they match a pure typed-ID pattern OR
# contain no alphabetic character at all.
ALPHA_TOKEN = re.compile(r"[A-Za-z]+")


def classify_name(name) -> str:
    if name is None:
        return "id_only"
    s = str(name).strip()
    if not s:
        return "id_only"
    # If the whole string matches a typed-ID / numeric pattern → id_only
    if ID_REGEX.fullmatch(s):
        return "id_only"
    # Otherwise: must contain at least one alpha character
    if not ALPHA_TOKEN.search(s):
        return "id_only"
    return "readable"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[E1c] Loading nodes from {KG_NODES.relative_to(PROJECT_ROOT)}")
    nodes = pd.read_parquet(KG_NODES)
    print(f"[E1c] nodes shape: {nodes.shape}")
    print(f"[E1c] columns: {list(nodes.columns)}")

    # If name column missing, try common alternatives
    if "name" not in nodes.columns:
        for alt in ("node_name", "label", "display_name"):
            if alt in nodes.columns:
                nodes = nodes.rename(columns={alt: "name"})
                break
        else:
            print(f"[E1c] WARNING: no 'name' column found; treating all as id_only")
            nodes["name"] = None

    nodes["readable"] = nodes["name"].apply(classify_name)

    # Per (kind × source_kg) tabulation
    has_src = "source_kg" in nodes.columns
    group_cols = ["kind"] + (["source_kg"] if has_src else [])
    tab = (
        nodes.groupby(group_cols + ["readable"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "readable" not in tab.columns:
        tab["readable"] = 0
    if "id_only" not in tab.columns:
        tab["id_only"] = 0
    tab["total"] = tab["readable"] + tab["id_only"]
    tab["readable_pct"] = (tab["readable"] / tab["total"] * 100).round(2)
    tab["id_only_pct"] = (tab["id_only"] / tab["total"] * 100).round(2)
    tab = tab.sort_values("total", ascending=False)

    out_csv = OUT_DIR / "node_readability.csv"
    tab.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[E1c] saved per-kind table → {out_csv.name}")
    print(tab.to_string(index=False))

    # Summary by kind only
    summary_by_kind = (
        nodes.groupby("kind")["readable"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    if "readable" in summary_by_kind.columns:
        summary_by_kind["readable_pct"] = (summary_by_kind["readable"] * 100).round(2)
    if "id_only" in summary_by_kind.columns:
        summary_by_kind["id_only_pct"] = (summary_by_kind["id_only"] * 100).round(2)

    # JSON summary
    overall_readable = (nodes["readable"] == "readable").mean() * 100
    summary = {
        "experiment": "E1c node-name readability",
        "merged_kg": str(KG_NODES.relative_to(PROJECT_ROOT)),
        "n_nodes_total": int(len(nodes)),
        "overall_readable_pct": round(float(overall_readable), 2),
        "per_kind": {
            str(k): {
                "n": int((nodes["kind"] == k).sum()),
                "readable_pct": round(
                    float((nodes.loc[nodes["kind"] == k, "readable"] == "readable").mean() * 100),
                    2,
                ),
            }
            for k in nodes["kind"].dropna().unique()
        },
        "kinds_meeting_i4_threshold": [
            str(k)
            for k in nodes["kind"].dropna().unique()
            if (nodes.loc[nodes["kind"] == k, "readable"] == "readable").mean() >= 0.80
        ],
        "kinds_failing_i4_threshold": [
            str(k)
            for k in nodes["kind"].dropna().unique()
            if (nodes.loc[nodes["kind"] == k, "readable"] == "readable").mean() < 0.80
        ],
        "biomedical_relevant_kinds": [
            "Drug", "Side Effect", "Gene", "Protein", "Disease",
            "Anatomy", "Pathway", "drug_effect", "Phenotype", "Compound",
        ],
    }
    out_json = OUT_DIR / "node_readability.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[E1c] saved summary → {out_json.name}")
    print(f"  overall readable: {overall_readable:.2f}%")
    print(f"  kinds ≥80% readable: {summary['kinds_meeting_i4_threshold']}")
    print(f"  kinds <80% readable: {summary['kinds_failing_i4_threshold']}")

    # Spot-check samples
    spot_rows = []
    for k in nodes["kind"].dropna().unique():
        sub = nodes[nodes["kind"] == k]
        for cls in ("readable", "id_only"):
            cls_sub = sub[sub["readable"] == cls]
            if len(cls_sub) == 0:
                continue
            n_sample = min(10, len(cls_sub))
            sample = cls_sub.sample(n_sample, random_state=42)
            for _, r in sample.iterrows():
                spot_rows.append(
                    {
                        "kind": k,
                        "class": cls,
                        "id": r["id"],
                        "name": r["name"],
                        "source_kg": r.get("source_kg", ""),
                    }
                )
    spot_df = pd.DataFrame(spot_rows)
    out_spot = OUT_DIR / "node_readability_sample.csv"
    spot_df.to_csv(out_spot, index=False, encoding="utf-8-sig")
    print(f"[E1c] saved spot-check samples → {out_spot.name}")

    # i4 verdict
    bio_kinds = set(summary["biomedical_relevant_kinds"])
    bio_present = [k for k in summary["per_kind"] if k in bio_kinds]
    bio_pass = [
        k for k in bio_present
        if summary["per_kind"][k]["readable_pct"] >= 80
    ]
    print(f"\n[E1c i4 verdict]")
    print(f"  Biomedical-relevant kinds present in merged KG: {bio_present}")
    print(f"  Of those, ≥80% readable: {bio_pass}")
    if len(bio_pass) >= len(bio_present) * 0.7:
        print(f"  → i4 feasibility CONFIRMED ({len(bio_pass)}/{len(bio_present)} of biomedical kinds pass)")
    else:
        print(f"  → i4 feasibility WEAK ({len(bio_pass)}/{len(bio_present)} of biomedical kinds pass)")


if __name__ == "__main__":
    main()
