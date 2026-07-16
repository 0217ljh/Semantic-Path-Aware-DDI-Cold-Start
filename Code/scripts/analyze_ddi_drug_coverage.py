"""DDI-benchmark <-> KG-substrate drug coverage report (read-only QA, OUTSIDE the KG).

Codex (thread 019f19d3): DDI-pool membership is NOT baked into the KG (it spans drug
kinds and varies by dataset). This report surfaces the benchmark-to-substrate coverage
gap per dataset: which DDI drugs are present in / missing from the unified v12 KG
(kg_id = 'drug:' + drugbank_id). It does NOT modify the KG.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_ddi_drug_coverage.py
"""
from __future__ import annotations

import glob
import os

import pandas as pd

V12 = "Code/data/KG/_merged_kg_dedup_v12"
DDI_ROOT = "Code/data/ddi_unified"


def main() -> None:
    nd = pd.read_parquet(f"{V12}/nodes__dedup.parquet")
    kg_drugs = set(nd[nd["kind"].astype(str) == "drug"]["id"].astype(str))   # 'drug:DBxxxxx'
    print(f"v12 KG drug nodes: {len(kg_drugs)}")
    print("=" * 64)
    print("per-dataset DDI drug coverage (kg_id = 'drug:'+drugbank_id)")
    print("=" * 64)

    any_missing = set()
    for drugs_file in sorted(glob.glob(f"{DDI_ROOT}/*/*/inductive/S1/drugs.parquet")):
        ds = drugs_file.split(os.sep)
        task, dataset = ds[-5], ds[-4]
        df = pd.read_parquet(drugs_file)
        raw = df["drugbank_id"]
        n_null = int(raw.isna().sum())                      # null/unmapped id = dataset hygiene, NOT a KG gap
        ids = set(raw.dropna().astype(str)) - {"<NA>", "nan", ""}
        kg_ids = {f"drug:{x}" for x in ids}
        present = kg_ids & kg_drugs
        missing = ids - {x.split(":", 1)[1] for x in present}
        any_missing |= {(task, dataset, m) for m in missing}
        print(f"  {task}/{dataset}: {len(ids)} mapped drugs | present_in_kg {len(present)} | "
              f"MISSING {len(missing)}" + (f" {sorted(missing)[:8]}" if missing else "")
              + (f" | null-id rows (dataset artifact, not KG gap): {n_null}" if n_null else ""))

    print("\n" + "=" * 64)
    print(f"TOTAL distinct (dataset, missing-drug) gaps: {len(any_missing)}")
    miss_drugs = sorted(set(m for _, _, m in any_missing))
    print(f"distinct DrugBank ids in some DDI task but ABSENT from KG substrate: {len(miss_drugs)}")
    print(f"  {miss_drugs}")
    print("\nPolicy: v12 normalization covers the 8048 KG drugs only. These missing drugs")
    print("are a benchmark->substrate coverage gap; loader/eval must decide policy")
    print("(skip pair / zero KG-neighborhood / external stub). NOT auto-folded into the KG.")


if __name__ == "__main__":
    main()
