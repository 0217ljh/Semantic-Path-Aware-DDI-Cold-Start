"""Show N PD cases: DDInter clinical mechanism text + LLM-extracted chain.

Read-only. Joins the LLM chain extraction (Code/runs/pd_mechanism/chains.jsonl)
with the DDInter clinical mechanism text (Code/data/_cache/pd_mechanism_200.csv)
and prints a few illustrative cases side by side.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/show_pd_mechanism_cases.py
"""
from __future__ import annotations

import json
import pathlib
import textwrap

import pandas as pd

JSONL = "Code/runs/pd_mechanism/chains.jsonl"
PD_CSV = "Code/data/_cache/pd_mechanism_200.csv"
WANT = ["potassium", "serotonin", "qt"]  # try to pick one case each


def main() -> None:
    text_by_key = {}
    src = pd.read_csv(PD_CSV)
    for _, r in src.iterrows():
        text_by_key[f"{r['drug_a_id']}|{r['drug_b_id']}"] = r["original_text"]

    recs = []
    for line in pathlib.Path(JSONL).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("parsed"):
                recs.append(r)

    chosen, used = [], set()
    for kw in WANT:
        for r in recs:
            p = r["parsed"]
            blob = (p.get("category", "") + p.get("convergence", "")).lower()
            if kw in blob and r["key"] not in used:
                chosen.append(r); used.add(r["key"]); break
    for r in recs:  # pad to 3
        if len(chosen) >= 3:
            break
        if r["key"] not in used:
            chosen.append(r); used.add(r["key"])

    for i, r in enumerate(chosen[:3], 1):
        p = r["parsed"]
        print("=" * 78)
        print(f"CASE {i}: {r['a']}  +  {r['b']}")
        print(f"  DDI effect : {r['effect']}")
        print(f"  KG shared-target chain present? {r['kg_chain']}")
        print("\n  --- DDInter clinical mechanism (clinician-written) ---")
        print(textwrap.fill(str(text_by_key.get(r["key"], "(text not found)")),
                            width=74, initial_indent="    ", subsequent_indent="    "))
        print("\n  --- LLM-extracted convergence chain ---")
        print(f"    A target     : {p.get('a_target')}  [{p.get('a_action')}]")
        print(f"    convergence  : {p.get('convergence')}")
        print(f"    B target     : {p.get('b_target')}  [{p.get('b_action')}]")
        print(f"    category     : {p.get('category')}")
        print(f"    molecular_grounded (both targets KG-mappable): {p.get('molecular_grounded')}")
        print(f"    chain        : {p.get('chain')}")
        print()


if __name__ == "__main__":
    main()
