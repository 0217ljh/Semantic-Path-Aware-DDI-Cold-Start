"""Surface the RELIABLE mechanism source for PD DDIs (DDInter clinical text).

Read-only, mechanism-FIRST (not KG-first). Pulls 200 PD pairs that carry a
DDInter clinician-written mechanism description, prints concrete examples and
the ddi_type distribution, and writes the 200-sample working set to disk for a
follow-up per-sample structured-mechanism extraction (LLM stage).

The point: assess whether PD DDIs HAVE a real mechanism (from authoritative
clinical text), independent of whether the current KG happens to encode it.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_mechanism_source.py
"""
from __future__ import annotations

import pandas as pd

CORPUS = "Notes/Log/insight-discovery/exp1-0512/filtered_ddinter_corpus.csv"
PKPD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
KE = "Code/data/KG/drugbank/enriched/ddi_key_entities.csv"
OUT = "Code/data/_cache/pd_mechanism_200.csv"


def main() -> None:
    corp = pd.read_csv(CORPUS)
    print("corpus rows:", len(corp), "| cols:", list(corp.columns))
    pk = pd.read_csv(PKPD)[["ddi_type", "pk_pd_label"]]
    corp = corp.merge(pk, on="ddi_type", how="left")

    pd_corp = corp[(corp["pk_pd_label"] == "PD")
                   & corp["original_text"].astype(str).str.len().gt(20)].copy()
    print(f"\nPD pairs with DDInter mechanism text: {len(pd_corp)}")

    # bring in DrugBank shared-target chain flag (for cross-reference only)
    ke = pd.read_csv(KE, usecols=["drug_a_id", "drug_b_id", "has_key_entity",
                                  "key_entity_name", "mechanism_chain"])
    pd_corp = pd_corp.merge(ke, on=["drug_a_id", "drug_b_id"], how="left")

    samp = pd_corp.sample(n=min(200, len(pd_corp)), random_state=42).reset_index(drop=True)
    has_kg = samp["has_key_entity"].astype(str).str.lower().eq("true")
    print(f"sampled {len(samp)} PD | with DrugBank shared-target chain: "
          f"{has_kg.sum()} ({has_kg.mean()*100:.0f}%) | "
          f"WITHOUT KG chain but WITH clinical mechanism: {(~has_kg).sum()}")

    print("\n=== ddi_type (effect) distribution of the 200 PD ===")
    print(samp["ddi_type"].value_counts().head(15).to_string())

    print("\n=== 18 concrete PD mechanisms from DDInter (reliable source) ===")
    show = samp.sample(n=18, random_state=1)
    for _, r in show.iterrows():
        kg = "KGchain" if str(r["has_key_entity"]).lower() == "true" else "noKGchain"
        print(f"\n[{kg}] {r['drug_a_name']} + {r['drug_b_name']}  ({r['ddi_type'][:55]})")
        print(f"   MECH: {str(r['original_text'])[:300]}")

    cols = ["drug_a_id", "drug_b_id", "drug_a_name", "drug_b_name", "ddi_type",
            "pk_pd_label", "original_text", "has_key_entity", "key_entity_name",
            "mechanism_chain"]
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    samp[cols].to_csv(OUT, index=False)
    print(f"\nsaved 200-PD working set -> {OUT}")


if __name__ == "__main__":
    main()
