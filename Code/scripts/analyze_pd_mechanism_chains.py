"""Do PD DDIs have a reliable, KG-anchorable mechanism activation chain?

Read-only. Uses the pre-computed DrugBank-derived key-entity extraction
(`ddi_key_entities.csv`), which gives, per DDI pair, the shared mechanism
entity (target/enzyme/transporter), each drug's action on it, and an explicit
`mechanism_chain` string + `chain_type`. Reports, for PD pairs: the fraction
with an identifiable chain, the chain-type / confidence / action-pattern /
entity-type distributions, the top mechanism entities, and concrete examples.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_mechanism_chains.py
"""
from __future__ import annotations

import pandas as pd

KE = "Code/data/KG/drugbank/enriched/ddi_key_entities.csv"


def main() -> None:
    df = pd.read_csv(KE)
    print("total rows:", len(df), "| cols:", list(df.columns))
    print("pk_pd counts:\n", df["pk_pd_label"].value_counts(dropna=False))

    pd_df = df[df["pk_pd_label"] == "PD"].copy()
    n = len(pd_df)
    has = pd_df["has_key_entity"].astype(str).str.lower().eq("true")
    print(f"\n=== PD pairs: {n} | with mechanism chain: {has.sum()} "
          f"({has.mean()*100:.1f}%) ===")

    chain = pd_df[has]
    print("\n--- chain_type (PD with chain) ---")
    print(chain["chain_type"].value_counts().head(12).to_string())
    print("\n--- confidence (PD with chain) ---")
    print(chain["confidence"].value_counts(dropna=False).to_string())
    print("\n--- key_entity_type (PD with chain) ---")
    print(chain["key_entity_type"].value_counts(dropna=False).to_string())
    print("\n--- action pattern (drug_a action -- drug_b action) ---")
    ap = (chain["action_drug_a"].astype(str) + " / " + chain["action_drug_b"].astype(str))
    print(ap.value_counts().head(12).to_string())
    print("\n--- top key entities for PD chains ---")
    print(chain["key_entity_name"].value_counts().head(15).to_string())

    # 200-sample view (honor the explicit request) + concrete chains
    samp = pd_df.sample(n=min(200, n), random_state=42)
    sh = samp["has_key_entity"].astype(str).str.lower().eq("true")
    print(f"\n=== random 200 PD sample: {sh.sum()}/200 have a chain "
          f"({sh.mean()*100:.0f}%) ===")
    print("\n--- 12 concrete PD mechanism chains (high confidence) ---")
    hi = chain[chain["confidence"] == "high"]
    for _, r in hi.head(12).iterrows():
        print(f"  [{r['chain_type']}] {str(r['mechanism_chain'])[:110]}")

    # what do the 93% WITHOUT a chain look like? top ddi_types
    no = pd_df[~has]
    print(f"\n=== PD WITHOUT chain ({len(no)}): top ddi_types ===")
    print(no["ddi_type"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
