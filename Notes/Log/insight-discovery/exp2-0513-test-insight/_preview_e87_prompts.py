"""Preview complete E8.7 prompts (Stage 1 selector + Stage 2 reasoner)
for a PD-pos, PK-pos, and NEG example."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import importlib
e8 = importlib.import_module("08_llm_oracle")
e8f = importlib.import_module("08f_llm_oracle_paircond_named")

import pandas as pd

# Match E8.7 sample sizes
e8.N_PD_POS = 60
e8.N_PK_POS = 60
e8.N_NEG = 60

nodes = pd.read_parquet(e8.NODES)
edges = pd.read_parquet(e8.EDGES)
drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
id2name = dict(zip(nodes["id"], nodes["name"].fillna("")))
mol_feats, eff_feats = e8.build_features(edges, nodes, drug_set)
pkpd_df = pd.read_csv(e8.PKPD)
pkpd_labels = dict(zip(pkpd_df["ddi_type"], pkpd_df["pk_pd_label"]))
pairs = e8.sample_pairs(pkpd_labels, mol_feats, eff_feats, id2name, seed=42)

samples = {}
for cls in ["PD", "PK", "NEG"]:
    for p in pairs:
        if p["mech_class"] == cls:
            samples[cls] = p
            break

print("=" * 80)
print("STAGE 1 SYSTEM PROMPT (identical for all pairs)")
print("=" * 80)
print(e8f.SELECTOR_SYSTEM)

for cls, p in samples.items():
    eff_a = eff_feats.get(p["drug_a_id"], [])[:30]
    eff_b = eff_feats.get(p["drug_b_id"], [])[:30]
    print()
    print("=" * 80)
    label = "POSITIVE" if p["mech_class"] != "NEG" else "NEGATIVE (no DDI)"
    print(f"STAGE 1 USER PROMPT - {p['mech_class']} {label}")
    print(f"Pair: {p['drug_a_name']} (drug A) + {p['drug_b_name']} (drug B)")
    if p.get("ddi_type"):
        print(f"Ground-truth ddi_type: {p['ddi_type'][:120]}")
    print("=" * 80)
    print(e8f.selector_prompt_named(p["drug_a_name"], p["drug_b_name"], eff_a, eff_b))

print()
print("=" * 80)
print("STAGE 2 SYSTEM PROMPT (identical for all pairs)")
print("=" * 80)
print(e8f.REASONER_SYSTEM)

print()
print("=" * 80)
print("STAGE 2 USER PROMPT (example: selector returned 1 composition pair)")
print("=" * 80)
example_pairs = [("Blood fibrinogen decreased", "Bleeding tendency")]
print(e8f.reasoner_prompt_named("Pentoxifylline", "Cilostazol",
                                 e8f.format_pairs_for_reasoner(example_pairs)))

print()
print("=" * 80)
print("STAGE 2 USER PROMPT (example: selector returned NONE)")
print("=" * 80)
print(e8f.reasoner_prompt_named("Galantamine", "Lincomycin",
                                 e8f.format_pairs_for_reasoner([])))
