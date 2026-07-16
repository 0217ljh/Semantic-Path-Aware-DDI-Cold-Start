"""E1b — Path endpoint layer by PK/PD class on merged KG.

Supports i1 (PK and PD are two structurally different reasoning paradigms).
For each positive DDI pair with a known PK or PD label, find the shared 1-hop
intermediate nodes between the two drugs on the merged KG, classify each
intermediate's KG `kind` into one of:
    - "molecular"     : Gene, gene/protein, Protein, Pathway, pathway,
                        Molecular Function, molecular_function, Compound,
                        Pharmacologic Class
    - "effect_system" : Side Effect, effect/phenotype, drug_effect, Symptom,
                        Disease, disease, Anatomy, anatomy, Phenotype
    - "other"         : Biological Process, biological_process,
                        Cellular Component, cellular_component, exposure

Test the hypothesis: PK pairs concentrate intermediates in "molecular"; PD pairs
concentrate in "effect_system". Chi-square is used as omnibus only; effect sizes
(odds ratios) are the primary claim.

Side outputs:
  - audit_sample.csv  : 50 random ddi_types with their PK/PD label for hand-verify

Run via:
  /home/lakestar_ljh/miniconda3/envs/project_1/bin/python \\
      Notes/Log/insight-discovery/exp2-0513-test-insight/02_path_endpoint_layer.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")


PROJECT_ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
PKPD = PROJECT_ROOT / "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
SPLITS = PROJECT_ROOT / "Code/data/KG/drugbank/splits/seed42"

# ---------------------------------------------------------------------------
# Layer mapping (committed in script, per plan)
# ---------------------------------------------------------------------------
LAYER_MAP = {
    # molecular
    "Gene": "molecular",
    "gene/protein": "molecular",
    "Protein": "molecular",
    "Pathway": "molecular",
    "pathway": "molecular",
    "Molecular Function": "molecular",
    "molecular_function": "molecular",
    "Compound": "molecular",
    "Pharmacologic Class": "molecular",
    # effect / system
    "Side Effect": "effect_system",
    "effect/phenotype": "effect_system",
    "drug_effect": "effect_system",
    "Symptom": "effect_system",
    "Disease": "effect_system",
    "disease": "effect_system",
    "Anatomy": "effect_system",
    "anatomy": "effect_system",
    "Phenotype": "effect_system",
    # other / ambiguous
    "Biological Process": "other",
    "biological_process": "other",
    "Cellular Component": "other",
    "cellular_component": "other",
    "exposure": "other",
}


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Load
    # -----------------------------------------------------------------------
    print("[E1b] Loading nodes / edges / PK-PD labels / splits ...")
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    pkpd = pd.read_csv(PKPD)
    train = pd.read_parquet(SPLITS / "train.parquet")[["drug_a_id", "drug_b_id", "ddi_type"]]
    test_s2 = pd.read_parquet(SPLITS / "test_s2.parquet")[["drug_a_id", "drug_b_id", "ddi_type"]]
    print(f"[E1b] nodes={len(nodes)}  edges={len(edges)}  pkpd={len(pkpd)}")
    print(f"[E1b] train={len(train)}  test_s2={len(test_s2)}")

    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    print(f"[E1b] drug nodes: {len(drug_set)}")

    # -----------------------------------------------------------------------
    # 2. Build drug-1hop adjacency (drug_id → set of (neighbor_id, neighbor_kind))
    # -----------------------------------------------------------------------
    print("[E1b] Building drug 1-hop adjacency ...")
    adj: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        src_is_drug = src in drug_set
        dst_is_drug = dst in drug_set
        # treat symmetric edges both ways; treat directed src→dst only one way
        if src_is_drug and not dst_is_drug:
            adj[src].add((dst, id2kind.get(dst, "Unknown")))
        if dst_is_drug and not src_is_drug:
            if not directed:
                adj[dst].add((src, id2kind.get(src, "Unknown")))
        # drug-drug edges: not used as 1-hop intermediate
    n_drugs_with_neighbors = sum(1 for d in drug_set if d in adj)
    print(f"[E1b] drugs with ≥1 non-drug neighbor: {n_drugs_with_neighbors}")

    # -----------------------------------------------------------------------
    # 3. Join positives with PK/PD label
    # -----------------------------------------------------------------------
    label_map = dict(zip(pkpd["ddi_type"], pkpd["pk_pd_label"]))
    print(f"[E1b] PK/PD label counts: {pkpd['pk_pd_label'].value_counts().to_dict()}")

    pairs = pd.concat([train, test_s2], ignore_index=True)
    pairs["pk_pd_label"] = pairs["ddi_type"].map(label_map)
    labeled = pairs[pairs["pk_pd_label"].isin(["PK", "PD"])].copy()
    print(f"[E1b] total positives: {len(pairs)}; PK/PD-labeled: {len(labeled)}")
    print(f"  PK-labeled: {(labeled['pk_pd_label']=='PK').sum()}")
    print(f"  PD-labeled: {(labeled['pk_pd_label']=='PD').sum()}")

    # -----------------------------------------------------------------------
    # 4. For each labeled pair, count shared 1-hop intermediates by layer
    # -----------------------------------------------------------------------
    print("[E1b] Counting shared 1-hop intermediates by layer ...")
    # accumulate (n_pairs_with_layer_X) and (total_intermediate_count_for_layer_X)
    counts = {
        "PK": {"molecular": 0, "effect_system": 0, "other": 0, "n_pairs": 0, "n_pairs_with_any": 0},
        "PD": {"molecular": 0, "effect_system": 0, "other": 0, "n_pairs": 0, "n_pairs_with_any": 0},
    }
    # also per-pair occurrence (binary: pair has ≥1 intermediate of layer X)
    per_pair_occ = {
        "PK": {"molecular": 0, "effect_system": 0, "other": 0},
        "PD": {"molecular": 0, "effect_system": 0, "other": 0},
    }

    for da, db, lab in zip(labeled["drug_a_id"], labeled["drug_b_id"], labeled["pk_pd_label"]):
        counts[lab]["n_pairs"] += 1
        ns_a = adj.get(da, set())
        ns_b = adj.get(db, set())
        shared = ns_a & ns_b
        if not shared:
            continue
        counts[lab]["n_pairs_with_any"] += 1
        layers_seen = set()
        for _, kind in shared:
            layer = LAYER_MAP.get(kind, "other")
            counts[lab][layer] += 1
            layers_seen.add(layer)
        for layer in layers_seen:
            per_pair_occ[lab][layer] += 1

    print(f"[E1b] PK pairs: {counts['PK']['n_pairs']}  with ≥1 shared neighbor: {counts['PK']['n_pairs_with_any']}")
    print(f"[E1b] PD pairs: {counts['PD']['n_pairs']}  with ≥1 shared neighbor: {counts['PD']['n_pairs_with_any']}")

    # -----------------------------------------------------------------------
    # 5. Print raw counts FIRST (so we always see them even if chi2 fails)
    # -----------------------------------------------------------------------
    print(f"\n[E1b] Total intermediate counts (PK/PD × molecular/effect_system/other):")
    print(f"  PK: mol={counts['PK']['molecular']}  eff={counts['PK']['effect_system']}  other={counts['PK']['other']}")
    print(f"  PD: mol={counts['PD']['molecular']}  eff={counts['PD']['effect_system']}  other={counts['PD']['other']}")
    print(f"\n[E1b] Per-pair occurrence (PK/PD × layer):")
    print(f"  PK: mol={per_pair_occ['PK']['molecular']}  eff={per_pair_occ['PK']['effect_system']}  other={per_pair_occ['PK']['other']}")
    print(f"  PD: mol={per_pair_occ['PD']['molecular']}  eff={per_pair_occ['PD']['effect_system']}  other={per_pair_occ['PD']['other']}")

    # Chi-square — use 2×2 (mol vs effect_system only) to avoid zero "other" cells
    def _safe_chi2(t: np.ndarray) -> tuple[float, float, int]:
        # Drop columns that are zero across both rows
        nonzero = (t.sum(axis=0) > 0)
        t2 = t[:, nonzero]
        if t2.shape[1] < 2:
            return float("nan"), float("nan"), 0
        chi2, p, dof, _ = chi2_contingency(t2)
        return float(chi2), float(p), int(dof)

    table_total = np.array(
        [
            [counts["PK"]["molecular"], counts["PK"]["effect_system"]],
            [counts["PD"]["molecular"], counts["PD"]["effect_system"]],
        ]
    )
    chi2, p, dof = _safe_chi2(table_total)
    print(f"\n[E1b] Chi-square on 2×2 (PK/PD × mol/eff) intermediate counts:")
    print(f"  chi2={chi2:.2f}  dof={dof}  p={p:.3e}")

    table_pair = np.array(
        [
            [per_pair_occ["PK"]["molecular"], per_pair_occ["PK"]["effect_system"]],
            [per_pair_occ["PD"]["molecular"], per_pair_occ["PD"]["effect_system"]],
        ]
    )
    chi2_p, p_p, dof_p = _safe_chi2(table_pair)
    print(f"\n[E1b] Chi-square on 2×2 (PK/PD × mol/eff) per-pair counts:")
    print(f"  chi2={chi2_p:.2f}  dof={dof_p}  p={p_p:.3e}")

    # Effect size: relative concentration of each layer per class
    def pct(a: int, total: int) -> float:
        return 100.0 * a / total if total > 0 else 0.0

    pk_mol_pct = pct(per_pair_occ["PK"]["molecular"], counts["PK"]["n_pairs_with_any"])
    pk_eff_pct = pct(per_pair_occ["PK"]["effect_system"], counts["PK"]["n_pairs_with_any"])
    pd_mol_pct = pct(per_pair_occ["PD"]["molecular"], counts["PD"]["n_pairs_with_any"])
    pd_eff_pct = pct(per_pair_occ["PD"]["effect_system"], counts["PD"]["n_pairs_with_any"])

    print(f"\n[E1b] Per-pair % with ≥1 intermediate of layer:")
    print(f"  PK molecular: {pk_mol_pct:.2f}%   PK effect_system: {pk_eff_pct:.2f}%")
    print(f"  PD molecular: {pd_mol_pct:.2f}%   PD effect_system: {pd_eff_pct:.2f}%")
    print(f"  PK mol/eff ratio: {pk_mol_pct / pk_eff_pct if pk_eff_pct > 0 else float('inf'):.2f}")
    print(f"  PD mol/eff ratio: {pd_mol_pct / pd_eff_pct if pd_eff_pct > 0 else float('inf'):.2f}")

    # -----------------------------------------------------------------------
    # 6. Save outputs
    # -----------------------------------------------------------------------
    summary = {
        "experiment": "E1b path endpoint layer by PK/PD",
        "n_pairs_total": int(len(pairs)),
        "n_pk_labeled": int(counts["PK"]["n_pairs"]),
        "n_pd_labeled": int(counts["PD"]["n_pairs"]),
        "n_pk_with_shared_neighbor": int(counts["PK"]["n_pairs_with_any"]),
        "n_pd_with_shared_neighbor": int(counts["PD"]["n_pairs_with_any"]),
        "intermediate_count": {
            "PK": {k: counts["PK"][k] for k in ["molecular", "effect_system", "other"]},
            "PD": {k: counts["PD"][k] for k in ["molecular", "effect_system", "other"]},
        },
        "per_pair_occurrence": {
            "PK": per_pair_occ["PK"],
            "PD": per_pair_occ["PD"],
        },
        "per_pair_pct": {
            "PK_molecular_pct": round(pk_mol_pct, 2),
            "PK_effect_pct": round(pk_eff_pct, 2),
            "PD_molecular_pct": round(pd_mol_pct, 2),
            "PD_effect_pct": round(pd_eff_pct, 2),
        },
        "chi_square": {
            "intermediate_count_table_chi2": float(chi2),
            "intermediate_count_table_p": float(p),
            "per_pair_table_chi2": float(chi2_p),
            "per_pair_table_p": float(p_p),
        },
        "layer_map": LAYER_MAP,
    }
    (OUT_DIR / "path_endpoint_layer.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n[E1b] saved summary → path_endpoint_layer.json")

    # Also save the contingency tables as CSV
    cont = pd.DataFrame(
        {
            "class": ["PK", "PK", "PK", "PD", "PD", "PD"],
            "layer": ["molecular", "effect_system", "other"] * 2,
            "intermediate_count": [counts["PK"]["molecular"], counts["PK"]["effect_system"], counts["PK"]["other"],
                                   counts["PD"]["molecular"], counts["PD"]["effect_system"], counts["PD"]["other"]],
            "per_pair_count": [per_pair_occ["PK"]["molecular"], per_pair_occ["PK"]["effect_system"], per_pair_occ["PK"]["other"],
                               per_pair_occ["PD"]["molecular"], per_pair_occ["PD"]["effect_system"], per_pair_occ["PD"]["other"]],
        }
    )
    cont.to_csv(OUT_DIR / "path_endpoint_layer.csv", index=False, encoding="utf-8-sig")
    print(f"[E1b] saved contingency → path_endpoint_layer.csv")

    # -----------------------------------------------------------------------
    # 7. Audit sample for E1b's PK/PD label pre-step
    # -----------------------------------------------------------------------
    audit = pkpd.sample(min(50, len(pkpd)), random_state=42).copy()
    audit["manual_verify_label"] = ""  # human fills: PK | PD | unsure
    audit["notes"] = ""
    audit_path = OUT_DIR / "audit_sample.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"[E1b] saved audit sample (50 rows for hand-verify) → audit_sample.csv")

    # -----------------------------------------------------------------------
    # 8. Verdict
    # -----------------------------------------------------------------------
    pk_concentrates_molecular = pk_mol_pct > pk_eff_pct
    pd_concentrates_effect = pd_eff_pct > pd_mol_pct
    print(f"\n[E1b i1 verdict]")
    print(f"  PK concentrates molecular (mol > eff)?  {pk_concentrates_molecular}  ({pk_mol_pct:.1f}% vs {pk_eff_pct:.1f}%)")
    print(f"  PD concentrates effect_system (eff > mol)?  {pd_concentrates_effect}  ({pd_eff_pct:.1f}% vs {pd_mol_pct:.1f}%)")
    if pk_concentrates_molecular and pd_concentrates_effect:
        print(f"  → i1 path-endpoint hypothesis SUPPORTED")
    else:
        print(f"  → i1 path-endpoint hypothesis NOT FULLY SUPPORTED — investigate")


if __name__ == "__main__":
    main()
