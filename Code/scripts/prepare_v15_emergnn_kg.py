"""idea2 v2 — build v15→EmerGNN KG variants for the lever-A KG-swap (codex 019f1dd2).

EmerGNN's kg_builder_merged.py keeps ONLY drug-incident edges, so the v15 meso hierarchy
(protein→meso) is unreachable. codex ruling B>A>C: materialize the drug→protein→meso 2-hop
signal as 1-hop drug→meso SHORTCUT edges so EmerGNN's bipartite builder can use it. Emit 3
KG variant dirs, each with a file named exactly MERGED_EDGES (7-col edges parquet), v15
"drug:" prefix STRIPPED on src/dst so drug nodes = bare DBxxxx (match benchmark drug_ids):

  A2  _v15_micro/               drug-incident to protein/pclass/drug only (EXCLUDE drug→meso)
  A3  _v15_micro_meso_native/   A2 + native direct drug→meso (best no-shortcut arm)
  B   _v15_micro_meso_shortcut/ A3 + materialized shortcuts (4 families, hub-capped)

Shortcut spec (EXACT, codex 019f1dd2 — do NOT simplify):
  families: drug_targets_{pathway,gobp,disease,phenotype}   (NO anatomy/GO-MF/GO-CC/ontology)
  per drug d: direct protein neighbors P(d) (kg_v15 adj["protein"]); for each p, its direct
    p→meso of the 4 types (kg_v15 build_protein_meso); add (d, meso, drug_targets_<type>).
  hub control: global_protein_degree[type][m] = #distinct proteins linked to m via that type;
    drop m with degree > 99th pct within type; score = support_count / log1p(degree);
    keep per drug top pathway32/GO-BP64/disease32/phenotype32.
  native drug→meso keep ORIGINAL relation; do NOT merge with shortcut relations.

Usage: PYTHONPATH=Code/code-idea-2 python -u Code/scripts/prepare_v15_emergnn_kg.py
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import kg_v15

KG_ROOT = Path("Code/data/KG")
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"  # EmerGNN hardcoded filename
COLS = ["src", "src_kind", "dst", "dst_kind", "relation", "source_kg", "directed"]
# codex round-1 shortcut families: (meso_key in build_protein_meso, meso id prefix, dst_kind, top-K)
SHORTCUT_FAMILIES = [
    ("pathway", "path:", "pathway", 32, "drug_targets_pathway"),
    ("gobp", "go:", "biological_process", 64, "drug_targets_gobp"),
    ("disease", "mondo:", "disease", 32, "drug_targets_disease"),
    ("phenotype", "hp:", "effect/phenotype", 32, "drug_targets_phenotype"),
]
BENCH = "Code/data/ddi_unified/binary_cls/drugbank_latest_full/inductive/S2/fold0"


def strip_drug(x: str) -> str:
    return x[5:] if x.startswith("drug:") else x


def main():
    nd, ed = kg_v15.load_v15()
    kind = dict(zip(nd["id"].astype(str), nd["kind"].astype(str)))
    drugset = set(nd[nd["kind"].astype(str) == "drug"]["id"].astype(str))  # "drug:DBxxxx"
    MESO_KINDS = {"biological_process", "molecular_function", "cellular_component", "pathway",
                  "anatomy", "disease", "disease_group", "effect/phenotype", "side_effect",
                  "symptom", "exposure"}
    MICRO_KINDS = {"drug", "gene/protein", "pharmacologic_class"}

    s = ed["src"].astype(str); d = ed["dst"].astype(str)
    src_is_drug = s.isin(drugset); dst_is_drug = d.isin(drugset)
    di = ed[src_is_drug | dst_is_drug].copy()  # drug-incident only (builder drops the rest anyway)
    # non-drug endpoint kind per drug-incident edge
    di_s = di["src"].astype(str); di_d = di["dst"].astype(str)
    other = np.where(di_s.isin(drugset), di_d, di_s)
    other_kind = pd.Series(other, index=di.index).map(kind).fillna("?")
    both_drug = di_s.isin(drugset) & di_d.isin(drugset)

    micro_mask = both_drug | other_kind.isin(MICRO_KINDS)
    meso_mask = other_kind.isin(MESO_KINDS)
    print(f"v15 total edges {len(ed):,} | drug-incident {len(di):,} | "
          f"micro-incident {int(micro_mask.sum()):,} | meso-incident {int(meso_mask.sum()):,}")

    def emit(df, out_dir):
        out = df[COLS].copy()
        out["src"] = out["src"].astype(str).map(strip_drug)
        out["dst"] = out["dst"].astype(str).map(strip_drug)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out.to_parquet(Path(out_dir) / MERGED_EDGES, index=False)
        return out

    a2 = emit(di[micro_mask], KG_ROOT / "_v15_micro")
    a3 = emit(di[micro_mask | meso_mask], KG_ROOT / "_v15_micro_meso_native")

    # ---- B: materialized shortcuts ----
    adj, _ = kg_v15.build_typed_drug_adj(nd, ed)
    p2 = kg_v15.build_protein_meso(ed)  # {key: {prot: set(meso)}}
    # global protein-degree per meso within type + 99th-pct allowlist
    allowed, degree = {}, {}
    for key, *_ in SHORTCUT_FAMILIES:
        m2p = defaultdict(set)
        for prot, mesos in p2[key].items():
            for m in mesos:
                m2p[m].add(prot)
        deg = {m: len(ps) for m, ps in m2p.items()}
        degree[key] = deg
        if deg:
            thr = np.percentile(list(deg.values()), 99)
            allowed[key] = {m for m, dg in deg.items() if dg <= thr}
        else:
            allowed[key] = set()
        print(f"  shortcut {key}: {len(deg)} meso nodes, 99th-pct protein-degree={thr if deg else 0:.0f}, "
              f"kept {len(allowed[key])}")

    rows = []
    fam_counts = Counter()
    for d_pref in drugset:
        P = adj["protein"].get(d_pref, set())
        if not P:
            continue
        dbare = strip_drug(d_pref)
        for key, pref, dkind, K, relname in SHORTCUT_FAMILIES:
            support = Counter()
            for p in P:
                for m in p2[key].get(p, set()):
                    if m in allowed[key]:
                        support[m] += 1
            if not support:
                continue
            scored = [(m, cnt / math.log1p(degree[key][m])) for m, cnt in support.items()]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for m, _ in scored[:K]:
                rows.append((dbare, "Drug", m, dkind, relname, "v15_shortcut", True))
                fam_counts[relname] += 1
    sc = pd.DataFrame(rows, columns=COLS)
    b = pd.concat([a3, sc], ignore_index=True)
    Path(KG_ROOT / "_v15_micro_meso_shortcut").mkdir(parents=True, exist_ok=True)
    b.to_parquet(KG_ROOT / "_v15_micro_meso_shortcut" / MERGED_EDGES, index=False)

    # ---- verify ----
    bench_drugs = set()
    for sp in ("train", "val", "test"):
        t = pd.read_parquet(f"{BENCH}/{sp}.parquet")
        bench_drugs |= set(t["drug_a_id"].astype(str)) | set(t["drug_b_id"].astype(str))
    print(f"\n== VERIFY (benchmark binary S2 has {len(bench_drugs)} distinct drugs) ==")
    for name, df in [("A2 micro", a2), ("A3 micro+meso native", a3), ("B micro+meso shortcut", b)]:
        dr = set(df["src"]) | set(df["dst"])
        dr_drugs = dr & bench_drugs
        cov = len(dr_drugs) / len(bench_drugs) * 100
        print(f"  {name:<24} edges {len(df):>9,} | rels {df['relation'].nunique():>2} | "
              f"bench-drug coverage {len(dr_drugs)}/{len(bench_drugs)} ({cov:.1f}%)")
    print(f"  shortcut edges per family: {dict(fam_counts)} (total {sum(fam_counts.values()):,})")
    # id-strip sanity: benchmark drug present as bare id in A3
    sample = next(iter(bench_drugs))
    print(f"  id-strip sanity: sample bench drug {sample!r} appears in A3 src/dst: "
          f"{sample in set(a3['src']) or sample in set(a3['dst'])}")
    print(f"\nwrote 3 variants under {KG_ROOT}/ (A1 control = existing _merged_kg)")


if __name__ == "__main__":
    main()
