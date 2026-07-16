"""Re-measure PD convergence on the dedup-v2 KG vs the original merged KG.

Read-only. Answers: did deduping proteins (Entrez) + GO bioprocess/molfunc/cellcomp
+ anatomy change the PD convergence picture? Runs the same PD-vs-degree-matched-random
convergence test on BOTH the original merged KG and `_merged_kg_dedup_v2`, side by side,
and breaks the molecular-convergence connector by TYPE (biological_process = MERGED in
v2, vs disease / pathway = NOT merged) so we can see which layer moved.

Metrics per KG (200 PD pairs vs 200 degree-matched random pairs, seed 42):
  * shared-target rate (proteins, by node) and median target count
  * molecular convergence per connector type (BP / disease / pathway): rate + median
    connector degree, for PD and RANDOM, plus the PD-RANDOM gap (discriminability)
  * shared side-effect rate (effect/phenotype — NOT merged in v2; control)

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_convergence_v2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

PD_CSV = "Code/data/_cache/pd_mechanism_200.csv"
DDI = "Code/data/KG/drugbank/filtered/ddi_edges.csv"
PKPD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
PROT = KIND_ORDER.index("protein_gene")
PATH = KIND_ORDER.index("pathway")
DIS = KIND_ORDER.index("disease")
BP = KIND_ORDER.index("biological_process")
SE = KIND_ORDER.index("side_effect")
TARGET_BUCKET = 0

VERSIONS = {
    "original": ("Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet",
                 "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"),
    "dedup_v2": ("Code/data/KG/_merged_kg_dedup_v2/nodes__dedup.parquet",
                 "Code/data/KG/_merged_kg_dedup_v2/edges__dedup.parquet"),
}


def nb(kg, u):
    return kg.indices[kg.indptr[u]:kg.indptr[u + 1]]


def target_proteins(kg, d):
    # ALL protein-type neighbours of the drug (any relation). Using the db:target
    # bucket via drug_rel is LOSSY on the dedup KG: a protein reached by both
    # db:target (bucket 0) and het:CdG (bucket 5) collapses to one prot: node and
    # drug_rel's first-wins setdefault can drop the target bucket. All-neighbours
    # is robust to the relation merge and is the right notion for convergence.
    x = nb(kg, d)
    return [int(m) for m in x if kg.type_id[m] == PROT]


def conn_via(kg, prots, ctid):
    s = set()
    for u in prots:
        x = nb(kg, u)
        s.update(x[kg.type_id[x] == ctid].tolist())
    return s


def build_pairs_ids(kg_ref):
    """Return (pd_id_pairs, neg_id_pairs) as DRUG IDs, sampled ONCE on a reference
    KG so the exact same pairs are evaluated on every KG version (fair comparison).
    Degree-matched negatives use the reference KG's degree."""
    pdp_df = pd.read_csv(PD_CSV)
    pd_ids = [(r["drug_a_id"], r["drug_b_id"]) for _, r in pdp_df.iterrows()
              if r["drug_a_id"] in kg_ref.id_to_idx and r["drug_b_id"] in kg_ref.id_to_idx]
    pk = pd.read_csv(PKPD)[["ddi_type", "pk_pd_label"]]
    ddi = pd.read_csv(DDI).merge(pk, on="ddi_type", how="left")
    ddi = ddi[ddi["drug_a_id"].isin(kg_ref.id_to_idx) & ddi["drug_b_id"].isin(kg_ref.id_to_idx)]
    pool = np.array(sorted(set(ddi["drug_a_id"]) | set(ddi["drug_b_id"])))
    posset = set(map(frozenset, zip(ddi["drug_a_id"], ddi["drug_b_id"])))
    deg = {d: kg_ref.degree[kg_ref.id_to_idx[d]] for d in pool}
    dv = np.array([deg[d] for d in pool])
    edges = np.quantile(dv, np.linspace(0, 1, 11)); edges[-1] += 1
    binof = {d: int(np.clip(np.digitize(deg[d], edges) - 1, 0, 9)) for d in pool}
    bins = {k: [] for k in range(10)}
    for d in pool:
        bins[binof[d]].append(d)
    rng = np.random.default_rng(42)
    neg_ids = []
    for a, b in pd_ids:
        for _ in range(30):
            a2 = rng.choice(bins[binof.get(a, 0)] or pool)
            b2 = rng.choice(bins[binof.get(b, 0)] or pool)
            if a2 != b2 and frozenset((a2, b2)) not in posset:
                neg_ids.append((a2, b2)); break
    return pd_ids, neg_ids


def to_idx(kg, id_pairs):
    out = []
    for a, b in id_pairs:
        ia, ib = kg.id_to_idx.get(a), kg.id_to_idx.get(b)
        if ia is not None and ib is not None:
            out.append((ia, ib))
    return out


def measure(kg, pairs):
    out = {ct: {"has": [], "deg": []} for ct in ("BP", "DIS", "PATH")}
    sh_tgt, se_has = [], []
    for a, b in pairs:
        TA, TB = target_proteins(kg, a), target_proteins(kg, b)
        sh_tgt.append(len(set(TA) & set(TB)) > 0)
        for name, ct in (("BP", BP), ("DIS", DIS), ("PATH", PATH)):
            cc = conn_via(kg, TA, ct) & conn_via(kg, TB, ct)
            out[name]["has"].append(len(cc) > 0)
            out[name]["deg"] += [int(kg.degree[w]) for w in cc]
        na = nb(kg, a)[kg.type_id[nb(kg, a)] == SE]
        nbb = nb(kg, b)[kg.type_id[nb(kg, b)] == SE]
        se_has.append(len(np.intersect1d(na, nbb)) > 0)
    res = {"shared_target": np.mean(sh_tgt), "shared_se": np.mean(se_has)}
    for ct in out:
        res[ct + "_rate"] = np.mean(out[ct]["has"])
        res[ct + "_deg"] = int(np.median(out[ct]["deg"])) if out[ct]["deg"] else 0
    return res


def main():
    # sample pairs ONCE on the original KG; reuse identical drug-id pairs everywhere
    ref = MergedKG.from_parquet(*VERSIONS["original"])
    pd_ids, neg_ids = build_pairs_ids(ref)
    print(f"sampled (once, on original): PD {len(pd_ids)} pairs | RAND {len(neg_ids)} pairs")

    rows = {}
    for ver, (npath, epath) in VERSIONS.items():
        kg = ref if ver == "original" else MergedKG.from_parquet(npath, epath)
        pdp, neg = to_idx(kg, pd_ids), to_idx(kg, neg_ids)
        rows[ver] = {"PD": measure(kg, pdp), "RAND": measure(kg, neg),
                     "n_pd": len(pdp), "n_neg": len(neg)}
        print(f"[{ver}] loaded n_nodes={kg.n_nodes} | PD pairs {len(pdp)} | rand {len(neg)}")

    print("\n=== PD convergence: original vs dedup_v2 (PD% / RAND% / gap) ===")
    metrics = [("shared_target", "shared TARGET (protein)"),
               ("BP_rate", "converge via BIOPROCESS  [MERGED in v2]"),
               ("DIS_rate", "converge via DISEASE     [not merged]"),
               ("PATH_rate", "converge via PATHWAY     [not merged]"),
               ("shared_se", "shared SIDE-EFFECT       [not merged]")]
    for key, label in metrics:
        line = f"  {label:<42}"
        for ver in VERSIONS:
            pd_v = rows[ver]["PD"][key] * 100
            rd_v = rows[ver]["RAND"][key] * 100
            line += f" | {ver}: PD {pd_v:>3.0f}% RAND {rd_v:>3.0f}% gap {pd_v-rd_v:+.0f}"
        print(line)
    print("\n=== median connector degree (hub-ness of the convergence node) ===")
    for key, label in [("BP_deg", "bioprocess connector deg"),
                       ("DIS_deg", "disease connector deg"),
                       ("PATH_deg", "pathway connector deg")]:
        line = f"  {label:<28}"
        for ver in VERSIONS:
            line += f" | {ver}: PD {rows[ver]['PD'][key]} RAND {rows[ver]['RAND'][key]}"
        print(line)


if __name__ == "__main__":
    main()
