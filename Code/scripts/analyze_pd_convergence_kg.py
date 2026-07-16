"""Do the LLM convergence chains exist in the KG, and are they discriminative?

Read-only. Answers the question: for PD pairs whose mechanism is
drug_A -> target_A -> [convergence] <- target_B <- drug_B, is the chain
present in the KG, and is it the SAME for random pairs (i.e. non-discriminative
because the convergence routes through hubs)?

Part 1 — convergence-node presence: search KG node names for the LLM-named
convergence concepts; report whether they exist and their degree (hub?).

Part 2 — discriminative test on 200 PD pairs vs 200 DEGREE-MATCHED random pairs:
  (a) molecular convergence: do A's target proteins and B's target proteins
      share a KG connector (pathway/disease/bioprocess)? connector degree?
  (b) side-effect convergence: do both drugs share a side-effect node? degree?
Compares PD vs random rates + connector degrees → missing chain vs hub problem.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_convergence_kg.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

PD_CSV = "Code/data/_cache/pd_mechanism_200.csv"
NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
DDI = "Code/data/KG/drugbank/filtered/ddi_edges.csv"
PKPD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
PROT = KIND_ORDER.index("protein_gene")
PATH = KIND_ORDER.index("pathway")
DIS = KIND_ORDER.index("disease")
BP = KIND_ORDER.index("biological_process")
SE = KIND_ORDER.index("side_effect")
TARGET_BUCKET = 0  # drug_rel bucket for db:target / prime:drug_protein


def nb(kg, u):
    return kg.indices[kg.indptr[u]:kg.indptr[u + 1]]


def target_proteins(kg, d):
    rel = kg.drug_rel.get(d, {})
    return [m for m, b in rel.items() if b == TARGET_BUCKET and kg.type_id[m] == PROT]


def conn_via(kg, prots, ctid):
    s = set()
    for u in prots:
        x = nb(kg, u)
        s.update(x[kg.type_id[x] == ctid].tolist())
    return s


def main():
    kg = MergedKG.from_parquet()
    names = pd.read_parquet(NODES, columns=["id", "name", "kind"])
    nm_series = names["name"].fillna("").astype(str).str.lower()

    # ---- Part 1: convergence concepts in the KG? ----
    print("=== [1] LLM convergence concepts: present in KG? degree? ===")
    concepts = ["cns depression", "central nervous system depression", "sedation",
                "respiratory depression", "qt", "long qt", "hypotension",
                "blood pressure", "hypertension", "potassium", "hypokalemia",
                "serotonin", "blood glucose", "hyperglycemia", "bradycardia",
                "platelet", "bleeding", "constipation", "nephrotox"]
    for c in concepts:
        hit = np.where(nm_series.str.contains(c, regex=False).to_numpy())[0]
        if len(hit) == 0:
            print(f"  '{c:<26}' : NOT in KG")
            continue
        degs = kg.degree[hit]
        kinds = names["kind"].to_numpy()[hit]
        top = hit[np.argmax(degs)]
        print(f"  '{c:<26}' : {len(hit):>4} node(s) | maxdeg {int(degs.max()):>5} "
              f"({names['name'].iloc[top][:34]} / {kinds[np.argmax(degs)]})")

    # ---- pairs: 200 PD (same set as LLM) + degree-matched random ----
    pdp_df = pd.read_csv(PD_CSV)
    pdp = []
    for _, r in pdp_df.iterrows():
        a, b = kg.id_to_idx.get(r["drug_a_id"]), kg.id_to_idx.get(r["drug_b_id"])
        if a is not None and b is not None:
            pdp.append((a, b))
    # drug pool + positives for degree-matched negatives
    pk = pd.read_csv(PKPD)[["ddi_type", "pk_pd_label"]]
    ddi = pd.read_csv(DDI).merge(pk, on="ddi_type", how="left")
    ddi["ai"] = ddi["drug_a_id"].map(kg.id_to_idx)
    ddi["bi"] = ddi["drug_b_id"].map(kg.id_to_idx)
    ddi = ddi.dropna(subset=["ai", "bi"]).astype({"ai": int, "bi": int})
    pool = np.array(sorted(set(ddi.ai) | set(ddi.bi)))
    posset = set(map(frozenset, zip(ddi.ai, ddi.bi)))
    deg = kg.degree
    edges = np.quantile(deg[pool], np.linspace(0, 1, 11)); edges[-1] += 1
    binof = {int(d): int(np.clip(np.digitize(deg[d], edges) - 1, 0, 9)) for d in pool}
    bins = {k: [] for k in range(10)}
    for d in pool.tolist():
        bins[binof[d]].append(d)
    rng = np.random.default_rng(42)
    neg = []
    for a, b in pdp:
        for _ in range(30):
            a2 = int(rng.choice(bins[binof.get(int(a), 0)] or pool))
            b2 = int(rng.choice(bins[binof.get(int(b), 0)] or pool))
            if a2 != b2 and frozenset((a2, b2)) not in posset:
                neg.append((a2, b2)); break

    def measure(pairs):
        mol_has, mol_deg, se_has, se_deg = [], [], [], []
        for a, b in pairs:
            TA, TB = target_proteins(kg, a), target_proteins(kg, b)
            conns, degs = 0, []
            for ct in (PATH, DIS, BP):
                cc = conn_via(kg, TA, ct) & conn_via(kg, TB, ct)
                conns += len(cc)
                degs += [int(kg.degree[w]) for w in cc]
            mol_has.append(conns > 0)
            mol_deg.append(int(np.median(degs)) if degs else 0)
            na = nb(kg, a)[kg.type_id[nb(kg, a)] == SE]
            nbb = nb(kg, b)[kg.type_id[nb(kg, b)] == SE]
            sh = np.intersect1d(na, nbb)
            se_has.append(len(sh) > 0)
            se_deg.append(int(kg.degree[sh].max()) if len(sh) else 0)
        return (np.mean(mol_has), np.median([d for d in mol_deg if d]),
                np.mean(se_has), np.median([d for d in se_deg if d]))

    print(f"\n=== [2] PD ({len(pdp)}) vs degree-matched random ({len(neg)}) ===")
    for tag, pr in [("PD", pdp), ("RAND(deg-matched)", neg)]:
        mh, md, sh, sd = measure(pr)
        print(f"  {tag:<18} | molec-convergence: {mh*100:>4.0f}% of pairs, "
              f"median connector deg {md:.0f} | shared side-effect: {sh*100:>4.0f}% "
              f"of pairs, median max-SE deg {sd:.0f}")


if __name__ == "__main__":
    main()
