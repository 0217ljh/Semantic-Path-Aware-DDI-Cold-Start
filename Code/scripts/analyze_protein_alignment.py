"""Cross-namespace protein entity-alignment audit (NOT string matching).

Read-only. Canonical key = Entrez gene id. Resolves the three protein
namespaces in the merged KG to Entrez where possible and quantifies:

  Part A — het <-> prime fragmentation (EXACT, offline):
    Hetionet `het:Gene:Gene::<entrez>` and PrimeKG `prime:gene/protein:<entrez>`
    both encode Entrez in the id suffix (name = official symbol). So a gene
    present in both sources is a DUPLICATE pair of nodes. Count duplicates and
    spot-check whether the two copies are ever linked by an edge. DrugBank
    `db:target:BE...` carries no Entrez/symbol/UniProt (target table only has a
    full protein name) -> reported as un-unifiable offline (needs name xwalk).

  Part B — per-drug target namespace split (200 PD drugs):
    For each drug in the 200 PD pairs, bucket its protein neighbors by source
    (db / het / prime). Shows how much of a drug's target signal sits in the
    Entrez-less db:BE namespace vs the Entrez-known het/prime namespaces.

  Part C — LLM target -> Entrez anchoring (offline first pass):
    Build symbol->Entrez directly from het+prime node names (free, exact for
    official symbols). For each LLM-extracted a_target/b_target, try to resolve
    a symbol and check whether that Entrez is in the drug's actual KG neighbors
    ("anchored"). Unresolved strings are bucketed so the alias/family/process
    gap (the part that needs an external HGNC table) is quantified, not hidden.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_protein_alignment.py
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

import pandas as pd

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
CHAINS = "Code/runs/pd_mechanism/chains.jsonl"
OUT_CSV = "Code/runs/pd_mechanism/target_alignment.csv"
PROT = KIND_ORDER.index("protein_gene")

# tokens that are clearly family/complex/process, not a single gene product
FAMILY_HINTS = ("receptor", "channel", "gaba", "ampa", "nmda", "nicotinic",
                "adrenergic receptors", "muscarinic")
PROCESS_HINTS = ("secretion", "homeostasis", "depression", "tone", "metabolism",
                 "repolarization", "aggregation", "clearance", "system",
                 "pressure", "synthesis", "release", "uptake", "level")


def het_entrez(nid: str) -> str | None:
    return nid.split("::")[-1] if nid.startswith("het:Gene") else None


def prime_entrez(nid: str) -> str | None:
    return nid.split(":")[-1] if nid.startswith("prime:gene/protein:") else None


def candidate_symbols(target: str) -> list[str]:
    """Pull candidate gene-symbol tokens from an LLM target string."""
    toks: list[str] = []
    for grp in re.findall(r"\(([^)]*)\)", target):          # parenthetical
        toks += re.split(r"[\/,;]", grp)
    # bare ALL-CAPS-ish symbol token (e.g. "COMT", "SLC12A1")
    toks += re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", target)
    out, seen = [], set()
    for t in toks:
        t = t.strip().upper()
        if t and t not in seen and not t.isdigit():
            out.append(t); seen.add(t)
    return out


def classify_unresolved(target: str) -> str:
    low = target.lower()
    if any(h in low for h in PROCESS_HINTS) and "receptor" not in low:
        return "process/system"
    if any(h in low for h in FAMILY_HINTS):
        return "family/complex"
    return "other-name"


def main() -> None:
    kg = MergedKG.from_parquet()
    nd = pd.read_parquet(NODES)
    prot = nd[nd["kind"].astype(str).str.contains("protein|gene|Protein|Gene",
                                                  case=False, na=False)].copy()

    # ---- Part A: het <-> prime fragmentation via Entrez ----
    het = prot[prot["source_kg"] == "hetionet"].copy()
    prm = prot[prot["source_kg"] == "primekg"].copy()
    dbp = prot[prot["source_kg"] == "drugbank"].copy()
    het["entrez"] = het["id"].map(het_entrez)
    prm["entrez"] = prm["id"].map(prime_entrez)
    het = het.dropna(subset=["entrez"]); prm = prm.dropna(subset=["entrez"])
    het_set, prm_set = set(het["entrez"]), set(prm["entrez"])
    both = het_set & prm_set
    print("=== [A] protein-node fragmentation across sources (canonical = Entrez) ===")
    print(f"  DrugBank db:* protein nodes : {len(dbp):>6}  (no Entrez/symbol/UniProt -> NOT unifiable offline)")
    print(f"  Hetionet het:Gene nodes     : {len(het_set):>6}  (Entrez in id)")
    print(f"  PrimeKG prime:gene/protein  : {len(prm_set):>6}  (Entrez in id)")
    print(f"  genes in BOTH het & prime   : {len(both):>6}  <- DUPLICATE node pairs (same protein, 2 nodes)")
    print(f"  het-only                    : {len(het_set - prm_set):>6}")
    print(f"  prime-only                  : {len(prm_set - het_set):>6}")

    # spot-check: are the duplicate het/prime nodes ever directly edge-linked?
    h_by_e = dict(zip(het["entrez"], het["id"]))
    p_by_e = dict(zip(prm["entrez"], prm["id"]))
    linked = checked = 0
    for e in list(both)[:500]:
        hi, pi = kg.id_to_idx.get(h_by_e[e]), kg.id_to_idx.get(p_by_e[e])
        if hi is None or pi is None:
            continue
        checked += 1
        nbrs = kg.indices[kg.indptr[hi]:kg.indptr[hi + 1]]
        if pi in nbrs:
            linked += 1
    print(f"  duplicate pairs directly edge-linked: {linked}/{checked} sampled "
          f"-> {'MERGED' if linked == checked else 'NOT merged (separate nodes)'}")

    # ---- symbol -> entrez map (free, from het+prime names) ----
    sym2ent: dict[str, str] = {}
    for _, r in pd.concat([het[["entrez", "name"]], prm[["entrez", "name"]]]).iterrows():
        s = str(r["name"]).strip().upper()
        if s and s != "NAN":
            sym2ent.setdefault(s, r["entrez"])
    print(f"  symbol->Entrez dict built from KG names: {len(sym2ent)} symbols")

    # ---- load LLM chains (key = drug_a_id|drug_b_id) ----
    recs = []
    for line in pathlib.Path(CHAINS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("parsed"):
                recs.append(r)

    def prot_neighbors(d_idx):
        nb = kg.indices[kg.indptr[d_idx]:kg.indptr[d_idx + 1]]
        return [n for n in nb if kg.type_id[n] == PROT]

    # ---- Part B: per-drug target namespace split ----
    print(f"\n=== [B] per-drug protein-neighbor namespace split (200 PD drugs) ===")
    drug_ids = set()
    for r in recs:
        a, b = r["key"].split("|")
        drug_ids.update([a, b])
    nsplit = Counter()
    drugs_with_db_only = 0
    for d in drug_ids:
        di = kg.id_to_idx.get(d)
        if di is None:
            continue
        pn = prot_neighbors(di)
        srcs = Counter()
        for n in pn:
            nid = kg.node_ids[n]
            srcs[nid.split(":")[0]] += 1
        for k, v in srcs.items():
            nsplit[k] += v
        if srcs.get("db", 0) > 0 and srcs.get("het", 0) == 0 and srcs.get("prime", 0) == 0:
            drugs_with_db_only += 1
    print(f"  protein-neighbor edges by source namespace: {dict(nsplit)}")
    print(f"  drugs whose protein targets are ONLY in db:BE (Entrez-less): "
          f"{drugs_with_db_only}/{len(drug_ids)}")

    # ---- Part C: LLM target -> Entrez anchoring (offline first pass) ----
    print(f"\n=== [C] LLM target anchoring to KG (offline, symbol->Entrez from KG) ===")
    rows = []
    stat = Counter()
    unresolved_examples = Counter()
    for r in recs:
        a_id, b_id = r["key"].split("|")
        p = r["parsed"]
        for who, did, tgt in [("A", a_id, p.get("a_target") or ""),
                              ("B", b_id, p.get("b_target") or "")]:
            tgt = str(tgt)
            di = kg.id_to_idx.get(did)
            ent = None
            for s in candidate_symbols(tgt):
                if s in sym2ent:
                    ent = sym2ent[s]; break
            resolved = ent is not None
            anchored = False
            if resolved and di is not None:
                nb_ent = set()
                for n in prot_neighbors(di):
                    nid = kg.node_ids[n]
                    e = het_entrez(nid) or prime_entrez(nid)
                    if e:
                        nb_ent.add(e)
                anchored = ent in nb_ent
            if resolved:
                stat["resolved_symbol"] += 1
                stat["anchored" if anchored else "resolved_not_in_neighbors"] += 1
            else:
                bucket = classify_unresolved(tgt)
                stat[f"unresolved:{bucket}"] += 1
                unresolved_examples[(bucket, tgt[:48])] += 1
            rows.append({"pair": r["key"], "who": who, "target": tgt,
                         "entrez": ent, "resolved": resolved, "anchored": anchored})
    total = len(rows)
    print(f"  total LLM targets (200 pairs x 2)         : {total}")
    print(f"  resolved to a single Entrez (via KG names): {stat['resolved_symbol']} "
          f"({stat['resolved_symbol']/total*100:.0f}%)")
    print(f"    of those, ANCHORED in drug's neighbors  : {stat['anchored']} "
          f"({stat['anchored']/total*100:.0f}% of all targets)")
    print(f"    resolved but NOT in drug's neighbors    : {stat['resolved_not_in_neighbors']}")
    print(f"  UNresolved (need alias table / not a single gene):")
    for k in sorted(stat):
        if k.startswith("unresolved:"):
            print(f"      {k.split(':',1)[1]:<16}: {stat[k]}")
    print("  --- sample unresolved strings by bucket ---")
    for (bucket, ex), n in unresolved_examples.most_common(18):
        print(f"      [{bucket:<14}] x{n:<3} {ex}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote per-target resolution -> {OUT_CSV}")


if __name__ == "__main__":
    main()
