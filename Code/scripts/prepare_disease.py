"""Disease (MONDO/DOID) axis renormalization (v7 -> v8): MONDO-native fold + residual.

Codex-locked design (threads 019f14fa / 019f153a):
  * PrimeKG disease nodes = BERT name-clusters of MONDO terms. Fold each into the
    STANDARD MONDO is_a tree instead of keeping an artificial group layer:
      - members parsed from node id `prime:disease:<m1>_<m2>...` (= MONDO ids),
        alt_id-canonicalized; obsolete members dropped (flagged).
      - target = the UNIQUE most-specific common ancestor (LCA) in the MONDO is_a
        DAG, IF it is non-obsolete, specific (depth >= MIN_DEPTH) and not a metaclass/
        not_a_disease grouping bucket. Singletons -> their own MONDO term.
      - external edges (disease_protein/phenotype/drug/exposure) re-keyed to that
        `mondo:<id>` umbrella. No invented disaggregation (edges already group-granular).
  * RESIDUAL (no unique specific umbrella: multi-LCA / too-general / no-common /
    empty-after-obsolete) -> explicit group node `dgrp:<hash>` (stable hash of sorted
    members), members linked by `mondo:member_of_group`. Residual stays OUT of is_a DAG.
  * Hierarchy: directed `mondo:is_a` child->parent, authoritative from mondo.obo,
    materialized over the is_a closure of folded terms (scaffold ancestors flagged).
    No transitive-closure shortcuts.
  * Macro: PRIMARY = body-system (19 direct children of 'disease by body system or
    component' MONDO:7770006); SECONDARY = 'disease_grouping' subset (sparse tag).
    Both set-valued (DAG), via full-mondo closure, stored in sidecar.
  * disease->anatomy: authoritative MONDO axioms (disease_has_location /
    _has_inflammation_site / _arises_from_structure), 3 distinct directed disease->
    anatomy relations, uberon endpoints canonicalized to the anatomy axis' uberon:C.
  * DOID: het:Disease curated MONDO xref. MERGE the 135 unique 1:1 collision-free
    twins into mondo:<id> (atomic-fold makes this safe; Hetionet edges transfer +
    dedup; provenance source_doids in dis_meta). The 2 non-unique stay standalone `doid:<id>`.
  * Edge folding merges collisions -> dedup; provenance kept in dis_fold_map +
    n_folded_sources in dis_meta.

Acceptance = representation correctness/completeness (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v8/{nodes__dedup,edges__dedup,path_meta,
        go_meta,anat_meta,dis_meta,dis_fold_map}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_disease.py
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict, deque

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v7"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v8"
MONDO = "Code/data/_cache/mondo.obo"
UBERON = "Code/data/_cache/uberon-basic.obo"   # for canonicalizing disease->anatomy endpoints
BODY_SYSTEM_ROOT = 7770006          # MONDO:7770006 'disease by body system or component'
MIN_DEPTH = 4                       # umbrella must be at least this deep to be a fold target
OLD_HIER = "prime:disease_disease"
DISEASE_EXT_RELS = ("prime:disease_protein", "prime:disease_phenotype_positive",
                    "prime:disease_phenotype_negative", "prime:contraindication",
                    "prime:indication", "prime:off-label use", "prime:exposure_disease")


def mid(s):
    m = re.search(r"MONDO:(\d+)", s)
    return int(m.group(1)) if m else None


def uberon_id(s):
    m = re.search(r"UBERON:(\d+)", s)
    return int(m.group(1)) if m else None


# disease->anatomy MONDO predicates -> canonical KG relation names (kept distinct, codex)
DIS_ANAT_PREDS = {"disease_has_location": "mondo:disease_has_location",
                  "disease_has_inflammation_site": "mondo:disease_has_inflammation_site",
                  "disease_arises_from_structure": "mondo:disease_arises_from_structure"}


def parse_mondo(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"name": None, "obsolete": False, "replaced_by": None, "alt": [],
                   "is_a": set(), "subset": set(), "doid": set(),
                   "danat": {p: set() for p in DIS_ANAT_PREDS}}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: MONDO:"):
                cur["id"] = mid(line); terms[cur["id"]] = cur
            elif line.startswith("name:"):
                cur["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
            elif line.startswith("replaced_by: MONDO:"):
                cur["replaced_by"] = mid(line)
            elif line.startswith("alt_id: MONDO:"):
                cur["alt"].append(mid(line))
            elif line.startswith("is_a: MONDO:"):
                u = mid(line)
                if u is not None:
                    cur["is_a"].add(u)
            elif line.startswith("subset:"):
                cur["subset"].add(line.split()[1])
            elif line.startswith("xref: DOID:"):
                cur["doid"].add(line.split()[1])
            elif line.startswith("relationship: ") or line.startswith("intersection_of: "):
                # genus-differentia (intersection_of) and plain relationship both carry these preds
                parts = line.split(None, 2)
                if len(parts) >= 3 and parts[1] in DIS_ANAT_PREDS and "UBERON:" in parts[2]:
                    u = uberon_id(parts[2])
                    if u is not None:
                        cur["danat"][parts[1]].add(u)
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def parse_uberon_altmap(path):
    """uberon-basic.obo alt_id -> primary (to canonicalize disease->anatomy endpoints
    onto the same uberon:C ids the anatomy axis produced)."""
    ualt = {}
    cur_id = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur_id = None
        elif line.startswith("id: UBERON:"):
            cur_id = uberon_id(line)
        elif line.startswith("alt_id: UBERON:") and cur_id is not None:
            a = uberon_id(line)
            if a is not None:
                ualt[a] = cur_id
    return ualt


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v7 in: {len(nd)} nodes, {len(ed)} edges")
    terms, altmap = parse_mondo(MONDO)
    ualt = parse_uberon_altmap(UBERON)
    print(f"MONDO terms {len(terms)} | alt_id {len(altmap)} | obsolete {sum(t['obsolete'] for t in terms.values())} "
          f"| uberon alt_id {len(ualt)}")

    def canon(N):
        return altmap.get(N, N)

    def resolve(N):
        # alt_id canonicalization, then one clean replaced_by hop for obsolete terms
        c = altmap.get(N, N)
        t = terms.get(c)
        if t and t["obsolete"] and t["replaced_by"] is not None:
            rb = altmap.get(t["replaced_by"], t["replaced_by"])
            if rb in terms and not terms[rb]["obsolete"]:
                return rb
        return c

    nm_obo = {n: t["name"] for n, t in terms.items()}
    parents = {n: {canon(p) for p in t["is_a"] if canon(p) in terms} for n, t in terms.items()}
    child = defaultdict(set)
    for n, ps in parents.items():
        for p in ps:
            child[p].add(n)

    anc_cache = {}

    def anc_incl(n):                 # ancestors including self
        if n in anc_cache:
            return anc_cache[n]
        seen, q = {n}, deque(parents.get(n, ()))
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x); q.extend(parents.get(x, ()))
        anc_cache[n] = seen
        return seen

    depth_cache = {}

    def depth(n):
        if n in depth_cache:
            return depth_cache[n]
        if not parents.get(n):
            depth_cache[n] = 0; return 0
        d = 1 + min(depth(p) for p in parents[n])
        depth_cache[n] = d; return d

    # blacklist: grouping metaclasses / non-disease buckets (codex: depth cutoff alone insufficient)
    blacklist = {n for n, t in terms.items() if t["subset"] & {"metaclass", "not_a_disease"}}

    def valid_umbrella(c):
        return (c in terms and not terms[c]["obsolete"] and c not in blacklist
                and depth(c) >= MIN_DEPTH)

    # ---- body-system macro (19 direct children) + grouping subset ----
    macro_bs = {c for c in child.get(BODY_SYSTEM_ROOT, set()) if not terms[c]["obsolete"]}
    grouping = {n for n, t in terms.items() if "disease_grouping" in t["subset"] and not t["obsolete"]}
    print(f"body-system macro set: {len(macro_bs)} | disease_grouping subset: {len(grouping)}")

    def macro_anc(n, macroset):      # most-specific reachable macro terms (set-valued, DAG)
        R = anc_incl(n) & macroset
        return sorted(s for s in R if not any(sp != s and s in (anc_incl(sp)) for sp in R))

    # ============ FOLD each PrimeKG disease node ============
    is_dis = nd["kind"].astype(str) == "disease"
    dis_nodes = nd[is_dis].copy()
    fold_rows, fold_target = [], {}      # prime_id -> target id (mondo:X or dgrp:X)
    residual_members = defaultdict(set)  # dgrp id -> member mondo ids
    used_live = set()                    # live (non-obsolete) member terms to materialize in is_a tree
    obs_targets = set()                  # obsolete terms kept as a sole fold target (flagged, no clean replacement)

    def dgrp_id(members):
        h = hashlib.sha1("_".join(str(m) for m in sorted(members)).encode()).hexdigest()[:10]
        return f"dgrp:{h}"

    for pid in dis_nodes["id"].astype(str):
        raw = [int(x) for x in pid.replace("prime:disease:", "").split("_")]
        resolved = sorted({resolve(m) for m in raw if resolve(m) in terms})
        live = [m for m in resolved if not terms[m]["obsolete"]]
        dead = [m for m in resolved if terms[m]["obsolete"]]
        used_live |= set(live)
        n_obs = len(dead)
        reason = None; target = None; lca_depth = None; cand = []
        if len(resolved) == 1:
            m = resolved[0]; target = f"mondo:{m}"
            if terms[m]["obsolete"]:
                obs_targets.add(m); reason = "obsolete_kept"   # no clean replacement -> keep flagged
            else:
                reason = "singleton"
        elif len(live) == 0:
            target = dgrp_id(resolved); reason = "residual_all_obsolete"
            residual_members[target] |= set(resolved)
        elif len(live) == 1:
            target = f"mondo:{live[0]}"; reason = "single_after_obsolete"
        else:
            common = set.intersection(*[anc_incl(m) for m in live])
            msca = sorted({canon(c) for c in common if not any(c != o and c in anc_incl(o) for o in common)})
            cand = msca
            if len(msca) == 1 and valid_umbrella(msca[0]):
                target = f"mondo:{msca[0]}"; lca_depth = depth(msca[0])
                reason = "member_lca" if msca[0] in live else "external_lca"
            else:
                target = dgrp_id(live); residual_members[target] |= set(live)
                if len(msca) > 1:
                    reason = "residual_multi_lca"
                elif msca and not valid_umbrella(msca[0]):
                    reason = "residual_too_general"; lca_depth = depth(msca[0])
                else:
                    reason = "residual_no_common"
        fold_target[pid] = target
        fold_rows.append({"prime_id": pid, "n_members": len(resolved), "n_obsolete_dropped": n_obs,
                          "target": target, "reason": reason, "lca_depth": lca_depth,
                          "members": resolved, "candidate_lcas": cand})
    fold_map = pd.DataFrame(fold_rows)
    print("\nfold reason distribution:")
    for r, c in fold_map["reason"].value_counts().items():
        print(f"  {r:<32} {c}")
    n_target_mondo = fold_map["target"].astype(str).str.startswith("mondo:").sum()
    print(f"folded to mondo: {n_target_mondo} | residual dgrp: {len(fold_map)-n_target_mondo} "
          f"({fold_map[fold_map['target'].str.startswith('dgrp:')]['target'].nunique()} distinct dgrp)")

    # ============ DOID -> MONDO merge map (curated xref, 1:1 + collision-free only) ============
    is_hetd = nd["kind"].astype(str) == "Disease"
    het_nodes = nd[is_hetd].copy()
    doid2mondo, mondo2doid = defaultdict(set), defaultdict(set)
    for n, t in terms.items():
        if not t["obsolete"]:
            for d in t["doid"]:
                doid2mondo[d].add(canon(n)); mondo2doid[canon(n)].add(d)
    doid_map = {}                 # het:Disease id -> mondo:Y (merged) or doid:<num> (standalone)
    merged_mondo = set()          # MONDO terms that absorb a DOID twin
    doid_src = defaultdict(set)   # mondo id -> {DOID strings} (provenance for dis_meta)
    n_merge = n_standalone = 0
    for x in het_nodes["id"]:
        dkey = str(x).split("::")[-1]                   # 'DOID:xxx'
        ms = {m for m in doid2mondo.get(dkey, ()) if not terms[m]["obsolete"]}
        if len(ms) == 1 and len(mondo2doid[next(iter(ms))]) == 1:   # guardrail: unique twin, collision-free
            m = next(iter(ms))
            doid_map[x] = f"mondo:{m}"; merged_mondo.add(m); doid_src[m].add(dkey); n_merge += 1
        else:
            doid_map[x] = f"doid:{dkey.replace('DOID:', '')}"; n_standalone += 1
    print(f"\nDOID: {len(het_nodes)} het:Disease -> merged into MONDO twin {n_merge} | standalone doid: {n_standalone}")

    # ============ materialize mondo: node universe = is_a closure of folded terms + merged DOID twins ============
    mondo_umbrellas = {int(t.split(":")[1]) for t in fold_target.values() if t.startswith("mondo:")}
    live_umbrellas = {m for m in mondo_umbrellas if m in terms and not terms[m]["obsolete"]}
    seed = used_live | live_umbrellas | macro_bs | merged_mondo   # +DOID twins so they exist + carry is_a
    present_live = set()
    for s in seed:
        present_live |= anc_incl(s)
    present_live = {n for n in present_live if not terms[n]["obsolete"]}
    scaffold = present_live - seed                     # pure-ontology bridging nodes
    # obsolete terms still materialized (sole obsolete fold target / obsolete residual members), flagged, no is_a
    obs_materialize = set(obs_targets) | {m for ms in residual_members.values() for m in ms if terms[m]["obsolete"]}
    present = present_live | obs_materialize
    print(f"mondo: nodes {len(present)} (live {len(present_live)} [seed {len(seed & present_live)} + scaffold {len(scaffold)}] + obsolete-kept {len(obs_materialize)})")

    # ============ build node table ============
    keep_nodes = nd[~(is_dis | is_hetd)].copy()
    mondo_rows = pd.DataFrame({"id": [f"mondo:{n}" for n in sorted(present)], "kind": "disease",
                               "name": [nm_obo.get(n) for n in sorted(present)], "source_kg": "mondo"})
    dgrp_ids = sorted(residual_members)
    dgrp_rows = pd.DataFrame({"id": dgrp_ids, "kind": "disease_group",
                              "name": [f"disease group ({len(residual_members[g])} subtypes)" for g in dgrp_ids],
                              "source_kg": "primekg_bert"})
    sa_ids, sa_names = [], []
    for x, nmx in zip(het_nodes["id"], het_nodes["name"]):
        if doid_map[x].startswith("doid:"):
            sa_ids.append(doid_map[x]); sa_names.append(nmx)
    doid_rows = pd.DataFrame({"id": sa_ids, "kind": "disease", "name": sa_names, "source_kg": "hetionet_doid"})
    new_nodes = pd.concat([keep_nodes, mondo_rows, dgrp_rows, doid_rows], ignore_index=True)
    new_nodes = new_nodes.drop_duplicates("id")
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ============ remap edges ============
    def remap(x):
        x = str(x)
        if x in fold_target:
            return fold_target[x]
        if x in doid_map:
            return doid_map[x]
        return x

    keep_e = ed[ed["relation"] != OLD_HIER].copy()
    keep_e["src"] = keep_e["src"].map(remap)
    keep_e["dst"] = keep_e["dst"].map(remap)
    keep_e["src_kind"] = [kind_of.get(s, k) for s, k in zip(keep_e["src"], keep_e["src_kind"])]
    keep_e["dst_kind"] = [kind_of.get(d, k) for d, k in zip(keep_e["dst"], keep_e["dst_kind"])]
    n_before = len(keep_e)
    keep_e = keep_e.drop_duplicates(["src", "dst", "relation", "directed"])
    print(f"\nedge remap+dedup: {n_before} -> {len(keep_e)} (collapsed {n_before-len(keep_e)} fold-collisions)")

    # mondo:is_a (directed child->parent, among live present nodes; obsolete have none)
    isa = []
    for n in present_live:
        for p in parents.get(n, ()):
            if p in present_live:
                isa.append((f"mondo:{n}", f"mondo:{p}"))
    isa_df = pd.DataFrame({"src": [s for s, _ in isa], "src_kind": "disease",
                           "dst": [d for _, d in isa], "dst_kind": "disease",
                           "relation": "mondo:is_a", "source_kg": "mondo", "directed": True})
    # member_of_group (residual only): mondo:<member> -> dgrp:<g>
    mog = [(f"mondo:{m}", g) for g, ms in residual_members.items() for m in ms if m in present]
    mog_df = pd.DataFrame({"src": [s for s, _ in mog], "src_kind": "disease",
                           "dst": [d for _, d in mog], "dst_kind": "disease_group",
                           "relation": "mondo:member_of_group", "source_kg": "primekg_bert", "directed": True})
    # MONDO disease->anatomy axioms (3 distinct typed relations; uberon endpoints canonicalized)
    valid_uberon = {i for i in kind_of if str(i).startswith("uberon:")}
    danat = []
    for n in present_live:
        for pred, rel in DIS_ANAT_PREDS.items():
            for u in terms[n]["danat"][pred]:
                uid = f"uberon:{ualt.get(u, u)}"
                if uid in valid_uberon:
                    danat.append((f"mondo:{n}", uid, rel))
    danat = list(set(danat))
    danat_df = pd.DataFrame({"src": [s for s, _, _ in danat], "src_kind": "disease",
                            "dst": [d for _, d, _ in danat], "dst_kind": "anatomy",
                            "relation": [r for _, _, r in danat], "source_kg": "mondo", "directed": True})
    new_edges = pd.concat([keep_e, isa_df, mog_df, danat_df], ignore_index=True)
    bad = ~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    # ============ dis_meta sidecar ============
    n_folded = Counter(fold_target.values())
    rows = []
    for n in sorted(present):
        mid_ = f"mondo:{n}"; obs = bool(terms[n]["obsolete"])
        rows.append({"id": mid_, "name": nm_obo.get(n), "depth": (depth(n) if not obs else None),
                     "obsolete": obs, "is_scaffold": bool(n in scaffold),
                     "is_macro_body_system": bool(n in macro_bs),
                     "is_grouping": bool(n in grouping),
                     "macro_body_system_ancestors": (macro_anc(n, macro_bs) if not obs else []),
                     "grouping_ancestors": (macro_anc(n, grouping) if not obs else []),
                     "n_folded_sources": int(n_folded.get(mid_, 0)),
                     "source_doids": sorted(doid_src.get(n, set())),
                     "n_source_doids": len(doid_src.get(n, set()))})
    dis_meta = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in ("path_meta", "go_meta", "anat_meta"):
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)
    dis_meta.to_parquet(f"{OUT_DIR}/dis_meta.parquet", index=False)
    fold_map.to_parquet(f"{OUT_DIR}/dis_fold_map.parquet", index=False)

    # ============ verification ============
    print(f"\n=== v8 built (disease axis) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | disease(prime) {int(is_dis.sum())} -> "
          f"mondo {len(mondo_rows)} + dgrp {len(dgrp_rows)} | het:Disease {int(is_hetd.sum())} -> doid {len(doid_rows)}")
    print(f"  edges {len(ed)} -> {len(new_edges)} | dropped {OLD_HIER} {int((ed['relation']==OLD_HIER).sum())}")
    print(f"  +mondo:is_a {len(isa_df)} | +member_of_group {len(mog_df)} | +disease->anatomy {len(danat_df)} "
          f"{dict(danat_df['relation'].value_counts()) if len(danat_df) else {}}")
    print(f"  DOID merged into MONDO twin: {n_merge} | standalone doid: nodes: {len(doid_rows)} "
          f"| diseases carrying source_doids: {int((dis_meta['n_source_doids']>0).sum())}")
    print(f"  no prime:disease:* left: {int(new_nodes['id'].astype(str).str.startswith('prime:disease:').sum())==0}")
    print(f"  no het:Disease left: {int(new_nodes['id'].astype(str).str.startswith('het:Disease').sum())==0}")
    print(f"  obsolete-kept mondo nodes: {int(dis_meta['obsolete'].sum())} | scaffold: {int(dis_meta['is_scaffold'].sum())}")
    real = dis_meta[~dis_meta['is_scaffold'] & ~dis_meta['obsolete']]
    has_bs_real = real['macro_body_system_ancestors'].map(lambda x: bool(x))
    print(f"  body-system macro coverage (real disease terms): "
          f"{int(has_bs_real.sum())}/{len(real)} ({has_bs_real.mean()*100:.0f}%)")
    print(f"  disease terms with a disease->anatomy edge: {danat_df['src'].nunique() if len(danat_df) else 0}")
    nm_new = dict(zip(new_nodes['id'], new_nodes['name']))
    for q in ["mondo:19019", "mondo:4995", "mondo:5071"]:   # osteo imperfecta, cardiovascular disorder, nervous
        r = dis_meta[dis_meta['id'] == q]
        if len(r):
            print(f"  {q} '{nm_new.get(q)}' -> body-system {[nm_new.get('mondo:'+str(s)) for s in r['macro_body_system_ancestors'].iloc[0]]}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,path_meta,go_meta,anat_meta,dis_meta,dis_fold_map).parquet")


if __name__ == "__main__":
    main()
