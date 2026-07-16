"""Phenotype (HPO) axis renormalization (v8 -> v9): hp-native canonicalize + macro + bridges.

Codex-locked design (thread 019f170c) — a simpler replay of the disease axis (HPO
has NO BERT grouping, so no fold/LCA/dgrp):
  * prime:effect/phenotype:N -> hp:resolve(N) (alt_id canonicalized + merged; obsolete
    with clean replaced_by redirected, else kept FLAGGED with no hierarchy/macro, like
    GO/anatomy/disease).
  * REPLACE symmetrized prime:phenotype_phenotype with directed hp:is_a child->parent
    (HPO has only is_a). Materialize is_a closure scaffold so macro roll-up is walkable
    (flagged is_scaffold; no transitive shortcuts).
  * MACRO = 23 direct is_a children of HP:0000118 'Phenotypic abnormality' (top-level
    abnormality categories; NOT pure organ systems -> field name macro_top_level_abnormality).
    set-valued via is_a closure, in hp_meta sidecar. Nodes OUTSIDE the HP:0000118 subtree
    (Mode of inheritance / Clinical modifier / Frequency ...) kept (edge-referenced) but
    flagged is_abnormality=False and EXCLUDED from macro.
  * Side Effect / Symptom cross-ontology (parallel to disease's DOID, but weak coverage):
    MERGE only the collision-free 1:1 'Side Effect' (UMLS CUI) twins into hp:<id>; the
    non-unique + unmapped Side Effects stay standalone se:<cui> (kind side_effect) with an
    se:xref_hp bridge where an HPO xref exists; ALL 'Symptom' (MeSH, 0 HPO xref) stay
    standalone sym:<mesh> (kind symptom). Provenance source_cuis in hp_meta.
  * Cross-axis: add MONDO disease_has_feature -> hp edges (mondo:disease_has_feature,
    relationship + intersection_of), distinct from prime:disease_phenotype_*.

Acceptance = representation correctness/completeness (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v9/{nodes__dedup,edges__dedup,path_meta,go_meta,
        anat_meta,dis_meta,dis_fold_map,hp_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_phenotype.py
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict, deque

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v8"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v9"
HP = "Code/data/_cache/hp.obo"
MONDO = "Code/data/_cache/mondo.obo"
ABNORMALITY_ROOT = 118              # HP:0000118 'Phenotypic abnormality'
OLD_HIER = "prime:phenotype_phenotype"


def hid(s):
    m = re.search(r"HP:(\d+)", s)
    return int(m.group(1)) if m else None


def parse_hp(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"name": None, "obsolete": False, "replaced_by": None, "alt": [],
                   "is_a": set(), "umls": set()}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: HP:"):
                cur["id"] = hid(line); terms[cur["id"]] = cur
            elif line.startswith("name:"):
                cur["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
            elif line.startswith("replaced_by: HP:"):
                cur["replaced_by"] = hid(line)
            elif line.startswith("alt_id: HP:"):
                cur["alt"].append(hid(line))
            elif line.startswith("is_a: HP:"):
                u = hid(line)
                if u is not None:
                    cur["is_a"].add(u)
            elif line.startswith("xref: UMLS:"):
                cur["umls"].add(line.split()[1].replace("UMLS:", ""))
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def parse_mondo_feature(path, hp_canon, present_hp):
    """MONDO disease_has_feature -> HP (relationship + intersection_of), canon HP, keep present."""
    edges = []
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = None
        elif line.startswith("id: MONDO:"):
            m = re.search(r"MONDO:(\d+)", line); cur = int(m.group(1)) if m else None
        elif cur is not None and (line.startswith("relationship: disease_has_feature HP:")
                                  or line.startswith("intersection_of: disease_has_feature HP:")):
            h = hid(line)
            if h is not None:
                hc = hp_canon(h)
                if hc in present_hp:
                    edges.append((cur, hc))
    return edges


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v8 in: {len(nd)} nodes, {len(ed)} edges")
    terms, altmap = parse_hp(HP)
    print(f"HPO terms {len(terms)} | alt_id {len(altmap)} | obsolete {sum(t['obsolete'] for t in terms.values())}")

    def canon(N):
        return altmap.get(N, N)

    def resolve(N):
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

    def anc_incl(n):
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

    macro_set = {c for c in child.get(ABNORMALITY_ROOT, set()) if not terms[c]["obsolete"]}
    print(f"top-level abnormality macro set (children of HP:0000118): {len(macro_set)}")

    def macro_anc(n):
        R = anc_incl(n) & macro_set
        return sorted(s for s in R if not any(sp != s and s in anc_incl(sp) for sp in R))

    # ---- rename phenotype nodes prime:effect/phenotype:N -> hp:resolve(N) ----
    is_ph = nd["kind"].astype(str) == "effect/phenotype"
    ph_nodes = nd[is_ph].copy()
    ph_ren, used_live, obs_targets = {}, set(), set()
    for x in ph_nodes["id"].astype(str):
        N = int(x.split(":")[-1]); r = resolve(N)
        ph_ren[x] = f"hp:{r}"
        if r in terms and terms[r]["obsolete"]:
            obs_targets.add(r)
        else:
            used_live.add(r)
    print(f"phenotype nodes {len(ph_nodes)} -> hp: {len(set(ph_ren.values()))} distinct (alt/replaced merged {len(ph_nodes)-len(set(ph_ren.values()))})")

    # ---- Side Effect (UMLS) merge map: collision-free 1:1 into hp: ----
    is_se = nd["kind"].astype(str) == "Side Effect"
    is_sym = nd["kind"].astype(str) == "Symptom"
    se_nodes, sym_nodes = nd[is_se].copy(), nd[is_sym].copy()
    umls2hp, hp2umls = defaultdict(set), defaultdict(set)
    for n, t in terms.items():
        if not t["obsolete"]:
            for c in t["umls"]:
                umls2hp[c].add(canon(n)); hp2umls[canon(n)].add(c)
    se_map, merged_hp, se_src, se_xref = {}, set(), defaultdict(set), []
    n_se_merge = 0
    for x in se_nodes["id"].astype(str):
        cui = x.split("::")[-1]
        hs = {h for h in umls2hp.get(cui, ()) if not terms[h]["obsolete"]}
        if len(hs) == 1 and len(hp2umls[next(iter(hs))]) == 1:        # unique, collision-free
            h = next(iter(hs)); se_map[x] = f"hp:{h}"; merged_hp.add(h); se_src[h].add(cui); n_se_merge += 1
        else:
            se_map[x] = f"se:{cui}"
            for h in hs:
                se_xref.append((f"se:{cui}", f"hp:{canon(h)}"))           # bridge, not merge
    sym_map = {x: f"sym:{str(x).split('::')[-1]}" for x in sym_nodes["id"].astype(str)}
    print(f"Side Effect {len(se_nodes)} -> merged into hp: {n_se_merge} | standalone se: {len(se_nodes)-n_se_merge} "
          f"(xref bridges {len(se_xref)}) | Symptom {len(sym_nodes)} -> standalone sym:")

    # ---- node universe: is_a closure of (live phenotype + merged SE twins) + macro ----
    seed = used_live | {h for h in merged_hp if not terms[h]["obsolete"]} | macro_set
    present_live = set()
    for s in seed:
        present_live |= anc_incl(s)
    present_live = {n for n in present_live if not terms[n]["obsolete"]}
    scaffold = present_live - seed
    obs_materialize = {h for h in obs_targets if h in terms}
    present = present_live | obs_materialize
    non_abn = {n for n in present_live if ABNORMALITY_ROOT not in anc_incl(n) and n != ABNORMALITY_ROOT}
    print(f"hp: nodes {len(present)} (live {len(present_live)} [seed {len(seed & present_live)} + scaffold {len(scaffold)}] "
          f"+ obsolete-kept {len(obs_materialize)}) | non-abnormality-branch (no macro) {len(non_abn)}")

    # ---- node table ----
    keep_nodes = nd[~(is_ph | is_se | is_sym)].copy()
    hp_rows = pd.DataFrame({"id": [f"hp:{n}" for n in sorted(present)], "kind": "effect/phenotype",
                            "name": [nm_obo.get(n) for n in sorted(present)], "source_kg": "hpo"})
    se_ids = sorted({v for v in se_map.values() if v.startswith("se:")})
    se_name = {f"se:{str(x).split('::')[-1]}": nmx for x, nmx in zip(se_nodes["id"].astype(str), se_nodes["name"])}
    se_rows = pd.DataFrame({"id": se_ids, "kind": "side_effect",
                            "name": [se_name.get(i) for i in se_ids], "source_kg": "sider_umls"})
    sym_ids = sorted(set(sym_map.values()))
    sym_name = {f"sym:{str(x).split('::')[-1]}": nmx for x, nmx in zip(sym_nodes["id"].astype(str), sym_nodes["name"])}
    sym_rows = pd.DataFrame({"id": sym_ids, "kind": "symptom",
                             "name": [sym_name.get(i) for i in sym_ids], "source_kg": "hetionet_mesh"})
    new_nodes = pd.concat([keep_nodes, hp_rows, se_rows, sym_rows], ignore_index=True).drop_duplicates("id")
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap edges ----
    rename = {}
    rename.update(ph_ren); rename.update(se_map); rename.update(sym_map)

    def remap(x):
        return rename.get(str(x), str(x))

    keep_e = ed[ed["relation"] != OLD_HIER].copy()
    keep_e["src"] = keep_e["src"].map(remap)
    keep_e["dst"] = keep_e["dst"].map(remap)
    keep_e["src_kind"] = [kind_of.get(s, k) for s, k in zip(keep_e["src"], keep_e["src_kind"])]
    keep_e["dst_kind"] = [kind_of.get(d, k) for d, k in zip(keep_e["dst"], keep_e["dst_kind"])]
    n_before = len(keep_e)
    keep_e = keep_e.drop_duplicates(["src", "dst", "relation", "directed"])
    print(f"\nedge remap+dedup: {n_before} -> {len(keep_e)} (collapsed {n_before-len(keep_e)} merge-collisions)")

    # hp:is_a (directed child->parent among live present)
    isa = [(f"hp:{n}", f"hp:{p}") for n in present_live for p in parents.get(n, ()) if p in present_live]
    isa_df = pd.DataFrame({"src": [s for s, _ in isa], "src_kind": "effect/phenotype",
                           "dst": [d for _, d in isa], "dst_kind": "effect/phenotype",
                           "relation": "hp:is_a", "source_kg": "hpo", "directed": True})
    # se:xref_hp bridges (non-merged) -> only to phenotype terms we actually materialized
    se_xref = [(s, d) for s, d in set(se_xref) if d in kind_of]
    sx_df = pd.DataFrame({"src": [s for s, _ in se_xref], "src_kind": "side_effect",
                          "dst": [d for _, d in se_xref], "dst_kind": "effect/phenotype",
                          "relation": "se:xref_hp", "source_kg": "hpo", "directed": True})
    # MONDO disease_has_feature -> hp
    present_mondo = {int(i.split(":")[1]) for i in kind_of if str(i).startswith("mondo:")}
    feat = [(m, h) for m, h in parse_mondo_feature(MONDO, resolve, present) if f"mondo:{m}" in kind_of]
    feat_df = pd.DataFrame({"src": [f"mondo:{m}" for m, _ in feat], "src_kind": "disease",
                            "dst": [f"hp:{h}" for _, h in feat], "dst_kind": "effect/phenotype",
                            "relation": "mondo:disease_has_feature", "source_kg": "mondo", "directed": True})
    new_edges = pd.concat([keep_e, isa_df, sx_df, feat_df], ignore_index=True)
    new_edges = new_edges.drop_duplicates(["src", "dst", "relation", "directed"])
    bad = ~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    # ---- hp_meta sidecar ----
    rows = []
    for n in sorted(present):
        obs = bool(terms[n]["obsolete"]); abn = bool(ABNORMALITY_ROOT in anc_incl(n) or n == ABNORMALITY_ROOT)
        rows.append({"id": f"hp:{n}", "name": nm_obo.get(n), "depth": (depth(n) if not obs else None),
                     "obsolete": obs, "is_scaffold": bool(n in scaffold),
                     "is_abnormality": abn, "is_macro": bool(n in macro_set),
                     "macro_top_level_abnormality": (macro_anc(n) if (not obs and abn) else []),
                     "n_source_cuis": len(se_src.get(n, set())),
                     "source_cuis": sorted(se_src.get(n, set()))})
    hp_meta = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in ("path_meta", "go_meta", "anat_meta", "dis_meta", "dis_fold_map"):
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)
    hp_meta.to_parquet(f"{OUT_DIR}/hp_meta.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v9 built (phenotype axis) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | effect/phenotype {int(is_ph.sum())} -> hp {len(hp_rows)} "
          f"| Side Effect {int(is_se.sum())} -> hp(merged) {n_se_merge}+se {len(se_rows)} | Symptom {int(is_sym.sum())} -> sym {len(sym_rows)}")
    print(f"  edges {len(ed)} -> {len(new_edges)} | dropped {OLD_HIER} {int((ed['relation']==OLD_HIER).sum())}")
    print(f"  +hp:is_a {len(isa_df)} | +se:xref_hp {len(sx_df)} | +mondo:disease_has_feature {len(feat_df)}")
    print(f"  no prime:effect/phenotype:* left: {int(new_nodes['id'].astype(str).str.startswith('prime:effect/phenotype:').sum())==0}")
    print(f"  no het:Side Effect / Symptom left: {int(new_nodes['id'].astype(str).str.startswith(('het:Side Effect','het:Symptom')).sum())==0}")
    print(f"  obsolete-kept {int(hp_meta['obsolete'].sum())} | scaffold {int(hp_meta['is_scaffold'].sum())} | non-abnormality {int((~hp_meta['is_abnormality']).sum())}")
    real = hp_meta[~hp_meta['is_scaffold'] & ~hp_meta['obsolete'] & hp_meta['is_abnormality']]
    has_m = real['macro_top_level_abnormality'].map(lambda x: bool(x))
    print(f"  macro coverage (real abnormality terms): {int(has_m.sum())}/{len(real)} ({has_m.mean()*100:.0f}%)")
    nm_new = dict(zip(new_nodes['id'], new_nodes['name']))
    for q in ["hp:135", "hp:1626", "hp:707"]:   # Hypogonadism, cardiovascular, nervous
        r = hp_meta[hp_meta['id'] == q]
        if len(r):
            print(f"  {q} '{nm_new.get(q)}' -> macro {[nm_new.get('hp:'+str(s)) for s in r['macro_top_level_abnormality'].iloc[0]]}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,+sidecars,hp_meta).parquet")


if __name__ == "__main__":
    main()
