"""Anatomy v7 — concrete meta-path cases + necessity audit of the macro tightening.

Read-only. Two jobs:
  (A) for each anatomy meta-path (M_anat / C_anat_meso / C_anat_macro /
      C_anat_absent / develops_from) pull a REAL instance from _merged_kg_dedup_v7.
  (B) audit codex's macro tightening: recompute macro_ancestors under the FULL
      system set (58, no non-human/vascular filter) vs the TIGHTENED set (37) and
      report exactly what changed — did dropping non-human systems strand any
      human term? where do vessels land before/after?

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_anatomy_v7_cases.py
"""
from __future__ import annotations

import re
from collections import defaultdict, deque

import pandas as pd

V7 = "Code/data/KG/_merged_kg_dedup_v7"
UBERON = "Code/data/_cache/uberon-basic.obo"
ANATOMICAL_SYSTEM = 467
PRESENT = "prime:anatomy_protein_present"
ABSENT = "prime:anatomy_protein_absent"


def ubid(s):
    m = re.search(r"UBERON:(\d+)", s)
    return int(m.group(1)) if m else None


def parse_obo(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"name": None, "obsolete": False, "is_a": set(), "part_of": set()}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: UBERON:"):
                cur["id"] = ubid(line[4:]); terms[cur["id"]] = cur
            elif line.startswith("name:"):
                cur["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
            elif line.startswith("alt_id: UBERON:"):
                altmap.setdefault("pending", []).append((ubid(line[8:]), cur))
            elif line.startswith("is_a: UBERON:"):
                u = ubid(line[6:].split("!")[0].strip())
                if u is not None:
                    cur["is_a"].add(u)
            elif line.startswith("relationship: part_of UBERON:"):
                u = ubid(line.split()[2])
                if u is not None:
                    cur["part_of"].add(u)
    am = {}
    for a, t in altmap.get("pending", []):
        am[a] = t["id"]
    return terms, am


def build_system_sets(terms):
    child_isa = defaultdict(set)
    for n, t in terms.items():
        for p in t["is_a"]:
            child_isa[p].add(n)
    sys_desc, dq = set(), deque([ANATOMICAL_SYSTEM])
    while dq:
        x = dq.popleft()
        for c in child_isa.get(x, ()):
            if c not in sys_desc:
                sys_desc.add(c); dq.append(c)
    BASE_BAD = ("subdivision", "regional part", "division", "system of")
    full = {n for n in sys_desc if not terms[n]["obsolete"] and terms[n]["name"]
            and "system" in terms[n]["name"].lower()
            and not any(b in terms[n]["name"].lower() for b in BASE_BAD)}
    EXTRA_BAD = ("insect", "larval", "extraembryonic", "tracheal", "water vascular",
                 "open circulatory", "closed circulatory", "one-pass", "two-pass",
                 "embryonic cardiovascular")
    EXCLUDE_EXACT = {"arterial system", "venous system", "vascular system",
                     "pulmonary vascular system", "systemic arterial system",
                     "hindbrain venous system"}
    tight = {n for n in full if not any(b in terms[n]["name"].lower() for b in EXTRA_BAD)
             and terms[n]["name"].lower() not in EXCLUDE_EXACT}
    return full, tight


def main():
    nd = pd.read_parquet(f"{V7}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{V7}/edges__dedup.parquet")
    am = pd.read_parquet(f"{V7}/anat_meta.parquet")
    nm = dict(zip(nd["id"], nd["name"]))
    terms, altmap = parse_obo(UBERON)

    # KG anatomy canonical id set + parent map (is_a/part_of) from v7 edges
    rel = ed["relation"].astype(str)
    parent = defaultdict(set)
    for s, d, r in zip(ed["src"].astype(str), ed["dst"].astype(str), rel):
        if r in ("uberon:is_a", "uberon:part_of"):
            parent[int(s.split(":")[-1])].add(int(d.split(":")[-1]))
    kg_anat = {int(i.split(":")[-1]) for i in am["id"]}

    anc_cache = {}

    def ancestors(n):
        if n in anc_cache:
            return anc_cache[n]
        seen, q = set(), deque(parent.get(n, ()))
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x); q.extend(parent.get(x, ()))
        anc_cache[n] = seen
        return seen

    def macro_anc(n, sysset):
        R = (ancestors(n) | ({n} if n in sysset else set())) & sysset
        return sorted(s for s in R if not any(sp != s and s in (ancestors(sp) | {sp}) for sp in R))

    full_sys, tight_sys = build_system_sets(terms)
    full_kg = full_sys & kg_anat
    tight_kg = tight_sys & kg_anat

    # ============ (B) NECESSITY AUDIT ============
    print("=" * 70)
    print("(B) MACRO TIGHTENING AUDIT — full(58) vs tightened(37)")
    print("=" * 70)
    print(f"  system macros in KG: full={len(full_kg)}  tightened={len(tight_kg)}  "
          f"removed={len(full_kg - tight_kg)}")
    removed = sorted(full_kg - tight_kg)
    print(f"  REMOVED macros: {[nm.get(f'uberon:{n}') or terms[n]['name'] for n in removed]}")

    # coverage under each
    cov_full = cov_tight = 0
    stranded = []           # had macro under full, none under tight (would be lost)
    moved = []              # macro set changed
    for n in kg_anat:
        mf = macro_anc(n, full_kg)
        mt = macro_anc(n, tight_kg)
        if mf:
            cov_full += 1
        if mt:
            cov_tight += 1
        if mf and not mt:
            stranded.append(n)
        elif set(mf) != set(mt):
            moved.append((n, mf, mt))
    print(f"\n  coverage (>=1 macro): full={cov_full}/{len(kg_anat)}  "
          f"tightened={cov_tight}/{len(kg_anat)}  delta={cov_tight - cov_full}")
    print(f"  STRANDED by tightening (had macro, now none): {len(stranded)}")
    for n in stranded[:10]:
        print(f"     {nm.get(f'uberon:{n}')}  had->{[nm.get(f'uberon:{m}') for m in macro_anc(n, full_kg)]}")

    # non-human systems: did any human KG term ever roll into them?
    nonhuman = [n for n in removed if any(b in terms[n]["name"].lower() for b in
                ("insect", "larval", "extraembryonic", "tracheal", "water vascular",
                 "open circulatory", "closed circulatory", "one-pass", "two-pass", "embryonic"))]
    nh_used = {n: 0 for n in nonhuman}
    for n in kg_anat:
        for m in (set(ancestors(n)) | {n}) & set(nonhuman):
            nh_used[m] += 1
    print(f"\n  NON-HUMAN systems removed: {len(nonhuman)}")
    for n in nonhuman:
        print(f"     {nm.get(f'uberon:{n}') or terms[n]['name']:<40} #KG terms under it: {nh_used[n]}")

    # vascular sub-systems: where do their descendants go now?
    vasc = [n for n in removed if n not in nonhuman]
    print(f"\n  VASCULAR sub-systems removed: {[terms[n]['name'] for n in vasc]}")
    for q in ["vein", "artery", "aorta", "pulmonary artery", "great vein", "femoral artery"]:
        hit = next((n for n in kg_anat if str(nm.get(f"uberon:{n}")).lower() == q), None)
        if hit:
            mf = [nm.get(f"uberon:{m}") for m in macro_anc(hit, full_kg)]
            mt = [nm.get(f"uberon:{m}") for m in macro_anc(hit, tight_kg)]
            print(f"     {q:<16} full->{mf}   tightened->{mt}")

    # ============ (A) META-PATH CASES ============
    print("\n" + "=" * 70)
    print("(A) META-PATH CASES (real instances from v7)")
    print("=" * 70)

    # drug -> protein edges
    drug_kind = set(nd[nd["kind"].astype(str).str.lower().isin(["drug", "compound"])]["id"])
    d2p = defaultdict(set)
    for s, d in zip(ed["src"].astype(str), ed["dst"].astype(str)):
        if s in drug_kind and d.startswith("prot:"):
            d2p[s].add(d)
        elif d in drug_kind and s.startswith("prot:"):
            d2p[d].add(s)

    pres = ed[rel == PRESENT]
    p2a_present, a2p_present = defaultdict(set), defaultdict(set)
    for s, d in zip(pres["src"].astype(str), pres["dst"].astype(str)):
        a, p = (s, d) if s.startswith("uberon:") else (d, s)
        if p.startswith("prot:"):
            p2a_present[p].add(a); a2p_present[a].add(p)

    # M_anat: protein -> anatomy -> system
    print("\n[M_anat] protein -present-> anatomy -is_a/part_of*-> organ-system")
    shown = 0
    for p, ays in p2a_present.items():
        for a in ays:
            n = int(a.split(":")[-1])
            ms = macro_anc(n, tight_kg)
            if ms and len(a2p_present[a]) < 50:   # pick a specific (non-hub) tissue
                print(f"   {p}({nm.get(p)}) -> {nm.get(a)} -> {[nm.get(f'uberon:{m}') for m in ms]}")
                shown += 1
                break
        if shown >= 3:
            break

    # drug -> system rollup
    def drug_systems(dru):
        sys = set()
        for p in d2p.get(dru, ()):
            for a in p2a_present.get(p, ()):
                for m in macro_anc(int(a.split(":")[-1]), tight_kg):
                    sys.add(m)
        return sys

    def drug_anats(dru):
        a = set()
        for p in d2p.get(dru, ()):
            a |= p2a_present.get(p, set())
        return a

    drugs = [d for d in d2p if d in drug_kind][:400]
    # C_anat_meso: two drugs share a specific anatomy
    print("\n[C_anat_meso] two drugs' proteins present in the SAME anatomy")
    shown = 0
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            sh = drug_anats(drugs[i]) & drug_anats(drugs[j])
            sh = {a for a in sh if len(a2p_present[a]) < 30}
            if sh:
                a = sorted(sh, key=lambda x: len(a2p_present[x]))[0]
                print(f"   {nm.get(drugs[i])} + {nm.get(drugs[j])}  share-> {nm.get(a)}")
                shown += 1
                break
        if shown >= 3:
            break

    # C_anat_macro: two drugs share an organ-system
    print("\n[C_anat_macro] two drugs roll up to the SAME organ-system")
    shown = 0
    for i in range(0, 60):
        for j in range(i + 1, 60):
            if i >= len(drugs) or j >= len(drugs):
                continue
            sh = drug_systems(drugs[i]) & drug_systems(drugs[j])
            if sh:
                print(f"   {nm.get(drugs[i])} + {nm.get(drugs[j])}  share-> "
                      f"{[nm.get(f'uberon:{m}') for m in list(sh)[:3]]}")
                shown += 1
                break
        if shown >= 3:
            break

    # C_anat_absent: drug -> protein -absent-> anatomy
    print("\n[C_anat_absent] drug -> protein -absent-> anatomy (negative localization)")
    absent = ed[rel == ABSENT]
    p2a_absent = defaultdict(set)
    for s, d in zip(absent["src"].astype(str), absent["dst"].astype(str)):
        a, p = (s, d) if s.startswith("uberon:") else (d, s)
        if p.startswith("prot:"):
            p2a_absent[p].add(a)
    shown = 0
    for dru in drugs:
        for p in d2p.get(dru, ()):
            if p in p2a_absent:
                a = next(iter(p2a_absent[p]))
                print(f"   {nm.get(dru)} -> {p}({nm.get(p)}) -absent-> {nm.get(a)}")
                shown += 1
                break
        if shown >= 3:
            break

    # develops_from cross-link
    print("\n[develops_from] developmental lineage cross-link (NOT roll-up)")
    dv = ed[rel == "uberon:develops_from"].head(3)
    for s, d in zip(dv["src"].astype(str), dv["dst"].astype(str)):
        print(f"   {nm.get(s)} -develops_from-> {nm.get(d)}")


if __name__ == "__main__":
    main()
