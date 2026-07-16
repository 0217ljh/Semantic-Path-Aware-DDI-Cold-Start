"""Step 1 of multi-hop path discovery:

For each positive drug pair with DDInter mechanism text:
  1. Extract clinical entities mentioned in the text (L3 dictionary).
  2. For each entity term, find all matching KG nodes (Hetionet + PrimeKG + DrugBank).
  3. From drug_a and drug_b, run BFS to each matched KG node (max 4 hops).
  4. Record the path as an abstracted node-kind sequence.
  5. Aggregate path-type signatures across all positive pairs.

Output:
  - `ddinter_anchored_paths.json`: per-pair raw path records
  - `path_type_templates.parquet`: aggregated path-type signatures + counts +
    example concrete paths
  - Console summary
"""
from __future__ import annotations
import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
_ROOT = next(p for p in [_HERE, *_HERE.parents] if (p / "Code" / "my_code").is_dir())
sys.path.insert(0, str(_ROOT / "Code"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json, re, time
from collections import Counter, defaultdict
import pandas as pd

from my_code.kg_lib.loader import build_merged_kg

OUT = _HERE.parent
KG_ROOT = _ROOT / "Code" / "data" / "KG"
EVAL_FILE = OUT.parent / "eval_200_PKB_PDB.parquet"  # 100 PK-B + 100 PD-B with mech text

# ----------------------------------------------------------------------------
# 1. L3 entity dictionary (from finding1 analysis, data-driven)
# ----------------------------------------------------------------------------
L3_DICT = {
    "receptor_channel_enzyme": [
        "cyp450", "cyp 3a4", "cyp3a4", "cyp 2d6", "cyp2d6", "cyp 2c9", "cyp2c9",
        "cyp 2c19", "cyp2c19", "cyp 1a2", "cyp1a2", "cyp 3a5", "cyp3a5",
        "alpha-1 adrenergic", "alpha-2 adrenergic", "alpha adrenergic",
        "beta-1 adrenergic", "beta-2 adrenergic", "beta adrenergic",
        "dopamine receptor", "dopaminergic",
        "serotonin receptor", "5-ht receptor",
        "muscarinic receptor", "nicotinic receptor",
        "histamine receptor", "h1 receptor", "h2 receptor",
        "opioid receptor", "mu receptor", "kappa receptor",
        "gaba receptor", "nmda receptor",
        "p-glycoprotein", "p-gp",
    ],
    "drug_class": [
        "anticholinergic", "antihistamine", "antihistaminic",
        "sedative", "sedating", "hypnotic",
        "opioid", "opiate",
        "phenothiazine", "phenothiazines",
        "tricyclic antidepressant", "tcas",
        "ssri", "maoi",
        "beta-blocker", "beta blocker", "ace inhibitor",
        "statin", "nsaid", "corticosteroid",
        "thiazide", "loop diuretic",
        "anticoagulant", "antiplatelet",
        "cns depressant",
        "serotonergic", "neuroleptic", "antipsychotic",
        "anxiolytic", "muscle relaxant",
    ],
    "system_organ": [
        "central nervous system", "cns",
        "cardiovascular system", "cardiac",
        "hepatic", "liver",
        "renal", "kidney",
        "gastrointestinal", "pulmonary", "respiratory",
        "autonomic",
    ],
    "mechanism_phenomenon": [
        "qt prolongation", "qtc prolongation", "qt interval",
        "torsade de pointes", "ventricular arrhythmia", "cardiac arrhythmia",
        "sudden death", "bleeding", "hemorrhage",
        "orthostatic hypotension", "hypotension",
        "sedation", "somnolence", "drowsiness",
        "cns depression", "respiratory depression", "respiratory-depressant",
        "hepatotoxicity", "liver injury",
        "nephrotoxicity", "renal failure",
        "cardiotoxicity", "serotonin syndrome",
        "hyperkalemia", "hypokalemia",
        "rhabdomyolysis", "myopathy",
        "electrolyte imbalance",
        "tachycardia", "bradycardia",
        "extrapyramidal",
    ],
}
L3_ALL = [(t, layer) for layer, terms in L3_DICT.items() for t in terms]


def extract_entities(text):
    """Return list of (term, layer) found in text (word-bounded match)."""
    if not isinstance(text, str):
        return []
    t = text.lower()
    found = []
    for term, layer in L3_ALL:
        if re.search(r'\b' + re.escape(term) + r'\b', t):
            found.append((term, layer))
    return found


# ----------------------------------------------------------------------------
# 2. KG graph: load merged edges, build adjacency + relation lookup
# ----------------------------------------------------------------------------
print("=" * 80)
print("Loading merged KG and building graph index...")
print("=" * 80)
t0 = time.time()
out = build_merged_kg()
edges_df = out["edges"]
nodes_df = out["nodes"]
print(f"  edges: {len(edges_df):,}, nodes: {len(nodes_df):,}, cache: {out['cache_hit']}")

# Adjacency list (undirected — we want reachability, direction recorded separately)
adj = defaultdict(set)
# Relation store: (sorted pair) -> set of relations
rel_of = defaultdict(set)
for src, dst, rel in zip(edges_df["src"], edges_df["dst"], edges_df["relation"]):
    if src == dst:
        continue
    adj[src].add(dst)
    adj[dst].add(src)
    key = (src, dst) if src < dst else (dst, src)
    rel_of[key].add(rel)
print(f"  adjacency built in {time.time()-t0:.1f}s")

# Node kind lookup
node_kind = dict(zip(nodes_df["id"], nodes_df["kind"]))
node_name = dict(zip(nodes_df["id"], nodes_df["name"]))


# Build entity-name → KG-node-id index (lowercase substring match)
print("Building entity name index...")
t0 = time.time()
name_to_ids = defaultdict(list)
for nid, nname in zip(nodes_df["id"], nodes_df["name"]):
    if not isinstance(nname, str) or not nname:
        continue
    nlow = nname.lower()
    # Index by full name; substring lookup happens at query time
    name_to_ids[nlow].append(nid)

# Pre-compute lowercase name list for substring search
all_node_names_lc = list(name_to_ids.keys())
print(f"  indexed {len(all_node_names_lc):,} unique node names in {time.time()-t0:.1f}s")


def find_kg_nodes_for_term(term, max_matches=20):
    """Return up to N KG node IDs whose name contains the term (case-insensitive)."""
    t = term.lower()
    matches = []
    # Exact match first
    if t in name_to_ids:
        matches.extend(name_to_ids[t])
    # Substring match (limited scan for perf)
    if len(matches) < max_matches:
        for nname in all_node_names_lc:
            if t in nname and nname != t:
                matches.extend(name_to_ids[nname])
                if len(matches) >= max_matches:
                    break
    return matches[:max_matches]


# ----------------------------------------------------------------------------
# 3. Bidirectional BFS shortest path (much faster on hub-rich KGs)
# ----------------------------------------------------------------------------
def bidir_bfs(src, dst, max_depth=3):
    """Bidirectional BFS: alternates expansion from src and dst, meets in middle.

    Returns list of nodes [src, ..., dst] or None if not reachable within
    `max_depth` hops total.
    """
    if src == dst:
        return [src]
    if src not in adj or dst not in adj:
        return None

    # frontiers and parent maps, one for each side
    parent_f = {src: None}      # forward: src side
    parent_b = {dst: None}      # backward: dst side
    front_f = {src}
    front_b = {dst}
    meeting = None
    total_depth = 0
    while front_f and front_b and total_depth < max_depth:
        # always expand the smaller frontier
        if len(front_f) <= len(front_b):
            new_f = set()
            for u in front_f:
                for v in adj[u]:
                    if v not in parent_f:
                        parent_f[v] = u
                        new_f.add(v)
                        if v in parent_b:
                            meeting = v
                            front_f = new_f
                            break
                if meeting:
                    break
            if meeting:
                break
            front_f = new_f
        else:
            new_b = set()
            for u in front_b:
                for v in adj[u]:
                    if v not in parent_b:
                        parent_b[v] = u
                        new_b.add(v)
                        if v in parent_f:
                            meeting = v
                            front_b = new_b
                            break
                if meeting:
                    break
            if meeting:
                break
            front_b = new_b
        total_depth += 1
    if meeting is None:
        return None
    # reconstruct path: meeting → src (reverse) + meeting → dst
    path_left = [meeting]
    while parent_f.get(path_left[-1]) is not None:
        path_left.append(parent_f[path_left[-1]])
    path_left.reverse()
    path_right = []
    cur = parent_b.get(meeting)
    while cur is not None:
        path_right.append(cur)
        cur = parent_b.get(cur)
    full = path_left + path_right
    if len(full) - 1 > max_depth:
        return None
    return full


# Back-compat alias
bfs_shortest_path = bidir_bfs


def path_to_type(path):
    """Abstract path (list of node IDs) to (kinds_tuple, relations_tuple).
    Relations are sorted tuples of all relations between adjacent nodes
    (handles multi-edges)."""
    kinds = tuple(node_kind.get(n, "Unknown") for n in path)
    rels = []
    for u, v in zip(path[:-1], path[1:]):
        key = (u, v) if u < v else (v, u)
        rels.append(tuple(sorted(rel_of.get(key, set()))))
    return kinds, tuple(rels)


def signature_string(kinds, rels):
    parts = [kinds[0]]
    for r, k in zip(rels, kinds[1:]):
        rs = "|".join(r) if len(r) > 1 else (r[0] if r else "?")
        parts.append(f"—[{rs}]→")
        parts.append(k)
    return " ".join(parts)


# ----------------------------------------------------------------------------
# 4. Run on positive pairs
# ----------------------------------------------------------------------------
print(f"\n" + "=" * 80)
print(f"Loading eval set: {EVAL_FILE.name}")
print("=" * 80)
ds = pd.read_parquet(EVAL_FILE)
print(f"  rows: {len(ds)}")
print(f"  class dist: {ds['pkpd_4way'].value_counts().to_dict()}")

# Cap per-term KG nodes & max paths per (drug, entity) to keep tractable
MAX_KG_NODES_PER_TERM = 8       # was 15: trim to top matches per entity term
MAX_DEPTH = 3                   # was 4: 3-hop covers meaningful chains; 4 blows up

# Step 1: collect ALL unique (drug, kg_target) queries from all pairs (dedup!)
print(f"\nStep 1: collecting unique (drug, kg_target) queries from {len(ds)} pairs...")
query_set = set()    # set of (drug_id, kg_node_id)
# pair_meta[(drug, kg_node)] = list of (pair_id, side, term, layer, class) using it
pair_meta = defaultdict(list)
nproc = 0
t0 = time.time()
for _, row in ds.iterrows():
    drug_a, drug_b = row["drug_a_id"], row["drug_b_id"]
    cls = row["pkpd_4way"]
    text = row.get("ddinter_mechanism", "") or ""
    entities = extract_entities(text)
    if not entities:
        continue
    nproc += 1
    for term, layer in entities:
        kg_node_ids = find_kg_nodes_for_term(term, MAX_KG_NODES_PER_TERM)
        for kg_id in kg_node_ids:
            for drug, side in [(drug_a, "A"), (drug_b, "B")]:
                query_set.add((drug, kg_id))
                pair_meta[(drug, kg_id)].append(dict(
                    pair_id=row.get("pair_id", ""),
                    drug_side=side,
                    drug_name=(row["drug_a_name"] if side == "A" else row["drug_b_name"]),
                    term=term, layer=layer, class_label=cls,
                ))
print(f"  pairs with entities: {nproc}/{len(ds)}")
print(f"  unique (drug, kg_target) queries to BFS: {len(query_set):,}", flush=True)

# Step 2: BFS each unique query, dedup-friendly
print(f"\nStep 2: bidirectional BFS for each unique query (MAX_DEPTH={MAX_DEPTH})...", flush=True)
type_counter = Counter()
type_examples = defaultdict(list)
pair_records = []
query_path = {}   # (drug, kg_id) -> path (or None)
t1 = time.time()
n_done = 0
n_found = 0
for (drug, kg_id) in query_set:
    path = bidir_bfs(drug, kg_id, max_depth=MAX_DEPTH)
    query_path[(drug, kg_id)] = path
    n_done += 1
    if path is not None:
        n_found += 1
    if n_done % 500 == 0:
        elapsed = time.time() - t1
        rate = n_done / max(elapsed, 0.01)
        eta = (len(query_set) - n_done) / max(rate, 0.01)
        print(f"  BFS {n_done}/{len(query_set)}  (found {n_found}, {100*n_found/n_done:.0f}%)  "
              f"rate {rate:.0f}/s  ETA {eta:.0f}s", flush=True)
print(f"  BFS done: {n_found}/{n_done} found a path in {time.time()-t1:.0f}s", flush=True)

# Step 3: assemble records using the cached paths
print(f"\nStep 3: assembling per-pair records and aggregating path types...", flush=True)
for (drug, kg_id), path in query_path.items():
    if path is None:
        continue
    kinds, rels = path_to_type(path)
    sig = (kinds, rels)
    for meta in pair_meta[(drug, kg_id)]:
        type_counter[sig] += 1
        pair_records.append(dict(
            pair_id=meta["pair_id"],
            class_label=meta["class_label"],
            drug_side=meta["drug_side"], drug=drug,
            term=meta["term"], layer=meta["layer"],
            kg_node=kg_id, kg_kind=node_kind.get(kg_id, ""),
            path_length=len(path) - 1,
            sig_kinds=kinds, sig_rels=rels,
        ))
        if len(type_examples[sig]) < 3:
            type_examples[sig].append({
                "pair_id": meta["pair_id"],
                "drug": drug, "drug_name": meta["drug_name"],
                "term": meta["term"], "layer": meta["layer"],
                "kg_node_id": kg_id,
                "kg_node_name": node_name.get(kg_id, ""),
                "path_ids": list(path),
                "path_names": [node_name.get(n, n) for n in path],
            })

print(f"\nDone. {nproc} pairs had at least one entity match.")
print(f"Total path records: {len(pair_records):,}")
print(f"Unique path-type signatures: {len(type_counter):,}")
print(f"Elapsed: {time.time()-t0:.1f}s")

# ----------------------------------------------------------------------------
# 5. Save outputs
# ----------------------------------------------------------------------------
# Per-pair records
records_df = pd.DataFrame(pair_records)
records_df.to_parquet(OUT / "ddinter_anchored_paths.parquet", index=False)
print(f"\nSaved: ddinter_anchored_paths.parquet ({len(records_df)} records)")

# Path type templates
template_rows = []
for (kinds, rels), count in type_counter.most_common():
    template_rows.append(dict(
        signature=signature_string(kinds, rels),
        length=len(kinds) - 1,
        node_kinds=" → ".join(kinds),
        relations=" | ".join("/".join(r) for r in rels),
        count=count,
        examples=json.dumps(type_examples[(kinds, rels)], ensure_ascii=False),
    ))
templates_df = pd.DataFrame(template_rows)
templates_df.to_parquet(OUT / "path_type_templates.parquet", index=False)
templates_df.drop(columns=["examples"]).to_csv(OUT / "path_type_templates.csv", index=False)
print(f"Saved: path_type_templates.parquet ({len(templates_df)} unique types)")

# ----------------------------------------------------------------------------
# 6. Console summary
# ----------------------------------------------------------------------------
print(f"\n" + "=" * 80)
print(f"Path type frequency report (top 40)")
print(f"{'='*80}")
print(f"{'#':<4}{'Len':>4}  {'Count':>7}  Node kinds → ...")
print("-" * 100)
for i, (sig, count) in enumerate(type_counter.most_common(40)):
    kinds, _ = sig
    print(f"{i+1:<4}{len(kinds)-1:>4}  {count:>7}  {' → '.join(kinds)}")

# Length distribution
print(f"\nDistribution by path length:")
len_counter = Counter()
for sig, c in type_counter.items():
    kinds, _ = sig
    len_counter[len(kinds) - 1] += c
for k in sorted(len_counter):
    print(f"  {k}-hop: {len_counter[k]:>7,}")

# By class
print(f"\nPath-type signatures unique to one class (top 20 by count, length > 1):")
by_class = defaultdict(Counter)
for r in pair_records:
    by_class[r["class_label"]][(r["sig_kinds"], r["sig_rels"])] += 1
pkb_only = [(s, c) for s, c in by_class["PK-B"].most_common()
            if s not in by_class["PD-B"] and len(s[0])-1 > 1][:20]
pdb_only = [(s, c) for s, c in by_class["PD-B"].most_common()
            if s not in by_class["PK-B"] and len(s[0])-1 > 1][:20]
print(f"\n  PK-B-only signatures (len > 1, top 15):")
for sig, c in pkb_only[:15]:
    kinds, _ = sig
    print(f"    {c:>5}  ({len(kinds)-1}-hop)  {' → '.join(kinds)}")
print(f"\n  PD-B-only signatures (len > 1, top 15):")
for sig, c in pdb_only[:15]:
    kinds, _ = sig
    print(f"    {c:>5}  ({len(kinds)-1}-hop)  {' → '.join(kinds)}")

print(f"\nDone. Path type templates ready for inspection.")
