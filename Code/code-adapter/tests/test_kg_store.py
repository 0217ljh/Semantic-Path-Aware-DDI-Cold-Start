"""A1 acceptance test - KG store (adjacency + types + directed edges).

Success criteria: builds from _merged_kg matching tex (178029/7099528/59),
undirected view is symmetric + deduped + self-loop-free, directed view keeps
relations, types canonicalized (no unmapped), drug mask correct, cache
round-trips.

Run: wsl bash -ic "conda activate project_1 && cd <root> && \
     python Code/code-adapter/tests/test_kg_store.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER))

from kg.store import build_kg_store  # noqa: E402

_fails: list[str] = []


def check(cond, msg):
    (_fails.append(msg) if not cond else None)
    print(f"  {'ok  ' if cond else 'FAIL'}: {msg}")


def main() -> int:
    print("== build (fresh) ==")
    kg = build_kg_store(rebuild=True)
    m = kg.meta

    print("\n== tex alignment ==")
    check(kg.n_nodes == 178029, f"n_nodes == 178029 (got {kg.n_nodes})")
    check(m["n_edges"] + m["n_edges_dropped"] == 7099528,
          f"n_edges + dropped == 7099528 (got {m['n_edges']}+{m['n_edges_dropped']})")
    check(m["n_edges_dropped"] <= 2, f"dropped <= 2 (got {m['n_edges_dropped']})")
    check(len(kg.rel_names) == 59, f"relations == 59 (got {len(kg.rel_names)})")

    print("\n== canonicalization ==")
    check(m["n_unmapped_kinds"] == 0, f"no unmapped kinds (got {m['n_unmapped_kinds']}: {m['unmapped_kinds']})")
    check("Gene/Protein" in kg.type_names, "Gene/Protein type present")
    check("Drug" in kg.type_names, "Drug type present")
    # canonical Drug = drug 5854 + Drug 1900 + Compound 294 = 8048
    check(int(kg.drug_mask.sum()) == 8048, f"drug nodes == 8048 (got {int(kg.drug_mask.sum())})")

    print("\n== undirected view: symmetric, deduped, no self-loops ==")
    rng = np.random.default_rng(0)
    samp = rng.choice(kg.n_nodes, size=300, replace=False)
    sym_ok = True; self_ok = True; dedup_ok = True
    for i in samp:
        nb = kg.u_indices[kg.u_indptr[i]:kg.u_indptr[i + 1]]
        if i in nb:
            self_ok = False
        if len(nb) != len(set(nb.tolist())):
            dedup_ok = False
        for j in nb[:50]:                       # back-edge must exist
            jn = kg.u_indices[kg.u_indptr[j]:kg.u_indptr[j + 1]]
            if i not in jn:
                sym_ok = False; break
    check(sym_ok, "undirected symmetric (a in adj[b] <=> b in adj[a])")
    check(self_ok, "no self-loops in undirected view")
    check(dedup_ok, "neighbors deduped")

    print("\n== directed view keeps relations ==")
    # pick a drug with edges
    drugs = kg.drug_ids()
    found = None
    for d in drugs[:200]:
        rows = kg.edge_rows_from(d)
        if rows:
            found = (d, rows); break
    check(found is not None, "a drug has directed edges")
    if found:
        d, rows = found
        rels = {r for _, r, _ in rows}
        check(len(rels) >= 1, f"edge relations present (e.g. {sorted(rels)[:3]})")

    print("\n== string-id API ==")
    some = kg.get_id(samp[0])
    check(kg.get_idx(some) == samp[0], "get_idx/get_id round-trip")
    check(set(kg.idx2node[kg.neighbors_idx(samp[0])].tolist()) == set(kg.neighbors(some).tolist()),
          "neighbors_idx matches neighbors (idx-space fast path)")
    check(isinstance(kg.node_type(some), str), "node_type returns str")
    check(kg.node_type("___nope___") is None, "unknown node_type -> None")

    print("\n== cache round-trip ==")
    kg2 = build_kg_store(rebuild=False)          # should HIT
    check(kg2.n_nodes == kg.n_nodes and len(kg2.rel_names) == 59, "cache reload matches")

    print(f"\n{'='*50}\n{'ALL PASS' if not _fails else f'{len(_fails)} FAILURES'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
