"""B1a acceptance test - node name resolution with BE backfill."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER))

from kg.store import build_kg_store          # noqa: E402
from kg.node_names import build_node_names   # noqa: E402

_fails = []


def check(cond, msg):
    (_fails.append(msg) if not cond else None)
    print(f"  {'ok  ' if cond else 'FAIL'}: {msg}")


def main() -> int:
    kg = build_kg_store()
    names = build_node_names(kg, rebuild=True)
    check(len(names) == kg.n_nodes, "names aligned to n_nodes")

    # Gene/Protein name coverage should be ~100% after BE backfill
    gp_t = kg.type_names.index("Gene/Protein")
    gp = kg.type_id == gp_t
    gp_named = np.array([bool(names[i]) for i in np.flatnonzero(gp)])
    cov = 100 * gp_named.mean()
    check(cov >= 99.0, f"Gene/Protein name coverage >= 99% (got {cov:.1f})")

    # a specific BE-target resolves to its real name
    bi = kg.get_idx("db:target:BE0000048")
    check(bi is not None and names[bi] == "Prothrombin", f"BE0000048 -> Prothrombin (got {names[bi] if bi is not None else None!r})")

    # a Hetionet gene keeps its symbol
    sample_gene = None
    for i in np.flatnonzero(gp):
        nid = kg.idx2node[i]
        if str(nid).startswith("het:Gene"):
            sample_gene = (nid, names[i]); break
    check(sample_gene is not None and bool(sample_gene[1]), f"Hetionet gene has a symbol name (e.g. {sample_gene})")

    check(build_node_names(kg).shape == names.shape, "cache reload aligns")

    print(f"\n{'='*50}\n{'ALL PASS' if not _fails else f'{len(_fails)} FAILURES'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
