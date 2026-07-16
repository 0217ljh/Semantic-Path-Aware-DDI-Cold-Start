"""Mine candidate (rel_a, type, rel_b) meta-path schemas from the LOCKED adapter's
AND support — read-only, no model, no retrain.

Feeds the meta-path-discovery analysis (Notes/Log/metapath_discovery_analysis_design.md,
codex Option A): we want the NOVEL relation-typed shared-mediator schemas the coarse
PMP already sees, ranked by prevalence, with the asymmetric (rel_a != rel_b) and
cross-type ones flagged — these are the candidates for the "neural-discovered new
meta-path" headline. Importance/causal validation come later; this is candidate
NOMINATION from the real support substrate.

Schema = (rel_a_bucket, mediator_type, rel_b_bucket), canonicalized by sorting the
two arm buckets (DDI pair is unordered). Reports per schema:
  * coverage  = fraction of test_s2 pairs with >=1 mediator of this schema
  * mass      = total mediator instances (share of all support mediators)
  * asym      = rel_a != rel_b (relation-role-refined, the novel axis vs HIN-DDI)

Uses the SAME AND support config as analyze_spmn_v2_centrality_gap.py
(support_mode='and', l_max=3, k_per_type=64, n_max=400, with_copath=False).

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && python Code/scripts/analyze_spmn_v2_schema_candidates.py"
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from run_spmn_v2_aware import _load_support_cache  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    KIND_ORDER, REL_BUCKET, REL_OTHER, REL_2HOP, N_REL_BUCKETS,
)

SEEDS = (42, 43, 44)
L_MAX, K_PER_TYPE, N_MAX = 3, 64, 400


def _bucket_labels() -> dict[int, str]:
    """Human label per relation-bucket id (invert REL_BUCKET, group multi-map)."""
    by_bucket: dict[int, list[str]] = collections.defaultdict(list)
    for rel, b in REL_BUCKET.items():
        by_bucket[b].append(rel)
    lab = {b: "|".join(sorted(v)) for b, v in by_bucket.items()}
    lab[REL_OTHER] = "other"
    lab[REL_2HOP] = "2hop"
    for b in range(N_REL_BUCKETS):
        lab.setdefault(b, f"bucket{b}")
    return lab


def main() -> None:
    rlab = _bucket_labels()
    tlab = {i: k for i, k in enumerate(KIND_ORDER)}

    # accumulate across seeds
    cov_counter: collections.Counter = collections.Counter()   # schema -> pairs w/ >=1
    mass_counter: collections.Counter = collections.Counter()  # schema -> total instances
    n_pairs_total = 0
    total_med = 0

    for seed in SEEDS:
        try:
            data = _load_support_cache("and", seed, L_MAX, K_PER_TYPE, N_MAX, False)
        except Exception as e:
            print(f"[seed {seed}] support cache missing: {str(e)[:100]}")
            continue
        te = data["test_s2"]
        off = te["offsets"]; typ = te["typ"]; rela = te["rela"]; relb = te["relb"]
        n_pairs = len(off) - 1
        n_pairs_total += n_pairs
        total_med += len(typ)
        for i in range(n_pairs):
            s0, e0 = int(off[i]), int(off[i + 1])
            seen = set()
            for j in range(s0, e0):
                ra, rb, t = int(rela[j]), int(relb[j]), int(typ[j])
                a, b = (ra, rb) if ra <= rb else (rb, ra)   # canonicalize arms
                key = (a, t, b)
                mass_counter[key] += 1
                seen.add(key)
            for key in seen:
                cov_counter[key] += 1

    if n_pairs_total == 0:
        print("no support caches found — build them first via run_spmn_v2_standalone.py")
        sys.exit(1)

    print(f"[schema-candidates] pooled seeds {SEEDS}: test_s2 pairs={n_pairs_total} "
          f"total mediators={total_med}")
    print(f"  rel buckets: " + "; ".join(f"{b}={rlab[b]}" for b in range(N_REL_BUCKETS)))

    rows = []
    for key, cov in cov_counter.items():
        a, t, b = key
        rows.append((cov / n_pairs_total, mass_counter[key] / total_med, a, t, b, a != b))
    rows.sort(reverse=True)

    def show(title, filt):
        print(f"\n=== {title} ===")
        print(f"{'coverage':>8} {'mass':>7}  asym  schema (rel_a, type, rel_b)")
        shown = 0
        for cov, mass, a, t, b, asym in rows:
            if not filt(a, t, b, asym):
                continue
            print(f"{cov:8.3f} {mass:7.4f}  {'Y' if asym else '.'}    "
                  f"({rlab[a]}, {tlab.get(t, t)}, {rlab[b]})")
            shown += 1
            if shown >= 20:
                break

    show("TOP schemas by coverage (all)", lambda a, t, b, asym: True)
    show("TOP ASYMMETRIC schemas (rel_a != rel_b) — the relation-refined novelty axis",
         lambda a, t, b, asym: asym)
    show("TOP schemas NOT involving a 2-hop arm (both arms 1-hop, cleaner)",
         lambda a, t, b, asym: a != REL_2HOP and b != REL_2HOP)


if __name__ == "__main__":
    main()
