"""Domain-AUC decomposition of struct_feats: which block carries the train/test shift?

Localizes the covariate-shift collapse (best_epoch=1 on the AND adapter) to a sub-block
of the per-pair struct_feats vector. codex round-2 hypothesis: the shift is dominated by
the UNCAPPED density COUNT block (s_tau / log1p(s_tau) / AA / n_support / n_active), not
the capped-support co-path SHAPE block (sym_copath). Decision rule:
  * if the no-count (copath) block's domain GBDT-AUC falls near 0.55-0.60, the collapse
    fix is feature-level — drop/normalize the count block (and route density through the
    separate decomposed promiscuity branch we already built);
  * if it stays high (>0.68), higher-order capped shape ALSO shifts and a feature drop
    alone will not fix the collapse (need real augmentation).

Reuses ``_domain_auc`` from analyze_spmn_v2_domain_shift (same balanced GBDT/linear
methodology, so numbers are comparable to the established ~0.76). Runs on the AND support
cache (the adapter that actually collapses). Read-only, cache-only, no GPU.

struct_feats layout (symmetric_binary_vector, dim = 3*N + N(N+1)/2 + 2):
  [ s_tau(N) | log1p(s_tau)(N) | AA(N) | sym_copath_triu(N(N+1)/2) | n_support | n_active ]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code/scripts"))

from analyze_spmn_v2_domain_shift import _domain_auc  # noqa: E402
from my_code.models.spmn_v1.retrieval import N_TYPES  # noqa: E402

CACHE = (ROOT / "Code/data/_cache/"
         "spmn_v2_supports_and_seed42_lmax3_kpt64_nmax400_cp0.npz")


def _blocks(dim: int) -> dict[str, list[int]]:
    nt = N_TYPES
    count = list(range(0, 3 * nt)) + [dim - 2, dim - 1]   # s_tau, log, AA, n_support, n_active
    copath = list(range(3 * nt, dim - 2))                  # sym_copath upper-tri (capped shape)
    return {"full": list(range(dim)),
            "count_only": count,
            "no_count(copath)": copath}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    args = ap.parse_args()

    z = np.load(args.cache)
    tr, va, te = z["train__struct"], z["val_s2__struct"], z["test_s2__struct"]
    dim = tr.shape[1]
    exp = 3 * N_TYPES + N_TYPES * (N_TYPES + 1) // 2 + 2
    print(f"[decomp] cache {Path(args.cache).name}")
    print(f"[decomp] train={len(tr)} val={len(va)} test={len(te)} dim={dim} (expect {exp})")
    blocks = _blocks(dim)
    print("[decomp] blocks: " + ", ".join(f"{k}={len(v)}d" for k, v in blocks.items()))

    print(f"\n{'block':<20} {'tr-vs-val GBDT':>15} {'(linear)':>10} {'tr-vs-test GBDT':>16}")
    for name, idx in blocks.items():
        idx = np.array(idx)
        g_v = _domain_auc(tr[:, idx], va[:, idx])
        l_v = _domain_auc(tr[:, idx], va[:, idx], linear=True)
        g_t = _domain_auc(tr[:, idx], te[:, idx])
        print(f"{name:<20} {g_v:>15.4f} {l_v:>10.4f} {g_t:>16.4f}")

    print("\nread: if no_count(copath) GBDT ~0.55-0.60 -> shift is the COUNT block -> "
          "feature-level fix (drop/normalize density). If it stays >0.68 -> capped shape "
          "also shifts -> a count drop alone won't fix the collapse.")


if __name__ == "__main__":
    main()
