"""Mechanism-pair feature probe (auto-research round 1, codex-designed).

DIAGNOSTIC ONLY (not a model / not an ensemble): for the standalone AND adapter,
test whether explicit DDI-mechanism structure that the pooling normalizes away
carries signal COMPLEMENTARY to the adapter — to decide what low-param feature to
build INTO the adapter's struct_feats next.

For each S2 pair we derive, from the cached AND support (rel_a, rel_b, degree per
shared mediator), candidate per-pair features:
  * rel-pair counts  count[symm(rel_a, rel_b)]      (all mediators)
  * rel-pair NON-HUB counts (mediator degree <= per-seed bottom-quartile)
  * total non-hub shared count
Then per feature we report, per seed and 3-seed-aggregated:
  * single-feature AUROC vs y_true
  * best rank-blend LIFT over a small lambda grid:
      AUC( rank(adapter) + lambda * rank(feature) ) - AUC(adapter)
A feature with consistent positive 3-seed blend lift = complementary DDI signal
the substrate is missing -> build it as an explicit feature (single model, no
ensemble). Read-only; uses existing caches + saved adapter predictions + KG degree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
from my_code.models.spmn_v1.retrieval import MergedKG, N_REL_BUCKETS  # noqa: E402

CACHE = ROOT / "Code/data/_cache"
RUNS = ROOT / "Code/runs/spmn_v2_standalone"
# coarse rel buckets (retrieval.REL_BUCKET): 0 target,1 enzyme,2 transporter,
# 3 carrier,4 pathway,5 gene-het,6 side_effect,7 indication,8 contra,9 other,10 2hop.
REL_NAME = {0: "target", 1: "enzyme", 2: "transp", 3: "carrier", 4: "pathway",
            5: "gene", 6: "sideeff", 7: "indic", 8: "contra", 9: "other", 10: "2hop"}


def _ranks(x):
    # average-rank ties (rankdata) so sparse/zero-heavy features do NOT leak the
    # label via index-order tie-breaking (frames are ordered positives-then-negatives).
    return rankdata(x, method="average") / len(x)


def _blend_lift(adapter, feat, y, lambdas):
    ra = _ranks(adapter); rf = _ranks(feat)
    base = roc_auc_score(y, ra)
    best, best_l = 0.0, 0.0
    for lam in lambdas:
        a = roc_auc_score(y, ra + lam * rf)
        if a - base > best:
            best, best_l = a - base, lam
    return base, best, best_l


def _pair_cell_counts(rela, relb, deg, pair_id, n_pairs, nonhub_mask, n_rel):
    """(n_pairs, n_rel*n_rel) symmetric rel-pair counts: all and non-hub."""
    lo = np.minimum(rela, relb); hi = np.maximum(rela, relb)
    cell = lo * n_rel + hi
    n_cell = n_rel * n_rel
    all_c = np.zeros((n_pairs, n_cell)); nh_c = np.zeros((n_pairs, n_cell))
    np.add.at(all_c, (pair_id, cell), 1.0)
    np.add.at(nh_c, (pair_id, cell), nonhub_mask.astype(float))
    return all_c, nh_c, n_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--adapter-tag", default="aware0_and",
                    help="prediction npz prefix: <tag>_seed<N>_test_s2_scores.npz")
    args = ap.parse_args()
    lambdas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

    print("[probe] loading merged KG (degree) ...", flush=True)
    kg = MergedKG.from_parquet()
    deg = kg.degree

    # accumulate per-feature lift across seeds
    feat_lift: dict[str, list] = {}
    feat_auc: dict[str, list] = {}

    for seed in args.seeds:
        z = np.load(CACHE / f"spmn_v2_supports_and_seed{seed}_lmax3_kpt64_nmax400_cp0.npz")
        med = z["test_s2__med"]; rela = z["test_s2__rela"]; relb = z["test_s2__relb"]
        off = z["test_s2__offsets"]; y = z["test_s2__y"].astype(int)
        n = len(off) - 1
        pred = np.load(RUNS / f"{args.adapter_tag}_seed{seed}_test_s2_scores.npz",
                       allow_pickle=True)
        adapter = pred["y_score"].astype(float)
        assert len(adapter) == n, f"pred {len(adapter)} != pairs {n}"
        assert (pred["y_true"].astype(int) == y).all(), "pred/cache label order mismatch"

        mdeg = deg[med].astype(float)
        q25 = np.quantile(mdeg, 0.25)
        nonhub = mdeg <= q25
        pair_id = np.repeat(np.arange(n), np.diff(off))

        all_c, nh_c, nr = _pair_cell_counts(rela, relb, mdeg, pair_id, n, nonhub,
                                            N_REL_BUCKETS)
        # build candidate features dict
        cands: dict[str, np.ndarray] = {}
        for ra in range(N_REL_BUCKETS):
            for rb in range(ra, N_REL_BUCKETS):
                c = ra * N_REL_BUCKETS + rb
                nm = f"{REL_NAME[ra]}-{REL_NAME[rb]}"
                if all_c[:, c].sum() >= 50:           # skip near-empty cells
                    cands[f"all:{nm}"] = all_c[:, c]
                if nh_c[:, c].sum() >= 50:
                    cands[f"nonhub:{nm}"] = nh_c[:, c]
        cands["nonhub:TOTAL"] = nh_c.sum(1)
        cands["all:TOTAL"] = all_c.sum(1)

        for name, f in cands.items():
            base, lift, lam = _blend_lift(adapter, f, y, lambdas)
            try:
                au = roc_auc_score(y, f)
            except ValueError:
                au = float("nan")
            feat_lift.setdefault(name, []).append(lift)
            feat_auc.setdefault(name, []).append(au)
        print(f"[probe] seed{seed}: n={n} adapterAUC={roc_auc_score(y,adapter):.4f} "
              f"q25_deg={q25:.0f} cands={len(cands)}", flush=True)

    # aggregate: keep features present in ALL seeds, rank by mean lift
    rows = []
    ns = len(args.seeds)
    for name, lifts in feat_lift.items():
        if len(lifts) == ns:
            rows.append((name, float(np.mean(lifts)), [round(x, 4) for x in lifts],
                         sum(x > 0 for x in lifts), float(np.nanmean(feat_auc[name]))))
    rows.sort(key=lambda r: -r[1])
    print(f"\n[probe] top features by 3-seed mean rank-blend lift "
          f"(complementary to adapter):")
    print(f"  {'feature':<26} {'meanLift':>9} {'per-seed':>26} {'pos':>4} {'soloAUC':>8}")
    for name, ml, ps, pos, au in rows[:20]:
        print(f"  {name:<26} {ml:>+9.4f} {str(ps):>26} {pos:>3}/{ns} {au:>8.3f}")
    print("\nread: a feature with mean lift >= +0.003 AND 3/3 seeds positive AND solo "
          "AUC clearly >0.5 is a real complementary DDI-mechanism signal -> build it "
          "into struct_feats (single model, not ensemble).")


if __name__ == "__main__":
    main()
