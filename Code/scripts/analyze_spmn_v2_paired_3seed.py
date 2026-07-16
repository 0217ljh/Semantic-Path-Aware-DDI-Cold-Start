"""Per-seed paired comparison: adapter (AND) vs NBFNet vs rank-blend, 3 seeds.

For each seed in {42,43,44} we join, by drug pair, the spmn_v2 AND-adapter
standalone scores and the matched-protocol NBFNet (run_nbfnet_v1_71_3seed)
scores on the SAME S2 test set, then report:
  - standalone AUC / AUPRC for each model,
  - 50/50 rank-blend AUC and the convex-optimal blend weight (alpha on adapter),
  - error-overlap groups (both right / only-adapter / only-nbfnet / neither).
Finally aggregates mean +/- std across seeds. The story holds if the blend beats
both standalone models per seed (genuine complementarity) and the per-seed
adapter/nbfnet AUCs track each other's seed ordering. Read-only.

Adapter npz: Code/runs/spmn_v2_standalone/and_ent1_asym0_absdiff0_dist0_seed{N}_test_s2_scores.npz
NBFNet npz : latest Code/runs/*run_nbfnet_v1_71_3seed*seed{N}*/test_s2_scores.npz
Both auto-discovered; override per seed with --adapter-npz / --nbfnet-npz (repeat).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError("could not locate project root (Code/data/KG)")


ROOT = _find_root()
SEEDS = (42, 43, 44)
ADP_DIR = ROOT / "Code/runs/spmn_v2_standalone"
ADP_NAME = "and_ent1_asym0_absdiff0_dist0_seed{seed}_test_s2_scores.npz"
RUNS_DIR = ROOT / "Code/runs"


def _adapter_npz(seed: int) -> Path:
    p = ADP_DIR / ADP_NAME.format(seed=seed)
    if not p.is_file():
        raise FileNotFoundError(f"adapter npz for seed{seed} not found at {p}")
    return p


def _nbfnet_npz(seed: int) -> Path:
    cands = sorted(RUNS_DIR.glob(f"*run_nbfnet_v1_71_3seed*seed{seed}*/test_s2_scores.npz"))
    if not cands:
        raise FileNotFoundError(
            f"no NBFNet 3seed npz for seed{seed} under {RUNS_DIR} "
            f"(pattern *run_nbfnet_v1_71_3seed*seed{seed}*/test_s2_scores.npz)")
    return cands[-1]  # latest timestamp (lexicographic == chronological here)


def _load_pair_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    pa = z["pair_a"].astype(str)
    pb = z["pair_b"].astype(str)
    y = z["y_true"].astype(int)
    s = z["y_score"].astype(float)
    return pa, pb, y, s


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def _join(adp_path: Path, nbf_path: Path):
    """Align NBFNet onto the adapter's pair order; return y, s_adp, s_nbf."""
    pa, pb, y, s_adp = _load_pair_npz(adp_path)
    npa, npb, ny, ns = _load_pair_npz(nbf_path)
    lut = {}
    for a, b, yy, ss in zip(npa, npb, ny, ns):
        lut[(a, b)] = (ss, yy)
    s_nbf = np.full(len(y), np.nan)
    y_nbf = np.full(len(y), -1, dtype=int)
    for i in range(len(y)):
        v = lut.get((pa[i], pb[i])) or lut.get((pb[i], pa[i]))
        if v is not None:
            s_nbf[i], y_nbf[i] = v
    ok = ~np.isnan(s_nbf)
    if (y_nbf[ok] != y[ok]).any():
        raise ValueError("label mismatch between adapter and NBFNet on joined pairs")
    return y[ok], s_adp[ok], s_nbf[ok], int(ok.sum()), int(len(y))


def _best_blend(y, s_adp, s_nbf):
    ra, rn = _rank(s_adp), _rank(s_nbf)
    best_a, best_auc = 0.5, -1.0
    for a in np.linspace(0.0, 1.0, 21):
        auc = roc_auc_score(y, a * ra + (1 - a) * rn)
        if auc > best_auc:
            best_a, best_auc = float(a), float(auc)
    half = roc_auc_score(y, 0.5 * ra + 0.5 * rn)
    return half, best_a, best_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--adapter-npz", action="append", default=[],
                    help="seed=path override, repeatable (e.g. --adapter-npz 42=/abs/x.npz)")
    ap.add_argument("--nbfnet-npz", action="append", default=[],
                    help="seed=path override, repeatable")
    args = ap.parse_args()

    adp_ovr = dict(kv.split("=", 1) for kv in args.adapter_npz)
    nbf_ovr = dict(kv.split("=", 1) for kv in args.nbfnet_npz)

    rows = []
    print(f"{'seed':>4} {'n':>5} {'nbf_auc':>8} {'adp_auc':>8} {'blend50':>8} "
          f"{'blend*':>8} {'a*':>5} | {'both':>5} {'onlyA':>5} {'onlyN':>5} {'none':>5}")
    for seed in args.seeds:
        adp_path = Path(adp_ovr[str(seed)]) if str(seed) in adp_ovr else _adapter_npz(seed)
        try:
            nbf_path = Path(nbf_ovr[str(seed)]) if str(seed) in nbf_ovr else _nbfnet_npz(seed)
        except FileNotFoundError as e:
            print(f"{seed:>4}  -- skipped: {e}")
            continue
        y, s_adp, s_nbf, n_join, n_tot = _join(adp_path, nbf_path)
        nbf_auc = roc_auc_score(y, s_nbf)
        adp_auc = roc_auc_score(y, s_adp)
        nbf_ap = average_precision_score(y, s_nbf)
        adp_ap = average_precision_score(y, s_adp)
        half, a_star, blend_star = _best_blend(y, s_adp, s_nbf)
        adp_ok = (s_adp > 0.5) == (y > 0.5)
        nbf_ok = (s_nbf > 0.5) == (y > 0.5)
        both = int((adp_ok & nbf_ok).sum())
        onlyA = int((adp_ok & ~nbf_ok).sum())
        onlyN = int((~adp_ok & nbf_ok).sum())
        none = int((~adp_ok & ~nbf_ok).sum())
        if n_join != n_tot:
            print(f"  [warn] seed{seed}: joined {n_join}/{n_tot} pairs (rest unmatched)")
        print(f"{seed:>4} {n_join:>5} {nbf_auc:>8.4f} {adp_auc:>8.4f} {half:>8.4f} "
              f"{blend_star:>8.4f} {a_star:>5.2f} | {both:>5} {onlyA:>5} {onlyN:>5} {none:>5}")
        rows.append((nbf_auc, adp_auc, half, blend_star, nbf_ap, adp_ap))

    if len(rows) >= 2:
        arr = np.array(rows)
        names = ["nbf_auc", "adp_auc", "blend50", "blend*", "nbf_ap", "adp_ap"]
        print("\nmean +/- std across seeds:")
        for j, nm in enumerate(names):
            print(f"  {nm:<9} {arr[:, j].mean():.4f} +/- {arr[:, j].std(ddof=0):.4f}")
        print("\n  complementarity holds if blend* > max(nbf_auc, adp_auc) every seed;")
        print("  a* near 0.5 => balanced contribution, a*->1 adapter-dominant, ->0 nbf-dominant.")


if __name__ == "__main__":
    main()
