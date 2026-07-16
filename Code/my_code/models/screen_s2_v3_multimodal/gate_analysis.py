"""Complementarity gate — ANALYSIS (codex r19/r20).

Loads MNAH test_s2 logits + motif-only test_s2 logits, aligns by canonical pair,
and answers: does the (weak) molecular signal add COMPLEMENTARY information to MNAH?

Outputs:
  - MNAH AUC, motif AUC (sanity)
  - Spearman corr(MNAH logit, motif logit)  [low corr => potentially complementary]
  - ORACLE ensemble: grid over a*z(mnah)+b*z(motif), best test AUC = UPPER BOUND on fusion
  - Error-overlap: among positives MNAH ranks in its worst tercile, what fraction does
    motif rank in its best half? (the niche molecular could fix)

Decision (codex r20): oracle-ensemble gain over MNAH < +1.0pt => molecular not worth it.
NOTE: oracle ensemble is tuned ON test = optimistic upper bound. If even that is < +1pt,
the honest fusion gain is ~0 and we pivot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
GATE = ROOT / "Code/runs/_gate_molecular"


def _canon(a, b):
    return (a, b) if a <= b else (b, a)


def main():
    mnah = np.load(GATE / "mnah_logits.npz", allow_pickle=True)
    motif = np.load(GATE / "motif_only_logits.npz", allow_pickle=True)

    # build canonical-pair -> (logit,label) for each, then align on common pairs
    def index(npz, logit_key):
        d = {}
        for a, b, lg, y in zip(npz["pair_a"], npz["pair_b"], npz[logit_key], npz["y_true"]):
            d[_canon(str(a), str(b))] = (float(lg), int(y))
        return d

    mi = index(mnah, "mnah_logit")
    ti = index(motif, "motif_logit")
    common = [k for k in mi if k in ti]
    print(f"[gate-an] MNAH pairs={len(mi)} motif pairs={len(ti)} common={len(common)}")
    # sanity: labels must agree
    y = np.array([mi[k][1] for k in common])
    y2 = np.array([ti[k][1] for k in common])
    if not (y == y2).all():
        n_dis = int((y != y2).sum())
        print(f"[gate-an] WARNING {n_dis} label mismatches between dumps; using MNAH labels")
    lm = np.array([mi[k][0] for k in common])
    lt = np.array([ti[k][0] for k in common])

    auc_m = roc_auc_score(y, lm)
    auc_t = roc_auc_score(y, lt)
    rho, _ = spearmanr(lm, lt)
    print(f"[gate-an] MNAH AUC={auc_m:.4f}  motif AUC={auc_t:.4f}  Spearman corr={rho:.3f}")

    # z-normalize, oracle grid ensemble (UPPER BOUND, tuned on test)
    zm = (lm - lm.mean()) / (lm.std() + 1e-9)
    zt = (lt - lt.mean()) / (lt.std() + 1e-9)
    best = (auc_m, 1.0, 0.0)
    for b in np.linspace(0, 2, 41):
        auc = roc_auc_score(y, zm + b * zt)
        if auc > best[0]:
            best = (auc, 1.0, float(b))
    print(f"[gate-an] ORACLE ensemble best AUC={best[0]:.4f} (a=1, b={best[2]:.2f}) "
          f"=> gain over MNAH = {best[0]-auc_m:+.4f}")

    # error-overlap: positives MNAH ranks worst-tercile, motif ranks better-half
    pos_mask = y == 1
    lm_pos = lm[pos_mask]; lt_pos = lt[pos_mask]
    mnah_worst = lm_pos <= np.quantile(lm_pos, 1/3)
    motif_betterhalf = lt_pos >= np.median(lt_pos)
    niche = int((mnah_worst & motif_betterhalf).sum())
    niche_frac = niche / max(int(mnah_worst.sum()), 1)
    print(f"[gate-an] error-overlap niche: of {int(mnah_worst.sum())} MNAH-worst-tercile "
          f"positives, motif ranks {niche} ({niche_frac:.1%}) in its better half")

    verdict = ("PROCEED to alignment" if best[0] - auc_m >= 0.01
               else "PIVOT — molecular adds <1pt even at oracle")
    print(f"[gate-an] VERDICT: {verdict}")
    (GATE / "gate_analysis.json").write_text(json.dumps({
        "n_common": len(common), "mnah_auc": auc_m, "motif_auc": auc_t,
        "spearman_corr": float(rho), "oracle_ensemble_auc": best[0],
        "oracle_gain": best[0] - auc_m, "oracle_b": best[2],
        "niche_frac": niche_frac, "verdict": verdict,
        "note": "oracle ensemble tuned on TEST = optimistic upper bound on fusion gain.",
    }, indent=2))
    print(f"[gate-an] saved -> {GATE/'gate_analysis.json'}")


if __name__ == "__main__":
    main()
