"""Model-FREE premise for the meta-path-discovery story: is converging on a shared
mediator FAMILY (esp. biological_process) intrinsically discriminative for DDI?

No model, no GPU, no checkpoint — pure counts over the locked adapter's AND support
cache + per-pair labels. This is the F1 "premise" layer (design:
Notes/Log/story_position_accessibility_design.md): before we claim the model
DISCOVERS a family, show the family itself carries DDI signal.

Per mediator-type family τ and per pair, feature = # shared mediators of type τ
(and a hub-robust Adamic-Adar-style variant). Report univariate AUROC vs label,
pooled over 3 seeds, plus a support-size-matched control (does the family beat
"just more mediators"?).

Uses the SAME AND support config as the winning eval (and, l_max=3, kpt=64, nmax=400, cp0).

Run (no GPU needed, but use project env for consistent libs):
  wsl bash -ic "conda activate project_1 && cd <root> && python Code/scripts/analyze_spmn_v2_family_discriminativity.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from run_spmn_v2_aware import _load_support_cache  # noqa: E402
from my_code.models.spmn_v1.retrieval import KIND_ORDER  # noqa: E402

SEEDS = (42, 43, 44)
L_MAX, K_PER_TYPE, N_MAX = 3, 64, 400


def _auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def main() -> None:
    n_types = len(KIND_ORDER)
    # per-pair, per-type count + total support; pooled across seeds
    cnt_by_type = {t: [] for t in range(n_types)}
    supp_tot = []
    y_all = []

    for seed in SEEDS:
        try:
            data = _load_support_cache("and", seed, L_MAX, K_PER_TYPE, N_MAX, False)
        except Exception as e:
            print(f"[seed {seed}] cache missing: {str(e)[:80]}")
            continue
        te = data["test_s2"]
        off = te["offsets"]; typ = te["typ"]; y = te["y"]
        n = len(off) - 1
        for i in range(n):
            s0, e0 = int(off[i]), int(off[i + 1])
            tt = typ[s0:e0].astype(int)
            supp_tot.append(e0 - s0)
            y_all.append(int(y[i]))
            bc = np.bincount(tt, minlength=n_types)
            for t in range(n_types):
                cnt_by_type[t].append(int(bc[t]))

    y_all = np.asarray(y_all); supp_tot = np.asarray(supp_tot, float)
    if len(y_all) == 0:
        print("no cache found"); sys.exit(1)
    print(f"[family-discrim] pooled seeds {SEEDS}: pairs={len(y_all)} "
          f"pos={int(y_all.sum())} neg={int((1-y_all).sum())}")
    print(f"  total-support univariate AUROC (baseline 'more mediators'): "
          f"{_auc(y_all, supp_tot):.4f}")

    # per-family: raw-count AUROC + log1p; also PARTIAL (residualize count on total
    # support via rank, crude: AUROC on count within total-support tertiles averaged)
    tert = np.quantile(supp_tot, [1/3, 2/3])
    strata = np.digitize(supp_tot, tert)
    print(f"\n{'family':22} {'share_pairs':>11} {'count-AUROC':>11} {'matched-AUROC':>13}")
    rows = []
    for t in range(n_types):
        c = np.asarray(cnt_by_type[t], float)
        share = float((c > 0).mean())
        auc_raw = _auc(y_all, c)
        # support-matched: average within-stratum AUROC (removes "just more support")
        aucs = []
        for s in (0, 1, 2):
            m = strata == s
            if m.sum() >= 50 and len(np.unique(y_all[m])) == 2:
                aucs.append(_auc(y_all[m], c[m]))
        auc_matched = float(np.nanmean(aucs)) if aucs else float("nan")
        rows.append((auc_matched if not np.isnan(auc_matched) else -1,
                     KIND_ORDER[t], share, auc_raw, auc_matched))
    rows.sort(reverse=True)
    for _, name, share, auc_raw, auc_m in rows:
        print(f"{name:22} {share:11.3f} {auc_raw:11.4f} {auc_m:13.4f}")

    print("\nnote: matched-AUROC (support-size-stratified) > 0.5 means the FAMILY carries")
    print("DDI signal beyond 'more mediators overall'. This is the model-free premise;")
    print("model attention (Exp2) + causal deletion (Exp3) later show the model USES it.")


if __name__ == "__main__":
    main()
