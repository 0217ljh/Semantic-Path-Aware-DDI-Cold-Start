"""Consolidate the S2 cold-start rank-analysis results into one figure-ready JSON
for the introduction figure.

Figure spec (user 2026-07-06): 2-D, x = PCA rank of the model's pre-scorer pair
representation, y = performance. One colored curve per (model, task) = the
DEPLOYABLE transfer curve (fit on train/source, applied to cold, no cold labels).
Each (model, task) also has a horizontal UPPER-BOUND line = the model's OWN cold
head prediction. degree_only (hub) and within-cold (oracle) included for optional
context.

Reads three already-computed runs verbatim (no re-training); recomputes only the
EmerGNN head_cold + leak-free within-cold from its saved pilot_Z. Writes
Code/runs/_intro_figure/intro_rank_curves.json.

Run:
  python Code/scripts/export_intro_figure_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))
from analyze_rank_within_split import _within_split_curve, _balance  # noqa: E402

RUNS = ROOT / "Code" / "runs"
RGCN_BIN = RUNS / "2026-07-06_14-26-09__train_rgcn_rank_s2__seed42"
EMER_TRANSFER = RUNS / "2026-07-06_13-41-52__extract_emergnn_s2_sparse_transfer__ddi800_cold_s2_fold0"
EMER_EXTRACT = RUNS / "2026-07-06_02-16-48__extract_emergnn_s2_rank__ddi800_cold_s2_fold0"
RGCN_MC = RUNS / "2026-07-06_15-19-18__train_rgcn_multicls_s2__seed42"


def _within(Z, y, ranks):
    Zb, yb = _balance(Z, y)
    _, cur = _within_split_curve(Zb, yb, ranks, n_splits=5, seed=0)
    missing = [r for r in ranks if r not in cur["mean"]]
    assert not missing, f"within-cold missing ranks {missing} (would misalign the figure)"
    return [cur["mean"][r] for r in ranks]


def main() -> None:
    series = []

    # 1) binary R-GCN S2 (results.json already has everything)
    r = json.loads((RGCN_BIN / "results.json").read_text())
    series.append({
        "key": "rgcn_binary_s2", "model": "R-GCN", "task": "binary", "metric": "AUROC",
        "dim": 256, "ranks": r["ranks"], "transfer": r["transfer_mean"],
        "transfer_resid": r["transfer_resid_mean"],
        "head_upper_bound": r["head_cold"], "degree_only": r["degree_only"],
        "within_cold": r["within_cold"], "source_run": RGCN_BIN.name})

    # 2) binary EmerGNN S2 (transfer from sparse-transfer run; head + within
    #    recomputed leak-free from the extract run's saved pilot_Z)
    st = json.loads((EMER_TRANSFER / "sparse_transfer.json").read_text())
    z = np.load(EMER_EXTRACT / "pilot_Z.npz")
    Zc, yc = z["Z_cold"], z["y_cold"]
    head_emer = float(roc_auc_score(yc, z["logit_cold"]))
    within_emer = _within(Zc, yc, st["ranks"])
    series.append({
        "key": "emergnn_binary_s2", "model": "EmerGNN", "task": "binary", "metric": "AUROC",
        "dim": 128, "ranks": st["ranks"], "transfer": st["transfer_mean"],
        "transfer_resid": st["transfer_resid_mean"],
        "head_upper_bound": round(head_emer, 4), "degree_only": st["degree_only_mean"],
        "within_cold": within_emer, "source_run": EMER_TRANSFER.name})

    # 3) multi-class R-GCN S2 (macro-AUROC curves)
    m = json.loads((RGCN_MC / "results.json").read_text())
    series.append({
        "key": "rgcn_multicls_s2", "model": "R-GCN", "task": f"multi-class (K={m['K']})",
        "metric": "macro-AUROC", "dim": m["dim"], "ranks": m["ranks"],
        "transfer": m["transfer_macroauc_mean"], "transfer_top5": m["transfer_top5_mean"],
        "head_upper_bound": m["head_cold"]["macro_auroc"], "degree_only": m["degree_only_macroauc"],
        "within_cold": m["within_cold_macroauc"],
        "head_all_metrics": m["head_cold"], "source_run": RGCN_MC.name})

    out = {
        "description": "S2 cold-start rank analysis, intro figure data (leak-free).",
        "axes": {"x": "PCA rank of the model's pre-scorer pair representation",
                 "y": "AUROC (binary) / macro-AUROC (multi-class)"},
        "curve": "transfer = DEPLOYABLE (fit on train/source, applied to cold, NO cold labels); "
                 "peaks at low rank then decays.",
        "upper_bound": "head_upper_bound = model's OWN cold head prediction (horizontal line).",
        "context": "degree_only = hub baseline; within_cold = oracle (uses cold labels, secondary).",
        "series": series}
    out_dir = RUNS / "_intro_figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intro_rank_curves.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {out_dir/'intro_rank_curves.json'}")
    for s in series:
        tp = np.nanmax(s["transfer"])
        print(f"  {s['key']:22s} metric={s['metric']:11s} dim={s['dim']} "
              f"transfer_peak={tp:.3f} head={s['head_upper_bound']} degree={s['degree_only']}")


if __name__ == "__main__":
    main()
