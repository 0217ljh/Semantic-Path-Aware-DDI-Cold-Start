"""Analyze a v2 run: overall + PK/PD-subgroup test_s2 AUROC/AUPRC.

Reads a run's test_s2_scores.npz (dumped by run_v2.py), maps each positive pair to PK/PD via
the SAME path as run_stage3.py: canonical-pair -> ddi_type from the release seed42 split
parquets (which carry ddi_type; the legacy PKL test_s2 does not), then ddi_type -> pk_pd_label
from ddi_pk_pd_labels.csv. Negatives have no ddi_type; for subgroup AUROC we pair each PK/PD
positive set against the FULL shared negative pool (standard one-vs-neg subgroup eval).

Usage: python -u .../analyze_v2_pkpd.py --run-dir Code/runs/<run_id> [--label v2_main]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
SPLIT_DIR = ROOT / "Code/data/KG/drugbank/splits/seed42"
PKPD_LABELS = ROOT / "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"


def _canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pair_to_type() -> dict[tuple[str, str], str]:
    """Canonical-pair -> ddi_type from all seed42 release split parquets."""
    m: dict[tuple[str, str], str] = {}
    for pq in sorted(SPLIT_DIR.glob("*.parquet")):
        d = pd.read_parquet(pq)
        if "ddi_type" not in d.columns:
            continue
        for a, b, t in zip(d["drug_a_id"].astype(str), d["drug_b_id"].astype(str), d["ddi_type"]):
            m[_canon(a, b)] = t
    return m


def _auc_safe(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    npz = np.load(run_dir / "test_s2_scores.npz", allow_pickle=True)
    pa = [str(x) for x in npz["pair_a"]]
    pb = [str(x) for x in npz["pair_b"]]
    y = npz["y_true"].astype(int)
    s = npz["y_score"].astype(float)

    p2t = _pair_to_type()
    lab = pd.read_csv(PKPD_LABELS).set_index("ddi_type")["pk_pd_label"].to_dict()

    pk_pd = []
    for a, b in zip(pa, pb):
        t = p2t.get(_canon(a, b))
        pk_pd.append(lab.get(t, "UNK") if t is not None else "UNK")
    pk_pd = np.array(pk_pd)

    pos = y == 1
    neg = y == 0
    res = {
        "label": args.label or run_dir.name,
        "overall_auc": _auc_safe(y, s),
        "overall_auprc": float(average_precision_score(y, s)),
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
    }
    for cls in ("PK", "PD", "Mixed"):
        sel_pos = pos & (pk_pd == cls)
        if sel_pos.sum() == 0:
            res[f"{cls}_auc"] = float("nan"); res[f"{cls}_n"] = 0
            continue
        yy = np.concatenate([np.ones(int(sel_pos.sum())), np.zeros(int(neg.sum()))])
        ss = np.concatenate([s[sel_pos], s[neg]])
        res[f"{cls}_auc"] = _auc_safe(yy, ss)
        res[f"{cls}_n"] = int(sel_pos.sum())
    res["pos_with_label"] = int((pk_pd[pos] != "UNK").sum())
    res["pos_unk"] = int((pk_pd[pos] == "UNK").sum())

    # per-channel subgroup AUROC + gate stratified stats (codex 019e6251 diagnostic)
    if "ch_g" in npz.files:
        g = npz["ch_g"].astype(float)
        for ch in ("emergnn", "mol", "eff", "head", "combined"):
            key = f"ch_{ch}"
            if key not in npz.files:
                continue
            cs = npz[key].astype(float)
            for cls in ("PK", "PD"):
                sel = pos & (pk_pd == cls)
                if sel.sum() == 0:
                    continue
                yy = np.concatenate([np.ones(int(sel.sum())), np.zeros(int(neg.sum()))])
                ss = np.concatenate([cs[sel], cs[neg]])
                res[f"ch_{ch}_{cls}_auc"] = _auc_safe(yy, ss)
        # gate distribution overall + by subgroup
        def gstats(mask):
            v = g[mask]
            if v.size == 0:
                return None
            return {"mean": float(v.mean()), "std": float(v.std()),
                    "p10": float(np.quantile(v, .1)), "p50": float(np.quantile(v, .5)),
                    "p90": float(np.quantile(v, .9))}
        res["gate_all"] = gstats(np.ones_like(pos, dtype=bool))
        res["gate_PK_pos"] = gstats(pos & (pk_pd == "PK"))
        res["gate_PD_pos"] = gstats(pos & (pk_pd == "PD"))
        # correlation of gate with PK(=1)/PD(=0) among labeled positives
        labmask = pos & np.isin(pk_pd, ["PK", "PD"])
        if labmask.sum() > 5:
            gv = g[labmask]
            is_pk = (pk_pd[labmask] == "PK").astype(float)
            res["corr_gate_PK"] = float(np.corrcoef(gv, is_pk)[0, 1])
        res["corr_mol_eff_logit"] = float(np.corrcoef(npz["ch_mol"], npz["ch_eff"])[0, 1])

    print(json.dumps(res, indent=2))
    out = run_dir / "pkpd_analysis.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[analyze] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
