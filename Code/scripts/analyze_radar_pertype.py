"""Per-DrugBank-type radar (NO retrain): the "universal improvement" figure. Each spoke = one
DrugBank interaction type; radial = per-type AUPR of {type-T positives vs all negatives}. Two
lines: backbone-alone score vs backbone+adapter (design-R) score. The adapter line envelops the
backbone line -> the adapter improves cold-start on (almost) EVERY interaction type.

This is REAL performance per mechanism type (not representation separability), matching an AUPR
radar. Types come from the multiclass drugbank_ryu labels mapped onto the binary cold-test pairs.

Usage: python Code/scripts/analyze_radar_pertype.py --ckpt <design-R best.pt> --min-pos 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "scripts"))
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
import torch                                                  # noqa: E402
from sklearn.metrics import average_precision_score           # noqa: E402

from specs import TaskSpec                                    # noqa: E402
from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import rebuild_composer              # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402


@torch.no_grad()
def scores(comp, reps, which):
    x = torch.as_tensor(reps, dtype=torch.float32, device=comp.device)
    logit = comp.head(x) if which == "design" else comp.backbone.model.out(x)
    return logit.detach().cpu().numpy().reshape(-1)


def pair_type_map(dataset, fold):
    """{(a,b) -> DrugBank type id} from the multiclass split (positives carry the interaction type)."""
    dm = load_rank_data(dataset, TaskSpec.multiclass(200), fold)
    return {tuple(map(str, p)): int(t) for p, t in zip(dm.cold_test_pairs, dm.cold_test_labels)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--min-pos", type=int, default=15)       # min type-T positives to include a spoke
    ap.add_argument("--metric", default="aupr", choices=["aupr", "auroc"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs); y = np.asarray(data.cold_test_labels).astype(int)
    p_bb = comp._encode_backbone(te, proto.fact_context(data)).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)
    s_bb = scores(comp, p_bb, "backbone")
    s_dr = scores(comp, p_prime, "design")

    tmap = pair_type_map(args.dataset, args.fold)
    ptype = np.array([tmap.get(tuple(map(str, p)), -1) for p in te])   # -1 for negatives/unmapped
    neg = y == 0
    from sklearn.metrics import average_precision_score as ap_, roc_auc_score
    scorefn = ap_ if args.metric == "aupr" else roc_auc_score

    types = [t for t in sorted(set(ptype[ptype >= 0]))
             if int(((ptype == t) & (y == 1)).sum()) >= args.min_pos]
    rows = []
    for t in types:
        pos = (ptype == t) & (y == 1)
        sel = pos | neg                                       # type-T positives vs ALL negatives
        yt = pos[sel].astype(int)
        rows.append((t, int(pos.sum()),
                     float(scorefn(yt, s_bb[sel])), float(scorefn(yt, s_dr[sel]))))
    bb_v = np.array([r[2] for r in rows]); dr_v = np.array([r[3] for r in rows])
    above = int((dr_v > bb_v).sum())
    print(f"[radar] {len(rows)} DrugBank types (>= {args.min_pos} pos) | metric={args.metric} | "
          f"adapter > backbone on {above}/{len(rows)} ({100*above/max(len(rows),1):.0f}%) | "
          f"mean {args.metric}: backbone {bb_v.mean():.3f} -> adapter {dr_v.mean():.3f} "
          f"(+{dr_v.mean()-bb_v.mean():.3f})")

    # radar
    N = len(rows)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(7.2, 7.2)); ax = fig.add_subplot(111, polar=True)
    for v, col, lab, lw, ls in [(bb_v, "#7f7f7f", "backbone-alone", 1.6, "-"),
                                (dr_v, "#d62728", "backbone + adapter", 2.0, "-")]:
        vc = np.concatenate([v, v[:1]])
        ax.plot(ang_c, vc, ls, color=col, lw=lw, label=lab)
        ax.fill(ang_c, vc, color=col, alpha=0.08)
    ax.set_xticks(ang); ax.set_xticklabels([str(r[0]) for r in rows], fontsize=7)
    ax.set_ylim(0, 1); ax.set_rlabel_position(0)
    ax.set_title(args.title or f"Per-DrugBank-type {args.metric.upper()} "
                 f"(adapter improves {above}/{N})\n{Path(args.ckpt).parent.parent.name}", fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=9)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / f"radar_{args.metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[radar] -> {out}")
    np.savez(out.with_suffix(".npz"), types=[r[0] for r in rows], npos=[r[1] for r in rows],
             backbone=bb_v, adapter=dr_v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
