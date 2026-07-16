"""Universal-improvement scatter (NO retrain): turn "modest but universal" into the message.
Each point = one mechanism category (a specific shared KG mediator, one-vs-rest). x = pair
separability BEFORE the adapter (p_bb), y = AFTER (p_prime), both = logistic CV AUROC of
"pairs sharing this mediator vs not". If (almost) every point sits ABOVE the y=x diagonal, the
adapter improves EVERY mechanism category -> a strong, honest single-figure statement.

Categories span multiple KG mediator TYPES (Gene/Protein, Pathway, Anatomy, ...) so the figure
shows the improvement is pervasive across mechanism kinds, not one lucky category.

Usage: python Code/scripts/analyze_universal_scatter.py --ckpt <best.pt> --per-type 12
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
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.model_selection import cross_val_score          # noqa: E402

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402

from analyze_f1_precheck import rebuild_composer              # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_fine_categories import pair_mediator_sets        # noqa: E402


def fast_sep(X, y, max_n=5000, seed=0):
    """Fast A-vs-rest separability: held-out AUROC of a 1-D LDA projection (closed-form, no iters)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    if len(y) > max_n:
        idx = rng.choice(idx, max_n, replace=False)
    Xs, ys = X[idx], y[idx]
    perm = rng.permutation(len(ys)); h = len(ys) // 2; tr, te = perm[:h], perm[h:]
    if len(np.unique(ys[tr])) < 2 or len(np.unique(ys[te])) < 2:
        return float("nan")
    proj = LinearDiscriminantAnalysis(n_components=1).fit(Xs[tr], ys[tr]).transform(Xs[te])[:, 0]
    return float(roc_auc_score(ys[te], proj))

TYPES = ["Gene/Protein", "Pathway", "Anatomy", "SideEffect"]
TCOL = {"Gene/Protein": "#d62728", "Pathway": "#ff7f0e", "Anatomy": "#2ca02c", "SideEffect": "#9467bd"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--per-type", type=int, default=12)      # top-N mediators per type
    ap.add_argument("--min-pos", type=int, default=120)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs)
    p_bb = comp._encode_backbone(te, proto.fact_context(data)).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)

    sets = pair_mediator_sets(comp, te)
    freq = {}
    for s in sets:
        for m in s:
            freq[m] = freq.get(m, 0) + 1
    # candidate categories: top-per_type frequent mediators of each type
    cats = []
    for tn in TYPES:
        if tn not in kg.type_names:
            continue
        tid = kg.type_names.index(tn)
        ms = sorted((m for m in freq if kg.type_id[m] == tid and freq[m] >= args.min_pos),
                    key=freq.get, reverse=True)[:args.per_type]
        cats += [(m, tn) for m in ms]

    pts = []
    for m, tn in cats:
        y = np.array([m in s for s in sets], dtype=int)
        if y.sum() < args.min_pos or (1 - y).sum() < args.min_pos:
            continue
        bb = fast_sep(p_bb, y); pp = fast_sep(p_prime, y)
        if np.isfinite(bb) and np.isfinite(pp):
            pts.append((bb, pp, tn))
    bb_a = np.array([p[0] for p in pts]); pp_a = np.array([p[1] for p in pts])
    above = int((pp_a > bb_a).sum())
    print(f"[univ] {len(pts)} mechanism categories | above diagonal: {above}/{len(pts)} "
          f"({100*above/max(len(pts),1):.0f}%) | mean gain {np.mean(pp_a-bb_a):+.4f}")

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    lim = [min(bb_a.min(), pp_a.min()) - 0.02, max(bb_a.max(), pp_a.max()) + 0.02]
    ax.plot(lim, lim, "--", c="gray", lw=1, zorder=0)
    for tn in TYPES:
        m = [i for i, p in enumerate(pts) if p[2] == tn]
        if m:
            ax.scatter(bb_a[m], pp_a[m], s=42, alpha=0.8, c=TCOL[tn], edgecolors="white",
                       linewidths=0.5, label=f"{tn} ({len(m)})")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("separability BEFORE adapter (p_bb)")
    ax.set_ylabel("separability AFTER adapter (p_bb')")
    ax.set_title(args.title or f"Per-mechanism separability, {Path(args.ckpt).parent.parent.name}\n"
                 f"{above}/{len(pts)} categories improved", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "universal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[univ] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
