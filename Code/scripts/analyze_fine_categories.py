"""Fine-grained category sweep (NO retrain): find the SPECIFIC mechanism categories on which the
adapter visibly separates pairs. Categories = "pairs sharing a SPECIFIC KG mediator" (e.g. a
specific enzyme like CYP3A4 vs a specific transporter) -> very fine, biologically interpretable.
For every pair of top-frequency shared mediators (A,B) we take disjoint groups (share A not B /
share B not A) and measure A-vs-B linear separability (logistic CV AUROC) in p_bb vs p_prime.
The categories with the biggest p_prime-p_bb separability gain are the ones to plot.

Only SOME categories need to show the effect (per PI). Ranks + renders the top-improving pairs.

Usage: python Code/scripts/analyze_fine_categories.py --ckpt <best.pt> --top-med 14 --render 4
"""
from __future__ import annotations

import argparse
import itertools
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import rebuild_composer              # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from kg.mediators import shared_mediators_idx                 # noqa: E402


def pair_mediator_sets(comp, pairs):
    kg = comp.adapter_model.kg; nbhd = comp.adapter_model.nbhd
    hp = comp.adapter_model.hp; mode = str(hp.get("med_mode", "and")); tau = int(hp.get("tau", 2))
    sets = []
    for u, v in pairs:
        ui, vi = kg.get_idx(str(u)), kg.get_idx(str(v))
        if ui is None or vi is None:
            sets.append(set()); continue
        med = shared_mediators_idx(kg, nbhd, int(ui), int(vi), mode=mode, tau=tau)["med_idx"]
        sets.append(set(int(m) for m in med))
    return sets


def sep_auroc(X, y, folds=3):
    """A-vs-B linear separability (logistic CV AUROC) — how separable the two mechanism groups are."""
    clf = LogisticRegression(max_iter=200, C=1.0)
    try:
        return float(cross_val_score(clf, X, y, cv=folds, scoring="roc_auc").mean())
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--top-med", type=int, default=14)       # candidate mediators by pair-frequency
    ap.add_argument("--med-type", default="Gene/Protein")    # restrict candidates to this node type
    ap.add_argument("--max-bb", type=float, default=1.0)     # only keep pairs with bb sep <= this
    ap.add_argument("--min-grp", type=int, default=60)        # min disjoint group size
    ap.add_argument("--render", type=int, default=4)
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
    if args.med_type:                                        # restrict to one node type (e.g. proteins)
        tid = kg.type_names.index(args.med_type)
        freq = {m: c for m, c in freq.items() if kg.type_id[m] == tid}
    top = sorted(freq, key=freq.get, reverse=True)[:args.top_med]
    names = {m: (str(kg.idx2node[m]), kg.type_names[kg.type_id[m]]) for m in top}
    print(f"[fine] top-{len(top)} shared mediators (name, type, #pairs):")
    for m in top:
        print(f"    {names[m][0]:<24} {names[m][1]:<12} {freq[m]}")

    # sweep disjoint mediator PAIRS -> separability gain
    results = []
    for a, b in itertools.combinations(top, 2):
        ga = np.array([a in s and b not in s for s in sets])
        gb = np.array([b in s and a not in s for s in sets])
        if ga.sum() < args.min_grp or gb.sum() < args.min_grp:
            continue
        idx = np.where(ga | gb)[0]
        y = ga[idx].astype(int)
        aur_bb = sep_auroc(p_bb[idx], y)
        if aur_bb > args.max_bb:                             # focus on categories the backbone struggles with
            continue
        aur_pp = sep_auroc(p_prime[idx], y)
        results.append((aur_pp - aur_bb, aur_bb, aur_pp, a, b, int(ga.sum()), int(gb.sum())))
    results.sort(reverse=True)
    print(f"\n=== A-vs-B separability gain (logistic CV AUROC, p_prime - p_bb) — top pairs ===")
    print(f"{'gain':>7} {'bb':>6} {'plug':>6} {'nA':>5} {'nB':>5} | mechanism A  vs  B")
    for gain, ab, ap_, a, b, na, nb in results[:12]:
        print(f"{gain:>+7.3f} {ab:>6.3f} {ap_:>6.3f} {na:>5} {nb:>5} | "
              f"{names[a][0]}({names[a][1]}) vs {names[b][0]}({names[b][1]})")

    # render the top-improving mechanism pairs (LDA-2, p_bb vs p_prime)
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters"
    out.mkdir(parents=True, exist_ok=True)
    for rank, (gain, ab, ap_, a, b, na, nb) in enumerate(results[:args.render]):
        ga = np.array([a in s and b not in s for s in sets])
        gb = np.array([b in s and a not in s for s in sets])
        idx = np.where(ga | gb)[0]; y = ga[idx].astype(int)
        na_, nb_ = names[a][0], names[b][0]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
        for k, (X, ttl) in enumerate([(p_bb[idx], "M_bb (before)"), (p_prime[idx], "M_plug (after)")]):
            Z = LinearDiscriminantAnalysis(n_components=1).fit(X, y).transform(X)[:, 0]
            r = X - X.mean(0)
            d = LinearDiscriminantAnalysis(n_components=1).fit(X, y).coef_[0]; d = d / (np.linalg.norm(d)+1e-9)
            res = r - np.outer(r @ d, d); _, _, Vt = np.linalg.svd(res, full_matrices=False)
            XY = np.column_stack([Z, res @ Vt[0]])
            for c, nm, col in [(0, nb_, "#1f77b4"), (1, na_, "#d62728")]:
                m = y == c
                ax[k].scatter(XY[m, 0], XY[m, 1], s=6, alpha=0.5, c=col, label=nm, linewidths=0)
            ax[k].set_title(ttl, fontsize=10); ax[k].set_xticks([]); ax[k].set_yticks([])
        ax[1].legend(markerscale=2, fontsize=8)
        fig.suptitle(f"{names[a][0]} ({names[a][1]}) vs {names[b][0]} ({names[b][1]})  "
                     f"sep AUROC {ab:.2f}->{ap_:.2f}", fontsize=10)
        fig.tight_layout(); png = out / f"fine_{rank}_{na_}_vs_{nb_}.png".replace("/", "-")
        fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"[fine] rendered -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
