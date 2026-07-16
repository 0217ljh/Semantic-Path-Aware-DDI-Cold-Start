"""R-GCN FACT-protocol: (checkpoint x rank) -> AUROC surface.

Probes EVERY milestone checkpoint during training to see the rank DYNAMICS
(user 2026-07-05: expect warm's usable rank to GROW with training while cold's
SHRINKS — cold peaks early). At log-spaced training steps we extract train/warm/
cold pair reps on the fixed FACT KG and compute BOTH the transfer probe
(PCA+logreg fit on Z_train, applied to warm/cold) and the within-split probe
(CV inside each split) AUROC-vs-rank curve. Output = a (milestone x rank) surface
per {probe, split} plus a 3D plot.

Reuses train_rgcn_rank_fact_protocol's split + fact-KG logic and the two analyze
scripts' probe math (no re-implementation of the model / data protocol).

Run (project root, WSL conda env project_1):
  python Code/scripts/train_rgcn_rank_epoch_surface.py --fold fold0 --epochs 600 --amp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("root")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
from train_rgcn_rank_pilot import build_relation_edges, RGCNPairModel  # noqa: E402
from train_gcn_rank_pilot import load_data, _pairs_to_idx  # noqa: E402
from train_rgcn_rank_fact_protocol import _split_fact_target, _canon_pos_set  # noqa: E402
from analyze_rank_within_split import _within_split_curve, _balance  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _transfer_curve(Ztr, ytr, Zw, yw, Zc, yc, ranks):
    mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-8
    Str, Sw, Sc = (Ztr - mu) / sd, (Zw - mu) / sd, (Zc - mu) / sd
    center = Str.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Str - center, full_matrices=False)
    ytr = ytr.astype(int)
    warm, cold = [], []
    for r in ranks:
        Ptr = (Str - center) @ Vt[:r].T
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ptr, ytr)
        warm.append(float(roc_auc_score(yw, clf.predict_proba((Sw - center) @ Vt[:r].T)[:, 1])))
        cold.append(float(roc_auc_score(yc, clf.predict_proba((Sc - center) @ Vt[:r].T)[:, 1])))
    return warm, cold


def _within_curve(Z, y, ranks):
    Zb, yb = _balance(Z, y)
    rks, cur = _within_split_curve(Zb, yb, ranks, n_splits=3, seed=0)
    return rks, [cur["mean"][r] for r in rks]


def _milestones(total: int) -> list[int]:
    early = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    reg = list(range(100, total + 1, 100))
    return sorted({s for s in (early + reg + [total]) if 1 <= s <= total})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=600, help="optimizer STEPS (full-batch)")
    ap.add_argument("--fact-frac", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-bases", type=int, default=16)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", type=int, nargs="*", default=RANKS)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}", flush=True)

    kg = MergedKG.from_parquet(DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
    n = kg.n_nodes; id2i = kg.id_to_idx
    logdeg_bio = np.log1p(kg.degree.astype(np.float64))
    x = np.zeros((n, N_TYPES + 1), dtype=np.float32)
    x[np.arange(n), kg.type_id.astype(np.int64)] = 1.0
    x[:, N_TYPES] = ((logdeg_bio - logdeg_bio.mean()) / (logdeg_bio.std() + 1e-8)).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    edge_index, edge_type, num_relations, rel_names, _ = build_relation_edges(DEFAULT_EDGES_PATH, id2i)

    tr, warm_val, warm_test, cold_test = load_data(args.fold, args.warm_frac, args.seed)
    fact_pos, train_target, _, _ = _split_fact_target(tr, args.fact_frac, args.seed)
    # leak guard
    assert not (_canon_pos_set(fact_pos) & _canon_pos_set(warm_test)), "fact/warm leak"
    assert not (_canon_pos_set(fact_pos) & _canon_pos_set(cold_test)), "fact/cold leak"

    fa = fact_pos["drug_a_id"].astype(str).map(id2i).to_numpy(np.int64)
    fb = fact_pos["drug_b_id"].astype(str).map(id2i).to_numpy(np.int64)
    ddi_rel = num_relations
    aug_ei = torch.cat([edge_index, torch.from_numpy(
        np.vstack([np.concatenate([fa, fb]), np.concatenate([fb, fa])]))], dim=1).to(device)
    aug_et = torch.cat([edge_type, torch.from_numpy(
        np.full(2 * len(fa), ddi_rel, dtype=np.int64))]).to(device)
    aug_nr = num_relations + 1
    print(f"[kg] nodes={n} aug_relations={aug_nr} aug_edges={aug_ei.shape[1]} "
          f"fact_ddi={len(fa)}", flush=True)

    ia_tr, ib_tr, y_tr = _pairs_to_idx(train_target, id2i)
    packs = {"train": (ia_tr, ib_tr, y_tr),
             "warm": _pairs_to_idx(warm_test, id2i),
             "cold": _pairs_to_idx(cold_test, id2i)}
    ia_tr_t = torch.as_tensor(ia_tr, device=device); ib_tr_t = torch.as_tensor(ib_tr, device=device)
    y_tr_t = torch.as_tensor(y_tr, device=device)

    model = RGCNPairModel(x.shape[1], args.hidden, args.bottleneck, args.dropout,
                          num_relations=aug_nr, num_bases=args.num_bases,
                          num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT / "Code" / "runs" / f"{ts}__train_rgcn_rank_epoch_surface__seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _extract_curves(step):
        model.eval()
        h = model.encode(x, aug_ei, aug_et)
        Z = {}
        for name, (ia, ib, y) in packs.items():
            hu, hv = h[torch.as_tensor(ia, device=device)], h[torch.as_tensor(ib, device=device)]
            raw = torch.cat([hu * hv, (hu - hv).abs()], dim=-1).cpu().numpy().astype(np.float32)
            Z[name] = (raw, y)
        tw, tc = _transfer_curve(Z["train"][0], Z["train"][1], Z["warm"][0], Z["warm"][1],
                                 Z["cold"][0], Z["cold"][1], args.ranks)
        _, ww = _within_curve(Z["warm"][0], Z["warm"][1], args.ranks)
        _, wc = _within_curve(Z["cold"][0], Z["cold"][1], args.ranks)
        model.train()
        return {"transfer_warm": tw, "transfer_cold": tc, "within_warm": ww, "within_cold": wc}

    mile = _milestones(args.epochs)
    print(f"[milestones] {mile}", flush=True)
    surface = {"steps": mile, "ranks": args.ranks,
               "transfer_warm": [], "transfer_cold": [], "within_warm": [], "within_cold": []}
    mile_set = set(mile)
    for step in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(x, aug_ei, aug_et, ia_tr_t, ib_tr_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr_t)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step in mile_set:
            c = _extract_curves(step)
            for k in ("transfer_warm", "transfer_cold", "within_warm", "within_cold"):
                surface[k].append(c[k])
            print(f"[step {step}] loss={loss.item():.4f} "
                  f"tW={c['transfer_warm'][-1]:.3f} tC={c['transfer_cold'][-1]:.3f} "
                  f"wW={c['within_warm'][-1]:.3f} wC={c['within_cold'][-1]:.3f}", flush=True)

    (run_dir / "surface.json").write_text(json.dumps(surface, indent=2))

    # -- 3D plot: 2x2 (rows=probe, cols=split) --------------------------------
    S = np.array(surface["steps"], dtype=float)
    Rk = np.array(args.ranks, dtype=float)
    Xg, Yg = np.meshgrid(np.log10(S), np.log2(Rk), indexing="ij")
    fig = plt.figure(figsize=(13, 10))
    panels = [("transfer_warm", "transfer  WARM"), ("transfer_cold", "transfer  COLD"),
              ("within_warm", "within-split  WARM"), ("within_cold", "within-split  COLD")]
    for i, (key, title) in enumerate(panels):
        Zg = np.array(surface[key])  # (n_steps, n_ranks)
        axp = fig.add_subplot(2, 2, i + 1, projection="3d")
        axp.plot_surface(Xg, Yg, Zg, cmap="viridis", edgecolor="none", alpha=0.9)
        axp.set_xlabel("log10(step)"); axp.set_ylabel("log2(rank)"); axp.set_zlabel("AUROC")
        axp.set_title(title); axp.view_init(elev=25, azim=-60)
    fig.suptitle("R-GCN FACT-protocol: (training step x rank) -> AUROC")
    fig.tight_layout()
    fig.savefig(run_dir / "rank_epoch_surface.png", dpi=140)
    print(f"[done] {run_dir}\n  surface.json + rank_epoch_surface.png", flush=True)
    print("expect: warm saturation-rank GROWS with step; cold's SHRINKS / peaks early", flush=True)


if __name__ == "__main__":
    main()
