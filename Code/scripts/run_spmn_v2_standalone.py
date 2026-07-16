"""SPMN v2 — standalone adapter A/B (asymmetric vs non-asymmetric hyper-edge).

Trains ``StandaloneHead`` over ``StructuralVariableCore`` on cold-start S2 binary
DDI, mirroring the phase-2c loop but with selectable support mode and the
asymmetric-stratification switch. Used to answer "does asymmetric common-
reachability help on its own, before any backbone fusion".

Recommended A/B (the clean within-support test of the asymmetry mechanism):
  # SUM support, asymmetric OFF (baseline). First run builds the support cache:
  python Code/scripts/run_spmn_v2_standalone.py --support-mode sum --no-asym \
      --k-per-type 128 --n-max 1200 --workers 8
  # SUM support, asymmetric ON (cache HIT -> trains immediately):
  python Code/scripts/run_spmn_v2_standalone.py --support-mode sum --use-asym \
      --k-per-type 128 --n-max 1200 --workers 8

The support precompute is a CPU loop over ~115k pairs (depth-4 BFS + struct
features). It is single-threaded unless ``--workers > 1``, which fans the
per-pair work across processes (each loads its own KG). Progress prints every
2000 pairs with an ETA.

Note on the cap: ``build_pair_support`` ranks mediators by
``-(d_a + d_b) - deg_penalty*log(deg)`` and keeps top ``k_per_type`` / ``n_max``.
This down-ranks high-sum ASYMMETRIC mediators, so for the SUM experiment raise
``--k-per-type`` / ``--n-max`` (e.g. 128 / 1200) or the asymmetric cells get
capped away before the model ever sees them.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_REL_BUCKETS, N_TYPES, build_pair_support,
)
from my_code.models.spmn_v1.struct_features import (  # noqa: E402
    compute_struct_features, symmetric_binary_dim, symmetric_binary_vector,
)
from my_code.models.spmn_v2 import StandaloneHead, StructuralVariableCore, SupportBatch  # noqa: E402
from my_code.utils.train_progress import TrainProgress  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
THREE_SEED_DIR = ROOT / "Code/data/coldddi_legacy/800drug_3seed"
CACHE_DIR = ROOT / "Code/data/_cache"


def _load_frames(seed: int) -> dict:
    """Prefer the regenerated 3-seed binary frames (seeds 42/43/44, baked
    pos+neg parquet); fall back to the legacy single-seed pkl for seed 42."""
    d = THREE_SEED_DIR / f"seed{seed}"
    if d.is_dir():
        print(f"[spmn-v2] frames: 3-seed parquet {d}", flush=True)
        return {name: pd.read_parquet(d / f"{name}.parquet")
                for name in ("train", "val_s2", "test_s2")}
    print(f"[spmn-v2] frames: legacy pkl seed{seed}", flush=True)
    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{seed}.pkl"))
    return {"train": _binary_frame(ds.splits.train, ds.get_train_negatives()),
            "val_s2": _binary_frame(ds.splits.val_s2, ds.get_negatives("val_s2")),
            "test_s2": _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))}

# Per-worker KG handle (set by _worker_init under spawn, or directly in the
# single-process path). build_pair_support / struct features are read-only on
# the KG, so this is safe to share/parallelise.
_KG: MergedKG | None = None


def _worker_init() -> None:
    global _KG
    if _KG is None:
        _KG = MergedKG.from_parquet()


def _worker_pair(task):
    """Compute one pair's struct vector + mediator arrays. Returns None for OOV."""
    a_id, b_id, l_max, support_and, k_per_type, n_max, with_copath = task
    kg = _KG
    ai = kg.id_to_idx.get(a_id); bi = kg.id_to_idx.get(b_id)
    if ai is None or bi is None or ai == bi:
        return None
    sup = build_pair_support(kg, ai, bi, l_max=l_max, support_and=support_and,
                             k_per_type=k_per_type, n_max=n_max)
    sv = symmetric_binary_vector(compute_struct_features(kg, sup, with_copath=with_copath))
    if sup.n_support:
        ra, rb = kg.support_relations(sup)
        return (sv, sup.global_idx.astype(np.int64), sup.type_id.astype(np.int64),
                ra, rb, sup.d_a.astype(np.int64), sup.d_b.astype(np.int64))
    return (sv, None, None, None, None, None, None)


def _binary_frame(pos, neg):
    p = pos[["drug_a_id", "drug_b_id"]].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _absorb(res, i, struct, bins, counts):
    if res is None:
        return
    sv, med, typ, ra, rb, da, db = res
    struct[i] = sv
    if med is not None and len(med):
        bins["med"].append(med); bins["typ"].append(typ)
        bins["rela"].append(ra); bins["relb"].append(rb)
        bins["da"].append(da); bins["db"].append(db)
        counts[i] = len(med)


def _precompute(kg, frame, l_max, support_and, k_per_type, n_max, with_copath,
                tag, workers):
    global _KG
    n = len(frame)
    struct = np.zeros((n, symmetric_binary_dim()), dtype=np.float32)
    y = frame["label"].to_numpy().astype(np.float32)
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    tasks = [(a_ids[i], b_ids[i], l_max, support_and, k_per_type, n_max, with_copath)
             for i in range(n)]
    bins = {k: [] for k in ("med", "typ", "rela", "relb", "da", "db")}
    counts = np.zeros(n, dtype=np.int64)
    t0 = time.time()

    def _tick(i):
        if (i + 1) % 2000 == 0 or i + 1 == n:
            el = time.time() - t0; rate = (i + 1) / max(el, 1e-9)
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  [{tag}] {i+1}/{n} {el:.0f}s ({rate:.0f}/s eta {eta:.0f}s)", flush=True)

    if workers and workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_worker_init) as ex:
            for i, res in enumerate(ex.map(_worker_pair, tasks, chunksize=256)):
                _absorb(res, i, struct, bins, counts); _tick(i)
    else:
        _KG = kg
        for i, task in enumerate(tasks):
            _absorb(_worker_pair(task), i, struct, bins, counts); _tick(i)

    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"  [{tag}] done {n} in {time.time()-t0:.0f}s", flush=True)
    return {"struct": struct, "y": y, "med": cat(bins["med"]), "typ": cat(bins["typ"]),
            "rela": cat(bins["rela"]), "relb": cat(bins["relb"]),
            "da": cat(bins["da"]), "db": cat(bins["db"]), "offsets": offsets}


def _gather(split, idx, device):
    off = split["offsets"]
    med, typ, rela, relb, da, db = (split["med"], split["typ"], split["rela"],
                                    split["relb"], split["da"], split["db"])
    mL, tL, raL, rbL, daL, dbL, pL = [], [], [], [], [], [], []
    for local, p in enumerate(idx):
        s, e = off[p], off[p + 1]
        if e > s:
            mL.append(med[s:e]); tL.append(typ[s:e])
            raL.append(rela[s:e]); rbL.append(relb[s:e])
            daL.append(da[s:e]); dbL.append(db[s:e])
            pL.append(np.full(e - s, local, dtype=np.int64))
    t = lambda L: torch.as_tensor(np.concatenate(L), device=device) if L \
        else torch.zeros(0, dtype=torch.long, device=device)
    struct = torch.as_tensor(split["struct"][idx], device=device)
    y = torch.as_tensor(split["y"][idx], device=device)
    batch = SupportBatch(med_id=t(mL), pair_idx=t(pL), type_idx=t(tL), rel_a=t(raL),
                         rel_b=t(rbL), d_a=t(daL), d_b=t(dbL), struct_feats=struct,
                         n_pairs=len(idx))
    return batch, y


@torch.no_grad()
def _predict(model, split, device, batch_size):
    model.eval()
    n = len(split["y"]); probs = np.zeros(n)
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        batch, _ = _gather(split, idx, device)
        probs[idx] = torch.sigmoid(model(batch)).cpu().numpy()
    return probs


def _evaluate(model, split, device, batch_size):
    probs = _predict(model, split, device, batch_size)
    y = split["y"]
    return {"auc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--support-mode", choices=["and", "sum"], default="sum")
    ap.add_argument("--l-max", type=int, default=None,
                    help="default: 3 for AND (=> box l<=2), 5 for SUM (=> d_a+d_b<=5)")
    ap.add_argument("--use-asym", dest="use_asym", action="store_true")
    ap.add_argument("--no-asym", dest="use_asym", action="store_false")
    ap.set_defaults(use_asym=False)
    ap.add_argument("--no-entity-embed", dest="use_entity_embed",
                    action="store_false",
                    help="drop the free per-entity embedding -> purely structural "
                         "message (type+relation+dist). Tests whether entity_embed "
                         "is the memorisation/collapse driver.")
    ap.set_defaults(use_entity_embed=True)
    ap.add_argument("--use-absdiff-embed", action="store_true")
    ap.add_argument("--use-dist-attn", action="store_true",
                    help="distance-conditioned attention: learned zero-init bias "
                         "over the (d_a,d_b) grid added to within-tau attention "
                         "logits (soft dynamic receptive field). Uses cached "
                         "d_a/d_b — no support rebuild needed.")
    ap.add_argument("--k-per-type", type=int, default=64)
    ap.add_argument("--n-max", type=int, default=400)
    ap.add_argument("--with-copath", action="store_true",
                    help="include cross-type co-path struct features. OFF by "
                         "default: on SUM supports it does a BFS per mediator "
                         "(~hundreds/pair) and is intractable. Orthogonal to the "
                         "asymmetric-pooling mechanism this script tests.")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes for the support precompute (1 = single-thread)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save-predictions", action="store_true",
                    help="save test_s2 per-pair scores (pair_a/pair_b/y_true/"
                         "y_score) for E0 complementarity vs NBFNet")
    args = ap.parse_args()

    support_and = args.support_mode == "and"
    l_max = args.l_max if args.l_max is not None else (3 if support_and else 5)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = _load_frames(args.seed)

    cache = CACHE_DIR / (f"spmn_v2_supports_{args.support_mode}_seed{args.seed}"
                         f"_lmax{l_max}_kpt{args.k_per_type}_nmax{args.n_max}"
                         f"_cp{int(args.with_copath)}.npz")
    keys = ("struct", "y", "med", "typ", "rela", "relb", "da", "db", "offsets")
    if cache.is_file() and not args.no_cache:
        print(f"[spmn-v2] support cache HIT: {cache}", flush=True)
        z = np.load(cache)
        data = {name: {k: z[f"{name}__{k}"] for k in keys} for name in frames}
    else:
        print(f"[spmn-v2] building {args.support_mode} supports (l_max={l_max}, "
              f"k_per_type={args.k_per_type}, n_max={args.n_max}, "
              f"with_copath={args.with_copath}, workers={args.workers}) ...", flush=True)
        kg = None if args.workers > 1 else MergedKG.from_parquet()
        data = {name: _precompute(kg, fr, l_max, support_and, args.k_per_type,
                                  args.n_max, args.with_copath, name, args.workers)
                for name, fr in frames.items()}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{f"{name}__{k}": v for name, d in data.items()
                                      for k, v in d.items()})
        print(f"[spmn-v2] support cache saved: {cache}", flush=True)

    mu = data["train"]["struct"].mean(0, keepdims=True)
    sd = data["train"]["struct"].std(0, keepdims=True) + 1e-6
    for name in data:
        data[name]["struct"] = ((data[name]["struct"] - mu) / sd).astype(np.float32)

    core = StructuralVariableCore(
        n_entities=178029, n_types=N_TYPES, n_rel_buckets=N_REL_BUCKETS,
        struct_dim=symmetric_binary_dim(), d=args.d, hidden=args.hidden,
        dropout=args.dropout, use_entity_embed=args.use_entity_embed,
        use_asym=args.use_asym, use_absdiff_embed=args.use_absdiff_embed,
        use_dist_attn=args.use_dist_attn, max_dist=l_max - 1)
    model = StandaloneHead(core, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    tag = (f"{args.support_mode}_ent{int(args.use_entity_embed)}"
           f"_asym{int(args.use_asym)}_absdiff{int(args.use_absdiff_embed)}"
           f"_dist{int(args.use_dist_attn)}")
    print(f"[spmn-v2] variant={tag} l_max={l_max} d={args.d} out_dim={core.out_dim}", flush=True)

    train = data["train"]; n_train = len(train["y"])
    n_steps = (n_train + args.batch - 1) // args.batch
    prog = TrainProgress(args.epochs, log_step_every=100,
                         total_steps_per_epoch=n_steps, prefix=f"[spmn-v2 {tag}] ")
    best_auc, best_state, best_epoch = -1.0, None, -1
    val_curve = []  # per-epoch generalization curve (persisted -> see collapse offline)
    for epoch in range(args.epochs):
        model.train(); prog.epoch_start(epoch)
        perm = np.random.permutation(n_train)
        ep_loss = 0.0
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            batch, y = _gather(train, idx, device)
            loss = loss_fn(model(batch), y)
            opt.zero_grad(); loss.backward(); opt.step()
            prog.step(loss.item()); ep_loss += loss.item()
        val = _evaluate(model, data["val_s2"], device, args.batch)
        prog.log_eval(val, scope="epoch"); prog.epoch_end(extra={"val_auc": val["auc"]})
        val_curve.append({"epoch": epoch + 1, "val_auc": val["auc"],
                          "val_auprc": val["auprc"]})
        if val["auc"] > best_auc:
            best_auc = val["auc"]; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, data["val_s2"], device, args.batch)
    test = _evaluate(model, data["test_s2"], device, args.batch)
    print("\n=== SPMN v2 standalone (asymmetric adapter A/B) ===")
    print(f"  variant: {tag}  l_max={l_max}")
    print(f"  best val_s2: AUC={val['auc']:.4f} AUPRC={val['auprc']:.4f}")
    print(f"  test_s2:     AUC={test['auc']:.4f} AUPRC={test['auprc']:.4f}")
    print("  refs: AND no-rel pool 0.7720 | rel-head 0.7794 | NBFNet merged 0.7943")
    res_dir = ROOT / "Code/runs/spmn_v2_standalone"; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{tag}_seed{args.seed}_lmax{l_max}_d{args.d}.json").write_text(json.dumps({
        "seed": args.seed, "variant": tag, "support_mode": args.support_mode,
        "l_max": l_max, "use_asym": args.use_asym,
        "use_absdiff_embed": args.use_absdiff_embed, "k_per_type": args.k_per_type,
        "n_max": args.n_max, "d": args.d, "epochs": args.epochs,
        "best_epoch": best_epoch, "val_s2": val, "test_s2": test,
        "val_curve": val_curve}, indent=2))

    if args.save_predictions:
        probs = _predict(model, data["test_s2"], device, args.batch)
        fr = frames["test_s2"]
        pred_path = res_dir / f"{tag}_seed{args.seed}_test_s2_scores.npz"
        np.savez(pred_path,
                 pair_a=fr["drug_a_id"].astype(str).to_numpy(),
                 pair_b=fr["drug_b_id"].astype(str).to_numpy(),
                 y_true=data["test_s2"]["y"].astype(np.float64),
                 y_score=probs.astype(np.float32))
        print(f"  saved test_s2 predictions -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()
