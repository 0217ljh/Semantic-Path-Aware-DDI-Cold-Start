"""SPMN v2 adapter (aware_step=0) on the UNIFIED benchmark — one fold.

Ports the adapter (``spmn_v2`` aware, ``aware_step=0`` = pure-coarse: branch off,
mech off, hub gate off) from the legacy 800-drug pkl pipeline onto the project's
UNIFIED benchmark leaf so its test metric is directly comparable to the migrated
baselines. The ONLY thing that changes vs the legacy aware0 run is the data source:
instead of ``PairDataset.from_pkl`` + the 3-seed frames, we load one unified fold's
baked train/val/test parquets directly. The retrieval (merged-KG AND corridor
support), the model, and the training loop are all reused from the existing scripts
by IMPORT — no model / retrieval logic is duplicated or modified here.

Locked config (matches ``Code/runs/spmn_v2_standalone/aware0_and_seed42_lmax3_d32.json``):
  aware_step=0, support_mode="and", l_max=3, d=32, epochs=80,
  use_branch=False, use_mech=False (i.e. naked-AND coarse adapter).

Unified leaf (binary, ddi800/partial, cold S2):
  Code/data/ddi_unified/binary_cls/drugbank_latest_partial/inductive/S2/<fold>/
  train.parquet / val.parquet / test.parquet with cols
  (pair_id, drug_a_id, drug_b_id, y_bin); y_bin 1=pos, 0=neg. The baked positives
  AND negatives are the FIXED training pairs (no per-epoch negative resampling).
  Drug ids are DrugBank ids in the SAME namespace as the merged-KG node ids, so
  retrieval maps pairs -> KG nodes with no id translation. S2 val/test drug pools
  are disjoint from train (cold-start), and the merged KG is DDI-masked, so KG use
  is leakage-safe.

The support cache is keyed by FOLD (not seed) so folds do not collide and re-runs
hit the cache. It is built via the SAME ``_precompute`` machinery as the standalone
script (imported), saved next to the other spmn_v2 caches under ``Code/data/_cache``.

Example (one fold):
  python Code/scripts/run_spmn_v2_unified.py --fold fold0
  python Code/scripts/run_spmn_v2_unified.py --fold fold1 --epochs 80 --d 32 --l-max 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Sibling-script reuse: importing these modules runs their top-level sys.path
# setup + ROOT discovery but not their main() (guarded by __main__). We reuse the
# standalone support-precompute/cache machinery and the shared batch gather.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_spmn_v2_standalone import (  # noqa: E402
    CACHE_DIR, ROOT, _gather, _precompute,
)

sys.path.insert(0, str(ROOT / "Code"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_REL_BUCKETS, N_TYPES,
)
from my_code.models.spmn_v1.struct_features import symmetric_binary_dim  # noqa: E402
from my_code.models.spmn_v2 import density  # noqa: E402
from my_code.models.spmn_v2.aware_heads import DecomposedStandaloneHead  # noqa: E402
from my_code.models.spmn_v2.core import StructuralVariableCore  # noqa: E402
from my_code.utils.train_progress import TrainProgress  # noqa: E402

#: Unified benchmark leaf for the binary / ddi800-partial / cold-S2 setting.
UNIFIED_LEAF = (
    ROOT / "Code/data/ddi_unified/binary_cls/drugbank_latest_partial/inductive/S2"
)
KEYS = ("struct", "y", "med", "typ", "rela", "relb", "da", "db", "offsets")


def _load_unified_frames(fold: str) -> dict:
    """Load one unified fold's baked train/val/test pairs as adapter ``frames``.

    Maps the unified schema (``drug_a_id``, ``drug_b_id``, ``y_bin``) to the frame
    schema the downstream support-build + ``_gather`` expect (``drug_a_id``,
    ``drug_b_id``, ``label``). Positives AND baked negatives are kept verbatim (the
    fixed training pairs); no negatives are resampled. Keys mirror the aware/
    standalone frame keys (``train`` / ``val_s2`` / ``test_s2``) so cached-y row
    order maps 1:1 onto the frames row order.
    """
    fold_dir = UNIFIED_LEAF / fold
    if not fold_dir.is_dir():
        raise FileNotFoundError(
            f"unified fold not found: {fold_dir}\n  expected one of fold0..fold4 "
            f"under {UNIFIED_LEAF}")
    file_by_key = {"train": "train.parquet", "val_s2": "val.parquet",
                   "test_s2": "test.parquet"}
    frames: dict[str, pd.DataFrame] = {}
    for key, fname in file_by_key.items():
        df = pd.read_parquet(fold_dir / fname)
        if "y_bin" not in df.columns:
            raise KeyError(f"{fold_dir/fname} missing y_bin; cols={list(df.columns)}")
        fr = df[["drug_a_id", "drug_b_id"]].copy()
        fr["label"] = df["y_bin"].astype(int).to_numpy()
        frames[key] = fr.reset_index(drop=True)
    print(f"[spmn-v2-unified] frames: {fold} "
          f"train={len(frames['train'])} val_s2={len(frames['val_s2'])} "
          f"test_s2={len(frames['test_s2'])}", flush=True)
    return frames


def _support_cache_path(fold: str, l_max: int, k_per_type: int, n_max: int,
                        with_copath: bool) -> Path:
    """AND support cache keyed by FOLD (mirrors the standalone cache naming but
    substitutes ``unified_partial_S2_<fold>`` for the ``seed{N}`` token)."""
    return CACHE_DIR / (
        f"spmn_v2_supports_and_unified_partial_S2_{fold}"
        f"_lmax{l_max}_kpt{k_per_type}_nmax{n_max}_cp{int(with_copath)}.npz")


def _build_or_load_supports(frames: dict, fold: str, l_max: int, k_per_type: int,
                            n_max: int, with_copath: bool, workers: int,
                            no_cache: bool) -> dict:
    """Load the per-fold AND support cache, or build it via the standalone
    ``_precompute`` (same retrieval machinery) and save it. AND mode fixes
    ``support_and=True`` to match the legacy aware0 substrate."""
    cache = _support_cache_path(fold, l_max, k_per_type, n_max, with_copath)
    if cache.is_file() and not no_cache:
        print(f"[spmn-v2-unified] support cache HIT: {cache}", flush=True)
        z = np.load(cache)
        return {name: {k: z[f"{name}__{k}"] for k in KEYS} for name in frames}
    print(f"[spmn-v2-unified] building AND supports (l_max={l_max}, "
          f"k_per_type={k_per_type}, n_max={n_max}, with_copath={with_copath}, "
          f"workers={workers}) ...", flush=True)
    kg = None if workers > 1 else MergedKG.from_parquet()
    data = {name: _precompute(kg, fr, l_max, True, k_per_type, n_max,
                              with_copath, name, workers)
            for name, fr in frames.items()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{f"{name}__{k}": v for name, d in data.items()
                                  for k, v in d.items()})
    print(f"[spmn-v2-unified] support cache saved: {cache}", flush=True)
    return data


@torch.no_grad()
def _predict(model, split, device, batch_size) -> np.ndarray:
    """Batched sigmoid probs over a split (branch off -> no pair_feats)."""
    model.eval()
    n = len(split["y"]); probs = np.zeros(n)
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        batch, _ = _gather(split, idx, device)
        probs[idx] = torch.sigmoid(model(batch, None)).cpu().numpy()
    return probs


def _evaluate(model, split, device, batch_size) -> dict:
    probs = _predict(model, split, device, batch_size)
    y = split["y"]
    return {"auc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0",
                    choices=["fold0", "fold1", "fold2", "fold3", "fold4"],
                    help="which unified S2 fold to run (default fold0)")
    ap.add_argument("--seed", type=int, default=42,
                    help="torch/numpy seed for the training loop (data comes from "
                         "the baked fold parquets, not this seed)")
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--k-per-type", type=int, default=64)
    ap.add_argument("--n-max", type=int, default=400)
    ap.add_argument("--with-copath", action="store_true")
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
                    help="save test per-pair scores (pair_a/pair_b/y_true/y_score)")
    args = ap.parse_args()

    l_max = args.l_max
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = _load_unified_frames(args.fold)
    data = _build_or_load_supports(frames, args.fold, l_max, args.k_per_type,
                                   args.n_max, args.with_copath, args.workers,
                                   args.no_cache)

    # alignment guard: the cache stores y in the SAME frames order it was built
    # from; assert current-frames labels == cached y so a stale/reordered cache is
    # caught (mirrors the aware script's frame/cache misalignment guard).
    for name in data:
        fy = frames[name]["label"].to_numpy().astype(np.float32)
        cy = data[name]["y"].astype(np.float32)
        if len(fy) != len(cy) or not np.array_equal(fy, cy):
            raise SystemExit(
                f"[spmn-v2-unified] frame/cache MISALIGNMENT for {name}: current "
                f"frames row order does not match the support cache (labels differ). "
                f"Delete the cache and rebuild: {_support_cache_path(args.fold, l_max, args.k_per_type, args.n_max, args.with_copath)}")

    # standardise struct with TRAIN stats only (matches standalone + aware).
    mu = data["train"]["struct"].mean(0, keepdims=True)
    sd = data["train"]["struct"].std(0, keepdims=True) + 1e-6
    for name in data:
        data[name]["struct"] = ((data[name]["struct"] - mu) / sd).astype(np.float32)

    # aware_step=0 base config: clean naked AND (entity_embed on; asym/absdiff/
    # dist/hub-gate all off; branch off). Identical core_kwargs to the aware
    # script's Step 0 path, and DecomposedStandaloneHead(use_branch=False)
    # reproduces StandaloneHead exactly.
    core_kwargs = dict(
        n_entities=178029, n_types=N_TYPES, n_rel_buckets=N_REL_BUCKETS,
        struct_dim=symmetric_binary_dim(), d=args.d, hidden=args.hidden,
        dropout=args.dropout, use_entity_embed=True, use_asym=False,
        use_absdiff_embed=False, use_dist_attn=False, max_dist=l_max - 1)
    core = StructuralVariableCore(**core_kwargs)
    n_pair_feats = density.N_PAIR_FEATS
    model = DecomposedStandaloneHead(
        core, n_pair_feats=n_pair_feats, hidden=args.hidden,
        dropout=args.dropout, use_branch=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    tag = f"aware0_and_unified_partial_S2_{args.fold}"
    print(f"[spmn-v2-unified] variant={tag} l_max={l_max} d={args.d} "
          f"out_dim={core.out_dim} use_branch=False", flush=True)

    train = data["train"]; n_train = len(train["y"])
    n_steps = (n_train + args.batch - 1) // args.batch
    prog = TrainProgress(args.epochs, log_step_every=100,
                         total_steps_per_epoch=n_steps, prefix=f"[unified {tag}] ")
    best_auc, best_state, best_epoch = -1.0, None, -1
    val_curve = []  # per-epoch generalization curve (persisted per convention)
    for epoch in range(args.epochs):
        model.train(); prog.epoch_start(epoch)
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            batch, y = _gather(train, idx, device)
            loss = loss_fn(model(batch, None), y)
            opt.zero_grad(); loss.backward(); opt.step()
            prog.step(loss.item())
        val = _evaluate(model, data["val_s2"], device, args.batch)
        prog.log_eval(val, scope="epoch"); prog.epoch_end(extra={"val_auc": val["auc"]})
        val_curve.append({"epoch": epoch + 1, "val_auc": val["auc"],
                          "val_auprc": val["auprc"]})
        if val["auc"] > best_auc:
            best_auc = val["auc"]; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, data["val_s2"], device, args.batch)
    test = _evaluate(model, data["test_s2"], device, args.batch)
    print("\n=== SPMN v2 adapter aware0 on UNIFIED benchmark ===")
    print(f"  variant: {tag}  l_max={l_max}  d={args.d}")
    print(f"  best val_s2: AUC={val['auc']:.4f} AUPRC={val['auprc']:.4f}")
    print(f"  test_s2:     AUC={test['auc']:.4f} AUPRC={test['auprc']:.4f}")

    res_dir = ROOT / "Code/runs/spmn_v2_unified"; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{args.fold}.json").write_text(json.dumps({
        "fold": args.fold, "seed": args.seed, "variant": tag, "aware_step": 0,
        "support_mode": "and", "l_max": l_max, "use_branch": False,
        "use_mech": False, "n_pair_feats": n_pair_feats, "k_per_type": args.k_per_type,
        "n_max": args.n_max, "d": args.d, "epochs": args.epochs,
        "dataset_id": "drugbank_latest_partial", "split_code": "S2",
        "best_epoch": best_epoch, "val_s2": val, "test_s2": test,
        "val_curve": val_curve}, indent=2))
    print(f"  wrote {res_dir / f'{args.fold}.json'}", flush=True)

    if args.save_predictions:
        probs = _predict(model, data["test_s2"], device, args.batch)
        fr = frames["test_s2"]
        pred_path = res_dir / f"{args.fold}_test_scores.npz"
        np.savez(pred_path,
                 pair_a=fr["drug_a_id"].astype(str).to_numpy(),
                 pair_b=fr["drug_b_id"].astype(str).to_numpy(),
                 y_true=data["test_s2"]["y"].astype(np.float64),
                 y_score=probs.astype(np.float32))
        print(f"  saved test predictions -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()
