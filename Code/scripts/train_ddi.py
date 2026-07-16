"""Stage 5 - UNIFIED training entry for the code-adapter RankModel pipeline.

One command trains ANY code-adapter model through ``model.runner.run_training`` (which
owns the loop + tee-logging to train.log/_logs + best-epoch select + checkpoints +
per-sample cold-test predictions + results.json). The only per-run variation is which
RankModel to build, which cold-start protocol, and the hp - all resolved here:

  --mode backbone_only  : a bare baseline wrapper (WRAPPERS[backbone]); produces the
                          model/best.pt that an adapter_frozen run later loads.
  --mode adapter_frozen : AdapterBackboneComposer over a FROZEN backbone (needs
                          --backbone-ckpt = a prior backbone_only run's model/best.pt).
  --mode adapter_joint  : AdapterBackboneComposer, backbone trained jointly with the adapter.
  --mode adapter_alone  : AdapterRankModel (H_base = structural, no backbone).

Protocol is resolved by model_meta.resolve_protocol (user-locked rule): adapter_alone ->
the METHOD's protocol; every backbone-bearing mode -> the BACKBONE's native protocol.

Examples:
  # 1) baseline (writes Code/runs/<run>/model/best.pt)
  python Code/scripts/train_ddi.py --mode backbone_only --backbone rgcn \
      --dataset drugbank_ryu --task binary --fold fold0 --epochs 1500

  # 2) +adapter frozen on that baseline (design R, correct only the final stream)
  python Code/scripts/train_ddi.py --mode adapter_frozen --backbone rgcn \
      --backbone-ckpt Code/runs/backbone_only__rgcn__drugbank_ryu__binary__fold0/model/best.pt \
      --dataset drugbank_ryu --task binary --fold fold0 --adapter-layers last --epochs 500

  # 3) +adapter joint, per-layer correction
  python Code/scripts/train_ddi.py --mode adapter_joint --backbone knowddi \
      --dataset drugbank_ryu --task multiclass --fold fold0 --adapter-layers default

  # 4) adapter-alone
  python Code/scripts/train_ddi.py --mode adapter_alone \
      --dataset drugbank_ryu --task binary --fold fold0 --epochs 800

  # arbitrary hp override (repeatable): --set lr=5e-4 --set batch_size=128 --set arm_mode=pathway
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))

import json  # noqa: E402

from specs import TaskSpec  # noqa: E402
from data.loader import load_rank_data, SUPPORTED_DATASETS, SUPPORTED_FOLDS  # noqa: E402
from model_meta import AdapterMode, BASELINES, Protocol, resolve_protocol  # noqa: E402
from model.contracts import CheckpointPolicy  # noqa: E402
from model.runner import run_training  # noqa: E402
from model.wrappers import WRAPPERS  # noqa: E402

_UNIFIED = ROOT / "Code" / "data" / "ddi_unified"
_MODES = ("backbone_only", "adapter_frozen", "adapter_joint", "adapter_alone")


def _coerce(v: str):
    """Coerce a --set value string to bool/int/float/str (in that order)."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _parse_set(pairs: list[str]) -> dict:
    out = {}
    for kv in pairs or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects k=v, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _n_classes(dataset: str) -> int:
    """Canonical multiclass K = meta.json labels.n_labels for this dataset (cold S2)."""
    meta_p = _UNIFIED / "multi_cls" / dataset / "inductive" / "S2" / "meta.json"
    if not meta_p.is_file():
        raise SystemExit(f"multiclass meta.json not found: {meta_p}")
    return int(json.loads(meta_p.read_text(encoding="utf-8"))["labels"]["n_labels"])


def _build_task(task_arg: str, dataset: str) -> TaskSpec:
    return TaskSpec.binary() if task_arg == "binary" else TaskSpec.multiclass(_n_classes(dataset))


def _build(args, task):
    """(model, hp, protocol) for the requested mode. hp merges over each model's own
    _DEFAULTS; --set overrides win last."""
    base_hp = {"dataset": args.dataset, "fold": args.fold, "seed": args.seed}
    if args.batch_size is not None:
        base_hp["batch_size"] = int(args.batch_size)
    if getattr(args, "graphsage_hoist", False):
        base_hp["graphsage_hoist"] = True           # KnowDDI reads it; other backbones ignore it
    over = _parse_set(args.set)

    # adapter-mode memory/speed knobs (NOT injected for backbone_only baselines, which stay
    # paper-faithful full-P1). max_med caps mediators; binary_pos_cap subsamples binary targets.
    def _adapter_knobs() -> dict:
        k = {"max_med": int(args.max_med)}
        if task.is_binary and args.binary_pos_cap and int(args.binary_pos_cap) > 0:
            k["binary_pos_cap"] = int(args.binary_pos_cap)
        return k

    if args.mode == "adapter_alone":
        from adapter.rank_model import AdapterRankModel
        hp = {**base_hp, "hbase": "structural", **_adapter_knobs(), **over}
        return AdapterRankModel(), hp, resolve_protocol(AdapterMode.ALONE, None)

    if args.backbone not in BASELINES:
        raise SystemExit(f"--backbone must be one of {sorted(BASELINES)} for mode {args.mode}")
    meta = BASELINES[args.backbone]

    if args.mode == "backbone_only":
        hp = {**base_hp, **over}                    # baselines stay paper-faithful (no knobs)
        for k in ("binary_pos_cap", "max_med"):     # guard: don't let --set smuggle adapter knobs in
            if k in hp:
                print(f"[train_ddi] WARN backbone_only is paper-faithful; dropping {k}={hp[k]} "
                      f"(it would alter the baseline). Use an adapter mode if you want it.", flush=True)
                hp.pop(k)
        if getattr(args, "backbone_pos_cap", None) and task.is_binary:
            hp["binary_pos_cap"] = int(args.backbone_pos_cap)   # explicit slow-baseline opt-in
            print(f"[train_ddi] WARN --backbone-pos-cap={args.backbone_pos_cap}: backbone_only will "
                  f"SUBSAMPLE {args.backbone_pos_cap} positives/epoch (negs balanced 1:1). This is a "
                  f"DELIBERATE deviation from full-P1 paper-faithful, for tractable slow baselines "
                  f"(TIGER/EmerGNN). Reproducible per-epoch negs.", flush=True)
        return WRAPPERS[args.backbone](), hp, meta.set_protocol

    # adapter_frozen | adapter_joint -> the composer (design R)
    from adapter.composer import AdapterBackboneComposer
    frozen = args.mode == "adapter_frozen"
    hp = {**base_hp, "backbone": args.backbone, "backbone_freeze": frozen,
          "adapter_layers": args.adapter_layers, **_adapter_knobs(), **over}
    if frozen:
        if not args.backbone_ckpt:
            raise SystemExit("adapter_frozen needs --backbone-ckpt (a backbone_only run's model/best.pt)")
        hp["backbone_ckpt"] = args.backbone_ckpt
    amode = AdapterMode.FROZEN if frozen else AdapterMode.JOINT
    return AdapterBackboneComposer(), hp, resolve_protocol(amode, meta)


def _run_id(args) -> str:
    bb = args.backbone if args.mode != "adapter_alone" else "adapter"
    tag = f"{args.mode}__{bb}__{args.dataset}__{args.task}__{args.fold}"
    if args.mode in ("adapter_frozen", "adapter_joint"):
        tag += f"__{args.adapter_layers}"
    return tag


def _append_index(run_dir: Path, results: dict, args, protocol, fit_s: float, status: str, log) -> None:
    """Append one row to Code/runs/_logs/index.csv (CLAUDE.md logging convention)."""
    try:
        logs = run_dir.parent / "_logs"; logs.mkdir(parents=True, exist_ok=True)
        idx = logs / "index.csv"
        ct = (results or {}).get("cold_test", {}) if results else {}
        row = {"run_id": run_dir.name, "mode": args.mode, "backbone": args.backbone or "",
               "dataset": args.dataset, "task": args.task, "fold": args.fold,
               "adapter_layers": (args.adapter_layers if args.mode.startswith("adapter_") and
                                  args.mode != "adapter_alone" else ""),
               "protocol": getattr(protocol, "value", str(protocol)),
               "epochs": args.epochs, "seed": args.seed,
               "cold_best_val": (results or {}).get("cold_best_val", ""),
               "best_epoch": (results or {}).get("best_epoch", ""),
               "primary_metric": next(iter(ct.values()), "") if ct else "",
               "fit_time_s": round(fit_s, 1), "status": status, "run_dir": str(run_dir)}
        new = not idx.is_file()
        with open(idx, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    except OSError as e:  # best-effort; never abort a run over the index row
        log(f"[warn] index.csv append failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified code-adapter training entry (stage 5).")
    ap.add_argument("--mode", required=True, choices=_MODES)
    ap.add_argument("--backbone", choices=sorted(BASELINES), default=None,
                    help="required for every mode except adapter_alone")
    ap.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))
    ap.add_argument("--task", required=True, choices=["binary", "multiclass"])
    ap.add_argument("--fold", default="fold0", choices=list(SUPPORTED_FOLDS))
    ap.add_argument("--adapter-layers", default="last", choices=["last", "default"])
    ap.add_argument("--backbone-ckpt", default=None, help="adapter_frozen: prior backbone_only best.pt")
    ap.add_argument("--epochs", type=int, default=100,
                    help="non-RGCN baselines + adapter converge by ~100; RGCN can use more")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    # -- adapter-mode memory/speed knobs (safe defaults for a 32GB card; see 2026-07-13) --
    ap.add_argument("--max-med", type=int, default=32,
                    help="adapter: cap shared mediators per pair (0=all=paper-faithful but "
                         "blows VRAM on hub pairs). 32 bounds M=batch*32. MODELING knob.")
    ap.add_argument("--binary-pos-cap", type=int, default=4000,
                    help="adapter+binary: per-epoch positive-target subsample (negs balanced "
                         "1:1 -> ~8k pairs/epoch). 0/none=full seen_ddi. MODELING knob.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the model's default batch_size (None=model default)")
    ap.add_argument("--backbone-pos-cap", type=int, default=None,
                    help="backbone_only+binary: SUBSAMPLE this many positives/epoch (negs 1:1) so a "
                         "slow per-pair baseline (TIGER/EmerGNN) is tractable. DELIBERATE deviation "
                         "from full-P1 paper-faithful; reproducible per-epoch negs. Omit=full P1.")
    ap.add_argument("--graphsage-hoist", action="store_true",
                    help="KnowDDI only: compute the full-KG GraphSAGE once per epoch (grad "
                         "accumulation) instead of per minibatch. ~11x faster/epoch (51->~5min) "
                         "but turns minibatch SGD into full-batch GD (1 step/epoch) -> needs more "
                         "epochs. Ignored by non-KnowDDI backbones. TRAINING-PROTOCOL deviation.")
    ap.add_argument("--save-last", action="store_true", help="also persist the final-epoch state")
    ap.add_argument("--run-dir", default=None, help="override the auto run dir (Code/runs/<run_id>)")
    ap.add_argument("--resume-from", default=None,
                    help="path to a prior run's model/best.pt (recommended) to LOAD + CONTINUE "
                         "training (for a run that hit the epoch cap without converging). Optimizer "
                         "state is reset (warm restart); prior best is preserved so best.pt only "
                         "improves. (last.pt also works: its cold_val is re-evaluated as the floor.) "
                         "Run this backbone_only mode with the same backbone/dataset/fold.")
    ap.add_argument("--set", action="append", default=[], metavar="k=v",
                    help="hp override (repeatable), e.g. --set lr=5e-4 --set arm_mode=pathway")
    args = ap.parse_args()

    task = _build_task(args.task, args.dataset)
    model, hp, protocol = _build(args, task)
    if hp.get("graphsage_hoist"):
        if args.backbone != "knowddi":
            print(f"[train_ddi] WARN --graphsage-hoist only affects KnowDDI; backbone="
                  f"{args.backbone} ignores it.", flush=True)
        else:
            print("[train_ddi] WARN --graphsage-hoist: KnowDDI trains as FULL-BATCH GD "
                  "(1 optimizer step/epoch, not minibatch SGD). ~11x faster/epoch but needs "
                  "more epochs to converge. Documented training-protocol deviation.", flush=True)
    if protocol is Protocol.P2_EMERGING and hp.get("binary_pos_cap"):
        print(f"[train_ddi] WARN binary_pos_cap={hp['binary_pos_cap']} has NO effect under "
              f"P2_EMERGING (native protocol here); P2's per-epoch targets are the emerging "
              f"pairs, not the full seen_ddi. Speed there depends on emerging_ratio.", flush=True)
    run_dir = Path(args.run_dir) if args.run_dir else (ROOT / "Code" / "runs" / _run_id(args))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointPolicy(save_best=True, save_last=bool(args.save_last),
                            to_disk=True, save_dir=run_dir / "model")

    print(f"[train_ddi] {run_dir.name}  model={model.__class__.__name__} "
          f"protocol={getattr(protocol, 'value', protocol)}  task={task.kind}(K={task.n_classes})",
          flush=True)
    data = load_rank_data(args.dataset, task, args.fold)
    print(f"[train_ddi] data: train_pos={len(data.seen_ddi)} cold_val={len(data.cold_val_pairs)} "
          f"cold_test={len(data.cold_test_pairs)}", flush=True)

    t0 = time.perf_counter()
    results, status = None, "failed"
    try:
        results = run_training(model, data, protocol, run_dir, hp,
                               epochs=args.epochs, eval_every=args.eval_every,
                               patience=args.patience, ckpt=ckpt, seed=args.seed,
                               resume_from=args.resume_from)
        status = "ok"
    finally:
        _append_index(run_dir, results, args, protocol, time.perf_counter() - t0, status, print)

    print(f"[train_ddi] DONE {run_dir.name} status={status} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
