"""Run a baseline on 800-drug seed42 PKL, evaluate S0/S1/S2, save results.

Saves under: Code/runs/<run_id>/   (run_id = <timestamp>__run_baseline__<tag>__seed<N>)
  - results.json  (AUC/NLL per setting + config)
  - predictions_s0.parquet / s1 / s2
  - train.log     (full stdout+stderr captured)

Mirror log under: Code/runs/_logs/<run_id>.log
Index row appended to: Code/runs/_logs/index.csv

Example:
  python run_baseline.py --baseline emergnn --backbone-kg-source merged \\
      --epochs 5 --batch 16 --n-dim 32 --tag emergnn_merged_e5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))  # so `from my_code.utils...` resolves

from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
MERGED_KG = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"


def _normalize_baseline(arg: str) -> tuple[str, str]:
    """Resolve --baseline arg into (base_name, task).

    Task suffix convention (per user spec 2026-05-18):
      * bare name (e.g. ``emergnn``)   → defaults to ``bc`` (binary cls)
      * ``<name>-bc``                  → binary classification (explicit)
      * ``<name>-mcc``                 → multi-class classification (K-way single-label)
      * ``<name>-mlc``                 → multi-label classification (reserved, not yet wired)
      * ``<name>_mc`` (legacy)         → DEPRECATED alias for ``-mcc``,
                                         still accepted with stderr warning

    Returns:
        (base_name, task) where task in {"bc", "mcc", "mlc"}.
    """
    if arg.endswith("-bc"):
        return arg[: -len("-bc")], "bc"
    if arg.endswith("-mcc"):
        return arg[: -len("-mcc")], "mcc"
    if arg.endswith("-mlc"):
        return arg[: -len("-mlc")], "mlc"
    if arg.endswith("_mc"):
        new = arg[: -len("_mc")] + "-mcc"
        print(
            f"[run_baseline] DEPRECATION: '--baseline {arg}' is the old "
            f"underscore form; use '--baseline {new}' instead. (Still works "
            f"for now to keep old scripts running.)",
            file=sys.stderr,
            flush=True,
        )
        return arg[: -len("_mc")], "mcc"
    return arg, "bc"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",
                   choices=[
                       # Bare name = binary cls (default task)
                       "emergnn", "hdn_ddi", "tiger",
                       # Explicit -bc = binary cls (same as bare)
                       "emergnn-bc", "hdn_ddi-bc", "tiger-bc",
                       # -mcc = multi-class single-label (K-way)
                       "emergnn-mcc", "hdn_ddi-mcc", "tiger-mcc",
                       # Legacy _mc = DEPRECATED alias for -mcc (kept for back-compat)
                       "emergnn_mc", "hdn_ddi_mc", "tiger_mc",
                       # DEPRECATED 2026-05-20: emergnn-multimode was promoted
                       # to the official "emergnn" binary baseline; the alias
                       # is still accepted (re-routes to "emergnn") for any
                       # in-flight scripts. emergnn-kgonly stays accepted
                       # but routes to the deprecated reference experiment
                       # at baseline/emergnn/deprecate/binary_cls_kgonly/.
                       "emergnn-multimode", "emergnn-kgonly",
                   ],
                   required=True,
                   help="Baseline + task. Format: '<name>' (default binary) or "
                        "'<name>-bc' (binary) or '<name>-mcc' (multi-class). "
                        "Legacy '<name>_mc' kept as deprecated alias. "
                        "Future: '<name>-mlc' for multi-label cls (reserved).")
    p.add_argument("--backbone-kg-source", "--kg-source", dest="backbone_kg_source", choices=["drugbank", "merged", "none"], default="drugbank",
                   help="KG source. emergnn: 'drugbank' (paper) or 'merged'. "
                        "tiger: 'merged' for dual-channel BKG, 'none' for mol-only. "
                        "hdn_ddi: ignored (uses molecules only).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=None,
                   help="Override default batch size")
    p.add_argument("--n-dim", type=int, default=None,
                   help="Override default hidden dim (emergnn only)")
    p.add_argument("--tag", type=str, default=None,
                   help="Run name tag (defaults to <baseline>_<kg-source>_e<epochs>)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tiger-cold-start-patch", action="store_true",
                   help="TIGER baseline only: enable the project-extension "
                        "cold-start center-node patch (NOT in paper Su et al. "
                        "AAAI 2024). Default off → paper-faithful. Set this "
                        "for S1/S2 cold-start runs where unseen drugs need "
                        "mol-channel projection. Ignored by other baselines.")
    p.add_argument("--tiger-extractor", type=str, default="randomWalk",
                   choices=["randomWalk", "khop-subtree", "probability"],
                   help="TIGER baseline only: which subgraph extractor to use. "
                        "Paper Sec 'Biomedical Knowledge Graph Channel' has 3 "
                        "variants (TIGER-DW / TIGER-KS / TIGER-P). Default "
                        "randomWalk (back-compat). Ignored by other baselines.")
    p.add_argument("--mol-pkl", type=str, default=None,
                   help="Path to pre-built 3-level hierarchical mol-graphs pkl "
                        "(produced by baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py). "
                        "Required by hdn_ddi / hdn_ddi-mcc per CLAUDE.md "
                        "§'Baseline 规范'; ignored by other baselines. "
                        "Default project pkl: Code/baseline/hdn_ddi/_data/necessary/hdn_ddi_mol_graphs__mine.pkl")
    p.add_argument("--no-save-pred", action="store_true",
                   help="Skip saving per-pair prediction parquets")
    p.add_argument("--log-step-every", type=int, default=50,
                   help="Print rolling-mean loss every N training steps "
                        "(per CLAUDE.md 训练进度日志规范). Default 50.")
    p.add_argument("--eval-strategy", choices=["no", "epoch", "steps"], default="epoch",
                   help="When to run val eval during training (HF Trainer style). Default 'epoch'.")
    p.add_argument("--eval-steps", type=int, default=500,
                   help="Eval interval (steps) when --eval-strategy=steps. Default 500.")
    p.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="no",
                   help="When to save intermediate checkpoints. Default 'no' (only final save).")
    p.add_argument("--save-steps", type=int, default=500,
                   help="Save interval (steps) when --save-strategy=steps. Default 500.")
    p.add_argument("--save-total-limit", type=int, default=3,
                   help="Max number of intermediate checkpoints to keep. Default 3.")
    p.add_argument("--no-load-best-at-end", action="store_true",
                   help="Disable load_best_model_at_end (default: load best val_auc state).")
    args = p.parse_args()

    # Resolve the user's --baseline arg (which may include task suffix or
    # legacy _mc alias) into canonical (base_name, task).
    base_name, task = _normalize_baseline(args.baseline)
    # 2026-05-20: emergnn-multimode was promoted to the official "emergnn"
    # binary baseline. Accept the old alias and re-route after printing
    # a one-shot DEPRECATION warning to stderr (see block below).
    if base_name == "emergnn-multimode":
        print(
            f"[run_baseline] DEPRECATION: '--baseline emergnn-multimode' "
            f"was promoted to '--baseline emergnn' on 2026-05-20. The "
            f"alias still works (re-routes to 'emergnn') but is deprecated; "
            f"update your scripts to use the bare name.",
            file=sys.stderr,
            flush=True,
        )
        base_name = "emergnn"
    args.base_name = base_name  # store on args so _run() can read it
    args.task = task             # "bc" | "mcc" | "mlc"

    tag = args.tag or f"{args.baseline}_{args.backbone_kg_source}_e{args.epochs}"

    with RunLogger(script="run_baseline", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            baseline=args.baseline,
            kg_source=args.backbone_kg_source if base_name == "emergnn" else "",
            epochs=args.epochs,
        )
        _run(args, rl)


def _augment_splits_with_ddi_type(ds, edges_csv_path: Path):
    """Attach `ddi_type` column to every split DataFrame in ds.splits via
    join with ColdDDI source `ddi_edges.csv`. Required for multi-class task
    since legacy 800-drug PKL bundle only has binary `label`.
    """
    edges = pd.read_csv(edges_csv_path, usecols=["drug_a_id", "drug_b_id", "ddi_type"])
    # Build bidirectional lookup (DDI is symmetric)
    key_to_type = {}
    for a, b, t in zip(edges["drug_a_id"], edges["drug_b_id"], edges["ddi_type"]):
        key_to_type[(str(a), str(b))] = str(t)
        key_to_type[(str(b), str(a))] = str(t)
    for name in ["train", "val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2"]:
        df = getattr(ds.splits, name)
        if df is None or len(df) == 0:
            continue
        # only positives have a ddi_type; negatives won't appear in ddi_edges → NaN
        ddi_type = df.apply(
            lambda r: key_to_type.get((str(r["drug_a_id"]), str(r["drug_b_id"])), None),
            axis=1,
        )
        df_new = df.copy()
        df_new["ddi_type"] = ddi_type
        setattr(ds.splits, name, df_new)
        miss = df_new["ddi_type"].isna().sum()
        if miss > 0:
            # For PKL-positive rows label=1 (in train/val/test_s* the positives),
            # miss should be 0; for any neg rows in val/test the column will be NaN
            # but we don't use neg in multi-class eval anyway.
            print(f"  [_augment] {name}: {miss}/{len(df_new)} rows w/o ddi_type "
                  f"(neg pairs — expected)")


def _run(args, rl):
    run_dir = rl.run_dir
    print(f"[run] config: {vars(args)}")

    # ----- load data -----
    print(f"[run] loading {PKL.name} ...")
    t0 = time.time()
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    print(f"[run] loaded in {time.time()-t0:.1f}s: drugs={len(ds.drug_set)}  "
          f"train={len(ds.splits.train)}  test_s2={len(ds.splits.test_s2)}")

    # ----- multi-class: attach ddi_type column via join with ddi_edges.csv -----
    is_mc = args.task == "mcc"
    if is_mc:
        edges_csv = ROOT / "Code/data/KG/drugbank/filtered/ddi_edges.csv"
        if not edges_csv.is_file():
            raise FileNotFoundError(
                f"Multi-class task requires {edges_csv} to recover ddi_type from "
                f"800-drug PKL pairs (legacy bundle only has binary label)."
            )
        print(f"[run] attaching ddi_type from {edges_csv.name} ...")
        _augment_splits_with_ddi_type(ds, edges_csv)
        n_types = ds.splits.train["ddi_type"].astype(str).nunique()
        print(f"[run] observed {n_types} unique ddi_types in train")

    # ----- instantiate baseline -----
    print(f"[run] instantiating {args.baseline} ...")
    common_kwargs = dict(
        n_epochs=args.epochs,
        log_step_every=args.log_step_every,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=not args.no_load_best_at_end,
        run_dir=str(rl.run_dir),
    )
    if args.base_name == "emergnn":
        kwargs = dict(common_kwargs, backbone_kg_source=args.backbone_kg_source)
        if args.n_dim is not None:
            kwargs["n_dim"] = args.n_dim
        if args.backbone_kg_source == "merged":
            kwargs["merged_kg_path"] = str(MERGED_KG)
        if is_mc:
            # multi-class still uses single-mode per-mode trainer; --batch
            # is honored.
            if args.batch is not None:
                kwargs["batch_size"] = args.batch
            from baseline.emergnn import EmerGNNMulticlassBaseline
            model = EmerGNNMulticlassBaseline(**kwargs)
        else:
            # Binary "emergnn" is the multimode wrapper since 2026-05-20.
            # The wrapper HARDCODES batch_size per sub-model (128 for S0,
            # 32 for S1/S2 — matches upstream evaluate.py dispatch) and
            # rejects any user-supplied --batch override. If the user
            # passed --batch, fail loudly rather than silently ignore.
            if args.batch is not None:
                raise ValueError(
                    f"--batch is not supported for '--baseline emergnn' "
                    f"(binary). The multimode wrapper hardcodes batch_size "
                    f"per S0/S1/S2 sub-model (128 / 32 / 32, matching "
                    f"upstream evaluate.py:54-70). Drop --batch."
                )
            from baseline.emergnn import EmerGNNBaseline
            model = EmerGNNBaseline(**kwargs)
    elif args.base_name == "emergnn-kgonly":
        # DEPRECATED 2026-05-20: kgonly is the losing candidate from the
        # multimode-vs-kgonly comparison. Kept routable for back-compat /
        # reference experiments. Imports from deprecate/.
        kwargs = dict(common_kwargs, backbone_kg_source=args.backbone_kg_source)
        if args.batch is not None:
            kwargs["batch_size"] = args.batch
        if args.n_dim is not None:
            kwargs["n_dim"] = args.n_dim
        if args.backbone_kg_source == "merged":
            kwargs["merged_kg_path"] = str(MERGED_KG)
        if is_mc:
            raise ValueError(
                "emergnn-kgonly has no multi-class variant; use 'emergnn-mcc'."
            )
        print(
            "[run_baseline] NOTE: 'emergnn-kgonly' is the deprecated Method B "
            "variant from the 2026-05-20 binary-baseline comparison. The "
            "official binary baseline is now plain 'emergnn'.",
            file=sys.stderr,
            flush=True,
        )
        from baseline.emergnn.deprecate.binary_cls_kgonly import (
            EmerGNNKGOnlyBaseline,
        )
        model = EmerGNNKGOnlyBaseline(**kwargs)
    elif args.base_name == "hdn_ddi":
        kwargs = dict(common_kwargs)
        if args.batch is not None:
            kwargs["batch_size"] = args.batch
        # CLAUDE.md §"Baseline 规范" §2 step 3: HDN-DDI auto-detects /
        # auto-builds the bit-faithful 3-level mol-graphs pkl via the
        # local `_data/necessary/build_hierarchical_pkl.py`. Pass an
        # override path via --mol-pkl if you want a non-default location.
        if args.mol_pkl:
            kwargs["mol_pkl_path"] = args.mol_pkl
        if is_mc:
            from baseline.hdn_ddi import HDNDDIMulticlassBaseline
            model = HDNDDIMulticlassBaseline(**kwargs)
        else:
            from baseline.hdn_ddi import HDNDDIBaseline
            model = HDNDDIBaseline(**kwargs)
    elif args.base_name == "tiger":
        kwargs = dict(common_kwargs)
        if args.batch is not None:
            kwargs["batch_size"] = args.batch
        # TIGER dual-channel default uses merged KG as BKG source.
        # ``--backbone-kg-source none`` falls back to mol-only mode (legacy).
        if args.backbone_kg_source == "none":
            # TIGER has its OWN `kg_source` parameter (independent of EmerGNN
            # backbone). Tiger accepts "merged" or "none" (mol-only). Do NOT
            # rename the kwarg key — Tiger's API is not part of this refactor.
            kwargs["kg_source"] = "none"
            kwargs["mol_only"] = True
        else:
            kwargs["kg_source"] = "merged"
            kwargs["merged_kg_path"] = str(MERGED_KG)
        # Cold-start patch is a PROJECT EXTENSION, NOT paper-native.
        # Default off (paper-faithful) per CLAUDE.md §"Baseline 规范" §4
        # codex 2026-05-18 fix #5. Set via ``--tiger-cold-start-patch``
        # to enable for S1/S2 cold-start runs.
        if args.tiger_cold_start_patch:
            kwargs["cold_start_patch"] = True
        kwargs["extractor"] = args.tiger_extractor
        if is_mc:
            from baseline.tiger import TIGERMulticlassBaseline
            model = TIGERMulticlassBaseline(**kwargs)
        else:
            from baseline.tiger import TIGERBaseline
            model = TIGERBaseline(**kwargs)
    else:
        raise ValueError(
            f"unknown baseline base_name={args.base_name!r} "
            f"(from --baseline {args.baseline!r}); supported: "
            f"emergnn / hdn_ddi / tiger"
        )

    # ----- fit -----
    print(f"[run] fitting ({args.epochs} epochs) ...")
    t_fit = time.time()
    # Pass `ds` itself as the val argument so the baseline's `_validate(val)`
    # can read `val.splits.val_s2` + `val.get_negatives("val_s2")` during
    # intermediate eval (per CLAUDE.md 训练中间 eval 规范).
    model.fit(ds, val=ds, kg=ds.kg)
    fit_time = time.time() - t_fit
    print(f"[run] fit done in {fit_time:.1f}s")

    # ----- evaluate S0/S1/S2 -----
    eval_results = {}
    if is_mc:
        # Multi-class eval: per-pair top-k acc / macro F1 / macro AUC on POSITIVES only.
        # No drug-replacement negatives in this task (model predicts which type, not yes/no).
        from my_code.utils.task_eval import eval_multiclass
        ddi_to_idx = model._ddi_type_to_idx  # built during fit()
        n_classes = model.n_classes
        for setting in ["s0", "s1", "s2"]:
            pos = getattr(ds.splits, f"test_{setting}")
            if "ddi_type" not in pos.columns:
                print(f"[eval] test_{setting} missing ddi_type column, skipping")
                continue
            pos = pos[["drug_a_id", "drug_b_id", "ddi_type"]].copy()
            # Drop test rows whose ddi_type was not seen in train (out-of-vocab)
            mask = pos["ddi_type"].astype(str).isin(ddi_to_idx)
            n_oov = int((~mask).sum())
            if n_oov:
                print(f"[eval] {setting}: {n_oov} pairs have OOV ddi_type, excluded")
            pos = pos[mask].reset_index(drop=True)
            if len(pos) == 0:
                continue

            print(f"[eval] predict_proba on test_{setting} (n={len(pos)} positives only) ...")
            t_pred = time.time()
            p = model.predict_proba(pos[["drug_a_id", "drug_b_id"]], kg=ds.kg)
            pred_time = time.time() - t_pred
            labels = np.array([ddi_to_idx[str(t)] for t in pos["ddi_type"]])
            m = eval_multiclass(p, labels, n_classes)
            m["pred_time_s"] = round(pred_time, 1)
            m["n_oov_excluded"] = n_oov
            eval_results[setting] = m
            print(f"[eval] {setting}: top1={m['top1_acc']:.4f}  top3={m['top3_acc']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}  macro_auc={m['macro_auc']:.4f}  ({pred_time:.1f}s)")

            if not args.no_save_pred:
                pred_df = pos.copy()
                pred_df["true_class_idx"] = labels
                pred_df["pred_class_idx"] = p.argmax(axis=1)
                pred_df["pred_top1_prob"] = p.max(axis=1)
                pred_df.to_parquet(run_dir / f"predictions_{setting}.parquet")
    else:
        # Binary eval: pos + neg labeled, AUC/NLL/F1.
        for setting in ["s0", "s1", "s2"]:
            pos = getattr(ds.splits, f"test_{setting}")[["drug_a_id", "drug_b_id"]].copy()
            neg = ds.negatives_by_split.get(f"test_{setting}")
            if neg is None:
                print(f"[eval] no negatives for test_{setting}, skipping")
                continue
            neg = neg[["drug_a_id", "drug_b_id"]].copy()
            pos["label"] = 1
            neg["label"] = 0
            pairs = pd.concat([pos, neg], ignore_index=True)
            y = pairs["label"].values

            print(f"[eval] predict_proba on test_{setting} (n={len(pairs)}) ...")
            t_pred = time.time()
            p = model.predict_proba(pairs[["drug_a_id", "drug_b_id"]], kg=ds.kg)
            pred_time = time.time() - t_pred
            eps = 1e-7
            p_clip = np.clip(p, eps, 1 - eps)
            auc = roc_auc_score(y, p)
            nll = log_loss(y, p_clip)
            pred_label = (p >= 0.5).astype(int)
            tp = ((pred_label == 1) & (y == 1)).sum()
            fp = ((pred_label == 1) & (y == 0)).sum()
            fn = ((pred_label == 0) & (y == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)

            eval_results[setting] = {
                "auc": float(auc),
                "nll": float(nll),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "pred_time_s": round(pred_time, 1),
            }
            print(f"[eval] {setting}: AUC={auc:.4f}  NLL={nll:.4f}  F1={f1:.4f}  ({pred_time:.1f}s)")

            if not args.no_save_pred:
                pairs["pred"] = p.astype(np.float32)
                pairs[["drug_a_id", "drug_b_id", "label", "pred"]].to_parquet(
                    run_dir / f"predictions_{setting}.parquet"
                )

    # ----- save results.json -----
    results = {
        "run_id": rl.run_id,
        "baseline": args.baseline,
        "base_name": args.base_name,
        "task": args.task,
        "backbone_kg_source": args.backbone_kg_source if args.base_name == "emergnn" else None,
        "config": vars(args),
        "fit_time_s": round(fit_time, 1),
        "eval": eval_results,
        "data": {
            "n_drug_set": len(ds.drug_set),
            "n_train": len(ds.splits.train),
            "n_test_s2": len(ds.splits.test_s2),
        },
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[run] saved results to {run_dir / 'results.json'}")
    print(f"[run] total time: {time.time() - t0:.1f}s")

    # ----- record metrics into the run_logger index row -----
    if is_mc:
        rl.set_metrics(
            fit_time_s=round(fit_time, 1),
            auc_s0=round(eval_results.get("s0", {}).get("macro_auc", float("nan")), 4),
            auc_s1=round(eval_results.get("s1", {}).get("macro_auc", float("nan")), 4),
            auc_s2=round(eval_results.get("s2", {}).get("macro_auc", float("nan")), 4),
            # For multi-class, reuse nll_* slots to record top1_acc instead
            nll_s0=round(eval_results.get("s0", {}).get("top1_acc", float("nan")), 4),
            nll_s1=round(eval_results.get("s1", {}).get("top1_acc", float("nan")), 4),
            nll_s2=round(eval_results.get("s2", {}).get("top1_acc", float("nan")), 4),
        )
    else:
        rl.set_metrics(
            fit_time_s=round(fit_time, 1),
            auc_s0=round(eval_results.get("s0", {}).get("auc", float("nan")), 4),
            auc_s1=round(eval_results.get("s1", {}).get("auc", float("nan")), 4),
            auc_s2=round(eval_results.get("s2", {}).get("auc", float("nan")), 4),
            nll_s0=round(eval_results.get("s0", {}).get("nll", float("nan")), 4),
            nll_s1=round(eval_results.get("s1", {}).get("nll", float("nan")), 4),
            nll_s2=round(eval_results.get("s2", {}).get("nll", float("nan")), 4),
        )


if __name__ == "__main__":
    main()
