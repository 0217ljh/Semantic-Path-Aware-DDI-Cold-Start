"""Top-level evaluator — paper §A.6.3 entry point.

Dispatches to a registered :class:`baseline.BaselineModel`,
fits it on a :class:`PairDataset`, runs evaluation on the requested
setting (S0 / S1 / S2 / all), and writes per-split metrics to ``--out``.

Layer 1 implements the CLI surface and the fit / score / save loop;
the actual metrics are computed by helpers from
:mod:`coldddi.eval.metrics` (added in a later layer).

CLI
---
::

    python evaluate.py \\
      --method {deepddi,ssi-ddi,dsn-ddi,hdn-ddi,emergnn,tiger,mkg-fenn,textddi} \\
      --data DIR \\
      --seed INT \\
      --setting {S0,S1,S2,all} \\
      [--device {cuda,cpu}] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import inspect

from baseline import (
    BaselineModel,
    NAME_TO_MODULE,
    ensure_imported,
    list_baselines,
    load_baseline,
)
from baseline.base import _REGISTRY
from data_utils.dataset import PairDataset

ALL_SETTINGS: tuple[str, ...] = ("S0", "S1", "S2")


def _setting_splits(setting: str) -> tuple[str, str]:
    """Map a setting label to its (val_split_name, test_split_name)."""
    s = setting.lower()
    return f"val_{s}", f"test_{s}"


def _evaluate_split(
    model: BaselineModel,
    pos: pd.DataFrame,
    neg: pd.DataFrame,
) -> dict[str, float]:
    """Score positives + negatives, return placeholder metrics dict.

    Layer 1 placeholder — real metric computation moves to
    :mod:`coldddi.eval.metrics` once the first baseline lands.
    """
    if len(pos) == 0:
        return {"n_pos": 0, "n_neg": int(len(neg))}
    pos_scores = model.predict_proba(pos[["drug_a_id", "drug_b_id"]])
    neg_scores = (
        model.predict_proba(neg[["drug_a_id", "drug_b_id"]])
        if len(neg)
        else np.empty(0)
    )
    return {
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "mean_pos_score": float(pos_scores.mean()),
        "mean_neg_score": float(neg_scores.mean()) if neg_scores.size else None,
    }


def run_evaluation(
    *,
    method: str,
    data_dir: Path,
    seed: int,
    settings: Sequence[str],
    out_dir: Path,
    checkpoint: Path | None = None,
    device: str = "cuda",
) -> dict:
    """Train (or load) a baseline and write metrics for each setting.

    Note on ``settings``: when multiple settings are passed, the model
    is **trained once and reused** to score every setting's val/test
    splits. This matches the legacy ``cold_start_split_fair`` design
    where train ⊆ G1×G1 is shared across S0/S1/S2 (no setting-specific
    re-training). Baselines that internally do val-driven early
    stopping should use only the first setting's val for that purpose.

    Parameters
    ----------
    method
        Registered baseline name (``baseline.list_baselines()``).
    data_dir
        Release-style directory consumed by
        :meth:`PairDataset.from_release_dir`.
    seed
        Which split seed to load.
    settings
        Iterable of ``"S0" / "S1" / "S2"``.
    out_dir
        Where ``metrics_seed{N}.json`` (and any checkpoint) is written.
    checkpoint
        If given, load instead of training.
    device
        Hint passed via baseline-specific kwargs (most baselines pick up
        ``CUDA_VISIBLE_DEVICES`` themselves).
    """
    # Lazy-import the baseline module *before* validating, so a fresh
    # process that only did `import evaluate` still finds it.
    ensure_imported(method)
    if method not in list_baselines():
        known = sorted(set(list_baselines()) | set(NAME_TO_MODULE))
        raise ValueError(
            f"Unknown method {method!r}; registered or declared baselines are {known}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] method = {method}", file=sys.stderr)
    print(f"[evaluate] loading PairDataset @ seed={seed} from {data_dir}",
          file=sys.stderr, flush=True)
    ds = PairDataset.from_release_dir(data_dir, seed=seed)

    # Train (or load).
    if checkpoint is not None:
        print(f"[evaluate] loading checkpoint {checkpoint}", file=sys.stderr)
        model = load_baseline(checkpoint)
    else:
        # Forward `device` only to baselines that accept it; not every
        # subclass needs it (e.g. a tree-based baseline).
        cls = _REGISTRY[method]
        kwargs: dict = {}
        if "device" in inspect.signature(cls).parameters:
            kwargs["device"] = device
        model = cls(**kwargs)
        print(f"[evaluate] training {method} on seed {seed}", file=sys.stderr)
        model.fit(ds, kg=ds.kg)

    # Evaluate every requested setting.
    metrics: dict[str, dict] = {}
    for setting in settings:
        if setting not in ALL_SETTINGS:
            raise ValueError(f"Unknown setting {setting!r}; expected one of {ALL_SETTINGS}")
        val_name, test_name = _setting_splits(setting)
        for split_name in (val_name, test_name):
            pos = getattr(ds.splits, split_name)
            neg = ds.get_negatives(split_name)
            metrics[split_name] = _evaluate_split(model, pos, neg)
            print(
                f"[evaluate] {split_name}: {metrics[split_name]}",
                file=sys.stderr,
            )

    out_path = out_dir / f"metrics_seed{seed}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] wrote {out_path}", file=sys.stderr)
    return metrics


def _ensure_baseline_imported(method: str) -> None:
    """Backward-compat alias for :func:`baseline.ensure_imported`.

    Older tests / code may import this; new code should use
    :func:`baseline.ensure_imported` directly. The two share
    the same ``NAME_TO_MODULE`` map.
    """
    if method not in NAME_TO_MODULE:
        raise ImportError(
            f"Could not import baseline module for {method!r}: "
            f"not declared in NAME_TO_MODULE."
        )
    ensure_imported(method)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Run a registered ColdDDI baseline on a release directory and "
            "write per-split metrics (Paper @A.6.3)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        required=True,
        help=(
            "Registered baseline name (will lazy-import "
            "`baseline.<method>`)."
        ),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--setting",
        choices=["S0", "S1", "S2", "all"],
        default="S2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional pre-trained baseline directory (skips fit()).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = list(ALL_SETTINGS) if args.setting == "all" else [args.setting]
    run_evaluation(
        method=args.method,
        data_dir=args.data,
        seed=args.seed,
        settings=settings,
        out_dir=args.out,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
