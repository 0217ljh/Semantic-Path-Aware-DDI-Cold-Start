"""Run directory layout: method branch + exp_name + timestamp. See docs/6-Init_Path."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class RunPaths:
    run_dir: Path
    meta_dir: Path
    logs_dir: Path
    ckpt_dir: Path
    metrics_dir: Path
    preds_dir: Path
    figures_dir: Path
    artifacts_dir: Path
    resolved_config_path: Path
    meta_json_path: Path
    cmd_path: Path
    last_ckpt_path: Path
    best_ckpt_path: Path
    val_metrics_path: Path
    test_metrics_path: Path
    val_preds_path: Path
    test_preds_path: Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def init_run_paths(
    cfg: dict,
    *,
    timestamp: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
) -> RunPaths:
    method_name = cfg.get("model", {}).get("name", "main")
    exp_name = cfg.get("exp_name") or "default"
    ts = timestamp or _timestamp()

    if "main" in method_name:
        branch = Path("runs") / method_name
    else:
        branch = Path("runs/baselines") / method_name

    run_dir = branch / exp_name / ts
    run_dir = Path(run_dir)

    meta_dir = run_dir / "meta"
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "ckpt"
    metrics_dir = run_dir / "metrics"
    preds_dir = run_dir / "preds"
    figures_dir = run_dir / "figures"
    artifacts_dir = run_dir / "artifacts"

    paths = RunPaths(
        run_dir=run_dir,
        meta_dir=meta_dir,
        logs_dir=logs_dir,
        ckpt_dir=ckpt_dir,
        metrics_dir=metrics_dir,
        preds_dir=preds_dir,
        figures_dir=figures_dir,
        artifacts_dir=artifacts_dir,
        resolved_config_path=meta_dir / "resolved_config.yaml",
        meta_json_path=meta_dir / "meta.json",
        cmd_path=meta_dir / "cmd.txt",
        last_ckpt_path=ckpt_dir / "last.pt",
        best_ckpt_path=ckpt_dir / "best.pt",
        val_metrics_path=metrics_dir / "val.json",
        test_metrics_path=metrics_dir / "test.json",
        val_preds_path=preds_dir / "val.csv",
        test_preds_path=preds_dir / "test.csv",
    )

    if rank == 0:
        for d in [
            meta_dir, logs_dir, ckpt_dir, metrics_dir, preds_dir,
            figures_dir, artifacts_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    return paths
