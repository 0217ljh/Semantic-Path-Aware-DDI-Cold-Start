"""Training callbacks: epoch timing, logging, etc. See docs/13-Log §8."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Protocol

from my_code.io.artifacts import append_epoch_log


class TrainCallback(Protocol):
    """Callback interface for training. Implement on_epoch_start / on_epoch_end; optional on_train_begin / on_train_end."""

    def on_epoch_start(self, epoch: int, runtime: Dict[str, Any]) -> None:
        ...

    def on_epoch_end(self, epoch: int, runtime: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        ...


def _optional_hook(cb: Any, name: str, *args, **kwargs) -> None:
    if hasattr(cb, name):
        getattr(cb, name)(*args, **kwargs)


class EpochTimeLogger:
    """Records epoch start/end time and duration; appends one JSON line per epoch via io.artifacts.append_epoch_log."""

    def __init__(self, paths: Any, rank: int = 0):
        self.paths = paths
        self.rank = rank
        self._epoch_log_path = getattr(paths, "metrics_dir", None) and (paths.metrics_dir / "epochs.jsonl")

    def on_epoch_start(self, epoch: int, runtime: Dict[str, Any]) -> None:
        runtime["_epoch_start_ts"] = time.perf_counter()

    def on_epoch_end(self, epoch: int, runtime: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        if self.rank != 0 or not self._epoch_log_path:
            return
        start_ts = runtime.get("_epoch_start_ts")
        end_ts = time.perf_counter()
        duration_s = (end_ts - start_ts) if start_ts is not None else None
        row = {
            "epoch": epoch,
            "epoch_start_ts": start_ts,
            "epoch_end_ts": end_ts,
            "duration_s": duration_s,
            **{k: v for k, v in metrics.items() if k not in ("_epoch_start_ts",)},
        }
        append_epoch_log(row, self._epoch_log_path)


def run_epoch_start(callbacks: list, epoch: int, runtime: Dict[str, Any]) -> None:
    for cb in callbacks:
        _optional_hook(cb, "on_epoch_start", epoch, runtime)


def run_epoch_end(callbacks: list, epoch: int, runtime: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    for cb in callbacks:
        _optional_hook(cb, "on_epoch_end", epoch, runtime, metrics)
