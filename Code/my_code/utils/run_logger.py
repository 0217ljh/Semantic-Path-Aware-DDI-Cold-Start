"""Unified run-logger per CLAUDE.md '日志收集规范'.

Provides:
  - `make_run_id(script, tag, seed)` -> stable run identifier
  - `RunLogger(script_name, tag, seed)` context manager that:
      * creates Code/runs/<run_id>/
      * mirrors stdout+stderr to:
          - <run_dir>/train.log
          - Code/runs/_logs/<run_id>.log
      * on exit, appends a row to Code/runs/_logs/index.csv

Usage:
    from my_code.utils.run_logger import RunLogger

    with RunLogger(script="run_baseline", tag="emergnn_merged_e5", seed=42) as rl:
        # Note: `kg_source` is the index.csv schema column name.  The corresponding
        # Python parameter at the trainer level is `backbone_kg_source`
        # (renamed 2026-06-05 for clarity — it controls the EmerGNN backbone's
        # KG vocabulary ONLY, not the PMP mediator cache or v1.5+ score path).
        # Callers should pass set_meta(kg_source=args.backbone_kg_source).
        rl.set_meta(baseline="emergnn", kg_source="merged", epochs=5)
        # ... train / eval / write results.json to rl.run_dir ...
        rl.set_metrics(auc_s0=0.93, auc_s1=0.81, auc_s2=0.72, nll_s2=0.55)
"""
from __future__ import annotations

import csv
import io
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


_INDEX_FIELDS = [
    "run_id", "timestamp", "script", "tag",
    "baseline", "kg_source", "seed", "epochs",
    "fit_time_s",
    "auc_s0", "auc_s1", "auc_s2",
    "nll_s0", "nll_s1", "nll_s2",
    "status",  # ok | error
    "run_dir",
]


def _find_project_root() -> Path:
    """Locate the project root (the dir containing Code/data/KG/)."""
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError("project root with Code/data/KG/ not found")


def _safe(s: str) -> str:
    """Filesystem-safe string for paths."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def make_run_id(script: str, tag: str, seed: int, timestamp: str | None = None) -> str:
    """Generate run identifier: <timestamp>__<script>__<tag>__seed<N>."""
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}__{_safe(script)}__{_safe(tag)}__seed{seed}"


class _Tee(io.TextIOBase):
    """Write to multiple streams at once (with flush)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


class RunLogger:
    """Context manager: create run_dir, tee stdout/stderr, write index row on exit.

    Attributes:
        run_id:   unique identifier for this run
        run_dir:  Path to Code/runs/<run_id>/
        logs_dir: Path to Code/runs/_logs/ (shared collection)
        meta:     dict of metadata to write into the index row (extend via set_meta)
        metrics:  dict of metrics to write into the index row (extend via set_metrics)
    """
    def __init__(
        self,
        script: str,
        tag: str,
        seed: int,
        *,
        project_root: Path | None = None,
    ):
        self.project_root = Path(project_root) if project_root is not None else _find_project_root()
        self.run_id = make_run_id(script, tag, seed)
        self.script = script
        self.tag = tag
        self.seed = seed
        self.run_dir = self.project_root / "Code" / "runs" / self.run_id
        self.logs_dir = self.project_root / "Code" / "runs" / "_logs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._train_log = (self.run_dir / "train.log").open("w", encoding="utf-8")
        self._mirror_log = (self.logs_dir / f"{self.run_id}.log").open("w", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self.meta: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": self.run_id.split("__", 1)[0],
            "script": script,
            "tag": tag,
            "seed": seed,
            "run_dir": str(self.run_dir.relative_to(self.project_root)),
        }
        self.metrics: dict[str, Any] = {}
        self.status: str = "ok"

    def set_meta(self, **kwargs):
        """Add metadata fields to the index row (e.g., baseline, kg_source, epochs)."""
        self.meta.update(kwargs)

    def set_metrics(self, **kwargs):
        """Add metric fields to the index row (e.g., auc_s2, nll_s2, fit_time_s)."""
        self.metrics.update(kwargs)

    def __enter__(self):
        # tee stdout/stderr to file mirrors
        sys.stdout = _Tee(self._orig_stdout, self._train_log, self._mirror_log)
        sys.stderr = _Tee(self._orig_stderr, self._train_log, self._mirror_log)
        print(f"[run_logger] run_id = {self.run_id}")
        print(f"[run_logger] run_dir = {self.run_dir}")
        print(f"[run_logger] mirror_log = {self.logs_dir}/{self.run_id}.log")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.status = f"error:{exc_type.__name__}"
            print(f"\n[run_logger] EXCEPTION: {exc_type.__name__}: {exc_val}", flush=True)
            try:
                traceback.print_exception(exc_type, exc_val, exc_tb)
            except Exception:
                pass

        # restore stdout/stderr before closing files
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        try:
            self._train_log.close()
        except Exception:
            pass
        try:
            self._mirror_log.close()
        except Exception:
            pass

        # append index row
        row = {f: "" for f in _INDEX_FIELDS}
        row.update(self.meta)
        row.update(self.metrics)
        row["status"] = self.status
        index_path = self.logs_dir / "index.csv"
        write_header = not index_path.exists()
        with index_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_INDEX_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
        # don't swallow exception
        return False
