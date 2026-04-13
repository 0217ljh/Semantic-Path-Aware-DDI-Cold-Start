"""Run log file setup; script summary from env. See docs/4-scripts §5.4."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, TextIO

# Format for application logs (pipeline, data, predict, eval, etc.)
_LOG_FMT = "[%(asctime)s] %(levelname)s %(name)s %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_app_logging(logs_dir: Path, *, to_stderr: bool = True) -> None:
    """Configure Python logging for my_code: pipeline.log (all) and error.log (errors only).
    Call after init_run_paths so logs_dir exists. Optionally also log to stderr."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = logs_dir / "pipeline.log"
    error_log = logs_dir / "error.log"

    log = logging.getLogger("my_code")
    log.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if called multiple times
    if any(h.baseFilename == str(pipeline_log) for h in log.handlers if getattr(h, "baseFilename", None)):
        return

    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    fh = logging.FileHandler(pipeline_log, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    eh = logging.FileHandler(error_log, encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    log.addHandler(eh)

    if to_stderr:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def setup_run_log(log_path: Path, rank: int = 0) -> Optional[TextIO]:
    """Create/open run log file; write script summary from env if present. Only rank 0."""
    if rank != 0:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")

    summary = os.environ.pop("SCRIPT_STARTUP_SUMMARY", None)
    if summary:
        f.write(f"[script] {summary}\n")
        f.flush()
    return f


# def write_script_summary_if_present(log_file_handle: TextIO) -> None:
#     import os
#     summary = os.environ.get("SCRIPT_STARTUP_SUMMARY")
#     if summary:
#         log_file_handle.write(f"[script] {summary}\n")
#         log_file_handle.flush()
