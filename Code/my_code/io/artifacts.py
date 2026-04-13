"""Write run artifacts: resolved_config, cmd, meta. See docs/6-Init_Path."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_resolved_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def write_cmd(cmd: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(cmd + "\n")


def write_meta_json(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def append_epoch_log(row: dict, path: Path) -> None:
    """Append one JSON line (epoch timing/metrics) to path. Used by training callbacks. See docs/13-Log §8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
