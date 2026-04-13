"""Merge data/model/training config and apply --set overrides. See docs/5-Config."""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, List, Optional


def _coerce(val: str) -> Any:
    v = val.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "none" or v == "null":
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def load_cfg(args: Any) -> dict:
    """
    Merge config from run file OR (data + model + training), then apply --set overrides.
    Returns a single dict with top-level keys: data, model, training; optional: mode, ckpt, run.
    """
    if getattr(args, "run", None):
        run = args.run
        if run.endswith(".yaml") or run.endswith(".yml"):
            path = Path(run)
            if not path.is_absolute():
                path = Path("configs").parent / run if not Path(run).exists() else Path(run)
            with open(path) as f:
                base = yaml.safe_load(f)
            if base is None:
                base = {}
            # Run file may have data/model/training at top level
            if "data" not in base and "model" not in base:
                with open(Path("configs/data") / Path(run).stem.split("/")[-1] + ".yaml") as fd:
                    base.setdefault("data", yaml.safe_load(fd))
                with open(Path("configs/model") / Path(run).stem.split("/")[-1] + ".yaml") as fm:
                    base.setdefault("model", yaml.safe_load(fm))
                with open(Path("configs/training") / Path(run).stem.split("/")[-1] + ".yaml") as ft:
                    base.setdefault("training", yaml.safe_load(ft))
        else:
            root = Path("configs")
            with open(root / "data" / f"{run}.yaml") as fd:
                data = yaml.safe_load(fd)
            with open(root / "model" / f"{run}.yaml") as fm:
                model = yaml.safe_load(fm)
            with open(root / "training" / f"{run}.yaml") as ft:
                training = yaml.safe_load(ft)
            base = {"data": data, "model": model, "training": training}
    else:
        with open(args.data) as f:
            data = yaml.safe_load(f)
        with open(args.model) as f:
            model = yaml.safe_load(f)
        with open(args.training) as f:
            training = yaml.safe_load(f)
        base = {"data": data, "model": model, "training": training}

    cfg = base
    for s in getattr(args, "set", []) or []:
        key, _, val = s.partition("=")
        keys = key.strip().split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = _coerce(val)

    cfg["mode"] = getattr(args, "mode", "train")
    if cfg["mode"] == "predict":
        cfg["ckpt"] = getattr(args, "ckpt", None)
    cfg["exp_name"] = getattr(args, "exp_name", None)
    return cfg
