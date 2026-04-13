"""predict(cfg, loaders, method, paths) -> PredictOutput. See docs/11-Prediction."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def predict(
    cfg: dict,
    loaders: dict,
    method: Any,
    paths: Any,
    callbacks: list = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = {"device": device}
    method.setup(cfg, runtime)
    ckpt_path = cfg.get("ckpt")
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        method.load_state_dict(state)
        logger.info("predictor: loaded ckpt=%s", ckpt_path)
    else:
        logger.info("predictor: no ckpt (cfg.ckpt=%s)", ckpt_path)

    preds_paths = {}
    n_samples = {}
    for split in ("val", "test"):
        loader = loaders.get(split)
        if not loader:
            continue
        path = paths.val_preds_path if split == "val" else paths.test_preds_path
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        count = 0
        for batch in loader:
            batch["split"] = split
            out = method.eval_step(batch, runtime)
            preds = out["preds"]
            targets = out["targets"]
            mask = out.get("mask")
            if mask is not None:
                for i in range(mask.sum().item()):
                    idx = mask.nonzero(as_tuple=True)[0][i].item()
                    rows.append({"id": idx, "y_true": targets[idx].item(), "y_pred": preds[idx].item()})
                    count += 1
        if rows:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "y_true", "y_pred"])
                w.writeheader()
                w.writerows(rows)
            preds_paths[split] = str(path)
            n_samples[split] = count
    return {
        "status": "ok",
        "ckpt_path": ckpt_path or "",
        "preds_path": preds_paths if len(preds_paths) != 1 else list(preds_paths.values())[0],
        "n_samples": n_samples,
    }
