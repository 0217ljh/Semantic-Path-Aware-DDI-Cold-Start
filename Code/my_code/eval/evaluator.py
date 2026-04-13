"""evaluate(cfg, predict_output, paths) -> EvalOutput. See docs/12-Evaluation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from my_code.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def evaluate(cfg: dict, predict_output: dict, paths: Any) -> Dict[str, Any]:
    metrics_config = cfg.get("evaluation", {}).get("metrics", ["accuracy"]) or ["accuracy"]
    if isinstance(metrics_config, str):
        metrics_config = [metrics_config]
    preds_path = predict_output.get("preds_path")
    if isinstance(preds_path, str):
        preds_path = {"test": preds_path}
    results = {}
    paths_written = {}
    for split, path in (preds_path or {}).items():
        path = Path(path)
        if not path.exists():
            continue
        rows = _read_preds_csv(path)
        m = compute_metrics(rows, metrics_config)
        results[split] = m
        out_path = paths.val_metrics_path if split == "val" else paths.test_metrics_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(m, f, indent=2)
        paths_written[split] = str(out_path)
        logger.info("evaluator: split=%s metrics=%s metrics_path=%s", split, m, out_path)
    if paths_written:
        logger.info("evaluator: metrics written paths=%s summary=%s", paths_written, results)
    return {"status": "ok", "metrics": results, "metrics_path": paths_written}


def _read_preds_csv(path: Path) -> List[dict]:
    import csv
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            row["y_true"] = int(row["y_true"])
            row["y_pred"] = int(row["y_pred"])
            rows.append(row)
    return rows
