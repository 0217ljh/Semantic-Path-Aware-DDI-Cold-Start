"""Unified pipeline: load_cfg -> paths -> data -> dataset/loaders -> method -> train/predict -> evaluate. See docs/3-Pipeline."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Project root on path for "my_code.*" imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_code.utils.config import load_cfg
from my_code.utils.seeding import set_global_seed
from my_code.io.run_paths import init_run_paths
from my_code.io import artifacts
from my_code.io import logging as io_logging
from my_code.pipeline.load_dataset import get_or_load_data
from my_code.data import build_dataset, build_collate_fn, build_loaders
from my_code.pipeline.registry import load_method
from my_code.train.trainer import train
from my_code.predict.predictor import predict
from my_code.eval.evaluator import evaluate

logger = logging.getLogger(__name__)


def main(args) -> int:
    t0 = time.perf_counter()
    cfg = load_cfg(args)
    set_global_seed(cfg.get("data", {}).get("global_seed", 42))

    paths = init_run_paths(cfg)
    rank = getattr(args, "rank", 0)

    # Write resolved config and cmd (doc 5-Config §8)
    if rank == 0:
        artifacts.write_resolved_config(cfg, paths.resolved_config_path)
        cmd = " ".join(sys.argv)
        artifacts.write_cmd(cmd, paths.cmd_path)
        artifacts.write_meta_json({"mode": cfg.get("mode", "train")}, paths.meta_json_path)
        log_handle = io_logging.setup_run_log(paths.logs_dir / "stdout_rank0.log", rank)
        if log_handle:
            sys.stdout = log_handle
        io_logging.setup_app_logging(paths.logs_dir, to_stderr=True)

    logger.info("config loaded")
    logger.info("paths initialized: run_dir=%s", paths.run_dir)

    try:
        # Data
        logger.info("data loading started")
        processed, features = get_or_load_data(cfg, rank=rank)
        datasets = build_dataset(cfg, processed, features)
        collate_fn = build_collate_fn(cfg, features)
        loaders = build_loaders(cfg, datasets, collate_fn)
        logger.info("data loaded; datasets=%s", list(datasets.keys()))

        # Method
        model = load_method(cfg["model"]["name"], cfg)
        logger.info("model loaded: %s", cfg.get("model", {}).get("name"))

        if cfg.get("mode") == "train":
            logger.info("train started")
            train_output = train(cfg, loaders, model, paths)
            logger.info("train ended: status=%s best_ckpt=%s", train_output.get("status"), train_output.get("best_ckpt_path"))
            if rank == 0 and train_output.get("best_ckpt_path"):
                cfg["ckpt"] = train_output["best_ckpt_path"]
            logger.info("predict started")
            predict_output = predict(cfg, loaders, model, paths)
            logger.info("predict ended: status=%s", predict_output.get("status"))
        else:
            logger.info("predict started")
            predict_output = predict(cfg, loaders, model, paths)
            logger.info("predict ended: status=%s", predict_output.get("status"))

        logger.info("evaluate started")
        eval_output = evaluate(cfg, predict_output, paths)
        logger.info("evaluate ended: status=%s", eval_output.get("status"))

        elapsed = time.perf_counter() - t0
        status = "ok" if eval_output.get("status") == "ok" else "error"
        logger.info("pipeline finished: status=%s elapsed_s=%.2f", status, elapsed)
        return 0 if status == "ok" else 1
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception("pipeline failed after %.2fs", elapsed)
        return 1


if __name__ == "__main__":
    # Minimal args for direct run
    class Args:
        run = "configs/runs/cora_gcn.yaml"
        mode = "train"
        set = []
        data = model = training = ckpt = None
    sys.exit(main(Args()))
