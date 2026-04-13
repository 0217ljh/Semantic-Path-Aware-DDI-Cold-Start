"""train(cfg, loaders, method, paths) -> TrainOutput. See docs/10-Training."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from my_code.io.artifacts import append_epoch_log, write_meta_json
from my_code.train.callbacks import EpochTimeLogger, run_epoch_end, run_epoch_start

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def train(
    cfg: dict,
    loaders: dict,
    method: Any,
    paths: Any,
    callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = {"device": device, "epoch": 0, "global_step": 0}
    method.setup(cfg, runtime)

    epochs = cfg.get("training", {}).get("epochs", 200)
    monitor = cfg.get("training", {}).get("monitor", "val_loss")
    log_interval = cfg.get("training", {}).get("log_interval", 10)
    save_steps = cfg.get("training", {}).get("save_steps", 0) or 0
    eval_interval = cfg.get("training", {}).get("eval_interval", 1) or 1
    best_metric = None
    best_epoch = None
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")
    if not train_loader:
        return {"status": "error", "best_ckpt_path": None, "last_ckpt_path": str(paths.last_ckpt_path)}

    train_jsonl_path = paths.metrics_dir / "train.jsonl"
    train_log_path = paths.logs_dir / "train.log"
    is_rank0 = True  # Single-process for now

    _callbacks = [EpochTimeLogger(paths, rank=0 if is_rank0 else -1)] + (callbacks or [])

    for epoch in range(epochs):
        run_epoch_start(_callbacks, epoch, runtime)
        method.model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_val_loss = None
        epoch_val_acc = None

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            total=len(train_loader) if hasattr(train_loader, "__len__") else None,
            unit="it",
            leave=True,
            file=sys.stderr,
            ncols=100,
        )
        for batch in pbar:
            out = method.train_step(batch, runtime)
            epoch_loss += out["loss"]
            n_batches += 1
            runtime["global_step"] = runtime.get("global_step", 0) + 1
            gs = runtime["global_step"]
            pbar.set_postfix(loss=f"{out['loss']:.4f}")
            if is_rank0 and save_steps > 0 and gs % save_steps == 0:
                torch.save(method.state_dict(), paths.ckpt_dir / f"step_{gs}.pt")
            if is_rank0 and n_batches % log_interval == 0:
                lr = method.optimizer.param_groups[0]["lr"] if getattr(method, "optimizer", None) else None
                step_log = {
                    "event": "step",
                    "epoch": epoch,
                    "step": runtime["global_step"],
                    "loss": out["loss"],
                    "grad_norm": out.get("grad_norm"),
                    "learning_rate": lr,
                }
                append_epoch_log(step_log, train_jsonl_path)
                if train_log_path:
                    train_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(train_log_path, "a") as f:
                        f.write(json.dumps(step_log, ensure_ascii=False) + "\n")

        # Validation（每 eval_interval 个 epoch 做一次）
        if val_loader and epoch % eval_interval == 0:
            val_metrics = _run_eval(method, val_loader, device, split="val")
            epoch_val_loss = val_metrics.get("loss", 0)
            epoch_val_acc = val_metrics.get("accuracy", 0)
            if is_rank0:
                with open(paths.val_metrics_path, "w") as f:
                    json.dump({"epoch": epoch, "val_loss": epoch_val_loss, "val_accuracy": epoch_val_acc}, f, indent=2)
                eval_log = {
                    "event": "eval",
                    "epoch": epoch,
                    "eval_loss": epoch_val_loss,
                    "eval_accuracy": epoch_val_acc,
                    "eval_runtime_s": val_metrics.get("runtime_s"),
                    "eval_samples_per_second": val_metrics.get("samples_per_second"),
                    "eval_steps_per_second": val_metrics.get("steps_per_second"),
                }
                append_epoch_log(eval_log, train_jsonl_path)
                if train_log_path:
                    train_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(train_log_path, "a") as f:
                        f.write(json.dumps(eval_log, ensure_ascii=False) + "\n")
                print(eval_log)
            # Best by monitor (higher is better for accuracy, lower for loss)
            if monitor == "val_accuracy" or "acc" in monitor.lower():
                improved = best_metric is None or epoch_val_acc > best_metric
            else:
                improved = best_metric is None or epoch_val_loss < best_metric
            if improved:
                best_metric = epoch_val_acc if "acc" in monitor.lower() else epoch_val_loss
                best_epoch = epoch
                if is_rank0:
                    torch.save(method.state_dict(), paths.best_ckpt_path)

        if is_rank0:
            torch.save(method.state_dict(), paths.last_ckpt_path)

        train_loss_avg = epoch_loss / max(n_batches, 1)
        metrics = {
            "train_loss": train_loss_avg,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc,
        }
        run_epoch_end(_callbacks, epoch, runtime, metrics)
        # 每 epoch 在终端打印 train_loss + val 指标（写到 stderr，否则 stdout 被重定向时终端看不到）
        if is_rank0:
            msg = f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss_avg:.4f}"
            if epoch_val_loss is not None:
                msg += f"  val_loss={epoch_val_loss:.4f}  val_acc={epoch_val_acc:.4f}"
            print(msg, file=sys.stderr)
            if train_log_path:
                train_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(train_log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")


    return {
        "status": "ok",
        "global_step": runtime.get("global_step", 0),
        "epochs_ran": epochs,
        "last_ckpt_path": str(paths.last_ckpt_path),
        "best_ckpt_path": str(paths.best_ckpt_path) if best_metric is not None else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
    }


def _run_eval(method, loader, device, split="val") -> dict:
    method.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch["split"] = split
            out = method.eval_step(batch, {"device": device})
            total_loss += out["loss"]
            mask = batch["val_mask"] if split == "val" else batch["test_mask"]
            preds = out["preds"]
            targets = out["targets"]
            correct += (preds[mask] == targets[mask]).sum().item()
            total += mask.sum().item()
            n_batches += 1
    runtime_s = time.perf_counter() - t0
    n_samples = total
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
        "runtime_s": runtime_s,
        "n_batches": n_batches,
        "n_samples": n_samples,
        "samples_per_second": n_samples / runtime_s if runtime_s > 0 else 0,
        "steps_per_second": n_batches / runtime_s if runtime_s > 0 else 0,
    }