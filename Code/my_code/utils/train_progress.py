"""Training-progress logger per CLAUDE.md '训练进度日志规范'.

Provides a `TrainProgress` class that any baseline / new model's `fit()`
loop should use to emit:
  - Per-step rolling-mean loss (interval = `log_step_every`, configurable)
  - Per-epoch summary (always)
  - Intermediate eval triggers (`should_eval_step` / `should_eval_epoch`)
  - Intermediate save triggers (`should_save_step` / `should_save_epoch`)
  - Unified eval-log formatter (`log_eval`)

Mirrors HuggingFace Trainer's logging/eval/save settings semantics.

Usage:
    from my_code.utils.train_progress import TrainProgress

    prog = TrainProgress(
        total_epochs=self.n_epochs,
        log_step_every=self.log_step_every,
        eval_strategy=self.eval_strategy,
        eval_steps=self.eval_steps,
        save_strategy=self.save_strategy,
        save_steps=self.save_steps,
        prefix="[hdn_ddi] ",
    )
    for epoch in range(self.n_epochs):
        prog.epoch_start(epoch)
        for batch in dataloader:
            loss = ...
            prog.step(loss.item())
            if prog.should_eval_step():
                metrics = self._eval(val)
                prog.log_eval(metrics, scope="step")
            if prog.should_save_step():
                self.save_checkpoint(step=prog.step_count)
        if prog.should_eval_epoch():
            metrics = self._eval(val)
            prog.log_eval(metrics, scope="epoch")
        if prog.should_save_epoch():
            self.save_checkpoint(epoch=epoch + 1)
        prog.epoch_end(extra=metrics if prog.should_eval_epoch() else None)
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


_VALID_STRATEGIES = {"no", "epoch", "steps"}


class TrainProgress:
    """Unified training-loop logger + eval/save interval decider.

    Args:
        total_epochs: total epoch count (for the `ep K/N` print format).
        log_step_every: print rolling-mean step loss every N steps (default 50).
        rolling_window: how many recent steps to average for the print
            (defaults to log_step_every).
        total_steps_per_epoch: optional; if known, included in `step X/Y` format.
        prefix: optional prefix prepended to every line (e.g., "[fit]").
        eval_strategy: "no" | "epoch" | "steps" (default "epoch").
        eval_steps: int, used when `eval_strategy="steps"` (default 500).
        save_strategy: "no" | "epoch" | "steps" (default "no").
        save_steps: int, used when `save_strategy="steps"` (default 500).
        flush: whether to flush stdout after each print (default True).
    """
    def __init__(
        self,
        total_epochs: int,
        log_step_every: int = 50,
        rolling_window: Optional[int] = None,
        total_steps_per_epoch: Optional[int] = None,
        prefix: str = "",
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        save_strategy: str = "no",
        save_steps: int = 500,
        flush: bool = True,
    ):
        if eval_strategy not in _VALID_STRATEGIES:
            raise ValueError(f"eval_strategy={eval_strategy!r} not in {_VALID_STRATEGIES}")
        if save_strategy not in _VALID_STRATEGIES:
            raise ValueError(f"save_strategy={save_strategy!r} not in {_VALID_STRATEGIES}")
        self.total_epochs = total_epochs
        self.log_step_every = max(1, int(log_step_every))
        self.rolling_window = max(1, int(rolling_window or self.log_step_every))
        self.total_steps_per_epoch = total_steps_per_epoch
        self.prefix = prefix
        self.eval_strategy = eval_strategy
        self.eval_steps = max(1, int(eval_steps))
        self.save_strategy = save_strategy
        self.save_steps = max(1, int(save_steps))
        self.flush = flush

        self.epoch: int = -1
        self.step_count: int = 0
        self.epoch_loss_sum: float = 0.0
        self.epoch_step_count: int = 0
        self.rolling_losses: deque = deque(maxlen=self.rolling_window)
        self.epoch_t0: float = 0.0

    # ----- per-step loss logging -----
    def epoch_start(self, epoch: int) -> None:
        self.epoch = epoch
        self.step_count = 0
        self.epoch_loss_sum = 0.0
        self.epoch_step_count = 0
        self.rolling_losses.clear()
        self.epoch_t0 = time.time()

    def step(self, loss: float) -> None:
        """Call after each training step. Pass scalar loss value (use loss.item())."""
        self.step_count += 1
        self.epoch_step_count += 1
        self.epoch_loss_sum += float(loss)
        self.rolling_losses.append(float(loss))
        if self.step_count % self.log_step_every == 0:
            mean_loss = sum(self.rolling_losses) / len(self.rolling_losses)
            step_str = f"step {self.step_count}"
            if self.total_steps_per_epoch is not None:
                step_str += f"/{self.total_steps_per_epoch}"
            line = (
                f"{self.prefix}[ep {self.epoch+1}/{self.total_epochs} "
                f"{step_str}] loss={mean_loss:.4f}"
            )
            print(line, flush=self.flush)

    def epoch_end(self, extra: Optional[dict] = None) -> None:
        elapsed = time.time() - self.epoch_t0
        mean_loss = (
            self.epoch_loss_sum / self.epoch_step_count
            if self.epoch_step_count > 0
            else float("nan")
        )
        parts = [
            f"{self.prefix}[ep {self.epoch+1}/{self.total_epochs}]",
            f"mean_loss={mean_loss:.4f}",
            f"time={elapsed:.1f}s",
        ]
        if extra:
            for k, v in extra.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
        print(" ".join(parts), flush=self.flush)

    # ----- eval interval triggers -----
    def should_eval_step(self) -> bool:
        return (
            self.eval_strategy == "steps"
            and self.step_count > 0
            and self.step_count % self.eval_steps == 0
        )

    def should_eval_epoch(self) -> bool:
        return self.eval_strategy == "epoch"

    def log_eval(self, metrics: dict, scope: str = "epoch") -> None:
        """Print a uniform eval-log line. scope='step' or 'epoch'.

        Output:
            [<prefix>][eval @ ep <E> step <S>] key1=val1 key2=val2 ...
        """
        if scope not in ("step", "epoch"):
            raise ValueError(f"scope must be 'step' or 'epoch', got {scope!r}")
        loc = f"ep {self.epoch+1}"
        if scope == "step":
            loc += f" step {self.step_count}"
        parts = [f"{self.prefix}[eval @ {loc}]"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" ".join(parts), flush=self.flush)

    # ----- save interval triggers -----
    def should_save_step(self) -> bool:
        return (
            self.save_strategy == "steps"
            and self.step_count > 0
            and self.step_count % self.save_steps == 0
        )

    def should_save_epoch(self) -> bool:
        return self.save_strategy == "epoch"

    def log_save(self, path, scope: str = "epoch") -> None:
        """Print a uniform save-log line."""
        loc = f"ep {self.epoch+1}"
        if scope == "step":
            loc += f" step {self.step_count}"
        print(f"{self.prefix}[save @ {loc}] {path}", flush=self.flush)
