"""NBFNet v1.71 trainer — first-wave runtime acceleration (subclass of v1.7).

Adds, on top of :class:`my_code.models.nbfnet_v1_7.nbfnet_trainer.NBFNetTrainer`:

  1. TF32 matmul/cudnn (Ampere+)                         -> faster matmul, ~no acc loss
  2. AMP autocast (bf16 default) around all forward       -> ~1.5-2x + half activations
  3. eval cadence (eval every N epochs)                   -> cut eval cost
  4. per-epoch train-time / eval-time split logging       -> profile where time goes

The v1.7 paper-faithful MODEL is reused unchanged (imported via the parent). The
BF recurrence math is identical; only matmul/activation precision differs under
TF32/AMP. Early-stop and ReduceLROnPlateau are stepped only on eval epochs so
their semantics stay consistent with the (now sparser) validation cadence.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from my_code.models.nbfnet_v1_7.nbfnet_trainer import NBFNetTrainer


class NBFNetTrainerV171(NBFNetTrainer):
    """v1.7 trainer + first-wave runtime acceleration (lossless / precision-only)."""

    def __init__(
        self,
        *args,
        use_amp: bool = True,
        amp_dtype: str = "bf16",          # "bf16" (preferred, no GradScaler) or "fp16"
        use_tf32: bool = True,
        eval_every_n_epochs: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_amp = bool(use_amp)
        self.amp_dtype = torch.bfloat16 if str(amp_dtype) == "bf16" else torch.float16
        self.use_tf32 = bool(use_tf32)
        self.eval_every_n_epochs = max(1, int(eval_every_n_epochs))

    # ------------------------------------------------------------------
    # Accel 1: TF32 (set before model build)
    # ------------------------------------------------------------------
    def init_model(self) -> None:
        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        super().init_model()
        print(
            f"[nbfnet-v1.71] accel: tf32={self.use_tf32} amp={self.use_amp} "
            f"amp_dtype={'bf16' if self.amp_dtype==torch.bfloat16 else 'fp16'} "
            f"eval_every_n_epochs={self.eval_every_n_epochs}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Accel 2: AMP autocast around ALL forward (train/val/predict all route
    # through _score_pairs, so one override covers everything). bf16 needs no
    # GradScaler. The BCE loss in the parent runs outside autocast (fine).
    # ------------------------------------------------------------------
    def _score_pairs(self, batch_a, batch_b, aug_edges, training):
        if self.use_amp and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                out = super()._score_pairs(batch_a, batch_b, aug_edges, training)
            # Upcast the final (B,) logits to fp32 so BCE / .numpy() / sigmoid in
            # the inherited _train_epoch / _validate / predict_proba work (numpy
            # has no bf16). Internal BF/matmul still ran in low precision.
            return out.float()
        return super()._score_pairs(batch_a, batch_b, aug_edges, training)

    # ------------------------------------------------------------------
    # Accel 3+4: eval cadence + train/eval timing split.
    # Override of NBFNetTrainer.fit — same logic, plus cadence + split timing.
    # ------------------------------------------------------------------
    def fit(
        self,
        val_pos_pairs: pd.DataFrame,
        val_neg_pairs: pd.DataFrame,
        drug_id_map: dict[str, int],
        run_dir: Path,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("init_model must be called before fit")
        if self.train_ddi_triplets is None:
            raise RuntimeError("setup_graph must be called before fit")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        val_pos_a, val_pos_b = self._pairs_to_node_idx(val_pos_pairs, drug_id_map)
        val_neg_a, val_neg_b = self._pairs_to_node_idx(val_neg_pairs, drug_id_map)
        val_a = np.concatenate([val_pos_a, val_neg_a])
        val_b = np.concatenate([val_pos_b, val_neg_b])
        val_y = np.concatenate([np.ones(len(val_pos_a)), np.zeros(len(val_neg_a))])
        print(
            f"[nbfnet-v1.71] val set. n_pos={len(val_pos_a)} n_neg={len(val_neg_a)}",
            flush=True,
        )

        fit_start = time.time()
        total_train_s = 0.0
        total_eval_s = 0.0

        for epoch in range(1, self.n_epochs + 1):
            (edge_src, edge_dst, edge_rel), epoch_targets = self._build_epoch_kg(epoch)
            if len(epoch_targets) == 0:
                print(
                    f"[nbfnet-v1.71] [ep {epoch}/{self.n_epochs}] 0 targets after "
                    f"shuffle_train(mode={self.shuffle_train_mode}); skip",
                    flush=True,
                )
                continue

            neg_rng = np.random.default_rng(self.seed + 100_000 + epoch)
            n_neg = self.neg_ratio * len(epoch_targets)
            neg_pairs = self._sample_negatives(epoch_targets, n_neg, neg_rng)

            # --- timed train ---
            t_train = time.time()
            epoch_loss = self._train_epoch(
                epoch_targets, neg_pairs, (edge_src, edge_dst, edge_rel), epoch,
            )
            train_s = time.time() - t_train
            total_train_s += train_s

            # --- eval cadence: eval on every Nth epoch and always the last ---
            do_eval = (
                self.eval_strategy == "epoch"
                and (epoch % self.eval_every_n_epochs == 0 or epoch == self.n_epochs)
            )
            if do_eval:
                t_eval = time.time()
                val_auc, val_ap = self._validate(val_a, val_b, val_y)
                eval_s = time.time() - t_eval
                total_eval_s += eval_s
                self.scheduler.step(val_auc)
                print(
                    f"[nbfnet-v1.71] [ep {epoch}/{self.n_epochs}] loss={epoch_loss:.4f} "
                    f"val_auc={val_auc:.4f} val_ap={val_ap:.4f} "
                    f"train_t={train_s:.1f}s eval_t={eval_s:.1f}s",
                    flush=True,
                )
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.best_epoch = epoch
                    self.best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self.save_state(run_dir / "best_model.pt")
                if self._should_early_stop(epoch):
                    print(
                        f"[nbfnet-v1.71] early stop at epoch {epoch} "
                        f"(best ep {self.best_epoch}, val_auc={self.best_val_auc:.4f})",
                        flush=True,
                    )
                    break
            else:
                print(
                    f"[nbfnet-v1.71] [ep {epoch}/{self.n_epochs}] loss={epoch_loss:.4f} "
                    f"train_t={train_s:.1f}s eval=skipped",
                    flush=True,
                )

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        fit_time = time.time() - fit_start
        print(
            f"[nbfnet-v1.71] fit done. best_val_auc={self.best_val_auc:.4f} "
            f"best_epoch={self.best_epoch} time={fit_time:.1f}s "
            f"(train={total_train_s:.1f}s eval={total_eval_s:.1f}s)",
            flush=True,
        )

        return {
            "best_val_auc": float(self.best_val_auc),
            "best_epoch": int(self.best_epoch),
            "fit_sec": float(fit_time),
            "train_sec": float(total_train_s),
            "eval_sec": float(total_eval_s),
        }
