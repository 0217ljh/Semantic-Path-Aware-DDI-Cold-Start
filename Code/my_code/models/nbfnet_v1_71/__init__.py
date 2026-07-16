"""NBFNet v1.71 — first-wave runtime acceleration over v1.7.

v1.71 is a thin subclass of the v1.7 trainer (:class:`NBFNetTrainer`). It reuses
the v1.7 paper-faithful model (:mod:`nbfnet_v1_7.nbfnet_model`) UNCHANGED and only
adds lossless / low-risk runtime accelerations in the trainer:

  - TF32 matmul/cudnn (Ampere+)
  - AMP autocast (bf16 by default) around all forward (train/val/predict)
  - configurable eval cadence (eval every N epochs) with early-stop/scheduler
    aligned to eval events
  - per-epoch train-time vs eval-time split logging

The BF math is unchanged (only matmul/activation precision differs under
TF32/AMP); v1.7 stays the paper-faithful reference (0.7911 merged-KG baseline).
"""
from my_code.models.nbfnet_v1_71.nbfnet_trainer import NBFNetTrainerV171

__all__ = ["NBFNetTrainerV171"]
