"""NBFNet v1.72 — dual-source signed-bilinear interference (action-aware).

v1.72 = v1.71 (accel) backbone + SIGNED fields + ACTION-signed edges + an
endpoint Hadamard interference readout. The interference core is signed+hadamard;
a built-in 2×2 (activation × combine) with action held ON enables clean
attribution. v1.7 / v1.71 are untouched.
"""
from my_code.models.nbfnet_v1_72.nbfnet_model import NBFNetV172, NBFLayerV172
from my_code.models.nbfnet_v1_72.nbfnet_trainer import NBFNetTrainerV172

__all__ = ["NBFNetV172", "NBFLayerV172", "NBFNetTrainerV172"]
