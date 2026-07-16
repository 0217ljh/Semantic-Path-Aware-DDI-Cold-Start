"""DDI-LoRA adapter (the proposed M_A.M_B typed semantic adapter). Paper Method.

Step-1 contract: AdapterConfig, AdapterBatch, MediatorAssignment (M_A),
TypedPrototypes (M_B), DDILoRAAdapter (read-out stubbed for step 2).
"""
from __future__ import annotations

from .config import AdapterConfig
from .batch import AdapterBatch
from .modules import DDILoRAAdapter, MediatorAssignment, TypedPrototypes, PathwayFeature
from .losses import AdapterLoss

__all__ = ["AdapterConfig", "AdapterBatch", "DDILoRAAdapter", "MediatorAssignment",
           "TypedPrototypes", "PathwayFeature", "AdapterLoss"]
