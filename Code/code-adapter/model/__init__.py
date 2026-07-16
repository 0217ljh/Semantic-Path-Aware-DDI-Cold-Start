"""Stage 4 model layer: RankModel seam + protocol execution + training runner."""
from .contracts import (CheckpointPolicy, EpochData, PairEncoding, RankModel,
                        ScoringContext, TrainEpochOutput)
from .protocol import ColdStartProtocol
from .runner import run_training

__all__ = ["RankModel", "ScoringContext", "PairEncoding", "EpochData",
           "TrainEpochOutput", "CheckpointPolicy", "ColdStartProtocol",
           "run_training"]
