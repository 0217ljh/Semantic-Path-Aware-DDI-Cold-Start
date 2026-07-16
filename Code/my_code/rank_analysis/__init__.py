"""Cold-start (S2) rank-analysis pipeline. Harness-owned protocol; models plug in
via the RankModel ABC (setup / train_epoch / encode_pairs)."""
from .specs import (TaskSpec, S2ProtocolSpec, ScoringContext, PairEncoding,
                    TrainEpochOutput)
from .data import RankData, load_rank_data, ref_degree_matrix
from .framework import RankModel, S2Protocol, RankHarness, RankRunWriter, EpochData

__all__ = ["TaskSpec", "S2ProtocolSpec", "ScoringContext", "PairEncoding",
           "TrainEpochOutput", "RankData", "load_rank_data", "ref_degree_matrix",
           "RankModel", "S2Protocol", "RankHarness", "RankRunWriter", "EpochData"]
