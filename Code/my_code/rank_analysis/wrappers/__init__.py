"""Per-model wrappers implementing the RankModel interface. One wrapper per model;
they all plug into the SAME shared RankHarness.

"wrapper" = model-specific glue (build / train_epoch / encode_pairs for THIS model).
The word "adapter" is reserved for the method's semantic M_A·M_B adapter, which is a
model component we attach, not this pipeline glue.
"""
from .rgcn import RGCNRankWrapper
from .emergnn import EmerGNNRankWrapper
from .knowddi import KnowDDIRankWrapper

WRAPPERS = {"rgcn": RGCNRankWrapper, "emergnn": EmerGNNRankWrapper,
            "knowddi": KnowDDIRankWrapper}

__all__ = ["RGCNRankWrapper", "EmerGNNRankWrapper", "KnowDDIRankWrapper", "WRAPPERS"]
