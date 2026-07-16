"""Per-baseline RankModel wrappers (backbone glue to the shared harness)."""
from .emergnn import EmerGNNRankWrapper
from .knowddi import KnowDDIRankWrapper
from .mkg_fenn import MKGFENNRankWrapper
from .rgcn import RGCNRankWrapper
from .ssi_ddi import SSIDDIRankWrapper
from .tiger import TIGERRankWrapper

#: registry key -> wrapper class (extended as baselines are ported)
WRAPPERS = {
    "rgcn": RGCNRankWrapper,
    "emergnn": EmerGNNRankWrapper,
    "knowddi": KnowDDIRankWrapper,
    "ssiddi": SSIDDIRankWrapper,      # matches model_meta BASELINES key
    "tiger": TIGERRankWrapper,        # matches model_meta BASELINES key
    "mkgfenn": MKGFENNRankWrapper,    # matches model_meta BASELINES key
}

__all__ = ["RGCNRankWrapper", "EmerGNNRankWrapper", "KnowDDIRankWrapper",
           "SSIDDIRankWrapper", "TIGERRankWrapper", "MKGFENNRankWrapper", "WRAPPERS"]
