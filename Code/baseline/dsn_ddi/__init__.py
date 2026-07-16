"""DSN-DDI baseline — paper Li et al. (BIB 2023), binary adaptation.

DSN-DDI extends SSI-DDI with an inter-graph attention layer that
operates over a complete bipartite graph between the two drug
molecules' atoms. Importing this submodule registers
:class:`DSNDDIBaseline` under the ``"dsn_ddi"`` name.
"""

from __future__ import annotations

from baseline.dsn_ddi.baseline import DSNDDIBaseline

__all__ = ["DSNDDIBaseline"]
