"""SSI-DDI baseline — paper Nyamabo et al. (BIB 2022), binary adaptation.

Importing this submodule registers :class:`SSIDDIBaseline` under the
``"ssi_ddi"`` name. Like the other baselines, the wrapper is intentionally
thin: molecular GAT blocks + co-attention + RESCAL come from
:mod:`baseline.ssi_ddi.layers` /
:mod:`baseline.ssi_ddi.models` (verbatim from the research repo,
import paths aside); molecular feature extraction is in
:mod:`baseline.ssi_ddi.mol_features`; all glue lives in
:class:`SSIDDIBaseline.fit`.
"""

from __future__ import annotations

from baseline.ssi_ddi.baseline import SSIDDIBaseline

__all__ = ["SSIDDIBaseline"]
