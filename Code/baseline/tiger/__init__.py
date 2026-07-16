"""TIGER baseline — paper Su et al. (AAAI 2024), binary + multi-class.

Default mode is **dual-channel** (molecule GraphTransformer + BKG-subgraph
GraphTransformer), paper-faithful to Su et al. AAAI 2024.

Two project-side adaptations exist but are **off by default** (set
explicitly to enable; default keeps paper-faithful behaviour for warm-
start S0 reproducibility):
  - ``cold_start_patch=True`` / CLI ``--tiger-cold-start-patch``: enable
    the cold-start center-node patch (`model._patch_unseen_center_nodes`)
    which replaces unseen drugs' BKG-center embedding with a projection
    of the mol-graph embedding. **NOT in the paper.** Useful for S1/S2.
  - ``extractor`` selector (``"randomWalk"`` default / ``"khop-subtree"`` /
    ``"probability"``) — all 3 paper extractors are wired (TIGER-DW /
    TIGER-KS / TIGER-P). CLI ``--tiger-extractor``.

BKG source: our merged DrugBank+Hetionet+PrimeKG parquet (passed via
``merged_kg_path``); the upstream ``dataset/drugbank/networks.txt`` path
is not required.

Backward-compat: pass ``mol_only=True`` (or ``kg_source='none'``) to
fall back to mol-only adapter (no KG, no BKG).

Importing this submodule registers:
  - :class:`TIGERBaseline` under ``"tiger"`` (binary, paper-native task).
  - :class:`TIGERMulticlassBaseline` under ``"tiger_mc"`` (K-way DDI
    type prediction — paper extension, NOT in original Su et al.).
"""

from __future__ import annotations

from baseline.tiger.binary_cls import TIGERBaseline
from baseline.tiger.multi_cls import TIGERMulticlassBaseline
from baseline.tiger.multi_label_cls import TIGERMultilabelBaseline

__all__ = ["TIGERBaseline", "TIGERMulticlassBaseline", "TIGERMultilabelBaseline"]
