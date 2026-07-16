"""HDN-DDI baseline — Sun & Zheng 2025, BMC Bioinformatics.

Layout — task-separated subfolders (per CLAUDE.md §"Baseline 目录布局
规范")::

  hdn_ddi/
  ├── __init__.py            ← this file (re-exports for external code)
  ├── _shared.py             ← cross-task helpers (drug_smiles_dict,
  │                            bipartite_edge_index_y1)
  ├── models.py              ← shared base HDN_DDI model class
  ├── layers.py              ← shared GAT / RESCAL / CoAttention layers
  ├── mol_features.py        ← shared BRICS-aware featurizer (66-dim,
  │                            3-level atoms/frags/super, paper Table S1)
  ├── binary_cls/            ← per-pair binary task (paper's native task)
  │   ├── __init__.py
  │   └── baseline.py        ← HDNDDIBaseline (paper-faithful BRICS)
  └── multi_cls/             ← per-pair K-way task (NOT in original paper)
      ├── __init__.py
      ├── baseline.py        ← HDNDDIMulticlassBaseline (BRICS + K-way head)
      └── model.py           ← HDN_DDI_MC (K-way RESCAL einsum subclass)

Registered variants:

* ``"hdn_ddi"``     — binary_cls (paper-native task), paper-faithful
                      BRICS-aware: refined BRICS 3-level graph +
                      66-dim Table-S1 features + y==1 substructure
                      bipartite + LambdaLR(0.96^epoch) +
                      per-epoch dynamic negatives.
* ``"hdn_ddi_mc"``  — multi_cls (K-way DDI type prediction, NOT in
                      original paper).  Inherits the binary baseline's
                      BRICS-aware encoder; only swaps the output head
                      to (B, K) softmax + cross-entropy + top1/macro-F1.

Both variants are **independently maintained copies** of the logic in
``Code/reproductions/HDN-DDI/`` (per CLAUDE.md §"复现 → baseline 迁移
规范" — no cross-folder imports, no shared files).
"""

from __future__ import annotations

from baseline.hdn_ddi.binary_cls import HDNDDIBaseline
from baseline.hdn_ddi.multi_cls import HDNDDIMulticlassBaseline
from baseline.hdn_ddi.multi_label_cls import HDNDDIMultilabelBaseline

__all__ = [
    "HDNDDIBaseline",
    "HDNDDIMulticlassBaseline",
    "HDNDDIMultilabelBaseline",
]
