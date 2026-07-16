"""MKG-FENN baseline — paper Hou et al. (Inf. Fusion 2024), binary
adaptation. Four parallel KG-GNN channels (drug-entity, drug-substructure,
drug-DDI, drug-property) fused into a 2-class softmax head.

Migrated from ColdDDI repo (D:\\My-Research\\03-Projects\\ColdDDI\\
Code-Released-Formal\\coldddi\\baselines\\mkg_fenn) on 2026-06-08.
Used as a cold-start multi-modal misalignment exemplar baseline for the
v1.6.x AAAI 2027 paper (see Notes/Log/v1_6x_paper_frame.md Part 2).

NOTE on imports: source code references `coldddi.baselines.*` namespace.
When activating this baseline in the new project, update imports in
baseline.py and model.py to match destination namespace (e.g.,
`baseline.mkg_fenn.*` or whatever the project registry expects).

Importing this submodule registers :class:`MKGFENNBaseline` under
``"mkg_fenn"``.
"""

from __future__ import annotations

# Lazy import — defer until imports adapted to destination project namespace.
# When ready, uncomment:
# from baseline.mkg_fenn.baseline import (
#     PAPER_HYPERPARAMS,
#     MKGFENNBaseline,
# )
#
# __all__ = ["MKGFENNBaseline", "PAPER_HYPERPARAMS"]

__all__: list[str] = []
