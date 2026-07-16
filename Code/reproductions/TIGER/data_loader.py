"""TIGER reproduction data loader — thin wrapper over the upstream
``data_process.py`` functions to assemble the inputs needed by
:class:`model.tiger.TIGER`.

Paper datasets live at the canonical reproduction path (per CLAUDE.md
§"复现代码 (Reproduction) 规范" §1):

    Code/reproductions/TIGER/_Original-Dataset/<dataset>/
    ├── drug_smiles.txt           # 1st line header, then  id\\tsmiles  per line
    ├── networks.txt              # 1st line header, then  head tail rel  per line
    ├── ddi.txt                   # rows: drug1 drug2 rel label (1:1 balanced)
    ├── entity2id.txt             # upstream-shipped entity vocab
    └── necessary/                # required preprocessing artefacts
        └── mol_sp__official.json # upstream-shipped SMILES SP cache (paper format)
        # mol_sp__mine.json       # written here when we rebuild from scratch

Download source (manual): Google Drive zip in ``_paper-and-GitHub/README.md``
→ ``https://drive.google.com/file/d/13ZFDZ28Eam5C5gs-yw-UZ6Yi_X2jkN69/``
(use ``gdown 13ZFDZ28Eam5C5gs-yw-UZ6Yi_X2jkN69``).

Runtime subgraph caches (e.g. ``<dataset>/khop-subtree/*.json``) are
written next to the inputs by ``data_process.generate_node_subgraphs``;
they're NOT classified as ``necessary/`` because they're builder output
on a per-extractor basis, not paper-shipped artefacts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from data_process import (
    generate_node_subgraphs,
    read_interactions,
    read_network,
    read_smiles,
    smile_to_graph,
)

# Canonical paper-dataset root (project rule: paper-original datasets live
# inside the reproduction folder under ``_Original-Dataset/``, NOT in
# ``Code/data/``). See CLAUDE.md §"复现代码 (Reproduction) 规范" §1.
REPO_DATA_DIR = Path(__file__).resolve().parent / "_Original-Dataset"


@dataclass
class TigerDataBundle:
    """All artefacts needed to build a :class:`TIGER` model + DataLoader."""

    interactions: np.ndarray  # (N, 2) — drug1_id, drug2_id (int)
    labels: np.ndarray  # (N,) — 0/1
    smile_graph: dict  # drug_id (str) → mol-graph tuple
    drug_subgraphs: dict  # drug_id (str) → subgraph tuple
    stats: dict  # see :func:`load_data`


def load_data(
    dataset: str,
    extractor: str = "randomWalk",
    *,
    data_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    khop: int = 2,
    fixed_num: int = 32,
    graph_fixed_num: int = 1,
) -> TigerDataBundle:
    """Load the 3 paper datasets (drugbank / kegg / ogbl-biokg) — or any
    user-supplied dataset that follows the upstream file convention.

    Returns the same artefacts that upstream ``main.py:load_data`` returns,
    plus a stats dict suitable for :class:`TIGER` construction.

    ``data_root`` defaults to :data:`REPO_DATA_DIR` (the canonical
    ``reproductions/TIGER/_Original-Dataset/``). Override only for testing.
    ``cache_root`` defaults to ``data_root`` so subgraph caches sit next to
    the inputs.
    """
    data_root = Path(data_root) if data_root is not None else REPO_DATA_DIR
    cache_root = Path(cache_root) if cache_root is not None else data_root
    data_path = data_root / dataset

    ligands = read_smiles(str(data_path / "drug_smiles.txt"))

    print("[tiger-repro] SMILES -> molecular graphs")
    smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(str(data_path), ligands)

    print("[tiger-repro] load BKG (networks.txt)")
    num_node, network_edge_index, network_rel_index, num_rel = read_network(
        str(data_path / "networks.txt")
    )

    print("[tiger-repro] load DDI interactions (ddi.txt)")
    interactions_label, all_contained_drugs = read_interactions(
        str(data_path / "ddi.txt"), smile_graph
    )
    interactions = interactions_label[:, :2]
    labels = interactions_label[:, 3]

    # Args shim for ``generate_node_subgraphs`` (matches upstream call).
    class _Args:
        pass
    args = _Args()
    args.extractor = extractor
    args.khop = khop
    args.fixed_num = fixed_num
    args.graph_fixed_num = graph_fixed_num

    print(f"[tiger-repro] generate {extractor} subgraphs (cache -> {cache_root})")
    drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(
        dataset,
        all_contained_drugs,
        network_edge_index,
        network_rel_index,
        num_rel,
        args,
        cache_root=str(cache_root),
    )

    # +1 for padding slot in Embedding layers (matches upstream main.py).
    stats = {
        "num_nodes": num_node + 1,
        "num_rel_mol": num_rel_mol_update + 1,
        "num_rel_graph": num_rel_update + 1,
        "num_interactions": len(interactions),
        "num_drugs_DDI": len(all_contained_drugs),
        "max_degree_graph": max_smiles_degree + 1,
        "max_degree_node": int(max_subgraph_degree) + 1,
    }

    print(f"[tiger-repro] data stats: {stats}")
    return TigerDataBundle(
        interactions=interactions,
        labels=labels,
        smile_graph=smile_graph,
        drug_subgraphs=drug_subgraphs,
        stats=stats,
    )
