"""Atom/bond featurizer + per-drug PyG molecular-graph builder for MRCGNN's TrimNet.

Ported + adapted from upstream
``Paper/Reference/Original-Code/MRCGNN/codes for MRCGNN/trimnet/data_preprocessing.py``
(the ``atom_features`` / ``bond_attr`` / ``get_mol_edge_list_and_feat_mtx`` functions
and the per-drug ``Data`` construction). File-independence (CLAUDE.md §Baseline):
COPY+adapt, no import of ``Paper/Reference/Original-Code/`` or ``reproductions/``.

Interface follows SSI-DDI's ``build_drug_graphs`` (``Code/baseline/ssi_ddi/mol_features.py``):
build from a ``{drug_id: smiles}`` dict, NOT the upstream ``data/drug_listxiao.csv``.
Drugs whose SMILES fail to parse (or have zero atoms) are skipped and reported.

Feature dims (faithful to upstream ``data_preprocessing.py:44-75``, ``explicit_H=True``,
``use_chirality=False``):
  * 44 one-hot atom symbol (with 'Unknown' fallback)   (upstream ``:48-52``)
  *  4 numeric: degree/10, implicit valence, formal charge, num radical  (``:53-54``)
  *  5 one-hot hybridization (SP/SP2/SP3/SP3D/SP3D2)   (``:55-59``)
  *  1 is-aromatic                                     (``:59``)
  *  1 explicit total Hs                               (``:61-62``)
  => 55 atom-feature dims total. This is EXACTLY upstream's ``TrimNet(55, 10, ...)``
     input dim (``trimnet/train.py:250``, ``models.py:122-123`` ``LayerNorm(55)`` / ``Linear(55, ...)``).

Bond dims (upstream ``bond_attr`` ``data_preprocessing.py:103-127``, ``use_chirality=True``):
  * 6 bond-type/conjugation/ring flags + 4 one-hot stereo => 10 edge-feature dims,
    matching upstream ``TrimNet(55, 10, ...)`` edge dim (``trimnet/train.py:250``).

NOTE (upstream quirk, preserved): upstream builds bond features via a nested O(n^2)
``bond_attr`` scan over ``GetBondBetweenAtoms`` (``data_preprocessing.py:103-127``) and a
SEPARATE undirected ``edge_list`` from ``mol.GetBonds()`` (``:137-140``). We preserve the
same two constructions and the same undirected doubling so the atom/edge counts fed to
TrimNet match upstream exactly.
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data

# Silence rdkit's noisy parse warnings for malformed SMILES.
RDLogger.DisableLog("rdApp.*")


# upstream data_preprocessing.py:50-52 — 44-symbol list incl. 'Unknown'
_ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
    "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]
# upstream data_preprocessing.py:56-58 — 5 hybridization types
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
# upstream data_preprocessing.py:122-123 — stereo one-hot categories
_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]

#: atom-feature width — MUST equal 55 (upstream TrimNet in_dim).
ATOM_FEATURE_DIM: int = len(_ATOM_SYMBOLS) + 4 + len(_HYBRIDIZATIONS) + 1 + 1
#: bond-feature width — MUST equal 10 (upstream TrimNet edge_in_dim).
BOND_FEATURE_DIM: int = 6 + len(_STEREO)


def _one_of_k_unk(x, allowable):
    """upstream one_of_k_encoding_unk (data_preprocessing.py:39-42)."""
    if x not in allowable:
        x = allowable[-1]
    return list(map(lambda s: x == s, allowable))


def atom_features(atom) -> torch.Tensor:
    """55-dim atom feature (upstream ``atom_features`` data_preprocessing.py:44-75,
    ``explicit_H=True``, ``use_chirality=False``)."""
    results = (
        _one_of_k_unk(atom.GetSymbol(), _ATOM_SYMBOLS)
        + [
            atom.GetDegree() / 10,
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
        ]
        + _one_of_k_unk(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [atom.GetIsAromatic()]
        + [atom.GetTotalNumHs()]
    )
    return torch.from_numpy(np.array(results, dtype=np.float32))


def bond_attr(mol, use_chirality: bool = True) -> np.ndarray:
    """Bond-feature matrix (upstream ``bond_attr`` data_preprocessing.py:103-127).

    Nested O(n^2) scan over ``GetBondBetweenAtoms(i, j)`` for i != j — preserved
    verbatim so the edge count matches the separate undirected ``edge_list`` doubling.
    """
    feat = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE,
                        bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE,
                        bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing(),
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + _one_of_k_unk(
                            str(bond.GetStereo()), _STEREO)
                    feat.append(bond_feats)
    return np.array(feat)


def mol_to_graph(mol, smiles: str) -> Data | None:
    """Build a per-drug PyG ``Data`` (upstream ``get_mol_edge_list_and_feat_mtx``
    data_preprocessing.py:128-143 + ``__create_graph_data`` :248-253).

    Returns ``None`` for empty / unparseable molecules — caller skips the drug.
    ``x`` = (n_atoms, 55) atom feats; ``edge_index`` = undirected bonds (2, 2E);
    ``edge_attr`` = (E_scan, 10) bond feats from the O(n^2) scan (upstream keeps the
    two constructions separate — preserved).
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    feat_pairs = [(a.GetIdx(), atom_features(a)) for a in mol.GetAtoms()]
    feat_pairs.sort()  # align feature rows to atom idx (upstream :133)
    _, feats = zip(*feat_pairs)
    x = torch.stack(feats)

    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
    edge_attr = torch.FloatTensor(bond_attr(MolFromSmiles(smiles)))
    if len(edge_list):
        undirected = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0)  # upstream :140
        edge_index = undirected.T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_drug_graphs(
    smiles_dict: dict[str, str],
) -> tuple[dict[str, Data], list[str]]:
    """Parse SMILES → ``{drug_id: pyg.Data}`` (mirrors SSI-DDI's ``build_drug_graphs``).

    Drugs whose SMILES fail to parse (or have zero atoms) are skipped and reported
    as the second return value — same filter semantics as upstream ``DrugDataset``
    against ``MOL_EDGE_LIST_FEAT_MTX`` (data_preprocessing.py:198-200).
    """
    graphs: dict[str, Data] = {}
    missing: list[str] = []
    for drug_id, smi in smiles_dict.items():
        if smi is None or not str(smi).strip():
            missing.append(drug_id)
            continue
        smi = str(smi).strip()
        mol = Chem.MolFromSmiles(smi)
        data = mol_to_graph(mol, smi)
        if data is None:
            missing.append(drug_id)
            continue
        graphs[drug_id] = data
    return graphs, missing


__all__ = [
    "atom_features",
    "bond_attr",
    "mol_to_graph",
    "build_drug_graphs",
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
]
