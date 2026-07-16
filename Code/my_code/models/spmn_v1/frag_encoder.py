"""SPMN v1 — molecular fragment featurizer (M1): SMILES -> BRICS fragment
atom-subgraphs, for a learnable fragment GNN (g_frag).

Faithful to the design: g_frag is a GNN over each BRICS fragment's atom graph
(NOT a fixed fingerprint). This module does the training-free featurization
(atom/bond features + BRICS fragment atom-index sets, all from ONE RDKit mol so
indices are consistent); the learnable GNN lives in :mod:`frag_gnn`.

BRICS fragmentation (validated): FindBRICSBonds -> FragmentOnBonds(addDummies=
False) -> GetMolFrags returns atom-index tuples in the original mol indexing,
covering all atoms. Drugs with no BRICS bonds = a single whole-molecule
fragment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

# Atom featurisation vocab (compact, standard for molecular GNNs).
_ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
_HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
           Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
           Chem.HybridizationType.SP3D2]
#: atom feature dim: element(12+other) + degree(0..5+) + charge(-2..+2) +
#: hybridization(5+other) + aromatic(1) + numH(0..4+) + in_ring(1)
ATOM_FEAT_DIM = (len(_ATOM_LIST) + 1) + 7 + 5 + (len(_HYBRID) + 1) + 1 + 5 + 1
BOND_FEAT_DIM = 4  # single / double / triple / aromatic


def _onehot(value, choices) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def _atom_features(atom: Chem.Atom) -> list[float]:
    f: list[float] = []
    f += _onehot(atom.GetSymbol(), _ATOM_LIST)
    f += _onehot(min(atom.GetDegree(), 6), list(range(7)))[:7]
    f += _onehot(int(np.clip(atom.GetFormalCharge(), -2, 2)), list(range(-2, 3)))[:5]
    f += _onehot(atom.GetHybridization(), _HYBRID)
    f += [1.0 if atom.GetIsAromatic() else 0.0]
    f += _onehot(min(atom.GetTotalNumHs(), 4), list(range(5)))[:5]
    f += [1.0 if atom.IsInRing() else 0.0]
    return f


def _bond_features(bond: Chem.Bond) -> list[float]:
    bt = bond.GetBondType()
    return [
        1.0 if bt == Chem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.BondType.AROMATIC else 0.0,
    ]


@dataclass
class FragmentGraph:
    """One BRICS fragment as an atom subgraph (local atom indexing)."""

    x: np.ndarray            # (n_atoms, ATOM_FEAT_DIM) float32
    edge_index: np.ndarray   # (2, n_edges) int64 — undirected (both dirs)
    edge_attr: np.ndarray    # (n_edges, BOND_FEAT_DIM) float32


def _brics_atom_fragments(mol: Chem.Mol) -> list[tuple[int, ...]]:
    n = mol.GetNumAtoms()
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds:
        return [tuple(range(n))]
    bidx = [mol.GetBondBetweenAtoms(a1, a2).GetIdx() for (a1, a2), _ in bonds]
    frag_mol = Chem.FragmentOnBonds(mol, bidx, addDummies=False)
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    return [tuple(f) for f in frags]


def smiles_to_fragment_graphs(smiles: str) -> list[FragmentGraph]:
    """SMILES -> list of BRICS-fragment atom subgraphs (>=1, or [] if bad).

    Returns [] for unparseable OR empty molecules (RDKit returns an empty mol,
    not None, for ""). Note feature SATURATION (not bug): degree>=6, numH>=4,
    formal charge outside [-2,2] are clamped into the endpoint bucket.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    atom_feats = [_atom_features(a) for a in mol.GetAtoms()]
    out: list[FragmentGraph] = []
    for atom_ids in _brics_atom_fragments(mol):
        local = {g: i for i, g in enumerate(atom_ids)}
        x = np.asarray([atom_feats[g] for g in atom_ids], dtype=np.float32)
        ei, ea = [], []
        for g in atom_ids:
            for bond in mol.GetAtomWithIdx(g).GetBonds():
                nb = bond.GetOtherAtomIdx(g)
                if nb in local and nb > g:           # internal bond, once
                    bf = _bond_features(bond)
                    ei.append([local[g], local[nb]]); ea.append(bf)
                    ei.append([local[nb], local[g]]); ea.append(bf)
        if ei:
            edge_index = np.asarray(ei, dtype=np.int64).T
            edge_attr = np.asarray(ea, dtype=np.float32)
        else:                                          # single-atom fragment
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)
        out.append(FragmentGraph(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return out


def build_drug_fragment_graph_cache(smiles_csv: str | Path) -> dict[str, list[FragmentGraph]]:
    """Map drugbank_id -> list of FragmentGraph (CSV cols: drugbank_id, smiles)."""
    import pandas as pd
    df = pd.read_csv(smiles_csv)
    return {
        str(did): smiles_to_fragment_graphs(str(smi))
        for did, smi in zip(df["drugbank_id"], df["smiles"])
    }


__all__ = [
    "FragmentGraph",
    "smiles_to_fragment_graphs",
    "build_drug_fragment_graph_cache",
    "ATOM_FEAT_DIM",
    "BOND_FEAT_DIM",
]
