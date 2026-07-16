"""HDN-DDI BRICS-aware molecular featurizer (paper-faithful, 66-dim, 3-level).

Independent re-implementation of the BRICS-decomposed hierarchical
molecular graph described in Sun & Zheng 2025 (HDN-DDI, BMC
Bioinformatics 2025) §Methods.  Functionally equivalent to
``baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py`` (lives
in this same package, but called as a subprocess CLI rather than
imported, so the two stay decoupled — see ``_shared.ensure_mol_graphs_pkl``).

What this builds
----------------

Per-drug PyG ``Data`` object with:

    x          : [n_nodes, 66]  float32  node features (paper Table S1)
    edge_index : [2, n_edges]   int64
    edge_attr  : [n_edges, 2]   int64
    y          : [n_nodes]      int64    node-type tags

Node types (``y``):

    y == 0   atom nodes                    (median ~24 per drug)
    y == 1   substructure / fragment nodes (median ~6 per drug, from refined BRICS)
    y == 2   super-node                    (exactly 1 per drug)

Edge categories (``edge_attr[:, 1]``):

    col1 == 0   cross-level edges  (atom↔fragment, fragment↔super)
    col1 == 1   intra-level non-aromatic chemical bonds (atom↔atom)
    col1 == 2   intra-level aromatic chemical bonds (atom↔atom)

Edge bond-type ids (``edge_attr[:, 0]``):

    {0: SINGLE, 1: DOUBLE, 2: TRIPLE, 3: AROMATIC,
     5: fragment↔super,   6: atom↔fragment}

Feature layout (66-dim, paper Additional file 1 Table S1)
---------------------------------------------------------

    [0..52]  atomic symbol one-hot (53 dims)
    [53]     degree
    [54]     implicit valence
    [55]     formal charge
    [56]     num radical electrons
    [57..63] hybridization one-hot (7 dims)
    [64]     aromatic flag
    [65]     total numH

Fragment / super-node features use the same 66-dim layout via
type-marker bits at the "pseudo-element" slots (50 = FRAGMENT marker,
51 = SUPER marker) plus the OTHER hybridization slot (63).

Fragmentation strategy (paper "refined BRICS")
----------------------------------------------

1. BRICS bonds (non-ring single bonds) → primary cuts
2. Multi-ring fragments with ≥ ``_LARGE_RING_ATOM_COUNT`` atoms get
   further split using RDKit SSSR (each smallest ring → own fragment)
3. Strict atom→fragment partition (each atom in exactly one fragment),
   matching paper's inclusion-edge definition
"""

from __future__ import annotations

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdchem
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------
# Feature dimension constants — paper Table S1
# ---------------------------------------------------------------------

ATOM_FEATURE_DIM: int = 66

_ATOM_SYMBOLS = [
    # 44 from the standard DSN-DDI / SSI-DDI / HDN-DDI list:
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
    "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In",
    "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
    # 9 extra drug-relevant elements (heavier metals / contrast-agent
    # nuclei / radiopharmaceuticals seen in DrugBank).  Order is a
    # best-effort extension since jcsun-00 never published the exact
    # 53-element list:
    "Ba", "Bi", "Sr", "Mo", "Ra", "Re", "Tc", "Gd", "Lu",
    # Catch-all so OOV symbols never crash:
    "Unknown",
]
assert len(_ATOM_SYMBOLS) == 53

_HYBRIDS = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.OTHER,
]
assert len(_HYBRIDS) == 7

# Feature block offsets (paper Table S1).
_OFFSET_ELEM      = 0     # 0..52  (53 dims)
_DIM_DEGREE       = 53
_DIM_VALENCE      = 54
_DIM_CHARGE       = 55
_DIM_RADICAL      = 56
_OFFSET_HYBRID    = 57    # 57..63 (7 dims)
_DIM_AROMATIC     = 64
_DIM_NUMH         = 65

# Pseudo-element slots for non-atom node markers (so HDN-DDI downstream
# can consume 66-dim vectors uniformly across node types).  Heavier-
# metal slots that real drugs basically never populate are reused as
# "FRAGMENT" / "SUPER" markers — matches what we observe in the
# official jcsun-00 pkl's y==1 / y==2 row features.
_FRAG_MARKER_DIM  = 50    # "Tc" slot → repurposed as FRAGMENT marker
_SUPER_MARKER_DIM = 51    # "Ra" slot → repurposed as SUPER marker
_HYBRID_OTHER_DIM = _OFFSET_HYBRID + (_HYBRIDS.index(rdchem.HybridizationType.OTHER))

# Threshold above which a multi-ring fragment gets further SSSR-split.
_LARGE_RING_ATOM_COUNT = 7


# ---------------------------------------------------------------------
# Per-atom feature vector (y == 0)
# ---------------------------------------------------------------------

def _one_hot(value, allowed, offset: int, vec: torch.Tensor) -> None:
    try:
        idx = allowed.index(value)
    except ValueError:
        idx = len(allowed) - 1
    vec[offset + idx] = 1.0


def _atom_features(atom: rdchem.Atom) -> torch.Tensor:
    """66-dim per paper Table S1."""
    vec = torch.zeros(ATOM_FEATURE_DIM, dtype=torch.float32)
    _one_hot(atom.GetSymbol(), _ATOM_SYMBOLS, _OFFSET_ELEM, vec)
    vec[_DIM_DEGREE]  = float(atom.GetDegree())
    vec[_DIM_VALENCE] = float(atom.GetImplicitValence())
    vec[_DIM_CHARGE]  = float(atom.GetFormalCharge())
    vec[_DIM_RADICAL] = float(atom.GetNumRadicalElectrons())
    _one_hot(atom.GetHybridization(), _HYBRIDS, _OFFSET_HYBRID, vec)
    if atom.GetIsAromatic():
        vec[_DIM_AROMATIC] = 1.0
    vec[_DIM_NUMH] = float(atom.GetTotalNumHs())
    return vec


# ---------------------------------------------------------------------
# Fragment / super-node feature vectors (y == 1 / y == 2)
# ---------------------------------------------------------------------

def _fragment_features(atom_indices, atom_feats: torch.Tensor) -> torch.Tensor:
    """66-dim pooled-from-atoms + FRAGMENT marker bit at slot 50."""
    atom_indices = list(atom_indices)
    vec = torch.zeros(ATOM_FEATURE_DIM, dtype=torch.float32)
    if atom_indices:
        sub = atom_feats[atom_indices]
        # Element slots: OR across constituents (multiple bits on).
        vec[_OFFSET_ELEM:_OFFSET_ELEM + 53] = (
            sub[:, _OFFSET_ELEM:_OFFSET_ELEM + 53].max(dim=0).values
        )
        # Scalar slots: SUM (more informative for "fragment size / total
        # charge / total radical count" than mean).
        vec[_DIM_DEGREE]  = sub[:, _DIM_DEGREE].sum()
        vec[_DIM_VALENCE] = sub[:, _DIM_VALENCE].sum()
        vec[_DIM_CHARGE]  = sub[:, _DIM_CHARGE].sum()
        vec[_DIM_RADICAL] = sub[:, _DIM_RADICAL].sum()
        # Aromatic: OR.
        vec[_DIM_AROMATIC] = float(sub[:, _DIM_AROMATIC].any().item())
        # numH: SUM.
        vec[_DIM_NUMH] = sub[:, _DIM_NUMH].sum()
    # Type marker: pseudo-element slot 50 + OTHER hybridization.
    vec[_FRAG_MARKER_DIM] = 1.0
    vec[_HYBRID_OTHER_DIM] = 1.0
    return vec


def _super_node_features() -> torch.Tensor:
    """66-dim pure marker: slot 51 (SUPER) + dim 63 (hybrid OTHER)."""
    vec = torch.zeros(ATOM_FEATURE_DIM, dtype=torch.float32)
    vec[_SUPER_MARKER_DIM] = 1.0
    vec[_HYBRID_OTHER_DIM] = 1.0
    return vec


# ---------------------------------------------------------------------
# Refined BRICS fragmentation (paper §Methods)
# ---------------------------------------------------------------------

def _ring_split_atom_sets(atom_indices, mol: Chem.Mol):
    """Split a multi-ring atom subset into per-SSSR-ring fragments.
    Non-ring 'linker' atoms get attached to a neighboring ring fragment.
    """
    sssr = Chem.GetSSSR(mol)
    rings_in_subset: list[set] = []
    for ring in sssr:
        ring_atoms = {int(a) for a in ring}
        if ring_atoms.issubset(atom_indices):
            rings_in_subset.append(ring_atoms)
    if len(rings_in_subset) <= 1:
        return [set(atom_indices)]

    output: list[set] = [set(r) for r in rings_in_subset]
    assigned: set = set().union(*output)
    leftover = atom_indices - assigned
    for ai in leftover:
        atom = mol.GetAtomWithIdx(ai)
        attached = False
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            for frag in output:
                if nb_idx in frag:
                    frag.add(ai)
                    attached = True
                    break
            if attached:
                break
        if not attached:
            output.append({ai})
    return [f for f in output if f]


def _refined_brics_fragment_atom_sets(mol: Chem.Mol):
    """Strict-partition refined BRICS: BRICS bonds + per-ring split for
    large fused-ring fragments.  Returns ``list[set[int]]`` of atom
    indices per fragment.  Fallback: one fragment = whole molecule when
    no cleavable bond exists.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return []

    # Step 1: BRICS bonds (non-ring only).
    cut_bond_indices: set = set()
    for (a, b), _label in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is not None and not bond.IsInRing():
            cut_bond_indices.add(bond.GetIdx())

    if cut_bond_indices:
        rw = Chem.RWMol(mol)
        frag_mol = Chem.FragmentOnBonds(
            rw, sorted(cut_bond_indices), addDummies=False
        )
        pieces = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
        brics_fragments = [
            {int(i) for i in atom_ids if i < n_atoms} for atom_ids in pieces
        ]
        brics_fragments = [f for f in brics_fragments if f]
    else:
        brics_fragments = [set(range(n_atoms))]

    # Step 2: SSSR split of large multi-ring fragments.
    final: list[set] = []
    sssr = Chem.GetSSSR(mol)
    for frag in brics_fragments:
        if len(frag) < _LARGE_RING_ATOM_COUNT:
            final.append(frag)
            continue
        n_internal_rings = sum(
            1 for ring in sssr if {int(a) for a in ring}.issubset(frag)
        )
        if n_internal_rings <= 1:
            final.append(frag)
            continue
        final.extend(_ring_split_atom_sets(frag, mol))

    return [f for f in final if f]


# ---------------------------------------------------------------------
# Per-molecule hierarchical Data builder
# ---------------------------------------------------------------------

def mol_to_data(smiles: str) -> Data | None:
    """Build a 3-level PyG ``Data`` for one drug.  Returns ``None`` for
    invalid / empty SMILES."""
    if smiles is None or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    atom_feats = torch.stack([_atom_features(a) for a in mol.GetAtoms()])
    fragment_sets = _refined_brics_fragment_atom_sets(mol)
    n_frags = len(fragment_sets)

    # Small-rigid-molecule fallback: ship frag + super only (no atom layer).
    drop_atom_layer = n_frags == 0
    if drop_atom_layer:
        fragment_sets = [set(range(n_atoms))]
        n_frags = 1

    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []

    if not drop_atom_layer:
        x_parts.append(atom_feats)
        y_parts.append(torch.zeros(n_atoms, dtype=torch.long))      # atoms y=0
    frag_feats = torch.stack(
        [_fragment_features(s, atom_feats) for s in fragment_sets]
    )
    x_parts.append(frag_feats)
    y_parts.append(torch.ones(n_frags, dtype=torch.long))           # frags y=1
    x_parts.append(_super_node_features().unsqueeze(0))
    y_parts.append(torch.full((1,), 2, dtype=torch.long))           # super y=2

    x = torch.cat(x_parts, dim=0)
    y = torch.cat(y_parts, dim=0)

    atom_offset = 0
    frag_offset = n_atoms if not drop_atom_layer else 0
    super_idx = frag_offset + n_frags

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_col0: list[int] = []
    edge_col1: list[int] = []

    # atom ↔ atom chemical bonds
    if not drop_atom_layer:
        for bond in mol.GetBonds():
            a = atom_offset + bond.GetBeginAtomIdx()
            b = atom_offset + bond.GetEndAtomIdx()
            is_aromatic = (
                bond.GetIsAromatic()
                or bond.GetBondType() == rdchem.BondType.AROMATIC
            )
            cat = 2 if is_aromatic else 1
            bt = bond.GetBondType()
            if bt == rdchem.BondType.SINGLE:
                col0 = 0
            elif bt == rdchem.BondType.DOUBLE:
                col0 = 1
            elif bt == rdchem.BondType.TRIPLE:
                col0 = 2
            elif bt == rdchem.BondType.AROMATIC:
                col0 = 3
            else:
                col0 = 0
            for s, d in ((a, b), (b, a)):
                edge_src.append(s); edge_dst.append(d)
                edge_col0.append(col0); edge_col1.append(cat)

    # atom ↔ fragment inclusion (col1=0, col0=6)
    if not drop_atom_layer:
        for fi, atom_ids in enumerate(fragment_sets):
            frag_node = frag_offset + fi
            for ai in atom_ids:
                a_node = atom_offset + ai
                for s, d in ((a_node, frag_node), (frag_node, a_node)):
                    edge_src.append(s); edge_dst.append(d)
                    edge_col0.append(6); edge_col1.append(0)

    # fragment ↔ super (col1=0, col0=5)
    for fi in range(n_frags):
        frag_node = frag_offset + fi
        for s, d in ((frag_node, super_idx), (super_idx, frag_node)):
            edge_src.append(s); edge_dst.append(d)
            edge_col0.append(5); edge_col1.append(0)

    # Paper §Methods is explicit: "no edges between substructure-level
    # nodes within a molecule" — we follow paper text, NOT the shipped
    # jcsun-00 pkl (which contains ~5000 spurious (1,1) edges).

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = (
        torch.tensor(list(zip(edge_col0, edge_col1)), dtype=torch.long)
        if edge_col0 else torch.zeros((0, 2), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------
# Top-level builder (drop-in shape compatible with mol_features.build_drug_graphs)
# ---------------------------------------------------------------------

def build_drug_graphs(
    smiles_dict: dict[str, str],
) -> tuple[dict[str, Data], list[str]]:
    """Parse SMILES → ``{drug_id: 3-level Data}``; report unparseable as
    ``missing``.  Drop-in callable from both
    :class:`baseline.hdn_ddi.binary_cls.baseline.HDNDDIBaseline` and
    :class:`baseline.hdn_ddi.multi_cls.baseline.HDNDDIMulticlassBaseline`.
    """
    graphs: dict[str, Data] = {}
    missing: list[str] = []
    for drug_id, smi in smiles_dict.items():
        data = mol_to_data(smi)
        if data is None:
            missing.append(drug_id)
            continue
        graphs[str(drug_id)] = data
    return graphs, missing
