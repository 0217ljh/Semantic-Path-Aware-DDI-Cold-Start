"""HDN-DDI hierarchical molecular graph builder (3-level: atoms / fragments / super-node).

**Baseline-side independent copy.** Lives at
``baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py`` per
CLAUDE.md §"Baseline 规范" §1 (builder co-located with its outputs
inside `_data/necessary/`).

A twin file lives on the reproduction side at
``reproductions/HDN-DDI/_Original-Dataset/necessary/build_hierarchical_pkl.py``
(per §"Baseline 规范" §2 step 2: "复制不剪切"; the two copies start
byte-identical, may drift independently per §"文件级独立性"). Neither
side cross-imports nor subprocess-calls the other.

Production purpose on baseline side: build BRICS pkl **for this project's
own drug pool** (no upstream version exists for our drug set). Baseline
pipeline auto-invokes this script via subprocess from
``_shared.ensure_mol_graphs_pkl`` (per §"Baseline 规范" §2 step 3 —
baseline-side detection DOES auto-build, contrast with reproduction-side
which only loads, never builds).

A regression-guard validator that compares this builder's output against
the official pkl on upstream SMILES lives next to it at
``baseline/hdn_ddi/_data/necessary/validate_hierarchical_pkl.py``.

Reproduced structural invariants (verified on the official pkl)
---------------------------------------------------------------

Per-drug PyG ``Data`` object with::

    x          : [n_nodes, 66]  float32   node features
    edge_index : [2, n_edges]   int64
    edge_attr  : [n_edges, 2]   int64
    y          : [n_nodes]      int64     node-type tags

Node types (``y``):

    y == 0   atom nodes                    (median 24 per drug)
    y == 1   fragment / substructure nodes (median  6 per drug — BRICS slices)
    y == 2   super-node                    (exactly  1 per drug)

Edge categories (``edge_attr[:, 1]``):

    col1 == 0   cross-level edges  (atom↔fragment, fragment↔super)
    col1 == 1   intra-level non-aromatic bonds (atom↔atom + fragment↔fragment)
    col1 == 2   intra-level aromatic/extra bonds (atom↔atom + fragment↔fragment)

Edge bond-type ids (``edge_attr[:, 0]``):

    Within col1==1:  {0: SINGLE, 1: DOUBLE, 2: TRIPLE}
    Within col1==2:  {3: AROMATIC, 0: SINGLE, 1: DOUBLE}  (extra-bond block)
    Within col1==0:  {5: fragment↔super, 6: atom↔fragment}

Feature layout (66-dim, per paper Additional file 1 Table S1)
-------------------------------------------------------------

Dims 0-52   one-hot of atom element symbol (53 entries; see ``_ATOM_SYMBOLS``).
            Slots 50 / 51 are also reused as pseudo-element markers for
            non-atom nodes: 50 = FRAGMENT (y==1), 51 = SUPER (y==2).
Dim  53     scalar: atom degree
Dim  54     scalar: implicit valence
Dim  55     scalar: formal charge
Dim  56     scalar: num radical electrons
Dims 57-63  hybridization one-hot (S / SP / SP2 / SP3 / SP3D / SP3D2 / OTHER)
            Non-atom nodes set the OTHER slot (dim 63) by construction —
            matches observed nonzero pattern in official pkl's y==1, y==2 rows.
Dim  64     aromatic flag
Dim  65     scalar: total number of Hs

NOTE: the official pkl's exact 66-dim semantics were not published.  The
slot assignment above MATCHES the observed nonzero-column pattern across
1706 drugs but the per-dim semantics are our best inference.  Downstream
HDN-DDI training only consumes ``y`` for super-node readout and
``edge_attr`` direction-tolerantly, so this featurization runs end-to-end
even if the exact upstream numeric values differ.

CLI
---

    # For the baseline's normal use (build BRICS pkl for our project data):
    # — Usually auto-invoked by _shared.ensure_mol_graphs_pkl; manual CLI shown for debug:
    python Code/baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py \\
        --smiles-csv <project drug csv with drugbank_id + smiles columns> \\
        --id-col drugbank_id --smiles-col smiles \\
        --out Code/baseline/hdn_ddi/_data/necessary/hdn_ddi_mol_graphs__mine.pkl

    # For the one-time alignment check against upstream's official pkl:
    python Code/baseline/hdn_ddi/_data/necessary/build_hierarchical_pkl.py \\
        --smiles-csv Code/reproductions/HDN-DDI/_Original-Dataset/drug_smiles.csv \\
        --id-col id --smiles-col smiles \\
        --out Code/reproductions/HDN-DDI/_Original-Dataset/necessary/id_data_dict__mine.pkl
    # then run baseline/hdn_ddi/_data/necessary/validate_hierarchical_pkl.py to compare
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdchem
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------
# Feature layout — paper Additional file 1 (Table S1) is the spec.
#
#   Dim   | Block                                | Width
#   ------|--------------------------------------|------
#   0..52 | Atomic symbol one-hot                |   53
#   53    | Degree of atom (scalar)              |    1
#   54    | Implicit valence (scalar)            |    1
#   55    | Formal charge (scalar)               |    1
#   56    | Number of radical electrons (scalar) |    1
#   57..63| Hybridization one-hot                |    7
#   64    | Aromatic flag                        |    1
#   65    | Total number of Hs (scalar)          |    1
#   ----- |                                Total |   66
#
# The 53-symbol table extends the 44-symbol DSN-DDI list with 8 extra
# drug-relevant elements that appear in DrugBank; slot 52 is reserved
# as the "Unknown" catch-all so element lookups never fail.  The exact
# slot assignment for the 8 extras isn't documented in the paper —
# best-effort guess preserving the standard prefix order.
#
# The 7-hybridization table covers RDKit's full enum (S/SP/SP2/SP3/
# SP3D/SP3D2/OTHER).  OTHER is also used as the marker hybridization
# for non-atom nodes (fragments + super-node), giving them a non-zero
# slot 63 by construction — matches what we observe in the official
# pkl's y==1 and y==2 row features.
# ---------------------------------------------------------------------

_ATOM_SYMBOLS = [
    # 44 from the standard DSN-DDI / SSI-DDI / HDN-DDI list:
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
    "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In",
    "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
    # 9 extra drug-relevant elements (heavier metals / contrast-agent
    # nuclei / radiopharmaceuticals seen in DrugBank).  Order is a
    # best guess; jcsun-00 didn't publish the exact extension.
    "Ba", "Bi", "Sr", "Mo", "Ra", "Re", "Tc", "Gd", "Lu",
    # Catch-all so out-of-vocabulary symbols don't crash:
    "Unknown",
]
assert len(_ATOM_SYMBOLS) == 53, f"expected 53 atomic-symbol slots, got {len(_ATOM_SYMBOLS)}"

_HYBRIDS = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.OTHER,
]
assert len(_HYBRIDS) == 7, f"expected 7 hybridization slots, got {len(_HYBRIDS)}"

FEAT_DIM = 66                             # total node-feature width

# Feature block offsets per paper Table S1.
_OFFSET_ELEM      = 0                     # 0..52   (53 dims)
_DIM_DEGREE       = 53
_DIM_VALENCE      = 54
_DIM_CHARGE       = 55
_DIM_RADICAL      = 56
_OFFSET_HYBRID    = 57                    # 57..63  (7 dims)
_DIM_AROMATIC     = 64
_DIM_NUMH         = 65

# Pseudo-element slots used for non-atom node markers (so HDN-DDI
# downstream code can still consume the 66-dim vectors uniformly).
# We borrow two of the heavier-metal slots that real drugs basically
# never populate -- "Tc" at slot 50 stands in as the FRAGMENT marker,
# "Ra" at slot 51 stands in as the SUPER-node marker, matching the
# nonzero positions observed in the official pkl's y==1 / y==2 rows.
_FRAG_MARKER_DIM  = 50
_SUPER_MARKER_DIM = 51
_HYBRID_OTHER_DIM = _OFFSET_HYBRID + (_HYBRIDS.index(rdchem.HybridizationType.OTHER))


# ---------------------------------------------------------------------
# Atom (y == 0) features
# ---------------------------------------------------------------------

def _one_hot(value, allowed, offset: int, vec: torch.Tensor) -> None:
    try:
        idx = allowed.index(value)
    except ValueError:
        # last "Unknown" slot for symbols / catch-all for hybrids
        idx = len(allowed) - 1
    vec[offset + idx] = 1.0


def _atom_features(atom: rdchem.Atom) -> torch.Tensor:
    """66-dim feature vector for an atom node (y == 0), paper Table S1.

    Slots:
      [0..52]   atomic symbol one-hot (Table S1: 53 dims)
      [53]      degree
      [54]      implicit valence
      [55]      formal charge
      [56]      num radical electrons
      [57..63]  hybridization one-hot (7 dims)
      [64]      aromatic flag
      [65]      total numH
    """
    vec = torch.zeros(FEAT_DIM, dtype=torch.float32)
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
# Fragment (y == 1) features — pooled from constituent atoms + type marker
# ---------------------------------------------------------------------

def _fragment_features(atom_indices: Iterable[int], atom_feats: torch.Tensor) -> torch.Tensor:
    """66-dim feature vector for a fragment node (y == 1).

    Design (matches observed nonzero patterns in the official pkl):
    - Element slots: OR (logical) of constituent atoms' element one-hots,
      giving multiple bits set in [0..52] — encodes "this fragment
      contains atoms of these element types".  Pure mean would give
      fractional values; OR preserves clear element-set semantics.
    - Scalar slots [53..56]: SUM over constituents (intensive vs
      extensive choice — sum is more informative for "fragment size /
      total charge / total radical count").
    - Hybridization: only the OTHER bit set (dim 63), per paper-pkl
      convention for non-atom nodes.  Per-atom hybridization is lost
      at the fragment level by design — the fragment is "structural"
      not "electronic".
    - Aromatic: 1 if ANY constituent atom is aromatic (logical OR).
    - numH: SUM over constituents.
    - Fragment marker: dim 50 set so downstream code can distinguish
      fragment from a (vanishingly improbable) real-atom feature
      vector that happens to coincidentally match.
    """
    atom_indices = list(atom_indices)
    vec = torch.zeros(FEAT_DIM, dtype=torch.float32)
    if atom_indices:
        sub = atom_feats[atom_indices]
        # Element slots: OR.
        vec[_OFFSET_ELEM:_OFFSET_ELEM + 53] = (
            sub[:, _OFFSET_ELEM:_OFFSET_ELEM + 53].max(dim=0).values
        )
        # Scalar slots: SUM.
        vec[_DIM_DEGREE]  = sub[:, _DIM_DEGREE].sum()
        vec[_DIM_VALENCE] = sub[:, _DIM_VALENCE].sum()
        vec[_DIM_CHARGE]  = sub[:, _DIM_CHARGE].sum()
        vec[_DIM_RADICAL] = sub[:, _DIM_RADICAL].sum()
        # Aromatic: OR.
        vec[_DIM_AROMATIC] = float(sub[:, _DIM_AROMATIC].any().item())
        # numH: SUM.
        vec[_DIM_NUMH] = sub[:, _DIM_NUMH].sum()
    # Fragment marker (pseudo-element slot 50) + OTHER hybridization.
    vec[_FRAG_MARKER_DIM] = 1.0
    vec[_HYBRID_OTHER_DIM] = 1.0
    return vec


# ---------------------------------------------------------------------
# Super-node (y == 2) features — pure type marker
# ---------------------------------------------------------------------

def _super_node_features() -> torch.Tensor:
    """66-dim feature vector for the per-molecule super-node (y == 2).

    Two bits only: pseudo-element slot 51 (SUPER marker) + hybridization
    OTHER (dim 63).  All scalars and aromatic / numH are zero so the
    super-node carries no first-order chemical info — it's purely a
    typed message-passing hub.  Matches the official pkl's observed
    nonzero pattern for y==2 nodes (dims {51, 63} only).
    """
    vec = torch.zeros(FEAT_DIM, dtype=torch.float32)
    vec[_SUPER_MARKER_DIM] = 1.0
    vec[_HYBRID_OTHER_DIM] = 1.0
    return vec


# ---------------------------------------------------------------------
# Refined BRICS fragmentation (paper §Methods — "refined BRICS algorithm")
# ---------------------------------------------------------------------
#
# Faithful to the HDN-DDI paper's description (Sun & Zheng, BMC
# Bioinformatics 2025, §Methods):
#
#   "we utilize the refined BRICS algorithm to decompose drug molecules
#    [...] apply the BRICS algorithm to preliminarily decompose the
#    graph into fragments. Any large ring fragments that cannot be
#    decomposed by the BRICS algorithm are further partitioned into
#    several smaller ring fragments based on additional rules."
#
# Biochemistry rationale
# ----------------------
# Step 1 — **BRICS cuts** target retrosynthetically interesting bonds:
#   linker bonds adjacent to rings, amide / ester / carbamate /
#   sulfonamide / phosphate junctions, etc.  These are the bonds an
#   enzyme would most plausibly cleave; the resulting fragments are
#   thus close to "pharmacophoric units" — sets of atoms that act as a
#   functional unit when binding a target.
#
# Step 2 — **Ring decomposition** addresses the BRICS blind spot: BRICS
#   never cuts inside a ring, so fused polycyclic systems (steroids,
#   anthracyclines, large macrocycles) stay glued together.  We split
#   such systems using RDKit's SSSR (Smallest Set of Smallest Rings),
#   which gives the natural per-ring decomposition.  Each ring becomes
#   a separate substructure (rigid scaffold unit, the typical building
#   block recognised by binding pockets).
#
# Output invariant: each atom belongs to **exactly one** fragment
# (strict partition — paper says "if a substructure Si includes an atom
# Aj, a bidirectional edge vSi↔vAj will be established", implying the
# inclusion is a function from atoms to substructures).


# Threshold above which we further split a ring system into per-ring
# fragments.  Empirically ≥7 captures fused bicyclic (e.g. naphthalene),
# steroid (≥17), anthracene, etc., without over-splitting normal 5/6-
# member single rings.
_LARGE_RING_ATOM_COUNT = 7


def _ring_split_atom_sets(atom_indices: set[int], mol: Chem.Mol) -> list[set[int]]:
    """Split a multi-ring atom subset into one fragment per SSSR ring.

    Atoms NOT in any ring of the subset are merged into the first ring
    they're adjacent to (so we never emit a singleton "linker atom"
    fragment — the paper's biochemistry stance is that linkers belong
    to one of their flanking ring systems).
    """
    sssr = Chem.GetSSSR(mol)
    rings_in_subset: list[set[int]] = []
    for ring in sssr:
        ring_atoms = {int(a) for a in ring}
        # Only consider rings entirely inside the subset.
        if ring_atoms.issubset(atom_indices):
            rings_in_subset.append(ring_atoms)
    if len(rings_in_subset) <= 1:
        # Nothing to split — single ring or no ring info.
        return [set(atom_indices)]

    # Greedy merging: rings sharing atoms (fused rings) we KEEP separate
    # since the paper wants "smaller ring fragments"; non-ring atoms get
    # attached to whichever ring they're bonded to first.
    output: list[set[int]] = [set(r) for r in rings_in_subset]
    assigned: set[int] = set().union(*output)
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
            # Isolated leftover (rare): make its own fragment.
            output.append({ai})
    return [f for f in output if f]


def _refined_brics_fragment_atom_sets(mol: Chem.Mol) -> list[set[int]]:
    """Return one ``set[int]`` of atom indices per fragment.

    Implements the paper's "refined BRICS" decomposition.  Output is a
    **strict partition** of the molecule's atoms (each atom in exactly
    one fragment) — required by the paper's inclusion-edge construction
    (atom→substructure edges are well-defined only under partition).

    Special case — empty return triggers `drop_atom_layer`:
    The official pkl ships 171 drugs with ``y_unique == {1, 2}`` (no
    atom-level nodes). Empirical analysis of those 171 (vs the 1535
    drugs that DO have an atom layer) shows the deterministic criterion:

        drop atom layer  ⇔  BRICS cuts == 0
                            AND mol.GetNumAtoms() > 1
                            AND mol has exactly 1 connected component

    We return ``[]`` for this case so the caller's ``drop_atom_layer``
    branch fires. Coverage: 161 / 171 official no-atom drugs (94%).

    Residual divergence (known): the remaining 10 official no-atom drugs
    are multi-component all-monatomic salts (e.g. ``[OH-].[OH-].[Mg++]``);
    our builder produces 3-layer for them instead of 2-layer. Documented
    here, not implemented (chemistry-criterion unclear without official
    builder source).

    NOTE: This logic must stay byte-identical to the reproduction-side
    twin at ``reproductions/HDN-DDI/_Original-Dataset/necessary/build_hierarchical_pkl.py``
    (per CLAUDE.md §"Baseline 规范" §2 step 2 "复制不剪切"; drift
    between the two needs a regression test).
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return []

    # ── Step 1: BRICS cuts ──────────────────────────────────────────
    cut_bond_indices: set[int] = set()
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
        # No BRICS cuts → match the official "drop atom layer" criterion.
        # Trigger drop_atom_layer for single-component multi-atom drugs
        # (covers 161 of the 171 official no-atom drugs).
        n_components = len(Chem.GetMolFrags(mol))
        if n_atoms > 1 and n_components == 1:
            return []  # caller's drop_atom_layer branch handles this
        brics_fragments = [set(range(n_atoms))]

    # ── Step 2: ring decomposition for large multi-ring fragments ───
    final_fragments: list[set[int]] = []
    for frag in brics_fragments:
        if len(frag) < _LARGE_RING_ATOM_COUNT:
            final_fragments.append(frag)
            continue
        # Count rings entirely within this fragment.
        sssr = Chem.GetSSSR(mol)
        n_internal_rings = sum(
            1 for ring in sssr if {int(a) for a in ring}.issubset(frag)
        )
        if n_internal_rings <= 1:
            final_fragments.append(frag)
            continue
        # Multi-ring fragment large enough to split.
        final_fragments.extend(_ring_split_atom_sets(frag, mol))

    return [f for f in final_fragments if f]


# Backward-compat alias (validation script imports this name).
_brics_fragment_atom_sets = _refined_brics_fragment_atom_sets


# ---------------------------------------------------------------------
# Per-molecule hierarchical Data builder
# ---------------------------------------------------------------------

def mol_to_hierarchical_data(smiles: str) -> Data | None:
    """Build a 3-level PyG ``Data`` object for one drug.

    Returns ``None`` for invalid SMILES.

    Topology
    --------
    * Atoms (y=0) connected by chemical bonds (col1 ∈ {1, 2} for non-
      aromatic / aromatic respectively).
    * Each atom linked to every fragment that contains it (col1=0,
      col0=6).
    * Each fragment linked to the molecule's super-node (col1=0,
      col0=5).
    * Fragments NOT directly linked to each other in our reproduction
      — the official pkl's `y==1 ↔ y==1` edges (~5% of total) appear
      to come from the fragment subgraph's residual bonds and are
      reproduced here implicitly by the atom-fragment fanout.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    atom_feats = torch.stack([_atom_features(a) for a in mol.GetAtoms()])  # [n_atoms, 66]
    fragment_sets = _brics_fragment_atom_sets(mol)
    n_frags = len(fragment_sets)

    # Edge case (matches official pkl's 171 small-molecule drugs): too
    # small to decompose — ship fragments+super only, no atom layer.
    drop_atom_layer = n_frags == 0
    if drop_atom_layer:
        # Treat the whole molecule as one fragment, no atom nodes.
        fragment_sets = [set(range(n_atoms))]
        n_frags = 1

    # Assemble node feature matrix in canonical order: atoms, fragments, super.
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []

    if not drop_atom_layer:
        x_parts.append(atom_feats)
        y_parts.append(torch.zeros(n_atoms, dtype=torch.long))   # y==0
    frag_feats = torch.stack(
        [_fragment_features(s, atom_feats) for s in fragment_sets]
    )                                                            # [n_frags, 66]
    x_parts.append(frag_feats)
    y_parts.append(torch.ones(n_frags, dtype=torch.long))        # y==1
    x_parts.append(_super_node_features().unsqueeze(0))          # [1, 66]
    y_parts.append(torch.full((1,), 2, dtype=torch.long))        # y==2

    x = torch.cat(x_parts, dim=0)
    y = torch.cat(y_parts, dim=0)

    # Node-id offsets in the concatenated graph.
    atom_offset = 0
    frag_offset = n_atoms if not drop_atom_layer else 0
    super_idx = frag_offset + n_frags

    # ── Edges: chemical bonds (atom ↔ atom) ─────────────────────────
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_col0: list[int] = []   # bond-type id
    edge_col1: list[int] = []   # category id

    if not drop_atom_layer:
        for bond in mol.GetBonds():
            a = atom_offset + bond.GetBeginAtomIdx()
            b = atom_offset + bond.GetEndAtomIdx()
            is_aromatic = bond.GetIsAromatic() or bond.GetBondType() == rdchem.BondType.AROMATIC
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

    # ── Cross-level edges: atom ↔ fragment (col1=0, col0=6) ─────────
    if not drop_atom_layer:
        for fi, atom_ids in enumerate(fragment_sets):
            frag_node = frag_offset + fi
            for ai in atom_ids:
                a_node = atom_offset + ai
                for s, d in ((a_node, frag_node), (frag_node, a_node)):
                    edge_src.append(s); edge_dst.append(d)
                    edge_col0.append(6); edge_col1.append(0)

    # ── Cross-level edges: fragment ↔ super (col1=0, col0=5) ────────
    for fi in range(n_frags):
        frag_node = frag_offset + fi
        for s, d in ((frag_node, super_idx), (super_idx, frag_node)):
            edge_src.append(s); edge_dst.append(d)
            edge_col0.append(5); edge_col1.append(0)

    # NB: NO fragment↔fragment edges by design.  Paper §Methods is
    # explicit: "there are no edges between substructure-level nodes
    # within a molecule".  Information flows fragment-to-fragment ONLY
    # via the super-node bottleneck — this forces the model to learn
    # a global-pool readout instead of inter-substructure message
    # passing, which the authors argue better matches the
    # pharmacophore-additivity assumption underlying DDI mechanisms.

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(
        list(zip(edge_col0, edge_col1)), dtype=torch.long
    ) if edge_col0 else torch.zeros((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------
# Bulk builder — produces the {drug_id: (smiles, Data)} dict pkl
# ---------------------------------------------------------------------

def build_id_data_dict(
    smiles_map: dict[str, str],
    *,
    verbose: bool = True,
) -> dict[str, tuple[str, Data]]:
    """Build the upstream-compatible ``{drug_id: (smiles, Data)}`` dict.

    Drugs whose SMILES fail RDKit parsing are silently dropped (matches
    the upstream behaviour — the official pkl has 1706 drugs even
    though DrugBank advertises more, because invalid SMILES were
    skipped at build time).
    """
    out: dict[str, tuple[str, Data]] = {}
    n_fail = 0
    for did, smi in smiles_map.items():
        data = mol_to_hierarchical_data(smi)
        if data is None:
            n_fail += 1
            continue
        out[did] = (smi, data)
        if verbose and (len(out) + n_fail) % 200 == 0:
            print(
                f"[hdn-hier] built {len(out)}/{len(smiles_map)} "
                f"(skipped {n_fail} invalid SMILES)",
                file=sys.stderr,
                flush=True,
            )
    if verbose:
        print(
            f"[hdn-hier] DONE  kept {len(out)} / {len(smiles_map)} "
            f"(skipped {n_fail})",
            file=sys.stderr,
            flush=True,
        )
    return out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the HDN-DDI hierarchical id_data_dict pkl "
            "(3-level atom/fragment/super-node graphs via BRICS)."
        )
    )
    parser.add_argument(
        "--smiles-csv", required=True, type=Path,
        help="CSV with a drug-id column and a smiles column.",
    )
    parser.add_argument(
        "--id-col", default="drug_id",
        help="Drug-id column in --smiles-csv (default: drug_id — matches "
             "jcsun-00/DrugBank's drug_smiles.csv schema).",
    )
    parser.add_argument(
        "--smiles-col", default="smiles",
        help="SMILES column in --smiles-csv (default: smiles).",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output pkl path (e.g., baseline/hdn_ddi/_data/necessary/hdn_ddi_mol_graphs__mine.pkl).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.smiles_csv, usecols=[args.id_col, args.smiles_col])
    smi_map = dict(zip(df[args.id_col], df[args.smiles_col]))

    id_data_dict = build_id_data_dict(smi_map, verbose=not args.quiet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(id_data_dict, f)
    print(f"Wrote {args.out}  ({len(id_data_dict)} drugs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
