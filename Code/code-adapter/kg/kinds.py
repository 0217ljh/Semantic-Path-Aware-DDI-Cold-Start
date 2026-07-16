"""Stage 2 / atom A1 support - EDITABLE node-kind canonicalization.

The merged KG unions 3 sources (DrugBank / Hetionet / PrimeKG) that spell the
same biological concept differently (e.g. proteins = ``gene/protein`` /
``Gene`` / ``Protein``). ``KIND_CANONICAL`` collapses the ~23 raw ``kind``
strings to ~12 canonical concepts, so a mediator has ONE type (typed prototype
M_B is per canonical type; drug-protein mechanism enzyme/transporter/target is
carried by the edge RELATION, not the node type).

This file is the swap point: to switch to a cleaned/dedup KG with a different
kind vocabulary, edit ``KIND_CANONICAL`` here (or pass ``kind_map=`` to
``build_kg_store``). ``DRUG_CANONICAL`` marks which canonical type is "drug"
(excluded from mediators downstream). Unmapped raw kinds fall to
``UNKNOWN_CANONICAL`` and are counted + logged by A1 (never silently dropped).
"""
from __future__ import annotations

UNKNOWN_CANONICAL = "Other"
DRUG_CANONICAL = "Drug"

#: raw ``kind`` (any source spelling) -> canonical concept.
KIND_CANONICAL: dict[str, str] = {
    # Drug
    "Drug": "Drug", "drug": "Drug", "Compound": "Drug",
    # Gene / protein (drug-protein mechanism is on the edge relation, not here)
    "Gene": "Gene/Protein", "gene": "Gene/Protein", "gene/protein": "Gene/Protein",
    "Protein": "Gene/Protein", "protein": "Gene/Protein",
    "enzyme": "Gene/Protein", "transporter": "Gene/Protein",
    "carrier": "Gene/Protein", "target": "Gene/Protein",
    # Pathway
    "Pathway": "Pathway", "pathway": "Pathway",
    # Side effect
    "Side Effect": "SideEffect", "side_effect": "SideEffect",
    "SideEffect": "SideEffect", "drug_effect": "SideEffect",
    # Phenotype / symptom
    "Symptom": "Phenotype", "symptom": "Phenotype",
    "Phenotype": "Phenotype", "effect/phenotype": "Phenotype",
    # Disease
    "Disease": "Disease", "disease": "Disease",
    # Anatomy
    "Anatomy": "Anatomy", "anatomy": "Anatomy",
    # Biological process / molecular function / cellular component
    "Biological Process": "BiologicalProcess", "biological_process": "BiologicalProcess",
    "Molecular Function": "MolecularFunction", "molecular_function": "MolecularFunction",
    "Cellular Component": "CellularComponent", "cellular_component": "CellularComponent",
    # Misc
    "Pharmacologic Class": "PharmacologicClass", "pharmacologic_class": "PharmacologicClass",
    "exposure": "Exposure", "Exposure": "Exposure",
}


def canonical_kind(raw: str, kind_map: dict | None = None) -> str:
    """Map one raw ``kind`` to its canonical concept (UNKNOWN_CANONICAL if absent)."""
    m = KIND_CANONICAL if kind_map is None else kind_map
    return m.get(str(raw), UNKNOWN_CANONICAL)


__all__ = ["KIND_CANONICAL", "DRUG_CANONICAL", "UNKNOWN_CANONICAL", "canonical_kind"]
