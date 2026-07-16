"""Stage 2 / Part B / atom B2a - relation vocabulary: merge + verbalize.

The 59 raw relations of the merged KG are cryptic source codes (db:* / het:* /
prime:*). For the frozen relation embedding z_r we (1) merge the relations that
are the SAME biological edge across sources, (2) drop the degenerate het:metaedge,
and (3) verbalize each canonical relation as a TYPED-TRIPLE sentence
"head_type <verb> tail_type" (no node names), following text-attributed-graph
edge-verbalization practice. 59 -> 47 canonical relations.

Merge policy (codex-reviewed): merge SOURCE-LEVEL synonyms, preserve MECHANISM-LEVEL
distinctions. So the 10 Hetionet=PrimeKG duplicate pairs merge, het:CbG+prime:drug_protein
merge into a generic "drug binds protein", but db:target (curated therapeutic target),
db:enzyme/transporter/carrier (PK carriers) and het:CuG/CdG (signed PD regulation)
stay separate.
"""
from __future__ import annotations

#: relations removed entirely (degenerate / non-informative)
RELATION_DROP = {"het:metaedge"}

#: raw relation -> canonical key. Raw relations NOT listed here are their own canonical.
RELATION_MERGE = {
    # 10 Hetionet == PrimeKG duplicate pairs
    "het:GiG": "protein_interacts_protein",   "prime:protein_protein": "protein_interacts_protein",
    "het:DrD": "disease_resembles_disease",   "prime:disease_disease": "disease_resembles_disease",
    "het:GpBP": "protein_participates_bioprocess", "prime:bioprocess_protein": "protein_participates_bioprocess",
    "het:GpCC": "protein_in_cellular_component",   "prime:cellcomp_protein": "protein_in_cellular_component",
    "het:GpMF": "protein_has_molecular_function",  "prime:molfunc_protein": "protein_has_molecular_function",
    "het:GpPW": "protein_in_pathway",         "prime:pathway_protein": "protein_in_pathway",
    "het:DaG": "disease_associated_protein",  "prime:disease_protein": "disease_associated_protein",
    "het:AeG": "anatomy_expresses_protein",   "prime:anatomy_protein_present": "anatomy_expresses_protein",
    "het:DpS": "disease_presents_phenotype",  "prime:disease_phenotype_positive": "disease_presents_phenotype",
    "het:CtD": "drug_treats_disease",         "prime:indication": "drug_treats_disease",
    # codex-approved source-synonym merge (generic drug-protein binding)
    "het:CbG": "drug_binds_protein",          "prime:drug_protein": "drug_binds_protein",
}

#: canonical relation -> typed-triple verbalization (head_type <verb> tail_type)
RELATION_SENTENCE = {
    # --- merged canonicals ---
    "protein_interacts_protein": "A protein interacts with another protein.",
    "disease_resembles_disease": "A disease resembles another disease.",
    "protein_participates_bioprocess": "A protein participates in a biological process.",
    "protein_in_cellular_component": "A protein is part of a cellular component.",
    "protein_has_molecular_function": "A protein has a molecular function.",
    "protein_in_pathway": "A protein participates in a biological pathway.",
    "disease_associated_protein": "A disease is associated with a protein.",
    "anatomy_expresses_protein": "A protein is expressed in an anatomical structure.",
    "disease_presents_phenotype": "A disease presents a phenotype.",
    "drug_treats_disease": "A drug treats a disease.",
    "drug_binds_protein": "A drug binds a protein.",
    # --- DrugBank (kept: mechanistic roles) ---
    "db:target": "A drug targets a protein.",
    "db:enzyme": "A drug is metabolized by an enzyme protein.",
    "db:transporter": "A drug is moved by a transporter protein.",
    "db:carrier": "A drug is carried by a carrier protein.",
    "db:pathway": "A drug acts in a biological pathway.",
    # --- Hetionet (kept: distinct verbs) ---
    "het:CuG": "A drug upregulates a protein.",
    "het:CdG": "A drug downregulates a protein.",
    "het:CrC": "A drug is structurally similar to another drug.",
    "het:CpD": "A drug palliates a disease.",
    "het:CcSE": "A drug causes a side effect.",
    "het:DdG": "A disease downregulates a protein.",
    "het:DuG": "A disease upregulates a protein.",
    "het:DlA": "A disease localizes to an anatomical structure.",
    "het:GcG": "A protein covaries with another protein.",
    "het:Gr>G": "A protein regulates another protein.",
    "het:AdG": "An anatomical structure downregulates a protein.",
    "het:AuG": "An anatomical structure upregulates a protein.",
    "het:PCiC": "A pharmacologic class includes a drug.",
    # --- PrimeKG (kept) ---
    "prime:contraindication": "A drug is contraindicated for a disease.",
    "prime:off-label use": "A drug is used off-label for a disease.",
    "prime:phenotype_protein": "A protein is associated with a phenotype.",
    "prime:disease_phenotype_negative": "A disease does not present a phenotype.",
    "prime:drug_effect": "A drug produces a phenotypic effect.",
    "prime:exposure_protein": "An environmental exposure affects a protein.",
    "prime:exposure_disease": "An environmental exposure is linked to a disease.",
    "prime:exposure_bioprocess": "An environmental exposure affects a biological process.",
    "prime:exposure_molfunc": "An environmental exposure affects a molecular function.",
    "prime:exposure_cellcomp": "An environmental exposure affects a cellular component.",
    "prime:anatomy_protein_absent": "A protein is absent from an anatomical structure.",
    "prime:phenotype_phenotype": "A phenotype is related to another phenotype.",
    "prime:bioprocess_bioprocess": "A biological process is related to another biological process.",
    "prime:molfunc_molfunc": "A molecular function is related to another molecular function.",
    "prime:cellcomp_cellcomp": "A cellular component is related to another cellular component.",
    "prime:exposure_exposure": "An environmental exposure is related to another exposure.",
    "prime:pathway_pathway": "A biological pathway is related to another pathway.",
    "prime:anatomy_anatomy": "An anatomical structure is related to another anatomical structure.",
}


def canonical_relation(raw: str) -> str | None:
    """Map a raw KG relation to its canonical key (None if dropped)."""
    if raw in RELATION_DROP:
        return None
    return RELATION_MERGE.get(raw, raw)


def canonical_relations() -> list[str]:
    """Sorted list of the 47 canonical relation keys."""
    return sorted(RELATION_SENTENCE)


def relation_sentence(canonical: str) -> str:
    return RELATION_SENTENCE[canonical]


__all__ = ["RELATION_DROP", "RELATION_MERGE", "RELATION_SENTENCE",
           "canonical_relation", "canonical_relations", "relation_sentence"]
