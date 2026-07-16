"""Build a {node_id -> text} mapping over the merged KG for TAG init.

Text sources by node kind (per first_step_plan.md §4.5):
  - Drug:              drug_profiles_flat.csv `text` column (rich profile)
  - Drug (name only):  drug_profiles_flat.csv `name` column
  - Other named nodes: merged KG nodes.parquet `name` column
  - ID-only nodes:     "" (encoder will emit zero vector)

Also produces a small `node_text_stats` table used by exp E1c (node readability).

Canonical kind mapping
----------------------
Hetionet uses TitleCase ("Drug", "Gene"); PrimeKG uses lowercase / snake_case
("drug", "gene/protein"). For Screen 1 we treat these as the same semantic
type so that:
  - variant D drug-profile lookup hits all drug-shaped nodes
  - variant B/F type-onehot uses one slot per biological concept
  - variant E within-kind shuffle pools all drug-like nodes together

If you want to ablate the raw-schema effect, use the canonical=False flag.
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np

# Project paths — resolve from this file location upward
_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
KG_ROOT = PROJECT_ROOT / "Code" / "data" / "KG"
MERGED_NODES = KG_ROOT / "_merged_kg" / "nodes__drugbank_hetionet_primekg.parquet"
DRUG_PROFILES_CSV = KG_ROOT / "drug_text" / "drug_profiles_flat.csv"

# Map raw KG `kind` -> canonical biological concept
_KIND_CANONICAL = {
    # Drug variants
    "Drug": "Drug", "drug": "Drug", "Compound": "Drug",
    # Gene / protein variants
    "Gene": "Gene/Protein", "gene": "Gene/Protein",
    "gene/protein": "Gene/Protein", "Protein": "Gene/Protein",
    "enzyme": "Gene/Protein", "transporter": "Gene/Protein",
    "carrier": "Gene/Protein", "target": "Gene/Protein",
    # Side effect / adverse
    "Side Effect": "SideEffect", "side_effect": "SideEffect",
    "SideEffect": "SideEffect", "drug_effect": "SideEffect",
    # Disease
    "Disease": "Disease", "disease": "Disease",
    # Anatomy
    "Anatomy": "Anatomy", "anatomy": "Anatomy",
    # Pathway
    "Pathway": "Pathway", "pathway": "Pathway",
    # Symptom / phenotype
    "Symptom": "Phenotype", "symptom": "Phenotype",
    "Phenotype": "Phenotype", "effect/phenotype": "Phenotype",
    # Biological process / molecular function / cellular component
    "BiologicalProcess": "BiologicalProcess",
    "Biological Process": "BiologicalProcess",
    "biological_process": "BiologicalProcess",
    "MolecularFunction": "MolecularFunction",
    "Molecular Function": "MolecularFunction",
    "molecular_function": "MolecularFunction",
    "CellularComponent": "CellularComponent",
    "Cellular Component": "CellularComponent",
    "cellular_component": "CellularComponent",
    # Other (kept as-is post-canonical, but stripped of case noise)
    "PharmacologicClass": "PharmacologicClass",
    "pharmacologic_class": "PharmacologicClass",
    "exposure": "Exposure",
}

# Kinds classified as "biomedically relevant" for E1c readability
# (uses canonical names from _KIND_CANONICAL)
RELEVANT_KINDS_CANONICAL = {
    "Drug", "Gene/Protein", "SideEffect", "Disease", "Anatomy",
    "Pathway", "Phenotype", "BiologicalProcess", "MolecularFunction",
    "CellularComponent", "PharmacologicClass", "Exposure",
}

# Regex: a "readable" name has at least one alphabetic word with vowel,
# AND is not just an ID-shaped token like "DB00006" or "ENSG00000..."
_ID_RE = re.compile(r"^(?:[A-Z]{1,5}\d{4,}|[A-Z0-9_:\-.]{1,8})$")
_VOWEL_WORD_RE = re.compile(r"[A-Za-z]*[aeiouAEIOU][A-Za-z]+")


def canonical_kind(raw_kind) -> str:
    """Map raw KG kind to canonical biological concept. NaN-safe."""
    if not isinstance(raw_kind, str):
        return "_unknown"
    return _KIND_CANONICAL.get(raw_kind, raw_kind)


def _clean_text(x) -> str:
    """Strict text-coercion: pd.isna -> ''; non-string -> str(); strip."""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    if x is None:
        return ""
    return str(x).strip()


def _is_readable_name(name: str) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    s = name.strip()
    if _ID_RE.match(s):
        return False
    if not _VOWEL_WORD_RE.search(s):
        return False
    return True


def build_node_text_table(canonical: bool = True) -> pd.DataFrame:
    """Return DataFrame columns: id, raw_kind, kind, source_kg, name, text, text_source, readable, relevant.

    `kind` is the canonical name if canonical=True, else the raw KG kind.
    """
    nodes = pd.read_parquet(MERGED_NODES)
    drug_profiles = pd.read_csv(DRUG_PROFILES_CSV)
    drug_text_by_id = {str(k): _clean_text(v) for k, v in zip(drug_profiles["drug_id"], drug_profiles["text"])}
    drug_name_by_id = {str(k): _clean_text(v) for k, v in zip(drug_profiles["drug_id"], drug_profiles["name"])}

    rows = []
    for _, n in nodes.iterrows():
        nid = _clean_text(n["id"])
        raw_kind = _clean_text(n["kind"]) or "_unknown"
        ckind = canonical_kind(raw_kind) if canonical else raw_kind
        name = _clean_text(n["name"])
        src = _clean_text(n["source_kg"])

        # Default: text = name (fallback)
        text = name
        text_source = "kg_name" if text else "empty"

        # Drug profile lookup uses canonical "Drug" kind (covers Drug/drug/Compound)
        if ckind == "Drug" and nid in drug_text_by_id:
            text = drug_text_by_id[nid]
            text_source = "drug_profile"
            if not name:
                name = drug_name_by_id.get(nid, "")

        rows.append(dict(
            id=nid, raw_kind=raw_kind, kind=ckind, source_kg=src,
            name=name, text=text, text_source=text_source,
            readable=_is_readable_name(text),
            relevant=(ckind in RELEVANT_KINDS_CANONICAL),
        ))
    return pd.DataFrame(rows)


def build_drug_name_only_table(df: pd.DataFrame | None = None) -> dict[str, str]:
    """Variant D-name text source: drug -> name; non-drug -> name (KG)."""
    if df is None:
        df = build_node_text_table()
    out = {}
    for _, n in df.iterrows():
        nid = n["id"]
        if n["kind"] == "Drug":
            # Drug profile has the canonical name; fall back to KG name
            out[nid] = _clean_text(n["name"]) or ""
        else:
            out[nid] = _clean_text(n["name"]) or ""
    return out


def build_typename_only_table(df: pd.DataFrame | None = None) -> dict[str, str]:
    """Variant F text source: every node -> canonical kind string only."""
    if df is None:
        df = build_node_text_table()
    return dict(zip(df["id"], df["kind"].astype(str)))


def build_shuffled_text_table(
    df: pd.DataFrame | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Variant E (FIXED per Codex round 1):

    Permute the FULL TEXT column within (canonical kind, text_source) groups.
    A node ends up with another node's text from the SAME canonical kind AND
    same text source (drug_profile / kg_name / empty), so:
      - text length distribution stays identical
      - profile-richness vs name-only distinction is preserved
      - the only thing that changes is semantic identity

    Uses a derangement where possible (no node keeps its own text); singleton
    groups keep their text unchanged.
    """
    if df is None:
        df = build_node_text_table()
    rng = np.random.default_rng(seed)
    out = {}
    for (kind, ts), grp in df.groupby(["kind", "text_source"]):
        ids = grp["id"].tolist()
        texts = grp["text"].astype(str).tolist()
        n = len(ids)
        if n <= 1:
            for i, nid in enumerate(ids):
                out[nid] = texts[i]
            continue
        for attempt in range(8):
            perm = rng.permutation(n)
            if not any(perm[i] == i for i in range(n)):
                break
        for i, nid in enumerate(ids):
            out[nid] = texts[perm[i]]
    return out


if __name__ == "__main__":
    df = build_node_text_table(canonical=True)
    print(f"Total nodes: {len(df):,}")
    print(f"With non-empty text: {(df['text'] != '').sum():,}")
    print(f"Readable: {df['readable'].sum():,}")
    print(f"Relevant kinds (canonical): {df['relevant'].sum():,}")
    print(f"  -> readable & relevant: {((df['readable']) & (df['relevant'])).sum():,}")
    print()
    print("Canonical kind distribution:")
    print(df["kind"].value_counts().head(15).to_string())
    print()
    print("Raw-kind -> canonical collapse mapping (top examples):")
    coll = df.groupby(["kind", "raw_kind"]).size().reset_index(name="n").sort_values("n", ascending=False).head(20)
    print(coll.to_string(index=False))
    print()
    print("Sample drug row (post-canonical):")
    drow = df[df["kind"] == "Drug"].head(1).iloc[0]
    print(f"  id={drow['id']}, raw_kind={drow['raw_kind']}, name={drow['name'][:30]}")
    print(f"  text={drow['text'][:120]}...")
