"""D1 builder — convert i4_typed_sets.json into drug→token KG edge triples.

Round 4 D1 (per Notes/Log/d1_llm_edge_design.md §2.2). Standalone CLI.

Produces two artefacts (deterministic, byte-stable across re-runs):

  Code/data/_cache/llm_edges/llm_drug_edges__seed42_drugbank.parquet
      cols: [drug_id, token_node_id, relation]  sorted (drug_id, relation, token_node_id)
  Code/data/_cache/llm_edges/llm_token_nodes__seed42_drugbank.json
      {token_node_id: {field, raw_token, n_drugs}}
  Code/data/_cache/llm_edges/_audit_rejected_tokens__seed42_drugbank.json
      {field: [(raw_token, reason), ...]}    rejected by drug-name audit

The builder:
  1. Loads i4_typed_sets.json (1530 drugs × 10 typed fields, verified schema 2026-05-30).
  2. Re-applies the v2i4_trainer.py:128-141 13-phrase sanitizer to free-text fields.
  3. Applies a drug-name audit (normalized-containment, see Notes/Log/d1_llm_edge_design.md
     §5 R3) sourced from Code/data/KG/drugbank/filtered/id2name.json (1900 drugs), OPTIONALLY
     restricted to the actual drug-pool universe of the split (e.g. 800 drugs from
     Code/data/coldddi_legacy/800drug/seed42.pkl). Restricting to the split universe
     dramatically reduces false positives where endogenous compounds (dopamine,
     cholesterol, testosterone, etc.) happen to be DrugBank entries but are NOT in the
     model's prediction pool — those endogenous-compound mentions in `primary_targets` /
     `pd_effects` carry biological evidence, not interaction leakage.
  4. Builds new node IDs `llm:<field>:<slug>` for each unique token.
  5. Writes parquet (drug→token, with relation=`llm:<field>`), drug as head.

Cold-start safety: per-drug intrinsic only. Builder reads NO DDI table.
File independence: does NOT modify mnah_trainer.py / v2i4_trainer.py / baseline/emergnn/*.

Usage (from project root):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\\
Semantic-Path-Aware-DDI-Cold-Start && python \\
Code/my_code/models/screen_s2_v3_multimodal/precompute_llm_edges.py"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]

# ---------------------------------------------------------------------
# Constants — match v2i4_trainer.py exactly (single source of truth here:
# v2i4_trainer.py:128-141 phrase list + free-text field tuple).
# ---------------------------------------------------------------------

I4_JSON = PROJECT_ROOT / "Code" / "data" / "_cache" / "llm_pharma" / "i4_typed_sets.json"
ID2NAME_JSON = PROJECT_ROOT / "Code" / "data" / "KG" / "drugbank" / "filtered" / "id2name.json"
DEFAULT_SPLIT_PKL = (
    PROJECT_ROOT / "Code" / "data" / "coldddi_legacy" / "800drug" / "seed42.pkl"
)

# Mirror v2i4_trainer.py:131-133 BAD tuple exactly.
SANITIZER_PHRASES: tuple[str, ...] = (
    "interact", "coadminist", "co-administ", "combined with", "combination with",
    "concomitant", "avoid with", "contraindicated", "with inhibitor", "with inducer",
    "increase levels", "decrease levels", "co-medic", "co-prescri",
)

# Mirror v2i4_trainer.py:134-135 FREE tuple exactly.
FREE_TEXT_FIELDS: tuple[str, ...] = (
    "therapeutic_class", "primary_targets", "pd_effects",
    "toxicity_mechanisms", "clearance",
)

ALL_FIELDS: tuple[str, ...] = (
    "cyp_substrate", "cyp_inhibitor", "cyp_inducer",
    "transporter_substrate", "transporter_inhibitor",
    "therapeutic_class", "primary_targets", "pd_effects",
    "toxicity_mechanisms", "clearance",
)

# Drug-name audit minimum length (avoid false positives on common short English
# words). Names shorter than this are not considered when checking containment.
MIN_NAME_LEN_FOR_AUDIT = 6

# Default output directory.
DEFAULT_OUT_DIR = PROJECT_ROOT / "Code" / "data" / "_cache" / "llm_edges"

# Default tag (seed42 + drugbank 5-bucket scope; matches the 0.7804 anchor).
DEFAULT_TAG = "seed42_drugbank"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_NON_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_NORM_RE = re.compile(r"[^a-z0-9]+")


def _slug(token: str) -> str:
    """Lowercase a token and reduce to [a-z0-9_-]; collapse whitespace to underscore."""
    s = token.strip().lower()
    s = s.replace(" ", "_")
    s = _NON_SLUG_RE.sub("", s)
    return s


def _norm(s: str) -> str:
    """Drug-name audit normalization: lowercase, alphanumerics joined by single space."""
    s = s.strip().lower()
    return _NORM_RE.sub(" ", s).strip()


def _sanitize_free_text(typed_sets: dict, fields: tuple[str, ...],
                        phrases: tuple[str, ...]) -> tuple[dict, int]:
    """Drop tokens containing interaction phrasing from free-text fields.

    Matches v2i4_trainer.py:128-141 behavior exactly (substring match, lowercase
    not enforced because v2i4 doesn't either — phrases are already lowercase
    and the tokens are typically lowercase, but we don't downcase here to stay
    byte-identical to v2i4's sanitize logic).
    """
    dropped = 0
    for drug_id, fdict in typed_sets.items():
        for field in fields:
            raw_list = fdict.get(field, []) or []
            kept = [t for t in raw_list if not any(p in t for p in phrases)]
            dropped += len(raw_list) - len(kept)
            fdict[field] = kept
    return typed_sets, dropped


def _load_split_drug_pool(split_pkl: Path) -> set[str]:
    """Load drug-id universe from the cold-start split pkl.

    Uses data_utils.PairDataset (same loader as run_v2i4.py:82). Returns the union
    of drug_a_id / drug_b_id across all split partitions (train / val_s* / test_s*).
    """
    sys.path.insert(0, str(PROJECT_ROOT / "Code"))
    from data_utils import PairDataset  # noqa
    ds = PairDataset.from_pkl(str(split_pkl))
    pool: set[str] = set()
    for _name, df in ds.splits.items():
        if "drug_a_id" in df.columns:
            pool.update(map(str, df["drug_a_id"]))
            pool.update(map(str, df["drug_b_id"]))
    return pool


def _build_drug_name_audit(
    id2name_path: Path,
    pool_filter: set[str] | None,
) -> tuple[set[str], list[str], int]:
    """Load id2name.json and produce (norm_set, long_norm_names, n_audited).

    If pool_filter is provided (e.g. the 800-drug split universe), restrict the
    audit name set to drugs whose drugbank_id is in the pool. This dramatically
    reduces false positives from endogenous-compound DrugBank entries that are
    not in the model's prediction pool.
    """
    if not id2name_path.is_file():
        raise FileNotFoundError(
            f"id2name.json not found at {id2name_path}. Required for D1 drug-name audit "
            "(see Notes/Log/d1_llm_edge_design.md §5 R3)."
        )
    id2name = json.loads(id2name_path.read_text(encoding="utf-8"))
    if pool_filter is not None:
        id2name = {did: nm for did, nm in id2name.items() if str(did) in pool_filter}
    norm_names = {_norm(name) for name in id2name.values() if name}
    norm_names = {n for n in norm_names if n}
    long_names = sorted({n for n in norm_names if len(n) >= MIN_NAME_LEN_FOR_AUDIT})
    return norm_names, long_names, len(id2name)


def _audit_token(token: str, norm_names: set[str], long_names: list[str]) -> str | None:
    """Return rejection reason if token must be dropped per the R3 audit; else None.

    Rules (whole-word boundary; built on the normalized token, where _norm collapses
    non-alphanumerics to single spaces so the token reads as space-separated words):

      (1) token's normalization equals any drug-name normalization.
      (2) any long (>= MIN_NAME_LEN_FOR_AUDIT) drug-name appears as a whole-word
          subsequence inside the token's normalization (i.e. wrapped by spaces or
          at the token boundary), AND the token is at most 4x that name's length.

    Whole-word matching is critical: in the unrestricted-substring variant we hit
    false positives where "epinephrine" matches "norepinephrine" (different
    molecule) or "choline" matches "anticholinergic" (medical-prefix compound
    word). The whole-word rule rejects "warfarin metabolism" (target is the drug
    warfarin) while keeping "norepinephrine reuptake inhibitor" (norepinephrine
    is its own whole word, distinct from "epinephrine").
    """
    t_norm = _norm(token)
    if not t_norm:
        return None
    # Rule 1
    if t_norm in norm_names:
        return f"exact_name_match={t_norm}"
    # Rule 2 — whole-word: wrap with spaces to test boundary.
    padded = " " + t_norm + " "
    for name in long_names:
        needle = " " + name + " "
        if needle in padded:
            if len(t_norm) <= 4 * len(name):
                return f"name_whole_word={name}_in_token={t_norm}"
    return None


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------

def build_llm_edges(
    typed_sets_path: Path = I4_JSON,
    id2name_path: Path = ID2NAME_JSON,
    out_dir: Path = DEFAULT_OUT_DIR,
    tag: str = DEFAULT_TAG,
    split_pkl: Path | None = DEFAULT_SPLIT_PKL,
) -> dict:
    """Build LLM-distilled drug→token edges for D1.

    Returns a summary dict (also used for logging/hashing).
    """
    print(f"[d1-builder] loading typed sets: {typed_sets_path}", flush=True)
    typed_sets = json.loads(typed_sets_path.read_text(encoding="utf-8"))
    n_drugs_input = len(typed_sets)

    typed_sets, n_dropped_phrase = _sanitize_free_text(
        typed_sets, FREE_TEXT_FIELDS, SANITIZER_PHRASES
    )
    print(
        f"[d1-builder] sanitizer-2 dropped {n_dropped_phrase} free-text tokens "
        f"with interaction language (matches v2i4_trainer.py:128-141)",
        flush=True,
    )

    pool_filter: set[str] | None = None
    if split_pkl is not None:
        pool_filter = _load_split_drug_pool(split_pkl)
        print(
            f"[d1-builder] audit-pool filter: {len(pool_filter)} drugs loaded from "
            f"{split_pkl} (audit drops only when token contains a drug name from "
            f"the prediction pool, not the full 1900-drug DrugBank)",
            flush=True,
        )

    norm_names, long_names, n_audit_drugs = _build_drug_name_audit(
        id2name_path, pool_filter
    )
    print(
        f"[d1-builder] drug-name audit source: {id2name_path} "
        f"(audit-pool size={n_audit_drugs}, {len(norm_names)} unique normalized names, "
        f"{len(long_names)} with length >= {MIN_NAME_LEN_FOR_AUDIT})",
        flush=True,
    )

    rejected_tokens: dict[str, list[tuple[str, str]]] = defaultdict(list)

    edge_rows: list[tuple[str, str, str]] = []  # (drug_id, token_node_id, relation)
    token_meta: dict[str, dict] = {}
    token_drug_counts: dict[str, Counter] = defaultdict(Counter)

    for drug_id, fdict in typed_sets.items():
        for field in ALL_FIELDS:
            raw_list = fdict.get(field, []) or []
            seen_for_drug: set[str] = set()
            for raw_token in raw_list:
                if not isinstance(raw_token, str):
                    continue
                token = raw_token.strip()
                if not token:
                    continue
                reason = _audit_token(token, norm_names, long_names)
                if reason is not None:
                    rejected_tokens[field].append((raw_token, reason))
                    continue
                slug = _slug(token)
                if not slug:
                    rejected_tokens[field].append((raw_token, "empty_after_slug"))
                    continue
                node_id = f"llm:{field}:{slug}"
                # Hard prefix-collision guard — none of the existing 5-bucket node IDs
                # carry these prefixes.
                assert not node_id.startswith(("db:", "het:", "prime:")), node_id

                if node_id in seen_for_drug:
                    continue  # within-drug duplicate token (e.g. "thrombin" listed twice)
                seen_for_drug.add(node_id)

                edge_rows.append((drug_id, node_id, f"llm:{field}"))
                token_drug_counts[node_id][drug_id] += 1

                if node_id not in token_meta:
                    token_meta[node_id] = {
                        "field": field,
                        "raw_token": token,
                        "n_drugs": 0,
                    }

    # Fill n_drugs after the full pass.
    for node_id, counter in token_drug_counts.items():
        token_meta[node_id]["n_drugs"] = len(counter)

    # Stable sort — deterministic output.
    edge_rows.sort(key=lambda r: (r[0], r[2], r[1]))
    edges_df = pd.DataFrame(
        edge_rows, columns=["drug_id", "token_node_id", "relation"]
    )
    # Defensive dedup at (drug, token, relation) level.
    before = len(edges_df)
    edges_df = edges_df.drop_duplicates(
        subset=["drug_id", "token_node_id", "relation"]
    ).reset_index(drop=True)
    if before != len(edges_df):
        print(
            f"[d1-builder] dropped {before - len(edges_df)} exact-duplicate edges "
            "(safety net; should be ~0)",
            flush=True,
        )

    # Per-field edge count.
    per_field_edges = (
        edges_df["relation"].value_counts().sort_index().to_dict() if len(edges_df) else {}
    )
    per_field_tokens = Counter(meta["field"] for meta in token_meta.values())
    per_field_rejected = {f: len(v) for f, v in rejected_tokens.items()}
    n_drugs_with_edge = edges_df["drug_id"].nunique() if len(edges_df) else 0

    print(
        f"[d1-builder] built {len(edges_df)} edges, {len(token_meta)} token nodes, "
        f"{n_drugs_with_edge} drugs have ≥1 edge",
        flush=True,
    )
    print(f"[d1-builder] per-relation edge count: {per_field_edges}", flush=True)
    print(f"[d1-builder] per-field token-node count: {dict(per_field_tokens)}", flush=True)
    print(f"[d1-builder] per-field rejected tokens (audit): {per_field_rejected}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir / f"llm_drug_edges__{tag}.parquet"
    nodes_path = out_dir / f"llm_token_nodes__{tag}.json"
    audit_path = out_dir / f"_audit_rejected_tokens__{tag}.json"

    edges_df.to_parquet(edges_path, index=False)
    nodes_path.write_text(
        json.dumps(token_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit_path.write_text(
        json.dumps(
            {f: rejected_tokens.get(f, []) for f in ALL_FIELDS},
            indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )

    # Determinism hash — for CP-2 reviewer to verify re-runs are byte-identical.
    edges_bytes = edges_path.read_bytes()
    edges_sha = hashlib.sha256(edges_bytes).hexdigest()[:16]

    summary = {
        "input_drugs": n_drugs_input,
        "n_edges": int(len(edges_df)),
        "n_token_nodes": int(len(token_meta)),
        "n_drugs_with_edge": int(n_drugs_with_edge),
        "per_relation_edges": per_field_edges,
        "per_field_token_nodes": dict(per_field_tokens),
        "per_field_rejected": per_field_rejected,
        "edges_sha16": edges_sha,
        "outputs": {
            "edges_parquet": str(edges_path),
            "token_nodes_json": str(nodes_path),
            "audit_json": str(audit_path),
        },
    }
    print(f"[d1-builder] summary: {json.dumps(summary, indent=2)}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build D1 LLM drug→token KG edges.")
    parser.add_argument("--typed-sets", type=Path, default=I4_JSON)
    parser.add_argument("--id2name", type=Path, default=ID2NAME_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument(
        "--split-pkl", type=Path, default=DEFAULT_SPLIT_PKL,
        help="Restrict drug-name audit to drugs in this split's universe. Pass empty "
        "string to disable (audit against full 1900-drug pool — over-aggressive).",
    )
    args = parser.parse_args(argv)
    split_pkl = args.split_pkl if str(args.split_pkl) else None
    build_llm_edges(
        typed_sets_path=args.typed_sets,
        id2name_path=args.id2name,
        out_dir=args.out_dir,
        tag=args.tag,
        split_pkl=split_pkl,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
