"""Generate PubMedBERT embedding caches for variants D, D-name, E (FIXED), F.

After this script finishes:
  Code/data/KG/_merged_kg/_cache/screen1_tag_init/
    d_full_text__pubmedbert.pt        # variant D
    d_name_only__pubmedbert.pt        # variant D-name
    e_shuffled_text__pubmedbert.pt    # variant E (FIXED: full-text within-kind shuffle)
    f_typename_only__pubmedbert.pt    # variant F (canonical kind)
    node_text_stats.parquet            # E1c motivation output

Per Codex round 1 CRITICAL #1 + #2: variant E now shuffles FULL TEXT (not just
name), and cache uses content_hash invalidation.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen1_tag_init import (  # noqa: E402
    node_text_builder as ntb,
    encoder as enc,
)


def main():
    print("=" * 72)
    print("Screen 1 — generating PubMedBERT embedding caches (round 2, post codex)")
    print("=" * 72)

    # Build the master text table once with canonical kind mapping
    print("\n[1/5] Building node text table (canonical kinds)...")
    df = ntb.build_node_text_table(canonical=True)
    node_ids = df["id"].tolist()
    print(f"  total nodes: {len(node_ids):,}")
    print(f"  drug-canonical nodes: {(df['kind'] == 'Drug').sum():,}")

    # Save E1c stats
    stats_path = ntb.KG_ROOT / "_merged_kg" / "_cache" / "screen1_tag_init" / "node_text_stats.parquet"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "raw_kind", "kind", "source_kg", "name", "text_source", "readable", "relevant"]].to_parquet(
        stats_path, index=False)
    print(f"  saved E1c stats -> {stats_path.name}")

    # Variant D
    print("\n[2/5] Encoding variant D (full text, max_len=256)...")
    enc.encode_pubmedbert(
        node_ids, df["text"].fillna("").astype(str).tolist(),
        tag="d_full_text", batch_size=64, max_length=256,
    )

    # Variant D-name
    print("\n[3/5] Encoding variant D-name (drug name / node name only, max_len=64)...")
    dname_table = ntb.build_drug_name_only_table(df)
    dname_texts = [dname_table.get(nid, "") for nid in node_ids]
    enc.encode_pubmedbert(
        node_ids, dname_texts,
        tag="d_name_only", batch_size=64, max_length=64,
    )

    # Variant E (FIXED): within-kind FULL-TEXT shuffle
    print("\n[4/5] Encoding variant E (within-kind FULL TEXT shuffle, max_len=256)...")
    shuf_table = ntb.build_shuffled_text_table(df, seed=0)
    shuf_texts = [shuf_table.get(nid, "") for nid in node_ids]
    enc.encode_pubmedbert(
        node_ids, shuf_texts,
        tag="e_shuffled_text", batch_size=64, max_length=256,
    )

    # Variant F: canonical kind string only
    print("\n[5/5] Encoding variant F (canonical kind string only, max_len=16)...")
    type_table = ntb.build_typename_only_table(df)
    type_texts = [type_table.get(nid, "") for nid in node_ids]
    enc.encode_pubmedbert(
        node_ids, type_texts,
        tag="f_typename_only", batch_size=64, max_length=16,
    )

    print("\n" + "=" * 72)
    print("All 4 caches ready (round 2, codex CRITICAL #1+#2 fixed).")


if __name__ == "__main__":
    main()
