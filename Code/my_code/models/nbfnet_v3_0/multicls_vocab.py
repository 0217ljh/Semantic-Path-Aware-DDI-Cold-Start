"""v3.0 multi-class DDI label vocabulary loader.

Reads `Code/data/_cache/ddi_type_map_v3_0.json` and exposes utility lookups for
mapping (drug_a, drug_b, ddi_type_string) -> multi-class index in [0, K) or
-1 if out-of-vocab.

Vocabulary derivation (frozen 2026-06-05).
- A_strict = ddi_type strings present in TRAIN split of EVERY seed in {42, 43, 44}
- B_top   = top-86 ddi_type strings by global frequency (sum over all 3 seeds x all splits)
- Final  = A_strict INTERSECT B_top = 80 classes
- Sorted by global frequency descending (idx=0 is the biggest class)
- Coverage 97.3 - 98.3% of positive pairs across all seeds x splits

Rationale.
- Drops 135 long-tail classes (median tail size << 10 examples) for which CE
  gradient is too noisy in cold-start S2
- Drops 6 classes that are too frequent to skip but missing from at least one
  seed's train (CE undefined for that seed)
- Out-of-vocab positives still contribute to BINARY loss (treated as
  "no mechanism label" by setting multi_labels = -1, see NBFNetJointDDI.joint_loss)
- Negatives have no ddi_type by definition; they are also labeled -1

Status. Initial 2026-06-05.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_VOCAB_PATH = "Code/data/_cache/ddi_type_map_v3_0.json"


def _project_root_from(start: Path) -> Path:
    """Walk up parents looking for the canonical project structure marker."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(
        f"could not locate project root from {start} (looking for Code/data/KG)"
    )


class MultiClsVocab:
    """Lookup helper for the v3.0 80-class DDI mechanism vocabulary.

    Attributes.
        K. number of classes (== 80 for v3.0 frozen vocab)
        idx_to_type. (K,) list of ddi_type strings (sorted by global freq desc)
        type_to_idx. dict[str, int]
        class_freq_global. dict[str, int] global frequency of each class
        source_note. string describing how the vocabulary was derived

    Usage.
        v = MultiClsVocab.load_default()
        idx = v.lookup_type("The metabolism of can be decreased when combined with")
        # idx is 0; for out-of-vocab strings, lookup_type returns None

        pair_idx = v.lookup_pair_via_table(a_drugbank_id, b_drugbank_id, pair_table)
        # pair_table = {"DBxxx|DByyy": "ddi_type_string", ...} from your parquet rows
    """

    def __init__(
        self,
        idx_to_type: list[str],
        type_to_idx: dict[str, int],
        class_freq_global: Optional[dict[str, int]] = None,
        source_note: str = "",
    ) -> None:
        self.idx_to_type = list(idx_to_type)
        self.type_to_idx = dict(type_to_idx)
        self.class_freq_global = dict(class_freq_global or {})
        self.source_note = source_note
        self.K = len(self.idx_to_type)

        if len(self.type_to_idx) != self.K:
            raise ValueError(
                f"type_to_idx size {len(self.type_to_idx)} != idx_to_type size {self.K}"
            )
        # Sanity. round-trip check
        for i, t in enumerate(self.idx_to_type):
            if self.type_to_idx.get(t) != i:
                raise ValueError(f"vocab round-trip failed at idx={i} type={t!r}")

    @classmethod
    def load(cls, json_path: Path | str) -> "MultiClsVocab":
        p = Path(json_path)
        if not p.is_file():
            raise FileNotFoundError(f"vocab file not found: {p}")
        d = json.loads(p.read_text())
        for key in ("idx_to_type", "type_to_idx"):
            if key not in d:
                raise KeyError(f"vocab json missing key {key!r}: {p}")
        return cls(
            idx_to_type=d["idx_to_type"],
            type_to_idx=d["type_to_idx"],
            class_freq_global=d.get("class_freq_global", {}),
            source_note=d.get("source", ""),
        )

    @classmethod
    def load_default(cls, anchor: Optional[Path] = None) -> "MultiClsVocab":
        """Load Code/data/_cache/ddi_type_map_v3_0.json relative to project root.

        Args.
            anchor. starting path to find project root from; defaults to this file.
        """
        start = anchor if anchor is not None else Path(__file__)
        root = _project_root_from(start)
        return cls.load(root / DEFAULT_VOCAB_PATH)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def lookup_type(self, ddi_type: str) -> Optional[int]:
        """Map a ddi_type string -> class index, or None if out-of-vocab."""
        if ddi_type is None:
            return None
        s = ddi_type.strip()
        if not s:
            return None
        return self.type_to_idx.get(s)

    def lookup_type_or_negative(self, ddi_type: Optional[str]) -> int:
        """Same as lookup_type but returns -1 instead of None.

        Convenient for batched tensor construction. The NBFNetJointDDI.joint_loss
        treats -1 as "no mechanism label" (skips multi-cls CE for that row).
        """
        idx = self.lookup_type(ddi_type)
        return -1 if idx is None else idx

    @staticmethod
    def canonical_pair(a: str, b: str) -> str:
        """Return canonical 'a|b' pair key with sorted ids (matches precompute_ddi_type_map.py)."""
        return f"{a}|{b}" if a <= b else f"{b}|{a}"

    def lookup_pair_via_table(
        self,
        a: str,
        b: str,
        pair_table: dict[str, str],
    ) -> int:
        """Look up multi-class index given a (drug_a, drug_b) pair and a pair->type table.

        Returns -1 if pair has no ddi_type or its ddi_type is out-of-vocab.
        """
        ddi_type = pair_table.get(self.canonical_pair(a, b))
        return self.lookup_type_or_negative(ddi_type)
