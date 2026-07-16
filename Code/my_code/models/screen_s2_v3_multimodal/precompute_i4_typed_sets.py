"""I4 prerequisite — parse LLM structured fields into normalized per-drug typed sets.

codex 019e67af recommended I4 as the final feature test: structured PK/PD fields (vs the
inert free-text PubMedBERT embedding) used to build EXPLICIT MECHANISTIC pairwise features
on top of MNAH.

Parses Code/data/_cache/llm_pharma/llm_pharma.jsonl (the leakage-sanitized distillation),
normalizes each field, emits per-drug sets for 10 mechanism axes. The downstream model
computes pair overlap features (shared CYP-substrate, inhibitor->substrate, etc.) on the fly.

Output: Code/data/_cache/llm_pharma/i4_typed_sets.json
  {drug_id: {cyp_substrate: [...], cyp_inhibitor: [...], ...}}
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
JSONL = ROOT / "Code/data/_cache/llm_pharma/llm_pharma.jsonl"
OUT = ROOT / "Code/data/_cache/llm_pharma/i4_typed_sets.json"

CYP_RE = re.compile(r"cyp[\s-]?(\d+[a-z]?\d*)")
TRANSPORTER_PATTERNS = [
    (r"p[\s-]?(?:gp|glycoprotein)|mdr1|abcb1", "pgp"),
    (r"bcrp|abcg2", "bcrp"),
    (r"mrp\s*(\d+)|abcc(\d+)", "mrp"),
    (r"oatp\s*(\d+[a-z]\d*)|slco(\d+[a-z]\d*)", "oatp"),
    (r"oat\s*(\d+)|slc22a(\d+)", "oat"),
    (r"oct\s*(\d+)", "oct"),
    (r"bsep|abcb11", "bsep"),
    (r"mate\s*(\d+)", "mate"),
    (r"ntcp|slc10a1", "ntcp"),
    (r"pept\s*(\d+)|slc15a(\d+)", "pept"),
]

# canonicalization for free-text fields (lowercase + strip punctuation + collapse whitespace)
_PUNCT = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def _canon_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _norm_cyp(items) -> list[str]:
    out = set()
    for v in items or []:
        for m in CYP_RE.finditer(str(v).lower()):
            out.add(f"cyp{m.group(1)}")
    return sorted(out)


def _norm_trans(items) -> list[str]:
    out = set()
    for v in items or []:
        low = str(v).lower()
        for pat, base in TRANSPORTER_PATTERNS:
            for m in re.finditer(pat, low):
                groups = [g for g in m.groups() if g]
                tok = base + (groups[0] if groups else "")
                out.add(tok)
    return sorted(out)


def _norm_free(items) -> list[str]:
    out = set()
    for v in items or []:
        c = _canon_text(v)
        if c and len(c) >= 3:
            out.add(c)
    return sorted(out)


def main() -> None:
    recs = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "error" in r:
            continue
        recs.append(r)
    print(f"[i4] parsed {len(recs)} records", flush=True)

    out: dict[str, dict] = {}
    cov = Counter()
    for r in recs:
        s = r.get("structured", {}) or {}
        d = {
            "cyp_substrate": _norm_cyp(s.get("cyp_substrate")),
            "cyp_inhibitor": _norm_cyp(s.get("cyp_inhibitor")),
            "cyp_inducer": _norm_cyp(s.get("cyp_inducer")),
            "transporter_substrate": _norm_trans(s.get("transporter_substrate")),
            "transporter_inhibitor": _norm_trans(s.get("transporter_inhibitor")),
            "therapeutic_class": _norm_free(s.get("therapeutic_class")),
            "primary_targets": _norm_free(s.get("primary_targets")),
            "pd_effects": _norm_free(s.get("pd_effects")),
            "toxicity_mechanisms": _norm_free(s.get("toxicity_mechanisms")),
            "clearance": _norm_free(s.get("clearance")),
        }
        for k, v in d.items():
            if v:
                cov[k] += 1
        out[r["drug_id"]] = d

    print("[i4] field non-empty coverage:")
    for k, c in cov.most_common():
        print(f"  {k:30s}  {c}/{len(recs)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    print(f"[i4] saved -> {OUT} ({len(out)} drugs)", flush=True)


if __name__ == "__main__":
    main()
