"""Embed the sanitized per-drug LLM pharmacology text with PubMedBERT (768-d).

Consumes llm_pharma.jsonl (from distill_llm_pharmacology.py), embeds the SANITIZED text
(leakage-redacted) via the shared screen1_tag_init encoder, saves per-drug embeddings.

Output: Code/data/_cache/llm_pharma/llm_text_pubmedbert.npz
  drug_ids (N,) str | emb (N,768) float32 | leakage_flag (N,) bool
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
JSONL = ROOT / "Code/data/_cache/llm_pharma/llm_pharma.jsonl"
OUT = ROOT / "Code/data/_cache/llm_pharma/llm_text_pubmedbert.npz"


def main() -> None:
    from my_code.models.screen1_tag_init.encoder import encode_pubmedbert

    recs = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "error" in r or not r.get("sanitized_text"):
            continue
        recs.append(r)
    # dedup by drug_id (keep last)
    by_id = {r["drug_id"]: r for r in recs}
    ids = sorted(by_id)
    texts = [by_id[d]["sanitized_text"] for d in ids]
    flags = np.array([bool(by_id[d].get("leakage_flag")) for d in ids], dtype=bool)
    print(f"[embed] {len(ids)} drugs, {int(flags.sum())} leakage-flagged", flush=True)

    emb = encode_pubmedbert(ids, texts, tag="llm_pharma_sanitized", batch_size=32,
                            max_length=256).float().numpy().astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, drug_ids=np.array(ids), emb=emb, leakage_flag=flags)
    print(f"[embed] saved -> {OUT} (emb {emb.shape})", flush=True)


if __name__ == "__main__":
    main()
