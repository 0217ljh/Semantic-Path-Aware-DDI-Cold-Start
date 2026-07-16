"""I1 — per-drug LLM-distilled INTRINSIC pharmacology (Claude), leakage-sanitized.

Substrate-change track B (user 2026-05-26): distill Claude's pretraining pharmacology
knowledge into a PER-DRUG feature (NOT pairwise -> cold-start honest, no DDI-edge leakage).

codex leakage protocol (thread 019e6734): the prompt FORBIDS naming any other drug / drug
class as an interacting partner and any DDI phrasing; only intrinsic properties (targets, MoA,
class, CYP/transporter roles, clearance, PD effects, toxicity). A sanitizer then scans each
output for (a) any OTHER drug name in our 1900-drug vocab and (b) DDI phrases; matches are
redacted and the response is leakage-flagged for audit.

Output JSONL (resumable): Code/data/_cache/llm_pharma/llm_pharma.jsonl
  {drug_id, name, raw, sanitized_text, structured{...}, leakage_flag, partner_hits, phrase_hits}
Keys read from API-KEY/API-KEY.txt (never printed/logged).

Usage:
  python -u .../distill_llm_pharmacology.py --limit 5      # smoke
  python -u .../distill_llm_pharmacology.py --workers 8    # full
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SMILES_CSV = ROOT / "Code/data/coldddi_legacy/800drug/drug_smiles__seed42.csv"
ID2NAME = ROOT / "Code/data/KG/drugbank/filtered/id2name.json"
KEYFILE = ROOT / "API-KEY/API-KEY.txt"
OUT_DIR = ROOT / "Code/data/_cache/llm_pharma"
OUT_JSONL = OUT_DIR / "llm_pharma.jsonl"

MODEL = "claude-haiku-4-5-20251001"

PROMPT = """Describe the intrinsic pharmacology of the drug below for representation learning.

Hard constraints (CRITICAL):
- Do NOT mention any other drug or drug class as an interacting partner, contraindicated co-medication, or combination therapy.
- Do NOT describe any drug-drug interaction.
- Do NOT use phrases like "when combined with", "coadministered with", "interacts with", "increases levels of [drug]", "avoid with", "contraindicated with".
- ONLY describe intrinsic properties of THIS drug: primary targets, mechanism of action, therapeutic class, metabolizing enzymes, transporter involvement, enzyme induction/inhibition effects, clearance route, pharmacodynamic effects, and major toxicity mechanisms.
- Express interaction-relevant biology only as an intrinsic property, e.g. "strong CYP3A4 inhibitor", never "increases exposure to [drug]".

Return STRICT JSON only, no prose outside JSON:
{{"free_text": "<= 120 words intrinsic pharmacology, following the constraints",
  "therapeutic_class": [], "primary_targets": [], "mechanism": "",
  "pd_effects": [], "toxicity_mechanisms": [],
  "cyp_substrate": [], "cyp_inhibitor": [], "cyp_inducer": [],
  "transporter_substrate": [], "transporter_inhibitor": [], "clearance": []}}

Drug: {name}
SMILES: {smiles}"""

DDI_PHRASES = [
    "interact", "coadminist", "co-administ", "combined with", "combination with",
    "concomitant", "avoid with", "contraindicated with", "when taken with",
    "increases exposure", "decreases exposure", "increases levels of", "decreases levels of",
    "increase the risk when", "potentiate", "co-medication", "co-prescri",
]


def _load_key() -> str:
    txt = KEYFILE.read_text(encoding="utf-8")
    m = re.search(r"sk-ant-[A-Za-z0-9_\-]+", txt)
    if not m:
        raise RuntimeError("no Anthropic key in API-KEY.txt")
    return m.group(0)


def _build_vocab() -> dict[str, str]:
    """lowercased other-drug-name -> drug_id, for partner-name leakage scan (len>=5)."""
    id2name = json.loads(ID2NAME.read_text(encoding="utf-8"))
    vocab = {}
    for did, nm in id2name.items():
        nm = str(nm).strip().lower()
        if len(nm) >= 5:
            vocab[nm] = did
    return vocab


def _sanitize(text: str, self_id: str, self_name: str, vocab: dict[str, str]) -> tuple[str, list, list]:
    low = text.lower()
    partner_hits = []
    for nm, did in vocab.items():
        if did == self_id or nm == (self_name or "").lower():
            continue
        if re.search(r"\b" + re.escape(nm) + r"\b", low):
            partner_hits.append(nm)
    phrase_hits = [p for p in DDI_PHRASES if p in low]
    san = text
    for nm in partner_hits:
        san = re.sub(r"(?i)\b" + re.escape(nm) + r"\b", "[REDACTED_DRUG]", san)
    return san, partner_hits, phrase_hits


def _parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


_lock = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=all drugs (smoke: 5)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=900)
    args = ap.parse_args()

    import pandas as pd
    import anthropic

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    id2name = json.loads(ID2NAME.read_text(encoding="utf-8"))
    sm = pd.read_csv(SMILES_CSV)
    id_col = "drugbank_id" if "drugbank_id" in sm.columns else sm.columns[0]
    smiles_map = dict(zip(sm[id_col].astype(str), sm["smiles"].astype(str)))

    drugs = [(d, id2name.get(d, d), smiles_map[d]) for d in smiles_map if d in id2name]
    if args.limit:
        drugs = drugs[: args.limit]

    done = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["drug_id"])
            except Exception:
                pass
    todo = [d for d in drugs if d[0] not in done]
    print(f"[llm] drugs total={len(drugs)} done={len(done)} todo={len(todo)} model={MODEL}", flush=True)

    vocab = _build_vocab()
    client = anthropic.Anthropic(api_key=_load_key())

    def work(item):
        did, name, smiles = item
        for attempt in range(5):
            try:
                r = client.messages.create(
                    model=MODEL, max_tokens=args.max_tokens, temperature=0.0,
                    messages=[{"role": "user",
                               "content": PROMPT.format(name=name, smiles=smiles)}],
                )
                raw = r.content[0].text
                obj = _parse_json(raw)
                free = str(obj.get("free_text", "")) if obj else raw
                san, ph, phr = _sanitize(free, did, name, vocab)
                return {
                    "drug_id": did, "name": name, "raw": raw,
                    "sanitized_text": san, "structured": {k: obj.get(k) for k in obj if k != "free_text"} if obj else {},
                    "leakage_flag": bool(ph or phr), "partner_hits": ph, "phrase_hits": phr,
                }
            except Exception as exc:
                if attempt == 4:
                    return {"drug_id": did, "name": name, "error": str(exc)[:200]}
                time.sleep(2 ** attempt)

    t0 = time.time()
    n_done = 0
    n_leak = 0
    with open(OUT_JSONL, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, it): it for it in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                with _lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                n_done += 1
                if rec.get("leakage_flag"):
                    n_leak += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    print(f"[llm] {n_done}/{len(todo)} leak_flagged={n_leak} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[llm] DONE todo={len(todo)} leak_flagged={n_leak} "
          f"({100*n_leak/max(1,len(todo)):.1f}%) -> {OUT_JSONL}", flush=True)


if __name__ == "__main__":
    main()
