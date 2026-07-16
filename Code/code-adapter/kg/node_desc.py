"""Stage 2 / Part B / atom B1c - node DESCRIPTION generation (LLM-compressed).

Only the two node classes that carry a rich LOCAL source get a description:
  - Gene/Protein with a DrugBank BE-id -> XML general/specific-function
  - Disease with a PrimeKG mondo definition -> disease_features text
All other mediators fall back to name+type (handled by the B3 encoder, not here).

The description is a single source-grounded sentence produced by GPT-4o under a
constrained-extractive policy (no outside knowledge, no drug names, no
interaction language, no restatement of the entity name/type). Results are
appended to a JSONL as they are produced (crash-safe, resumable), keyed by
node_id + prompt_version + source_sha so a later source/prompt change is visible.

Read-only over Code/data. Cache/output under kg/_cache/node_desc/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

_ADAPTER = Path(__file__).resolve().parents[1]
_OUT_DIR = _ADAPTER / "kg" / "_cache" / "node_desc"
_API_KEY_DIR = Path("/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/API-KEY")

PROMPT_VERSION = "desc_v4"
DEFAULT_MODEL = "gpt-4o"
WORD_MIN, WORD_MAX = 12, 30
_MAX_SOURCE_CHARS = 1400

# v4: like v3 the NODE NAME is a first-class knowledge handle (equal to source),
# which lets thin 2-word sources still yield an accurate description. v4 ADDS the
# constraint that the sentence must NOT begin with or restate the name/type - it
# states the function directly - because the name is encoded as its own field in
# B3, so restating it in the description is redundant with z_name.
SYSTEM_PROMPT = (
    "You write ONE factual clause describing the biological function of a "
    "biomedical entity. You are given the entity NAME, its TYPE, and SOURCE TEXT. "
    "Treat the name and source as equally important inputs and draw on "
    "well-established biological knowledge of the named entity. Describe ONLY its "
    "function and mechanism. Do NOT begin with or restate the entity name or its "
    "type. State the function directly, for example \"Catalyzes the conversion of "
    "dUMP to dTMP, a key step in DNA synthesis\". Write one declarative sentence "
    f"of {WORD_MIN} to {WORD_MAX} words, specific to this entity and factual, with "
    "no invented or speculative claims. Never mention any drug, medication, "
    "treatment, or drug-drug interaction, and never use the words drug-drug, drug "
    "interaction, DDI, or contraindication. Output only the sentence, with no "
    "prefix or quotation marks."
)


def build_user_message(name: str, type_canon: str, source_text: str) -> str:
    """Shared user-message construction (sample + batch must stay identical)."""
    return (f"Entity name: {name}\nEntity type: {type_canon}\n"
            f"Source text: {source_text}\n\nOne-sentence functional description:")
# Only GENUINE drug-drug-interaction phrasing is a leak. Bare "interact" is NOT
# banned: protein-protein / ligand-receptor "interacting with ..." is normal
# mechanism vocabulary in the source function text (removing it nuked 78% of
# valid protein descriptions in the batch-1 pilot).
_BANNED = re.compile(r"\b(drug[- ]drug|drug interaction[s]?|DDI|contraindicat\w*)\b", re.I)


# ----------------------------------------------------------------------------
# source assembly
# ----------------------------------------------------------------------------
def _clip(s: str, n: int = _MAX_SOURCE_CHARS) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


_BE_ID = re.compile(r"^db:(?:target|enzyme|carrier|transporter):(BE\d+)$")


def _be_of(node_id: str) -> str | None:
    """db:target:BE0000048 -> BE0000048 (only the exact DrugBank BE-id pattern)."""
    m = _BE_ID.match(str(node_id))
    return m.group(1) if m else None


def build_source_for_node(node_id: str, name: str, type_canon: str,
                          be_func: dict[str, dict], dis_text: dict[str, str]
                          ) -> tuple[str, str] | None:
    """Return (source_text, source_kind), or None when the node is not describable.

    Describable = DrugBank BE-proteins and diseases. Under prompt v3 the NAME is a
    first-class knowledge handle, so a describable node with NO local source is
    still generated (source_text=""), described from its name. Non-BE bulk genes
    and all other types return None (encoder falls back to name+type)."""
    if type_canon == "Gene/Protein":
        be = _be_of(node_id)
        if be is None:
            return None  # non-BE bulk gene: name+type only
        if be in be_func:
            r = be_func[be]
            src = _clip(f"{r.get('general_function','')} {r.get('specific_function','')}")
            if src:
                return src, "protein_xml"
        return "", "protein_name"  # BE-protein without XML function: describe from name
    if type_canon == "Disease":
        key = str(name).strip().lower()
        if key in dis_text:
            return _clip(dis_text[key]), "disease_features"
        return "", "disease_name"  # disease without PrimeKG text: describe from name
    return None


def assemble_manifest(node_ids, names, types, be_func, dis_text) -> pd.DataFrame:
    """Build the describable-node manifest (only nodes with a local source)."""
    recs = []
    for nid, nm, ty in zip(node_ids, names, types):
        got = build_source_for_node(nid, nm, ty, be_func, dis_text)
        if got is None:
            continue
        src, kind = got
        if not src and not str(nm).strip():
            continue  # nothing to describe from (no name, no source)
        recs.append({"node_id": str(nid), "name": str(nm), "type": str(ty),
                     "source_text": src, "source_kind": kind,
                     "source_sha": hashlib.sha1(src.encode()).hexdigest()[:12]})
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------------
def load_openai_key() -> str:
    """The key file is the source of truth (env may hold a stale key -> 401)."""
    for f in _API_KEY_DIR.glob("sk-proj-*"):
        return f.stem
    raise RuntimeError(f"no sk-proj-* key in {_API_KEY_DIR}")


def _chat(key: str, system: str, user: str, model: str) -> str:
    body = json.dumps({"model": model, "temperature": 0, "seed": 42,
                       "max_tokens": 80,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip().strip('"')


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def validate(desc: str) -> dict:
    """A description is invalid only if it uses genuine DDI phrasing or is empty.
    (A KG-wide drug-name lexicon was tried and dropped: it flags endogenous
    molecules like calcium/thrombin/vasopressin that are legitimate mechanism
    vocabulary, not drug leakage.)"""
    words = desc.split()
    return {"n_words": len(words),
            "word_ok": WORD_MIN <= len(words) <= WORD_MAX,
            "banned": bool(_BANNED.search(desc)),
            "empty": len(desc.strip()) == 0}


# ----------------------------------------------------------------------------
# incremental, resumable generation
# ----------------------------------------------------------------------------
def default_jsonl() -> Path:
    return _OUT_DIR / f"descriptions__{PROMPT_VERSION}.jsonl"


def audit_jsonl(jsonl: Path) -> Path:
    """Sibling file holding rejected (leaked/banned/empty) raw generations."""
    return jsonl.with_name(jsonl.name.replace(".jsonl", "__audit.jsonl"))


def desc_key(node_id: str, source_sha: str, model: str) -> str:
    """Stable generation key: a source/prompt/model change is NOT skipped as done."""
    raw = f"{node_id}|{PROMPT_VERSION}|{source_sha}|{model}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_done(jsonl: Path) -> set[str]:
    """Set of desc_keys already written to the canonical JSONL (any status)."""
    if not jsonl.is_file():
        return set()
    done = set()
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add(r.get("desc_key")
                         or desc_key(r["node_id"], r["source_sha"], r["model"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append(f, rec: dict) -> None:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def generate_batch(manifest: pd.DataFrame, jsonl: Path | None = None,
                   batch_size: int = 500, model: str = DEFAULT_MODEL,
                   key: str | None = None, log=print) -> dict:
    """Generate up to `batch_size` NOT-yet-done descriptions. Each accepted row is
    appended (flush+fsync) to the canonical JSONL with status 'ok'; a row that
    fails validation (genuine DDI phrasing or empty) is written to an audit JSONL
    AND recorded canonically as status 'fallback_empty' with an empty description,
    so the B3 encoder falls back to name+type and the node is never retried nor
    frozen with a leaked description. Resumable via desc_key."""
    jsonl = jsonl or default_jsonl()
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_jsonl(jsonl)
    key = key or load_openai_key()
    done = load_done(jsonl)
    keys = manifest.apply(lambda r: desc_key(r["node_id"], r["source_sha"], model), axis=1)
    todo = manifest[~keys.isin(done)].head(batch_size)
    log(f"[B1c] manifest {len(manifest)}, done {len(done)}, this batch {len(todo)}")

    stats = {"n_ok": 0, "n_fallback": 0, "banned": 0, "word_bad": 0, "errors": 0}
    with jsonl.open("a", encoding="utf-8") as f, audit.open("a", encoding="utf-8") as af:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            user = build_user_message(r.name, r.type, r.source_text)
            try:
                desc = _chat(key, SYSTEM_PROMPT, user, model)
            except Exception as e:  # noqa: BLE001 - never crash the batch on one node
                stats["errors"] += 1
                log(f"[B1c] ERROR {r.node_id}: {e}")
                continue
            v = validate(desc)
            k = desc_key(r.node_id, r.source_sha, model)
            base = {"desc_key": k, "node_id": r.node_id, "name": r.name,
                    "type": r.type, "source_kind": r.source_kind,
                    "source_sha": r.source_sha, "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            invalid = v["banned"] or v["empty"]
            if invalid:
                _append(af, {**base, "description": desc, "n_words": v["n_words"],
                             "banned": v["banned"],
                             "reason": "banned" if v["banned"] else "empty"})
                _append(f, {**base, "description": "", "n_words": 0,
                            "banned": v["banned"], "status": "fallback_empty"})
                stats["n_fallback"] += 1
                stats["banned"] += int(v["banned"])
            else:
                _append(f, {**base, "description": desc, "n_words": v["n_words"],
                            "banned": False, "status": "ok"})
                stats["n_ok"] += 1
                stats["word_bad"] += int(not v["word_ok"])
            if i % 50 == 0:
                log(f"[B1c] {i}/{len(todo)} ...")
    stats["n"] = stats["n_ok"] + stats["n_fallback"]
    stats["total_done"] = len(done) + stats["n"]
    stats["remaining"] = len(manifest) - stats["total_done"]
    return stats


__all__ = ["build_source_for_node", "assemble_manifest", "generate_batch",
           "load_done", "validate", "load_openai_key", "default_jsonl",
           "audit_jsonl", "desc_key", "build_user_message", "SYSTEM_PROMPT",
           "PROMPT_VERSION", "WORD_MIN", "WORD_MAX"]
