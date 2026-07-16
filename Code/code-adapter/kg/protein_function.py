"""Stage 2 / Part B / atom B1b-protein - DrugBank protein FUNCTION source.

One-pass streaming parse of the DrugBank full-database XML, collecting for every
bio-entity (target / enzyme / carrier / transporter) its BE-id and the nested
<polypeptide> functional annotations (general-function = rich mechanism text,
specific-function = short GO-style label, gene-name). This is the raw source
text later compressed into a node description for the ~874 BE-protein mediators.

Read-only over Code/data + the external DrugBank XML. Cached to kg/_cache.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache"
#: external DrugBank full database (1.6 GB); not under Code/data
DEFAULT_XML = Path("/mnt/f/Datasets/Formal_DDI/Durg_bank/Raw_data/full database.xml")
PROTFUNC_SCHEMA = "protein_function_v1"
_BIO = {"target", "enzyme", "carrier", "transporter"}


def _tag(t: str) -> str:
    return t.rsplit("}", 1)[-1]  # strip XML namespace


def _parse_xml(xml_path: Path, log=print) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    stack: list[str] = []
    ctx: dict | None = None
    ctx_depth = -1
    grab: str | None = None      # field currently being captured
    grab_tag: str | None = None  # tag that must close to consume it
    n_bio = 0
    for ev, el in ET.iterparse(str(xml_path), events=("start", "end")):
        tag = _tag(el.tag)
        if ev == "start":
            stack.append(tag)
            if tag in _BIO and ctx is None:
                ctx = {"be": None, "gen": None, "spec": None, "gene": None}
                ctx_depth = len(stack)
            if ctx is not None and grab is None:
                if tag == "id" and len(stack) == ctx_depth + 1 and ctx["be"] is None:
                    grab, grab_tag = "be", "id"
                elif tag == "general-function":
                    grab, grab_tag = "gen", "general-function"
                elif tag == "specific-function":
                    grab, grab_tag = "spec", "specific-function"
                elif tag == "gene-name" and ctx["gene"] is None:
                    grab, grab_tag = "gene", "gene-name"
        else:  # end
            if grab is not None and tag == grab_tag:
                txt = (el.text or "").strip()
                if not ctx[grab]:
                    ctx[grab] = txt
                grab = grab_tag = None
            if tag in _BIO and ctx is not None and len(stack) == ctx_depth:
                be = ctx["be"]
                if be and be not in rows and (ctx["gen"] or ctx["spec"]):
                    rows[be] = {"be": be, "gene_name": ctx["gene"] or "",
                                "general_function": ctx["gen"] or "",
                                "specific_function": ctx["spec"] or ""}
                n_bio += 1
                ctx = None
                ctx_depth = -1
                el.clear()
            stack.pop()
    log(f"[B1b/prot] bio entities scanned {n_bio}, unique BE with function {len(rows)}")
    return pd.DataFrame(rows.values())


def build_protein_function(xml_path: Path = DEFAULT_XML, rebuild: bool = False,
                           log=print) -> pd.DataFrame:
    """Return DataFrame [be, gene_name, general_function, specific_function]."""
    if not xml_path.is_file():
        raise FileNotFoundError(f"DrugBank XML not found: {xml_path}")
    st = xml_path.stat()
    fp = f"{int(st.st_mtime)}_{st.st_size}"
    cache = _CACHE / f"protein_function__{PROTFUNC_SCHEMA}__{fp}.parquet"
    if cache.is_file() and not rebuild:
        log(f"[B1b/prot] cache HIT: {cache}")
        return pd.read_parquet(cache)
    log("[B1b/prot] parsing DrugBank XML (one pass) ...")
    df = _parse_xml(xml_path, log=log)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    (cache.with_suffix(".meta.json")).write_text(
        json.dumps({"schema": PROTFUNC_SCHEMA, "xml_fp": fp, "n": int(len(df))}),
        encoding="utf-8")
    log(f"[B1b/prot] built + cached: {cache}")
    return df


__all__ = ["build_protein_function", "PROTFUNC_SCHEMA", "DEFAULT_XML"]
