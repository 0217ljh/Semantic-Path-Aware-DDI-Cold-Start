"""B1c entrypoint - generate node descriptions in resumable batches.

Run once per batch (default 500 new nodes). Each run:
  1. builds/loads the describable-node manifest (mediator union of latest_partial,
     restricted to Gene/Protein-with-XML-function and Disease-with-PrimeKG-text),
  2. generates up to --batch-size NOT-yet-done descriptions via GPT-4o,
  3. appends them to the JSONL (crash-safe) and prints a validation report.

Usage (from project root, WSL conda env):
  python Code/code-adapter/build_descriptions.py --batch-size 500
  python Code/code-adapter/build_descriptions.py --report-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(_ADAPTER))

from data.loader import load_rank_data                     # noqa: E402
from kg.store import build_kg_store                        # noqa: E402
from kg.node_names import build_node_names                 # noqa: E402
from kg.neighborhood import build_neighborhoods_idx        # noqa: E402
from kg.mediators import shared_mediators_idx              # noqa: E402
from kg.protein_function import build_protein_function     # noqa: E402
from kg.disease_text import build_disease_text             # noqa: E402
from kg import node_desc                                   # noqa: E402
from specs import TaskSpec                                 # noqa: E402

_OUT = _ADAPTER / "kg" / "_cache" / "node_desc"
_UNION = _OUT / "union_latest_partial.parquet"
_MANIFEST = _OUT / f"manifest__{node_desc.PROMPT_VERSION}.parquet"


def _mediator_union_ids(kg, log=print) -> np.ndarray:
    """Distinct mediator node-ids over all latest_partial pairs (AND, tau=2)."""
    if _UNION.is_file():
        log(f"[union] cache HIT: {_UNION}")
        return pd.read_parquet(_UNION)["node_id"].to_numpy().astype(str)
    d = load_rank_data("drugbank_latest_partial", TaskSpec.multiclass(165), "fold0")
    allp = np.concatenate([d.train_pairs, d.cold_val_pairs, d.cold_test_pairs], axis=0)
    drug_idx = sorted({kg.get_idx(x) for x in np.unique(allp.reshape(-1))} - {None})
    nb = build_neighborhoods_idx(kg, drug_idx, L=2)
    is_med = np.zeros(kg.n_nodes, dtype=bool)
    for a, b in allp:
        ia, ib = kg.get_idx(a), kg.get_idx(b)
        if ia is not None and ib is not None:
            is_med[shared_mediators_idx(kg, nb, ia, ib, mode="and", tau=2)["med_idx"]] = True
    ids = kg.idx2node[np.flatnonzero(is_med)].astype(str)
    _OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": ids}).to_parquet(_UNION)
    log(f"[union] built {len(ids)} distinct mediators -> {_UNION}")
    return ids


def _build_manifest(rebuild: bool, log=print) -> pd.DataFrame:
    if _MANIFEST.is_file() and not rebuild:
        log(f"[manifest] cache HIT: {_MANIFEST}")
        return pd.read_parquet(_MANIFEST)
    kg = build_kg_store()
    names = build_node_names(kg)
    union_ids = _mediator_union_ids(kg, log=log)
    idx = np.array([kg.get_idx(x) for x in union_ids])
    node_names = [names[i] for i in idx]
    node_types = [kg.type_names[kg.type_id[i]] for i in idx]
    be_func = {r.be: {"gene_name": r.gene_name, "general_function": r.general_function,
                      "specific_function": r.specific_function}
               for r in build_protein_function(log=log).itertuples(index=False)}
    dis_text = build_disease_text(log=log)
    man = node_desc.assemble_manifest(union_ids, node_names, node_types, be_func, dis_text)
    _OUT.mkdir(parents=True, exist_ok=True)
    man.to_parquet(_MANIFEST)
    log(f"[manifest] built {len(man)} describable nodes "
        f"({(man['source_kind']=='protein_xml').sum()} protein, "
        f"{(man['source_kind']=='disease_features').sum()} disease) -> {_MANIFEST}")
    return man


def _report(jsonl: Path, log=print) -> None:
    if not jsonl.is_file():
        log("[report] no jsonl yet")
        return
    df = pd.read_json(jsonl, lines=True)
    if len(df) == 0:
        log("[report] jsonl empty")
        return
    status = df.get("status", pd.Series(["ok"] * len(df)))
    ok = df[status == "ok"]
    fb = df[status == "fallback_empty"]
    log(f"\n{'='*70}\nVALIDATION REPORT  ({len(df)} rows: {len(ok)} ok, {len(fb)} fallback_empty)\n{'='*70}")
    if len(ok):
        log(f"  word count (ok rows): min {ok['n_words'].min()} / median {int(ok['n_words'].median())} / max {ok['n_words'].max()}")
        n_wbad = int(((ok['n_words'] < node_desc.WORD_MIN) | (ok['n_words'] > node_desc.WORD_MAX)).sum())
        log(f"  ok rows out of [{node_desc.WORD_MIN},{node_desc.WORD_MAX}] words: {n_wbad}")
    log(f"  rejected -> fallback_empty: banned {int(df['banned'].sum())}")
    audit = node_desc.audit_jsonl(jsonl)
    if audit.is_file() and audit.stat().st_size > 0:
        adf = pd.read_json(audit, lines=True)
        log(f"  audit file holds {len(adf)} rejected raw generations: {audit}")
        for r in adf.head(5).itertuples(index=False):
            log(f"    - [{r.reason}] {r.name} : {r.description}")
    log(f"  by source_kind: {df['source_kind'].value_counts().to_dict()}")
    if len(ok):
        log("\n  sample ok (every ~len/6):")
        step = max(1, len(ok) // 6)
        for r in ok.iloc[::step].head(6).itertuples(index=False):
            log(f"    [{r.type}] {r.name}: {r.description} ({r.n_words}w)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    jsonl = node_desc.default_jsonl()
    if args.report_only:
        _report(jsonl)
        return 0

    man = _build_manifest(args.rebuild_manifest)
    stats = node_desc.generate_batch(man, jsonl=jsonl, batch_size=args.batch_size)
    print(f"\n[batch done] ok {stats['n_ok']}, fallback_empty {stats['n_fallback']}, "
          f"errors {stats['errors']}, total {stats['total_done']}/{len(man)}, "
          f"remaining {stats['remaining']}")
    _report(jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
