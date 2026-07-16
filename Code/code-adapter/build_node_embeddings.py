"""B3/B4 entrypoint - build frozen z_m for the mediator union and sanity-check it.

Usage (project root, WSL conda env, offline HF):
  HF_HOME=/mnt/g/hf_cache HF_HUB_OFFLINE=1 \
    python Code/code-adapter/build_node_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(_ADAPTER))

import argparse                                    # noqa: E402

from kg.store import build_kg_store               # noqa: E402
from kg.node_names import build_node_names         # noqa: E402
from kg.text_encoder import make_encoder, ENCODERS  # noqa: E402
from kg.node_embed import build_node_embeddings    # noqa: E402

_UNION = _ADAPTER / "kg" / "_cache" / "node_desc" / "union_latest_partial.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="pubmedbert",
                    help=f"registry key {list(ENCODERS)} or a raw HF model id")
    ap.add_argument("--pooling", default=None, choices=[None, "mean", "cls"])
    args = ap.parse_args()

    kg = build_kg_store()
    names = build_node_names(kg)
    node_ids = pd.read_parquet(_UNION)["node_id"].astype(str).tolist()
    print(f"[build] mediator union: {len(node_ids)} nodes | encoder={args.encoder}")

    enc = make_encoder(args.encoder, pooling=args.pooling)
    Z, ids = build_node_embeddings(kg, node_ids, enc, names=names)

    # --- sanity ---
    print(f"\n[sanity] z_m shape {Z.shape}, dtype {Z.dtype}")
    norms = np.linalg.norm(Z, axis=1)
    print(f"[sanity] row-norm min {norms.min():.4f} max {norms.max():.4f} (expect ~1)")
    print(f"[sanity] any NaN: {bool(np.isnan(Z).any())}")

    id2row = {nid: i for i, nid in enumerate(ids)}
    typ = {nid: kg.type_names[kg.type_id[kg.get_idx(nid)]] for nid in ids}
    nm = {nid: str(names[kg.get_idx(nid)]) for nid in ids}

    def neighbors(query_name, k=6):
        hit = [nid for nid in ids if nm[nid].lower() == query_name.lower()]
        if not hit:
            print(f"\n[nn] '{query_name}' not in union"); return
        q = Z[id2row[hit[0]]]
        sims = Z @ q
        order = np.argsort(-sims)
        print(f"\n[nn] top-{k} cosine neighbors of '{query_name}' ({typ[hit[0]]}):")
        shown = 0
        for j in order:
            if ids[j] == hit[0]:
                continue
            print(f"    {sims[j]:.3f}  [{typ[ids[j]]}] {nm[ids[j]]}")
            shown += 1
            if shown >= k:
                break

    for q in ["Prothrombin", "Alzheimer's disease", "Tyrosine Metabolism",
              "Epidermal growth factor receptor"]:
        neighbors(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
