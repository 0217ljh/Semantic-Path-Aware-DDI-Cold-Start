"""Effect-layer neighbor sets per drug — input to the effect-channel cross-attention.

Component B of README_i1_routing_v2_learned.md: the effect channel does pair-conditional
select-compose over N_eff(a) x N_eff(b), the two drugs' effect-layer KG neighbors
(side-effect / phenotype / disease / anatomy / symptom). This precompute collects, per drug,
its 1-hop non-drug effect-layer neighbors from the merged (DDI-masked) KG and stores their
PubMedBERT(name) embedding row-indices in CSR form so the trainer can gather variable-length
neighbor sets without padding the whole matrix.

Edge-independent / leakage-safe by the same argument as k_u: merged KG has NO DDI edges.

Codex Reviewer decision (2026-05-25, thread 019e6224): store ALL effect kinds (Option 3
infrastructure) but the MAIN LINE trains/evals on the PD-composable subset
{Side Effect, effect/phenotype, Symptom} only (disease + anatomy excluded — they are
indication/localization, not composable adverse effects). Use a per-drug top-K=32 prefilter
ranked by inverse node frequency (generic effects like nausea link to many drugs -> low
salience) before the cross-attention. disease/anatomy kept only as an ablation axis.

Output: Code/data/_cache/effect_neighbors_pubmedbert.npz
  drug_ids   : (N_drug,) str   — DrugBank ids, sorted (same convention as k_u)
  indptr     : (N_drug+1,) int64 — CSR row pointer into nbr_rows
  nbr_rows   : (nnz,) int64     — row index into the PubMedBERT emb matrix (EMB_PT order)
  nbr_kind   : (nnz,) int8      — kind code: 0=side_effect 1=phenotype 2=symptom
                                  3=disease 4=anatomy (PD-composable = {0,1,2})
  nbr_drugdeg: (nnz,) int32     — # drugs linked to this effect node (inverse-freq salience;
                                  top-K prefilter ranks by SMALLEST drugdeg)
  n_eff      : (N_drug,) int64  — effect-neighbor count per drug (all kinds)
  emb_source : str              — path to the PubMedBERT .pt these rows index into
The downstream trainer MUST load EMB_PT and index emb[nbr_rows] to recover embeddings.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]

NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
EMB_PT = PROJECT_ROOT / "Code/data/KG/_merged_kg/_cache/screen1_tag_init/d_name_only__pubmedbert.pt"
OUT = PROJECT_ROOT / "Code/data/_cache/effect_neighbors_pubmedbert.npz"

# i1 effect-layer kinds (matched case-insensitively by keyword against nodes["kind"]).
# kind code: 0=side_effect 1=phenotype 2=symptom 3=disease 4=anatomy. PD-composable={0,1,2}.
EFFECT_KEYWORDS = ("side effect", "phenotype", "disease", "anatomy", "symptom")


def _kind_code(kind: str) -> int:
    k = str(kind).lower()
    if "side effect" in k:
        return 0
    if "phenotype" in k:
        return 1
    if "symptom" in k:
        return 2
    if "disease" in k:
        return 3
    if "anatomy" in k:
        return 4
    return -1


def _is_effect_kind(kind: str) -> bool:
    return _kind_code(kind) >= 0


def main() -> None:
    print("[eff] loading PubMedBERT embeddings + KG ...", flush=True)
    obj = torch.load(EMB_PT, map_location="cpu", weights_only=False)
    nid2row = {str(n): i for i, n in enumerate(obj["node_ids"])}

    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)  # merged KG = NO DDI edges (DDI-masked, verified)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    kind_code = {nid: _kind_code(k) for nid, k in id2kind.items()}

    print("[eff] collecting effect-layer 1-hop neighbors per drug ...", flush=True)
    nbrs: dict[str, set] = defaultdict(set)
    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        s_drug = src in drug_set
        d_drug = dst in drug_set
        if s_drug and not d_drug and kind_code.get(dst, -1) >= 0:
            nbrs[src].add(dst)
        if d_drug and not s_drug and not directed and kind_code.get(src, -1) >= 0:
            nbrs[dst].add(src)

    # drug-degree of each effect node = # distinct drugs linked to it (inverse-freq salience)
    drugdeg: dict[str, int] = defaultdict(int)
    for d, ns in nbrs.items():
        for n in ns:
            drugdeg[n] += 1

    drug_ids = sorted(drug_set)
    indptr = [0]
    nbr_rows: list[int] = []
    nbr_kind: list[int] = []
    nbr_drugdeg: list[int] = []
    n_eff = np.zeros(len(drug_ids), dtype=np.int64)
    for i, d in enumerate(drug_ids):
        cnt = 0
        for n in nbrs.get(d, ()):
            if n in nid2row:
                nbr_rows.append(nid2row[n])
                nbr_kind.append(kind_code[n])
                nbr_drugdeg.append(drugdeg[n])
                cnt += 1
        indptr.append(len(nbr_rows))
        n_eff[i] = cnt

    indptr_arr = np.array(indptr, dtype=np.int64)
    nbr_rows_arr = np.array(nbr_rows, dtype=np.int64)
    nbr_kind_arr = np.array(nbr_kind, dtype=np.int8)
    nbr_drugdeg_arr = np.array(nbr_drugdeg, dtype=np.int32)
    n_with = int((n_eff > 0).sum())
    pd_composable = int((nbr_kind_arr <= 2).sum())
    print(
        f"[eff] drugs={len(drug_ids)}, with >=1 effect neighbor={n_with}, "
        f"mean(eff>0)={n_eff[n_eff>0].mean():.1f}, max={int(n_eff.max())}, "
        f"total stored={len(nbr_rows_arr)}, PD-composable(kind<=2)={pd_composable} "
        f"({100*pd_composable/max(1,len(nbr_rows_arr)):.1f}%)",
        flush=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        drug_ids=np.array(drug_ids),
        indptr=indptr_arr,
        nbr_rows=nbr_rows_arr,
        nbr_kind=nbr_kind_arr,
        nbr_drugdeg=nbr_drugdeg_arr,
        n_eff=n_eff,
        emb_source=str(EMB_PT),
    )
    print(f"[eff] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
