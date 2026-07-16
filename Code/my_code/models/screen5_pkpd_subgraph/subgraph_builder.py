"""Build PK / PD subgraphs from merged KG.

Per first_step_plan.md §4.7 (Screen 5 ⑤a) + Codex rounds 3/4/final:

  PK subgraph contains:
    - Drug-(non-drug) edges where the non-drug endpoint is in LAYER_PK
    - Non-drug-only edges where AT LEAST ONE endpoint is in LAYER_PK
    - Drug-only edges per `overlap_mode` (include_both / strict_pk / etc.)

  PD subgraph contains:
    - Drug-(non-drug) edges where the non-drug endpoint is in LAYER_PD
    - Non-drug-only edges where AT LEAST ONE endpoint is in LAYER_PD
    - Drug-only edges per `overlap_mode`

  Cross-layer edges (Gene-Disease, Pathway-SideEffect, etc.) enter BOTH
  subgraphs. This is intentional: they are mechanistic bridges between
  PK and PD layers and provide useful information to both branches.

Variants used by run_screen5:
  S0 : full KG (anchor)
  S3 : PK ∥ PD with `include_both` (primary)
  S6 : two random subgraphs of similar size (structural control; with
       independent sampling, the two subsets may be disjoint or overlap
       freely — see `split_edges_random_2way` doc)
  S7 : PK ∥ PD with `strict_pk` (drug-drug edges to PK only) — strictness control
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen1_tag_init.node_text_builder import canonical_kind  # noqa: E402

LAYER_PK = {"Gene/Protein", "Pathway"}
LAYER_PD = {"SideEffect", "Disease", "Anatomy", "Phenotype"}


def _kind_set_for_endpoint(endpoint_kind: str, ckind: str) -> set[str]:
    """Return kinds present at this endpoint's canonical kind (singleton + Drug if drug)."""
    return {ckind}


def split_edges_pk_pd(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    overlap_mode: str = "include_both",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pk_edges, pd_edges) from merged-KG edges.

    overlap_mode:
      'include_both': drug-only edges go into BOTH subgraphs (default)
      'strict_pk'   : drug-only edges go into PK only
      'strict_pd'   : drug-only edges go into PD only
    """
    id2ckind = {str(i): canonical_kind(str(k)) for i, k in zip(nodes_df["id"], nodes_df["kind"])}

    src_ckind = edges_df["src"].astype(str).map(id2ckind).fillna("_unknown")
    dst_ckind = edges_df["dst"].astype(str).map(id2ckind).fillna("_unknown")

    # Per Codex CRITICAL #1 + Codex round 4 WARN: classify by the
    # NON-DRUG endpoint(s). Three cases:
    #   (a) both endpoints Drug: drug-only edge -> handled by overlap_mode
    #   (b) one endpoint Drug: the non-drug endpoint's kind determines layer
    #   (c) both endpoints non-Drug: edge belongs to layer L if EITHER
    #       endpoint's kind is in L (cross-layer edges like Disease-Gene
    #       enter BOTH PK and PD)
    is_src_drug = (src_ckind == "Drug")
    is_dst_drug = (dst_ckind == "Drug")
    is_drug_only = is_src_drug & is_dst_drug

    # Case (c): both endpoints non-drug — check both kinds
    both_non_drug = ~is_src_drug & ~is_dst_drug
    is_pk_pure = both_non_drug & (src_ckind.isin(LAYER_PK) | dst_ckind.isin(LAYER_PK))
    is_pd_pure = both_non_drug & (src_ckind.isin(LAYER_PD) | dst_ckind.isin(LAYER_PD))

    # Case (b): one endpoint Drug — use the OTHER endpoint's kind
    src_drug_only = is_src_drug & ~is_dst_drug   # src=Drug, dst=non-drug
    dst_drug_only = ~is_src_drug & is_dst_drug   # dst=Drug, src=non-drug
    is_pk_mixed = ((src_drug_only & dst_ckind.isin(LAYER_PK)) |
                   (dst_drug_only & src_ckind.isin(LAYER_PK)))
    is_pd_mixed = ((src_drug_only & dst_ckind.isin(LAYER_PD)) |
                   (dst_drug_only & src_ckind.isin(LAYER_PD)))

    is_pk_edge = is_pk_pure | is_pk_mixed
    is_pd_edge = is_pd_pure | is_pd_mixed

    # Build base membership
    pk_member = is_pk_edge.copy()
    pd_member = is_pd_edge.copy()

    # Handle Drug-only edges per overlap_mode
    if overlap_mode == "include_both":
        pk_member |= is_drug_only
        pd_member |= is_drug_only
    elif overlap_mode == "strict_pk":
        pk_member |= is_drug_only  # drug-only goes to PK only
    elif overlap_mode == "strict_pd":
        pd_member |= is_drug_only  # drug-only goes to PD only
    elif overlap_mode == "drop_drugdrug":
        pass  # neither side includes drug-only edges
    else:
        raise ValueError(f"unknown overlap_mode={overlap_mode!r}")

    return edges_df[pk_member].copy(), edges_df[pd_member].copy()


def split_edges_random_2way(
    edges_df: pd.DataFrame,
    sizes: tuple[int, int],
    seed: int = 0,
    allow_overlap: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Control split: two random subgraphs of given sizes.

    Per Codex final review: with allow_overlap=True (default), each subgraph
    is sampled INDEPENDENTLY (with replacement allowed only if requested
    size exceeds population). This is necessary when PK + PD sizes exceed
    edge count (the natural case with `include_both` mode and overlap).

    Args:
      sizes: (n_left, n_right) target sizes — each subgraph is sampled
             independently to match its target size.
      allow_overlap: if True, the two subgraphs may share edges (default,
             matches PK/PD include_both behavior). If False, edges go to
             at most one subgraph.

    Used as the S6 structural control. To compare to PK/PD subgraphs of
    sizes (6.4M, 4.6M), use allow_overlap=True since 6.4M + 4.6M > 7.1M total.
    """
    rng = np.random.default_rng(seed)
    n = len(edges_df)
    s1, s2 = sizes
    if not allow_overlap:
        if s1 + s2 > n:
            raise ValueError(f"sizes {sizes} sum > n={n} but allow_overlap=False")
        perm = rng.permutation(n)
        return edges_df.iloc[perm[:s1]].copy(), edges_df.iloc[perm[s1:s1+s2]].copy()
    # Independent sampling (with replacement only if size > n)
    idx_left = rng.choice(n, size=s1, replace=(s1 > n))
    idx_right = rng.choice(n, size=s2, replace=(s2 > n))
    return edges_df.iloc[idx_left].copy(), edges_df.iloc[idx_right].copy()
