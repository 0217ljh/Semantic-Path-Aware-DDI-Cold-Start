"""Per-drug reach marginals + decomposed pair features for the aware adapter.

The aware layer's promiscuity/regime branch ``b(v_ab)`` (Step 1) needs per-DRUG
marginal reach statistics — how many mediators a drug reaches and the type
distribution of that reach. These are NOT in the pair support cache (which holds
only the shared intersection), so this module computes them once per drug from the
symmetrised merged KG at the SAME depth the AND retrieval uses (``l_max - 1``),
then assembles the 6-dim pair feature vector ``v_ab``.

All quantities are functions of fixed KG structure + drug identity (no endpoint
embedding, seed-independent), which is exactly the transfer-stable / one-arm
covariate role the decomposed scorer is meant to absorb (see Notes/Log/
spmn_v2_theory_anchor.md, promiscuity tension).

v_ab (6 dims, codex Step 1 spec):
    reach_sum    = 0.5 * (log1p|r_a| + log1p|r_b|)
    reach_gap    = |log1p|r_a| - log1p|r_b||
    type_ent_sum = 0.5 * (H_a + H_b)        H = normalised type entropy of the reach
    type_ent_gap = |H_a - H_b|
    dom_sum      = 0.5 * (D_a + D_b)        D = dominant-type share of the reach
    dom_gap      = |D_a - D_b|

The caller standardises the assembled matrix with TRAIN-split stats only.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from my_code.models.spmn_v1.retrieval import MergedKG, N_TYPES

#: v_ab dimensionality (codex Step 1 decomposed branch).
N_PAIR_FEATS: int = 6
#: q_ab dimensionality (codex Step 3 pair-conditioned hub gate).
N_Q_FEATS: int = 4
#: raw DDI-mechanism-pair counts returned by build_mech_feature_matrix:
#: [shared-enzyme(PK), shared-target(PD), shared-transporter]. The runner adds a
#: log1p transform + the enzyme×target interaction, so the branch sees 4 mech dims.
N_MECH_FEATS: int = 3
#: coarse rel buckets the mech features key on (retrieval.REL_BUCKET):
_REL_TARGET: int = 0
_REL_ENZYME: int = 1
_REL_TRANSP: int = 2


def compute_drug_reach_stats(
    kg: MergedKG, drug_ids: Iterable[str], depth: int,
) -> dict[str, tuple[int, np.ndarray]]:
    """Per-drug reach marginals at ``depth`` hops (mediators only, drugs excluded).

    Returns ``{drug_id: (reach_size, type_counts[N_TYPES])}``. Distances/types come
    from the same symmetrised KG the AND retrieval uses, so ``depth`` should be
    ``l_max - 1`` to match the support reach. OOV drugs map to ``(0, zeros)``.
    """
    stats: dict[str, tuple[int, np.ndarray]] = {}
    for did in dict.fromkeys(str(d) for d in drug_ids):  # dedup, stable order
        idx = kg.id_to_idx.get(did)
        if idx is None:
            stats[did] = (0, np.zeros(N_TYPES, dtype=np.float64))
            continue
        nodes, dist = kg.neighborhood(idx, depth)
        keep = (~kg.is_drug[nodes]) & (dist >= 1)
        nodes = nodes[keep]
        counts = np.bincount(kg.type_id[nodes].astype(np.int64),
                             minlength=N_TYPES).astype(np.float64)
        stats[did] = (int(nodes.size), counts)
    return stats


def _drug_scalars(size: int, counts: np.ndarray) -> tuple[float, float, float]:
    """(log1p reach size, normalised type entropy, dominant-type share)."""
    log_reach = float(np.log1p(size))
    total = float(counts.sum())
    if total <= 0.0:
        return log_reach, 0.0, 0.0
    p = counts / total
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(N_TYPES))
    dom = float(p.max())
    return log_reach, ent, dom


def pair_features(a_id: str, b_id: str,
                  stats: dict[str, tuple[int, np.ndarray]]) -> np.ndarray:
    """6-dim raw ``v_ab`` from the two drugs' reach marginals (codex Step 1)."""
    za = stats.get(str(a_id), (0, np.zeros(N_TYPES)))
    zb = stats.get(str(b_id), (0, np.zeros(N_TYPES)))
    ra, ha, da = _drug_scalars(*za)
    rb, hb, db = _drug_scalars(*zb)
    return np.array([
        0.5 * (ra + rb), abs(ra - rb),
        0.5 * (ha + hb), abs(ha - hb),
        0.5 * (da + db), abs(da - db),
    ], dtype=np.float64)


def build_pair_feature_matrix(
    a_ids: Iterable[str], b_ids: Iterable[str],
    stats: dict[str, tuple[int, np.ndarray]],
) -> np.ndarray:
    """``(n, N_PAIR_FEATS)`` raw v_ab matrix aligned to the (a_ids, b_ids) order."""
    a_ids = list(a_ids); b_ids = list(b_ids)
    out = np.zeros((len(a_ids), N_PAIR_FEATS), dtype=np.float64)
    for i in range(len(a_ids)):
        out[i] = pair_features(a_ids[i], b_ids[i], stats)
    return out


def build_pair_regime_matrix(
    reach_a: Iterable[float], reach_b: Iterable[float],
    n_shared: Iterable[float], n_hub: Iterable[float],
) -> np.ndarray:
    """``(n, N_Q_FEATS)`` raw q_ab pair-regime features for the Step-3 hub gate.

    Columns (codex Step 3 spec), per pair:
      sparsity_ratio = log((1 + max(|r_a|,|r_b|)) / (1 + |J_ab|))
      shared_hub_frac = |J_hub| / max(1, |J_ab|)
      log1p(|J_hub|)
      log((1 + |J_ab|) / (1 + |J_nonhub|))
    |r_a|,|r_b| = per-drug reach sizes; |J_ab| = shared-support size; |J_hub|/
    |J_nonhub| = hub / non-hub mediator counts in the support. All from fixed KG
    structure (transfer-stable); the caller standardises with TRAIN stats only.
    """
    ra = np.asarray(reach_a, dtype=np.float64)
    rb = np.asarray(reach_b, dtype=np.float64)
    ns = np.asarray(n_shared, dtype=np.float64)
    nh = np.asarray(n_hub, dtype=np.float64)
    mx = np.maximum(ra, rb)
    nnh = np.maximum(0.0, ns - nh)
    sparsity = np.log((1.0 + mx) / (1.0 + ns))
    hub_frac = nh / np.maximum(1.0, ns)
    log_hub = np.log1p(nh)
    log_ratio = np.log((1.0 + ns) / (1.0 + nnh))
    return np.stack([sparsity, hub_frac, log_hub, log_ratio], axis=1)


def build_mech_feature_matrix(
    rela: np.ndarray, relb: np.ndarray, offsets: np.ndarray, n_pairs: int,
) -> np.ndarray:
    """``(n_pairs, N_MECH_FEATS)`` raw DDI-mechanism-pair counts per pair:
    [shared-enzyme count, shared-target count]. A shared-enzyme mediator is one
    BOTH drugs reach via a ``db:enzyme`` edge (rel bucket 1) — the PK / shared-CYP
    signal; shared-target is both via ``db:target`` (bucket 0) — the PD signal.
    Counts come straight from the cached support relation buckets (offsets + rela +
    relb), so no KG load is needed. The caller log1p's and standardises (train-only).
    """
    def _seg_count(mask):
        cum = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
        return (cum[offsets[1:]] - cum[offsets[:-1]]).astype(np.float64)

    enz = _seg_count((rela == _REL_ENZYME) & (relb == _REL_ENZYME))
    tgt = _seg_count((rela == _REL_TARGET) & (relb == _REL_TARGET))
    transp = _seg_count((rela == _REL_TRANSP) & (relb == _REL_TRANSP))
    return np.stack([enz, tgt, transp], axis=1)


#: e_mech dimensionality (codex richer uncapped evidence): per mechanism
#: {enzyme(PK), target(PD)}: [log1p(count), AA, log1p(nonhub count), norm-overlap].
N_MECH_EVIDENCE: int = 8


def build_mech_evidence_matrix(kg, a_ids, b_ids, hub_deg: int = 1000) -> np.ndarray:
    """``(n, 8)`` UNCAPPED richer mechanism evidence per pair (codex round-2). For
    enzyme(PK) and target(PD), the columns are
      [log1p(shared count), AA, log1p(shared NON-hub count), normalized overlap].
    Shared-X = mediators BOTH drugs reach 1-hop via a ``db:X`` edge. AA = sum
    1/log(deg). Non-hub = deg <= hub_deg. Normalized overlap =
    shared / sqrt(deg1_X(a) * deg1_X(b)) where deg1_X(d) = #1-hop X-neighbors of d
    (a specificity/Jaccard-style transfer-stable ratio). UNCAPPED (cap truncates
    these). Caller standardises with TRAIN stats only.
    """
    deg = kg.degree
    a_ids = [str(x) for x in a_ids]; b_ids = [str(x) for x in b_ids]
    # per-drug 1-hop enzyme/target neighbour counts (for overlap normalization)
    d1: dict[str, tuple[int, int]] = {}
    for did in set(a_ids) | set(b_ids):
        idx = kg.id_to_idx.get(did)
        if idx is None:
            d1[did] = (0, 0); continue
        r = kg.drug_rel.get(idx, {})
        ne = sum(1 for v in r.values() if v == _REL_ENZYME)
        nt = sum(1 for v in r.values() if v == _REL_TARGET)
        d1[did] = (ne, nt)

    out = np.zeros((len(a_ids), 8), dtype=np.float64)
    for i in range(len(a_ids)):
        ai = kg.id_to_idx.get(a_ids[i]); bi = kg.id_to_idx.get(b_ids[i])
        if ai is None or bi is None:
            continue
        ra = kg.drug_rel.get(ai); rb = kg.drug_rel.get(bi)
        if not ra or not rb:
            continue
        ec = eaa = enh = tc = taa = tnh = 0.0
        for m in (ra.keys() & rb.keys()):
            if kg.is_drug[m]:
                continue
            ba, bb = ra[m], rb[m]
            w = 1.0 / np.log(max(int(deg[m]), 2)); nh = float(deg[m] <= hub_deg)
            if ba == _REL_ENZYME and bb == _REL_ENZYME:
                ec += 1.0; eaa += w; enh += nh
            elif ba == _REL_TARGET and bb == _REL_TARGET:
                tc += 1.0; taa += w; tnh += nh
        nea, nta = d1[a_ids[i]]; neb, ntb = d1[b_ids[i]]
        ov_e = ec / np.sqrt(max(nea * neb, 1))
        ov_t = tc / np.sqrt(max(nta * ntb, 1))
        out[i] = [np.log1p(ec), eaa, np.log1p(enh), ov_e,
                  np.log1p(tc), taa, np.log1p(tnh), ov_t]
    return out


__all__ = [
    "N_PAIR_FEATS",
    "N_Q_FEATS",
    "N_MECH_FEATS",
    "N_MECH_EVIDENCE",
    "compute_drug_reach_stats",
    "pair_features",
    "build_pair_feature_matrix",
    "build_pair_regime_matrix",
    "build_mech_feature_matrix",
    "build_mech_evidence_matrix",
]
