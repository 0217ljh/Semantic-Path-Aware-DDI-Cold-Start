from __future__ import annotations

"""Self-contained KG-structural analysis for cold-start DDI.

Question. For pharmacodynamic (PD) DDIs with a named "core effect" (e.g. QTc
prolongation, CNS depression, bleeding), at what undirected hop-depth in the
merged biological KG does each drug of the pair reach its DDI's core-effect
node, and do both drugs reach it?

Pipeline.
  STEP 1  PD bucket on ddi_type + regex core-effect extraction.
  STEP 2  Curated (auditable) core-effect -> KG node mapping for the top-K
          core effects by sample mass. Emits a full mapping CSV.
  STEP 3  Sample ~100 PD positive rows whose core effect is mapped (seed 42).
  STEP 4  Depth-limited (L=4) undirected BFS from each drug to the core-effect
          node. Reports dA / dB depth distributions, both-reach fraction, the
          joint (dA,dB) cells, and core-effect node degree summary.

All numbers are produced from the actual run; nothing is hard-coded.
Run via WSL conda env project_1 (see project rules).
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]  # project root
DDI_CSV = ROOT / "Code/data/KG/drugbank/filtered/ddi_edges.csv"
NODES_PARQUET = ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"

RUN_DIR = ROOT / "Code/runs/2026-06-07__pd_core_effect_depth"
RUN_DIR.mkdir(parents=True, exist_ok=True)

EFFECT_KINDS = {"Side Effect", "effect/phenotype", "Symptom", "disease", "Disease"}
# Prefer specific phenotype/side-effect kinds over broad disease umbrella nodes.
KIND_PRIORITY = {"Side Effect": 0, "effect/phenotype": 1, "Symptom": 2, "disease": 3, "Disease": 4}

L = 4          # BFS depth limit
N_SAMPLE = 100  # sampled PD pairs
SEED = 42
TOP_K = 30      # curate mapping for the top-K core effects by mass

# PD bucket keywords on lowercased ddi_type.
PD_KEYS = (
    "risk or severity", "activities", "efficacy", "cns depression", "qtc",
    "hypertension", "hypotensive", "sedative", "adverse effects",
)

# Core-effect regexes, in priority order.
RE_RISK = re.compile(r"risk or severity of (.+?) can be")
RE_ACT = re.compile(r"(?:increase|decrease) the (.+?) activities")


def extract_core(ddi_type: str) -> str | None:
    tl = ddi_type.lower()
    m = RE_RISK.search(tl)
    if m:
        return m.group(1).strip()
    m = RE_ACT.search(tl)
    if m:
        return m.group(1).strip() + " [activity]"
    return None


# --------------------------------------------------------------------------- #
# STEP 2 curated synonym map: core_effect (lowercased, as extracted) ->
# normalized target NAME to look up exactly in the KG effect-node table.
# Verified by inspection against the node table (see report). The value is the
# exact node name (case-insensitive) we want; "[activity]" suffix dropped where
# the activity names a clinical effect with a node.
# Generic / non-localizable effects are mapped to None and EXCLUDED.
# --------------------------------------------------------------------------- #
CURATED: dict[str, str | None] = {
    "cns depression": "cns depression nos",
    "central nervous system depressant (cns depressant) [activity]": "cns depression nos",
    "adverse effects": None,                       # generic, excluded
    "qtc prolongation": "prolonged qt interval",
    "qtc-prolonging [activity]": "prolonged qt interval",
    "antihypertensive [activity]": "hypertension",  # axis: drugs acting on hypertension
    "methemoglobinemia": "methemoglobinemia",
    "hypertension": "hypertension",
    "hypotensive [activity]": "hypotension",
    "bleeding": "haemorrhage",
    "bleeding and hemorrhage": "haemorrhage",
    "gastrointestinal bleeding": "gastrointestinal haemorrhage",
    "nephrotoxicity": "nephrotoxicity",
    "tachycardia": "tachycardia",
    "hyperkalemia": "hyperkalemia",
    "arrhythmogenic [activity]": "arrhythmia",
    "serotonin syndrome": "serotonin syndrome",
    "myopathy, rhabdomyolysis, and myoglobinuria": "rhabdomyolysis",
    "gastrointestinal irritation": None,            # no clean node, excluded
    "hyperglycemia": "hyperglycemia",
    "sedative [activity]": "sedation",
    "hypoglycemia": "hypoglycemia",
    "hypoglycemic [activity]": "hypoglycemia",
    "bradycardic [activity]": "bradycardia",
    "hypokalemia": "hypokalaemia",
    "renal failure, hyperkalemia, and hypertension": "hyperkalemia",
    "hypotension": "hypotension",
    "neuromuscular blocking [activity]": "neuromuscular block prolonged",
    "anticoagulant [activity]": "haemorrhage",      # anticoagulant effect -> bleeding axis
    "immunosuppressive [activity]": None,           # no clean single node, excluded
    "orthostatic hypotensive [activity]": "hypotension",
    "sedation": "sedation",
    "neurotoxic [activity]": None,                  # broad, excluded
}


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ----------------------------------------------------------------- #
    # Load data
    # ----------------------------------------------------------------- #
    ddi = pd.read_csv(DDI_CSV)
    nodes = pd.read_parquet(NODES_PARQUET)
    edges = pd.read_parquet(EDGES_PARQUET, columns=["src", "dst"])

    # ----------------------------------------------------------------- #
    # STEP 1  PD bucket + core extraction
    # ----------------------------------------------------------------- #
    lt = ddi.ddi_type.str.lower()
    mask = pd.Series(False, index=ddi.index)
    for k in PD_KEYS:
        mask |= lt.str.contains(k, regex=False, na=False)
    pd_df = ddi[mask].copy()
    pd_df["core"] = pd_df.ddi_type.map(extract_core)

    n_pd_types = pd_df.ddi_type.nunique()
    n_types_with_core = pd_df.dropna(subset=["core"]).ddi_type.nunique()

    core_counts = (
        pd_df.dropna(subset=["core"]).groupby("core").size().sort_values(ascending=False)
    )

    # ----------------------------------------------------------------- #
    # STEP 2  curated mapping for top-K by mass
    # ----------------------------------------------------------------- #
    eff = nodes[nodes.kind.isin(EFFECT_KINDS)].copy()
    eff["nl"] = eff.name.str.lower().str.strip()

    # degree over ALL nodes (undirected, dedup, self-loop-free) -- compute later
    # once the graph is built; here we first resolve node ids.
    top_effects = core_counts.head(TOP_K)

    def resolve(target_name: str) -> pd.DataFrame:
        cand = eff[eff.nl == target_name.lower().strip()]
        if cand.empty:
            return cand
        return cand.assign(_pri=cand.kind.map(KIND_PRIORITY)).sort_values("_pri")

    mapping_rows = []
    for core_effect, n_samples in top_effects.items():
        tgt = CURATED.get(core_effect, "__AUTO__")
        if tgt is None:
            mapping_rows.append(dict(
                core_effect=core_effect, n_samples=int(n_samples), node_id=None,
                node_name=None, kind=None, match_method="unmapped_generic_excluded",
            ))
            continue
        if tgt == "__AUTO__":
            # Not in curated dict: try exact name == core (drop [activity]) only.
            tgt = core_effect.replace(" [activity]", "").strip()
            method = "exact_auto"
        else:
            method = "curated_synonym"
        cand = resolve(tgt)
        if cand.empty:
            mapping_rows.append(dict(
                core_effect=core_effect, n_samples=int(n_samples), node_id=None,
                node_name=None, kind=None, match_method=f"NO_NODE({tgt})",
            ))
            continue
        row = cand.iloc[0]
        mapping_rows.append(dict(
            core_effect=core_effect, n_samples=int(n_samples), node_id=row["id"],
            node_name=row["name"], kind=row["kind"], match_method=method,
        ))

    mapping = pd.DataFrame(mapping_rows)

    # ----------------------------------------------------------------- #
    # STEP 4a  build undirected binary self-loop-free graph
    # ----------------------------------------------------------------- #
    all_ids = nodes.id.to_numpy()
    id2idx = {nid: i for i, nid in enumerate(all_ids)}
    n_nodes = len(all_ids)

    s = edges.src.map(id2idx).to_numpy()
    d = edges.dst.map(id2idx).to_numpy()
    valid = ~(pd.isna(s) | pd.isna(d))
    s = s[valid].astype(np.int64)
    d = d[valid].astype(np.int64)
    keep = s != d  # drop self loops
    s, d = s[keep], d[keep]
    # symmetrize
    si = np.concatenate([s, d])
    di = np.concatenate([d, s])
    data = np.ones(si.shape[0], dtype=np.int8)
    A = csr_matrix((data, (si, di)), shape=(n_nodes, n_nodes))
    A.sum_duplicates()
    A.data[:] = 1  # binary

    # degree = number of distinct neighbors (undirected)
    degree = np.asarray((A != 0).sum(axis=1)).ravel()

    # attach degree to mapping
    mapping["degree"] = mapping.node_id.map(
        lambda nid: int(degree[id2idx[nid]]) if nid in id2idx else None
    )
    mapping = mapping[
        ["core_effect", "n_samples", "node_id", "node_name", "kind", "degree", "match_method"]
    ]
    mapping_csv = RUN_DIR / "core_effect_node_mapping.csv"
    mapping.to_csv(mapping_csv, index=False)

    # mapped core effects (have a node)
    mapped = mapping[mapping.node_id.notna()].set_index("core_effect")
    mapped_set = set(mapped.index)

    # ----------------------------------------------------------------- #
    # STEP 3  sample 100 PD rows whose core is mapped
    # ----------------------------------------------------------------- #
    cand_rows = pd_df[pd_df.core.isin(mapped_set)].copy()
    n_avail = len(cand_rows)
    take = min(N_SAMPLE, n_avail)
    sel_idx = rng.choice(cand_rows.index.to_numpy(), size=take, replace=False)
    sample = cand_rows.loc[sel_idx].copy()
    sample["effect_node_id"] = sample.core.map(mapped.node_id)
    sample["effect_node_name"] = sample.core.map(mapped.node_name)
    sample["effect_kind"] = sample.core.map(mapped["kind"])
    sample["effect_degree"] = sample.core.map(mapped.degree)

    # ----------------------------------------------------------------- #
    # STEP 4b  depth-limited BFS from each drug to its effect node
    # ----------------------------------------------------------------- #
    indptr, indices = A.indptr, A.indices

    def bfs_to_target(source_idx: int, target_idx: int) -> int:
        """Undirected depth-limited BFS. Returns hop-depth (1..L) or -1."""
        if source_idx == target_idx:
            return 0
        visited = np.zeros(n_nodes, dtype=bool)
        visited[source_idx] = True
        frontier = np.array([source_idx], dtype=np.int64)
        for depth in range(1, L + 1):
            if frontier.size == 0:
                break
            nbr = np.concatenate(
                [indices[indptr[u]:indptr[u + 1]] for u in frontier]
            ) if frontier.size else np.array([], dtype=np.int64)
            if nbr.size == 0:
                break
            nbr = np.unique(nbr)
            nbr = nbr[~visited[nbr]]
            if nbr.size == 0:
                break
            if target_idx in nbr:  # reached at this depth
                return depth
            visited[nbr] = True
            frontier = nbr
        return -1

    per_pair = []
    for r in sample.itertuples():
        a_idx = id2idx.get(r.drug_a_id)
        b_idx = id2idx.get(r.drug_b_id)
        e_idx = id2idx.get(r.effect_node_id)
        dA = bfs_to_target(a_idx, e_idx) if (a_idx is not None and e_idx is not None) else -2
        dB = bfs_to_target(b_idx, e_idx) if (b_idx is not None and e_idx is not None) else -2
        per_pair.append(dict(
            drug_a=r.drug_a_id, drug_b=r.drug_b_id, core_effect=r.core,
            effect_node_id=r.effect_node_id, effect_node_name=r.effect_node_name,
            effect_kind=r.effect_kind, effect_degree=int(r.effect_degree),
            dA=int(dA), dB=int(dB),
        ))
    pp = pd.DataFrame(per_pair)
    pp_csv = RUN_DIR / "per_pair_depths.csv"
    pp.to_parquet(RUN_DIR / "per_pair_depths.parquet", index=False)
    pp.to_csv(pp_csv, index=False)

    # ----------------------------------------------------------------- #
    # STEP 4c  summaries
    # ----------------------------------------------------------------- #
    def depth_dist(col: pd.Series) -> dict:
        c = Counter(col.tolist())
        out = {str(k): int(c.get(k, 0)) for k in [1, 2, 3, 4]}
        out["unreachable"] = int(c.get(-1, 0))
        out["drug_or_effect_missing"] = int(c.get(-2, 0))
        return out

    n = len(pp)
    dist_A = depth_dist(pp.dA)
    dist_B = depth_dist(pp.dB)

    def pct(x):
        return round(100.0 * x / n, 1) if n else 0.0

    reach_A = pp.dA.between(1, L).sum()
    reach_B = pp.dB.between(1, L).sum()
    both_reach = (pp.dA.between(1, L) & pp.dB.between(1, L)).sum()
    both_1hop = ((pp.dA == 1) & (pp.dB == 1)).sum()
    # both reach within each depth d (both <= d, both >=1)
    both_within = {}
    for dlim in [1, 2, 3, 4]:
        both_within[str(dlim)] = int(
            ((pp.dA.between(1, dlim)) & (pp.dB.between(1, dlim))).sum()
        )

    # joint (dA,dB) cells
    joint = (
        pp.groupby(["dA", "dB"]).size().reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    joint_cells = {f"({int(r.dA)},{int(r.dB)})": int(r.n) for r in joint.itertuples()}

    deg_used = pp.effect_degree
    degree_summary = dict(
        min=int(deg_used.min()), p25=int(deg_used.quantile(0.25)),
        median=int(deg_used.median()), mean=round(float(deg_used.mean()), 1),
        p75=int(deg_used.quantile(0.75)), max=int(deg_used.max()),
        n_distinct_effect_nodes=int(pp.effect_node_id.nunique()),
    )
    # degree per distinct effect node used
    deg_per_node = (
        pp.drop_duplicates("effect_node_id")[["effect_node_name", "effect_kind", "effect_degree"]]
        .sort_values("effect_degree", ascending=False)
    )

    summary = dict(
        params=dict(L=L, n_sample_requested=N_SAMPLE, n_sample_used=int(n),
                    seed=SEED, top_k_curated=TOP_K),
        step1=dict(
            n_pd_rows=int(len(pd_df)), n_pd_distinct_types=int(n_pd_types),
            n_pd_types_with_core=int(n_types_with_core),
            n_distinct_core_effects=int(len(core_counts)),
            total_mapped_core_samples=int(core_counts.sum()),
        ),
        step2=dict(
            n_curated_topk=int(len(mapping)),
            n_mapped_to_node=int(mapping.node_id.notna().sum()),
            n_excluded=int(mapping.node_id.isna().sum()),
            mapping_csv=str(mapping_csv),
        ),
        step3=dict(
            n_pd_rows_with_mapped_core=int(n_avail),
            n_sampled=int(n),
        ),
        step4=dict(
            dA_depth_dist=dist_A,
            dB_depth_dist=dist_B,
            dA_pct_1hop=pct(dist_A["1"]),
            dA_pct_deeper_2to4=pct(dist_A["2"] + dist_A["3"] + dist_A["4"]),
            dA_pct_unreachable=pct(dist_A["unreachable"]),
            reach_A=int(reach_A), reach_B=int(reach_B),
            pct_A_reaches=pct(reach_A), pct_B_reaches=pct(reach_B),
            both_reach_within_L=int(both_reach), pct_both_reach=pct(both_reach),
            both_1hop=int(both_1hop), pct_both_1hop=pct(both_1hop),
            both_reach_within_depth=both_within,
            joint_cells=joint_cells,
            effect_degree_summary=degree_summary,
        ),
    )
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ----------------------------------------------------------------- #
    # console report
    # ----------------------------------------------------------------- #
    print("=" * 70)
    print("STEP 1  PD bucket")
    print(f"  PD rows: {len(pd_df)}  distinct PD types: {n_pd_types}  "
          f"types with core: {n_types_with_core}")
    print(f"  distinct core effects: {len(core_counts)}  "
          f"mapped-core samples: {int(core_counts.sum())}")
    print("=" * 70)
    print("STEP 2  curated mapping (top-K):")
    print(mapping.to_string(index=False))
    print(f"  -> mapped to node: {mapping.node_id.notna().sum()} / {len(mapping)}; "
          f"excluded: {mapping.node_id.isna().sum()}")
    print(f"  mapping CSV: {mapping_csv}")
    print("=" * 70)
    print(f"STEP 3  PD rows with mapped core: {n_avail}; sampled: {n}")
    print("=" * 70)
    print("STEP 4  depth analysis")
    print(f"  dA dist: {dist_A}")
    print(f"  dB dist: {dist_B}")
    print(f"  dA %1-hop={summary['step4']['dA_pct_1hop']}  "
          f"%2-4={summary['step4']['dA_pct_deeper_2to4']}  "
          f"%unreach={summary['step4']['dA_pct_unreachable']}")
    print(f"  A reaches: {pct(reach_A)}%  B reaches: {pct(reach_B)}%  "
          f"BOTH reach: {pct(both_reach)}%  BOTH 1-hop: {pct(both_1hop)}%")
    print(f"  both-reach within depth: {both_within}")
    print(f"  joint (dA,dB) cells (top): "
          f"{dict(list(joint_cells.items())[:8])}")
    print(f"  effect-node degree summary: {degree_summary}")
    print("  degree per distinct effect node used:")
    print(deg_per_node.to_string(index=False))
    print("=" * 70)
    print("Artifacts:")
    for p in [mapping_csv, RUN_DIR / "per_pair_depths.csv",
              RUN_DIR / "per_pair_depths.parquet", RUN_DIR / "summary.json"]:
        print("  ", p)


if __name__ == "__main__":
    main()
