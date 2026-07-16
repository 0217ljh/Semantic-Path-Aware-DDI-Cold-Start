"""D2 builder — per-pair meeting-mediator set cache.

Round 4 D2 (per `Notes/Log/d2_meet_mask_design.md` §3.2; CP-1 PASS_WITH_NITS
at `_reviews/2026-06-01__d2_design__round1.md`). Standalone CLI.

Mediator definition (CP-1 round 2 clarified):
  M_ab = (n1[a] ∪ n2[a]) ∩ (n1[b] ∪ n2[b])
  i.e. all NON-DRUG entities reachable within 2 hops from BOTH a and b
  (union of 1-hop and 2-hop reach, then intersection).
  This is BROADER than MNAH's 22-d count cache (which uses n1∩n1 and n2∩n2
  separately) — D2 includes the cross-hop terms n1(a)∩n2(b) and n2(a)∩n1(b).

Produces two artefacts (deterministic, byte-stable across re-runs):

  Code/data/_cache/meet_mediators/meet_mediators__seed42_drugbank.parquet
      cols: [drug_a_id, drug_b_id, mediator_kg_ids, n_mediators]
      sorted by canonical (drug_a_id, drug_b_id)
  Code/data/_cache/meet_mediators/_summary__seed42_drugbank.json
      cardinality stats + R1 trigger status + per-kind mediator counts

Cache universe (per CP-1 R2 minor fix + §3.2 step 2):
  All drug pairs over the union of train + val_s2 + test_s2 + their negative
  parquets + train_negatives/epoch_0. For the 800-drug split this is the
  full Cartesian product (≈320k canonical pairs).

Cold-start safety (per CP-1 C2):
  Reads (a) merged KG edges parquet (with mask1 DDI masking; residual 4855
  het:CrC drug-drug edges filtered by n1/n2 builder), (b) split tables only
  for drug_a_id / drug_b_id column reads (pair universe enumeration). NO
  DDI label / interaction data is consulted.

R1 single threshold contract (per §3.2 step 5 / §6 R1):
  R1 triggers if mean n_mediators > 50 OR p95 > 200 OR max > 1000.
  When triggered, §6 one-sweep allowance is pre-committed to top-K filtering.

Usage (from project root):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\\
Semantic-Path-Aware-DDI-Cold-Start && python \\
Code/my_code/models/screen_s2_v3_multimodal/precompute_meet_mediators.py"
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

NODES = PROJECT_ROOT / "Code" / "data" / "KG" / "_merged_kg" / "nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code" / "data" / "KG" / "_merged_kg" / "edges__drugbank_hetionet_primekg__mask1.parquet"
# Source pair universe from the 800-drug split pkl that the v2i4 / D1 / D2
# trainers actually consume (matches PairDataset.from_pkl flow in
# run_v2i4.py:82 and run_v3_llm_edge.py). Earlier draft incorrectly used
# Code/data/KG/drugbank/splits/seed42 which is the full 1900-drug DrugBank
# split universe (1.8M Cartesian pairs); the 800-drug regime has 320k pairs.
SPLIT_PKL = PROJECT_ROOT / "Code" / "data" / "coldddi_legacy" / "800drug" / "seed42.pkl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "Code" / "data" / "_cache" / "meet_mediators"
DEFAULT_TAG = "seed42_drugbank"

# Kind groups copied from precompute_meet_features.py:38-51 (verified 2026-06-01;
# do NOT import to keep this builder independently maintainable; the constant
# is small + stable).
KIND_GROUPS: dict[str, list[str]] = {
    "protein_gene": ["Gene", "gene/protein", "Protein"],
    "pathway": ["Pathway", "pathway"],
    "side_effect": ["Side Effect", "effect/phenotype", "Symptom"],
    "disease": ["Disease", "disease"],
    "anatomy": ["Anatomy", "anatomy"],
    "compound": ["Compound"],
    "biological_process": ["Biological Process", "biological_process"],
    "molecular_function": ["Molecular Function", "molecular_function"],
    "cellular_component": ["Cellular Component", "cellular_component"],
    "pharmacologic_class": ["Pharmacologic Class"],
    "exposure": ["exposure"],
}
KIND_TO_GROUP: dict[str, str] = {}
for grp, kinds in KIND_GROUPS.items():
    for k in kinds:
        KIND_TO_GROUP[k] = grp

# R1 thresholds (single contract — design §3.2 step 5 + §6 R1).
R1_MEAN_TRIGGER = 50.0
R1_P95_TRIGGER = 200.0
R1_MAX_TRIGGER = 1000.0

# Top-K cap (preliminary K = 20 per design §6 R1 R1-trigger response).
# Applied at builder time if --top-k is set. Ranking criterion: low KG-degree
# first (most pair-specific mediators retained). Specificity rationale: a
# high-degree node like a popular gene is shared by many drug pairs and
# therefore poorly identifies any single pair; low-degree mediators carry
# more pair-discriminative information.
DEFAULT_TOP_K = 20


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _load_nodes(nodes_path: Path) -> tuple[dict[str, str], set[str]]:
    """Return (id2kind, drug_set).

    drug_set INCLUDES kind="Compound" alongside "Drug" and "drug" because
    Hetionet labels DrugBank drug IDs as Compound, and we need to exclude
    them from mediator sets to prevent drug-as-mediator leakage. Without
    this, the Path A vocab filter retains kind="Compound" nodes that are
    actually drugs (their IDs are in the drugbank 5-bucket vocab), giving
    a path A→B via drug-mediator that defeats the cold-start assumption.
    """
    print(f"[d2-builder] loading nodes: {nodes_path}", flush=True)
    nodes = pd.read_parquet(nodes_path)
    id2kind = dict(zip(nodes["id"].astype(str), nodes["kind"].astype(str)))
    drug_like_kinds = {"Drug", "drug", "Compound"}
    drug_set = set(nodes.loc[nodes["kind"].isin(drug_like_kinds), "id"].astype(str))
    print(
        f"[d2-builder] nodes={len(nodes)}, drug-like nodes "
        f"(kinds={drug_like_kinds})={len(drug_set)}",
        flush=True,
    )
    return id2kind, drug_set


def _build_neighbor_sets(
    edges: pd.DataFrame, id2kind: dict[str, str], drug_set: set[str]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build n1 / n2 NON-DRUG neighbor sets per drug.

    Mirrors precompute_meet_features.py:81-117 with the SAME drug-drug-edge
    filter (`src_is_drug and not dst_is_drug`) so the 4855 het:CrC residual
    drug-drug edges from refine-logs/LEAKAGE_AUDIT.txt:22 are excluded by
    construction.
    """
    print("[d2-builder] building 1-hop NON-DRUG neighbor sets ...", flush=True)
    n1: dict[str, set[str]] = defaultdict(set)
    fwd: dict[str, set[str]] = defaultdict(set)
    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        fwd[src].add(dst)
        if not directed:
            fwd[dst].add(src)
        src_is_drug = src in drug_set
        dst_is_drug = dst in drug_set
        # Drug-drug edges (e.g. het:CrC) are NEVER emitted into n1.
        if src_is_drug and not dst_is_drug:
            n1[src].add(dst)
        if dst_is_drug and not src_is_drug and not directed:
            n1[dst].add(src)
    print(
        f"[d2-builder] drugs with >=1 1-hop non-drug neighbor: "
        f"{sum(1 for d in drug_set if d in n1)}",
        flush=True,
    )

    print("[d2-builder] building 2-hop NON-DRUG neighbor sets ...", flush=True)
    n2: dict[str, set[str]] = defaultdict(set)
    t0 = time.time()
    for i, drug in enumerate(sorted(drug_set)):
        for mid in n1.get(drug, ()):
            for term in fwd.get(mid, ()):
                if term == drug or term in drug_set:
                    continue
                n2[drug].add(term)
        if (i + 1) % 200 == 0:
            print(
                f"    2-hop progress: {i+1}/{len(drug_set)}  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    print(
        f"[d2-builder] drugs with >=1 2-hop non-drug: "
        f"{sum(1 for d in drug_set if d in n2)} time={time.time()-t0:.1f}s",
        flush=True,
    )
    return n1, n2


def _build_drugbank_n1(split_pkl: Path) -> dict[str, frozenset[str]]:
    """Build per-drug 1-hop non-drug neighbor sets over the drugbank 5-bucket KG.

    Path B (Notes/Log/round4_d2_vocab_mismatch_discovery.md). Schema is
    bipartite drug↔entity (enzyme/target/transporter/carrier/pathway), so
    n2 is empty by construction; only n1 is needed.

    Output keys are drug_ids; values are frozensets of non-drug entity_ids
    in the drugbank 5-bucket vocab.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "Code"))
    from data_utils import PairDataset  # noqa: E402
    from baseline.emergnn._per_mode import _kg_to_kb_dict  # noqa: E402
    from baseline.emergnn.kg_builder import build_kg_from_kb  # noqa: E402

    ds = PairDataset.from_pkl(str(split_pkl))
    drug_ids: set[str] = set()
    for _, df in ds.splits.items():
        if "drug_a_id" in df.columns:
            drug_ids.update(df["drug_a_id"].astype(str))
            drug_ids.update(df["drug_b_id"].astype(str))
    if hasattr(ds.kg, "drug_ids"):
        drug_ids.update(ds.kg.drug_ids)
    drug_ids = sorted(drug_ids)
    kb = _kg_to_kb_dict(ds.kg)
    kg_artifacts = build_kg_from_kb(kb, drug_ids, keep_only_known_drugs=True)
    triplets = np.asarray(kg_artifacts["triplets"], dtype=np.int64)
    id2entity = {v: k for k, v in kg_artifacts["entity2id"].items()}

    drug_set = set(drug_ids)
    n1: dict[str, set[str]] = defaultdict(set)
    for h, t, _r in triplets:
        h_id, t_id = id2entity[int(h)], id2entity[int(t)]
        if h_id in drug_set and t_id not in drug_set:
            n1[h_id].add(t_id)
        # build_kg_from_kb always emits drug as head, so reverse case is rare;
        # handle defensively.
        elif t_id in drug_set and h_id not in drug_set:
            n1[t_id].add(h_id)

    out = {k: frozenset(v) for k, v in n1.items()}
    sizes = [len(v) for v in out.values()]
    print(
        f"[d2-builder] drugbank n1: {len(out)} drugs with mediators, "
        f"mean {np.mean(sizes):.1f}, max {np.max(sizes) if sizes else 0}",
        flush=True,
    )
    return out


def _drugbank_5bucket_entity_set(split_pkl: Path) -> set[str]:
    """Load drugbank 5-bucket entity vocab from the trainer's data flow.

    The D2 trainer inherits `_PerModeEmerGNN_V2I4 → _PerModeEmerGNN_MNAH →
    _PerModeEmerGNN` with `backbone_kg_source="drugbank"`, so its entity2id is built by
    `build_kg_from_kb(kb, drug_ids)` over the legacy 5-bucket schema
    (enzymes / targets / transporters / carriers / pathways).

    To keep mediator IDs in the trainer's namespace (CP-2-discovered vocab
    mismatch, see Notes/Log/round4_d2_vocab_mismatch_discovery.md), we
    explicitly enumerate the 5-bucket entity vocab here and filter mediator
    sets down to it at builder time.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "Code"))
    from data_utils import PairDataset  # noqa: E402
    from baseline.emergnn._per_mode import _kg_to_kb_dict  # noqa: E402
    from baseline.emergnn.kg_builder import build_kg_from_kb  # noqa: E402

    ds = PairDataset.from_pkl(str(split_pkl))
    # Drug pool: union of train + val + test splits (trainer-side build).
    drug_ids: set[str] = set()
    for _, df in ds.splits.items():
        if "drug_a_id" in df.columns:
            drug_ids.update(df["drug_a_id"].astype(str))
            drug_ids.update(df["drug_b_id"].astype(str))
    if hasattr(ds.kg, "drug_ids"):
        drug_ids.update(ds.kg.drug_ids)
    kb = _kg_to_kb_dict(ds.kg)
    kg_artifacts = build_kg_from_kb(kb, sorted(drug_ids), keep_only_known_drugs=True)
    vocab = set(kg_artifacts["entity2id"].keys())
    print(
        f"[d2-builder] drugbank 5-bucket vocab size = {len(vocab)} "
        f"(drug + enzyme + target + transporter + carrier + pathway)",
        flush=True,
    )
    return vocab


def _pair_universe(split_pkl: Path) -> set[tuple[str, str]]:
    """Enumerate canonical drug pairs over the 800-drug split universe.

    Loads via PairDataset.from_pkl (matches v2i4 / D1 / D2 trainer flow); takes
    the union of drug_a_id / drug_b_id across train + val_s2 + test_s2 + their
    negatives (registered in PairDataset.splits.items()), then returns the
    full Cartesian over the touched drug pool so epoch-regenerated train
    negatives can be looked up at trainer time without a cache miss.

    Per design §3.2 step 2 (CP-1 minor fix), restricted to the 800-drug pool
    that the trainer actually consumes (not the upstream 1900-drug DrugBank
    splits which would give 1.8M Cartesian pairs).
    """
    sys.path.insert(0, str(PROJECT_ROOT / "Code"))
    from data_utils import PairDataset  # noqa: E402

    print(f"[d2-builder] loading split pkl: {split_pkl}", flush=True)
    ds = PairDataset.from_pkl(str(split_pkl))
    all_drugs: set[str] = set()
    observed_pairs: set[tuple[str, str]] = set()
    for name, df in ds.splits.items():
        if "drug_a_id" not in df.columns:
            continue
        for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
            ca, cb = _canonical(a, b)
            observed_pairs.add((ca, cb))
            all_drugs.add(ca)
            all_drugs.add(cb)
    # Also fold in fixed-set negative pools so the cache catches them.
    for split_name in ("val_s2", "test_s2"):
        try:
            neg_df = ds.get_negatives(split_name)[["drug_a_id", "drug_b_id"]]
        except Exception as exc:  # noqa: BLE001
            print(f"[d2-builder] skip negatives {split_name}: {exc}", flush=True)
            continue
        for a, b in zip(neg_df["drug_a_id"].astype(str), neg_df["drug_b_id"].astype(str)):
            ca, cb = _canonical(a, b)
            observed_pairs.add((ca, cb))
            all_drugs.add(ca)
            all_drugs.add(cb)
    print(
        f"[d2-builder] pair-universe drug pool size: {len(all_drugs)}; "
        f"observed-pair count: {len(observed_pairs)}",
        flush=True,
    )
    # Cartesian for epoch-regenerated negatives.
    sorted_drugs = sorted(all_drugs)
    cartesian: set[tuple[str, str]] = set()
    for a, b in itertools.combinations(sorted_drugs, 2):
        cartesian.add((a, b))
    cartesian |= observed_pairs
    print(
        f"[d2-builder] full canonical-pair Cartesian: {len(cartesian)}",
        flush=True,
    )
    return cartesian


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------

def _compute_node_degrees(edges: pd.DataFrame) -> dict[str, int]:
    """Total degree per KG node (forward + reverse for undirected). Used for
    low-degree top-K mediator ranking."""
    print("[d2-builder] computing KG node degrees for top-K ranking ...", flush=True)
    degree: dict[str, int] = defaultdict(int)
    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        degree[src] += 1
        if not directed:
            degree[dst] += 1
        else:
            degree[dst] += 1  # directed: still counts as incident to dst
    return dict(degree)


def build_meet_mediators(
    nodes_path: Path = NODES,
    edges_path: Path = EDGES,
    split_pkl: Path = SPLIT_PKL,
    out_dir: Path = DEFAULT_OUT_DIR,
    tag: str = DEFAULT_TAG,
    top_k: int | None = None,
) -> dict:
    """Build per-pair meeting-mediator cache for D2.

    top_k: if not None, retain only the K lowest-degree mediators per pair
        (CP-1 R1 trigger response — see design §3.2 step 5). Applied AFTER
        full mediator-set computation so cardinality stats reflect both
        pre-cap and post-cap distributions.
    """
    id2kind, drug_set = _load_nodes(nodes_path)

    print(f"[d2-builder] loading edges: {edges_path}", flush=True)
    edges = pd.read_parquet(edges_path)
    print(f"[d2-builder] edges={len(edges)}", flush=True)

    # Path B (per Notes/Log/round4_d2_vocab_mismatch_discovery.md):
    # Build n1 directly over the drugbank 5-bucket KG (drug ↔ entity bipartite).
    # This bypasses the merged KG entirely for mediator computation, ensuring
    # mediator IDs are in the trainer's entity vocab by construction.
    # The merged-KG load above (id2kind, edges, n1_merged, n2_merged) is kept
    # for the cardinality cross-check + leakage audit, but mediators come from
    # the bipartite drugbank n1 below.
    n1_merged, n2_merged = _build_neighbor_sets(edges, id2kind, drug_set)
    node_degree = _compute_node_degrees(edges) if top_k is not None else {}

    trainer_vocab = _drugbank_5bucket_entity_set(split_pkl)
    trainer_vocab_non_drug = trainer_vocab - drug_set

    # Build per-drug n1 over drugbank 5-bucket bipartite KG (drug ↔ entity).
    print(
        "[d2-builder] building drugbank 5-bucket per-drug n1 (Path B) ...",
        flush=True,
    )
    n1_db = _build_drugbank_n1(split_pkl)

    pair_universe = _pair_universe(split_pkl)

    # Path B mediator source: use n1 over drugbank 5-bucket bipartite KG.
    # n_union[d] = frozenset n1_db[d] directly (no n2 because schema is
    # bipartite). All mediator IDs are in trainer entity vocab by construction.
    print(
        "[d2-builder] using drugbank n1 as per-drug reach set (Path B) ...",
        flush=True,
    )
    n_union: dict[str, frozenset[str]] = n1_db
    print(
        f"[d2-builder] built n_union for {len(n_union)} drugs (mean size = "
        f"{np.mean([len(v) for v in n_union.values()]):.1f})",
        flush=True,
    )

    # Compute mediators per pair: M_ab = n_union[a] ∩ n_union[b]
    # Mathematically equivalent to (n1 ∪ n2)(a) ∩ (n1 ∪ n2)(b) per design §3.2 step 3.
    print(
        f"[d2-builder] computing mediator sets for {len(pair_universe)} pairs ...",
        flush=True,
    )
    rows: list[dict] = []
    cardinalities_raw: list[int] = []
    cardinalities_post: list[int] = []
    per_kind_counts: Counter = Counter()
    leakage_assertions = 0
    t0 = time.time()
    for i, (a, b) in enumerate(sorted(pair_universe)):
        reach_a = n_union.get(a)
        reach_b = n_union.get(b)
        if reach_a is None or reach_b is None:
            mediator_set: frozenset[str] = frozenset()
        else:
            mediator_set = reach_a & reach_b
        # Hard leakage assertion per design C2: no mediator is a drug.
        if mediator_set & drug_set:
            leakage_assertions += 1
        raw_n = len(mediator_set)
        cardinalities_raw.append(raw_n)
        # Top-K low-degree filter (R1 trigger response).
        if top_k is not None and raw_n > top_k:
            # Rank by degree ascending; pick top_k smallest-degree mediators.
            mediator_list = sorted(
                mediator_set,
                key=lambda m: (node_degree.get(m, 0), m),
            )[:top_k]
        else:
            mediator_list = sorted(mediator_set)
        cardinalities_post.append(len(mediator_list))
        rows.append({
            "drug_a_id": a,
            "drug_b_id": b,
            "mediator_kg_ids": mediator_list,
            "n_mediators": len(mediator_list),
        })
        for m in mediator_list:
            kind = id2kind.get(m, "unknown")
            per_kind_counts[KIND_TO_GROUP.get(kind, "other")] += 1
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(pair_universe) - i - 1) / max(rate, 1e-6)
            print(
                f"    pair progress: {i+1}/{len(pair_universe)}  "
                f"elapsed={elapsed:.1f}s rate={rate:.0f}p/s eta={eta:.0f}s",
                flush=True,
            )

    if leakage_assertions > 0:
        raise RuntimeError(
            f"D2 leakage assertion failed: {leakage_assertions} pairs had a "
            "drug-id in their mediator set. n1/n2 builder filter is broken."
        )

    out_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    if top_k is not None:
        out_path = out_dir / f"meet_mediators__{tag}__topk{top_k}.parquet"
    else:
        out_path = out_dir / f"meet_mediators__{tag}.parquet"
    out_df.to_parquet(out_path, index=False)

    # Cardinality stats (R1 trigger check).
    def _card_stats(arr: list[int]) -> dict:
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
            "zero_fraction": float((a == 0).mean()),
        }

    raw_stats = _card_stats(cardinalities_raw)
    post_stats = _card_stats(cardinalities_post)
    # R1 trigger evaluated against POST (final) cardinality.
    r1_trigger = (
        post_stats["mean"] > R1_MEAN_TRIGGER
        or post_stats["p95"] > R1_P95_TRIGGER
        or post_stats["max"] > R1_MAX_TRIGGER
    )

    # Cross-check against 22-d MNAH count cache (sanity).
    try:
        mnah_cache = PROJECT_ROOT / "Code" / "data" / "_cache" / "meet_feat_drugbank_seed42_kgonly_v1.parquet"
        mnah_df = pd.read_parquet(mnah_cache)
        feat_cols = [c for c in mnah_df.columns if c.startswith("f_")]
        mnah_raw = np.expm1(mnah_df[feat_cols].values).sum(axis=1)
        mnah_mean = float(mnah_raw.mean())
        # Merge by canonical pair to compute per-pair ratio (D2 vs MNAH).
        # MNAH cache already canonical (precompute_meet_features.py:147).
        d2_idx = out_df.set_index(["drug_a_id", "drug_b_id"])["n_mediators"]
        mnah_idx = mnah_df.set_index(["drug_a_id", "drug_b_id"])
        common = d2_idx.index.intersection(mnah_idx.index)
        if len(common) > 0:
            d2_vals = d2_idx.loc[common].to_numpy(dtype=np.float64)
            mnah_vals = np.expm1(mnah_df.set_index(["drug_a_id", "drug_b_id"]).loc[common, feat_cols].values).sum(axis=1)
            ratios = d2_vals / np.maximum(mnah_vals, 1.0)
            median_ratio = float(np.median(ratios))
        else:
            median_ratio = float("nan")
    except FileNotFoundError:
        mnah_mean = float("nan")
        median_ratio = float("nan")
    except Exception as exc:
        print(f"[d2-builder] MNAH cross-check failed: {exc}", flush=True)
        mnah_mean = float("nan")
        median_ratio = float("nan")

    summary = {
        "n_pairs": int(len(out_df)),
        "n_drugs_in_universe": int(out_df[["drug_a_id", "drug_b_id"]].stack().nunique()),
        "leakage_assertions_violated": int(leakage_assertions),
        "top_k": top_k,
        "cardinality_raw": raw_stats,
        "cardinality_post_topk": post_stats,
        "r1_thresholds": {
            "mean_trigger": R1_MEAN_TRIGGER,
            "p95_trigger": R1_P95_TRIGGER,
            "max_trigger": R1_MAX_TRIGGER,
        },
        "r1_triggered_post_topk": bool(r1_trigger),
        "per_kind_mediator_counts_post_topk": dict(per_kind_counts),
        "cross_check_22d_mnah": {
            "mnah_mean": mnah_mean,
            "d2_to_mnah_median_ratio": median_ratio,
            "expected_ratio_range": "[1.0, 5.0] per design §3.2 step 5",
            "note": "ratio is computed against pre-topK raw mediator counts",
        },
        "outputs": {
            "parquet": str(out_path),
        },
    }

    summary_path = out_dir / f"_summary__{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Determinism hash for CP-2.
    edges_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    summary["parquet_sha16"] = edges_sha

    print(f"[d2-builder] summary: {json.dumps(summary, indent=2)}", flush=True)

    if r1_trigger:
        print(
            "[d2-builder] *** R1 TRIGGERED *** §6 one-sweep allowance pre-committed "
            "to top-K filtering (K=20 preliminary). Builder still wrote cache, but "
            "CP-2 must surface a top-K sweep before declaring PASS.",
            flush=True,
        )
    else:
        print(
            "[d2-builder] R1 not triggered; cardinality within healthy bounds.",
            flush=True,
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build D2 per-pair meeting-mediator cache.")
    p.add_argument("--nodes", type=Path, default=NODES)
    p.add_argument("--edges", type=Path, default=EDGES)
    p.add_argument("--split-pkl", type=Path, default=SPLIT_PKL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tag", type=str, default=DEFAULT_TAG)
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Retain only the K lowest-degree mediators per pair. None disables "
        "the filter; recommended K=20 when raw R1 trigger fires (see design §3.2 "
        "step 5).",
    )
    args = p.parse_args(argv)
    build_meet_mediators(
        nodes_path=args.nodes,
        edges_path=args.edges,
        split_pkl=args.split_pkl,
        out_dir=args.out_dir,
        tag=args.tag,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
