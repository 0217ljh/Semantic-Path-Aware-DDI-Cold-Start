"""Drug-wise cold-start S0 / S1 / S2 splits (Appendix B.1).

Implementation summary
----------------------
Three settings, all driven by a shared 50/50 (or ``drug_ratio``-controlled)
split of the drug universe into seen ``G_1`` and unseen ``G_2``:

* **S0** (transductive): both endpoints in ``G_1``; train + val + test all
  drawn from the same ``G_1 × G_1`` pool with a guarantee that every
  drug in ``G_1`` appears in train at least once.
* **S1** (semi-inductive): train edges are ``G_1 × G_1``; val/test edges
  cross between ``G_1`` and ``G_2`` (i.e. exactly one endpoint is unseen).
* **S2** (fully inductive): train edges are ``G_1 × G_1``; val/test edges
  are ``G_2 × G_2`` (both endpoints unseen at training time).

A single :class:`SplitFolds` object stores the train + 6 val/test
DataFrames + the two drug groups. The accompanying CLI / driver
function :func:`build_splits` wraps the three legacy step functions
(`Preprocessor/splits.py::cold_start_split_{0,1,2}_step`).

Persistence: per :class:`SplitFolds.save`, each split becomes its own
parquet under ``<out_dir>/seed{N}/<name>.parquet`` plus a ``manifest.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DRUG_RATIO: float = 1.5
DEFAULT_VAL_RATIO: float = 0.1

#: All seven splits, ordered (one train + six val/test buckets).
#: A single train serves all three settings (S0/S1/S2):
#: it is the ``G1 × G1`` pool minus the S0 val/test holdouts, so it is
#: guaranteed disjoint from every val/test bucket.
SPLIT_NAMES: tuple[str, ...] = (
    "train",
    "val_s0",
    "val_s1",
    "val_s2",
    "test_s0",
    "test_s1",
    "test_s2",
)

#: Mapping from a setting label ("s0" / "s1" / "s2") to its (val, test) names.
SETTINGS: dict[str, tuple[str, str]] = {
    "s0": ("val_s0", "test_s0"),
    "s1": ("val_s1", "test_s1"),
    "s2": ("val_s2", "test_s2"),
}


@dataclass
class SplitFolds:
    """Concrete :class:`data_utils.protocols.SplitFoldsProtocol` implementation.

    A single canonical ``train`` set serves all three cold-start
    settings (S0 / S1 / S2). It is constructed as
    ``G1 × G1`` **minus** the S0 val/test holdout (and capped at
    ``s0_train_budget`` when that is provided) so that:

    * ``train`` ∩ ``val_s0`` = ``train`` ∩ ``test_s0`` = ∅
      (S0 holdout is explicitly removed from train).
    * ``train`` ⊆ ``G1 × G1`` is automatically disjoint from
      ``val_s1`` / ``test_s1`` (cross edges) and from
      ``val_s2`` / ``test_s2`` (``G2 × G2`` edges).

    This matches the legacy ``cold_start_split_fair_step`` behaviour
    that produced the 800-drug / 1,900-drug bundles shipped in the
    paper, where ``split.train_idx`` is a single index list.
    """

    train: pd.DataFrame
    val_s0: pd.DataFrame
    val_s1: pd.DataFrame
    val_s2: pd.DataFrame
    test_s0: pd.DataFrame
    test_s1: pd.DataFrame
    test_s2: pd.DataFrame
    g1_drugs: list[str]
    g2_drugs: list[str]
    seed: int
    drug_ratio: float = DEFAULT_DRUG_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO

    # ------------------------------------------------------------------
    # Iteration helpers (used by serialization + tests)
    # ------------------------------------------------------------------

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [(name, getattr(self, name)) for name in SPLIT_NAMES]

    def total_pairs(self) -> int:
        return sum(len(df) for _, df in self.items())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Write each split as its own parquet + a manifest.json."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in self.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        manifest = {
            "seed": self.seed,
            "drug_ratio": self.drug_ratio,
            "val_ratio": self.val_ratio,
            "n_pairs": {name: int(len(df)) for name, df in self.items()},
            "g1_drugs": list(self.g1_drugs),
            "g2_drugs": list(self.g2_drugs),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def from_dir(cls, in_dir: Path) -> "SplitFolds":
        manifest = json.loads((in_dir / "manifest.json").read_text())
        kwargs = {
            name: pd.read_parquet(in_dir / f"{name}.parquet") for name in SPLIT_NAMES
        }
        return cls(
            **kwargs,
            g1_drugs=list(manifest["g1_drugs"]),
            g2_drugs=list(manifest["g2_drugs"]),
            seed=int(manifest["seed"]),
            drug_ratio=float(manifest.get("drug_ratio", DEFAULT_DRUG_RATIO)),
            val_ratio=float(manifest.get("val_ratio", DEFAULT_VAL_RATIO)),
        )

    def __repr__(self) -> str:
        sizes = {n: len(df) for n, df in self.items()}
        return (
            f"SplitFolds(seed={self.seed}, |G1|={len(self.g1_drugs)}, |G2|={len(self.g2_drugs)}, "
            f"sizes={sizes})"
        )


# ----------------------------------------------------------------------
# Group construction
# ----------------------------------------------------------------------


def _partition_drugs(
    all_drugs: list[str],
    *,
    seed: int,
    drug_ratio: float,
) -> tuple[list[str], list[str]]:
    """Shuffle and split the drug universe into ``(G1, G2)``.

    ``len(G1) = len(all_drugs) // drug_ratio`` (matches the legacy
    Preprocessor/splits.py behaviour with ``drug_ratio=1.5``, i.e.
    ~67% of drugs in G1).
    """
    drugs = sorted(set(all_drugs))
    rng = np.random.default_rng(seed)
    rng.shuffle(drugs)
    mid = max(1, int(len(drugs) // drug_ratio))
    return drugs[:mid], drugs[mid:]


def _coverage_anchor(
    edges: pd.DataFrame,
    g1: set[str],
    *,
    seed: int,
) -> list[int]:
    """S0 anchor: a deterministic index list that touches every drug in ``g1``.

    Mirrors the legacy ``cold_start_split_0_step`` "base_train" loop:
    iterate edges in a shuffled order, keep one whenever it covers a
    not-yet-seen drug.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(len(edges))
    rng.shuffle(order)

    a_arr = edges["drug_a_id"].astype(str).to_numpy()
    b_arr = edges["drug_b_id"].astype(str).to_numpy()

    unseen = set(g1)
    anchor: list[int] = []
    for i in order:
        if not unseen:
            break
        a, b = a_arr[i], b_arr[i]
        if a in unseen or b in unseen:
            anchor.append(int(i))
            unseen.discard(a)
            unseen.discard(b)
    return anchor


# ----------------------------------------------------------------------
# Per-setting builders (positive-only DataFrames keyed by drug_a_id/drug_b_id)
# ----------------------------------------------------------------------


def _build_s0(
    edges: pd.DataFrame,
    g1: set[str],
    *,
    seed: int,
    val_ratio: float,
    train_budget: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S0 transductive: train + val + test all in G1×G1.

    ``train_budget`` caps the train size to align with S1's train edge
    count (legacy default: 25,690). When ``None``, a 80/10/10 fallback
    is used.
    """
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    mask = a.isin(g1) & b.isin(g1)
    g1g1_idx = np.array(edges.index[mask].tolist())
    if len(g1g1_idx) == 0:
        empty = edges.iloc[0:0].reset_index(drop=True)
        return empty, empty, empty

    anchor = _coverage_anchor(edges.loc[mask].reset_index(drop=True), g1, seed=seed)
    # `anchor` indexes into the *masked* table; map back to global indices.
    g1g1_subset = edges.loc[mask].reset_index(drop=True)
    anchor_global = g1g1_subset.iloc[anchor].index.tolist()

    rng = np.random.default_rng(seed + 17)
    remaining = [int(i) for i in g1g1_subset.index.tolist() if i not in set(anchor_global)]
    rng.shuffle(remaining)

    target_train = train_budget if train_budget is not None else int(0.8 * len(g1g1_subset))
    n_extra = max(0, target_train - len(anchor_global))
    extra = remaining[:n_extra]
    rest = remaining[n_extra:]
    n_val = int(len(rest) * val_ratio)

    train_idx = anchor_global + extra
    val_idx = rest[:n_val]
    test_idx = rest[n_val:]

    return (
        g1g1_subset.iloc[train_idx].reset_index(drop=True),
        g1g1_subset.iloc[val_idx].reset_index(drop=True),
        g1g1_subset.iloc[test_idx].reset_index(drop=True),
    )


def _build_s1(
    edges: pd.DataFrame,
    g1: set[str],
    g2: set[str],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S1 semi-inductive: train = G1×G1, val/test = cross (G1×G2 ∪ G2×G1)."""
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    train_mask = a.isin(g1) & b.isin(g1)
    cross_mask = (a.isin(g1) & b.isin(g2)) | (a.isin(g2) & b.isin(g1))

    train_df = edges.loc[train_mask].reset_index(drop=True)
    cross_df = edges.loc[cross_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed + 29)
    cross_order = np.arange(len(cross_df))
    rng.shuffle(cross_order)
    n_val = int(len(cross_order) * val_ratio)
    val_idx = cross_order[:n_val]
    test_idx = cross_order[n_val:]
    return (
        train_df,
        cross_df.iloc[val_idx].reset_index(drop=True),
        cross_df.iloc[test_idx].reset_index(drop=True),
    )


def _build_s2(
    edges: pd.DataFrame,
    g1: set[str],
    g2: set[str],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S2 fully inductive: train = G1×G1, val/test = G2×G2."""
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    train_mask = a.isin(g1) & b.isin(g1)
    g2g2_mask = a.isin(g2) & b.isin(g2)

    train_df = edges.loc[train_mask].reset_index(drop=True)
    g2g2_df = edges.loc[g2g2_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed + 31)
    order = np.arange(len(g2g2_df))
    rng.shuffle(order)
    n_val = int(len(order) * val_ratio)
    val_idx = order[:n_val]
    test_idx = order[n_val:]
    return (
        train_df,
        g2g2_df.iloc[val_idx].reset_index(drop=True),
        g2g2_df.iloc[test_idx].reset_index(drop=True),
    )


# ----------------------------------------------------------------------
# Top-level builder
# ----------------------------------------------------------------------


def build_splits(
    ddi_edges: pd.DataFrame,
    *,
    seed: int,
    drug_ratio: float = DEFAULT_DRUG_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    s0_train_budget: int | None = None,
) -> SplitFolds:
    """Run S0 + S1 + S2 cold-start splitting in one pass.

    The drug universe is split into ``(G1, G2)`` once with the given
    ``seed`` and ``drug_ratio``; all three settings then share that
    partition. The label_col column (``"label"``), if present, is
    preserved verbatim in every output DataFrame.

    Parameters
    ----------
    ddi_edges
        Positive-only DataFrame with columns ``drug_a_id, drug_b_id``
        plus any extra columns to carry through.
    seed
        Random seed; the same seed always yields the same splits.
    drug_ratio
        ``len(G1) = len(drugs) // drug_ratio``; default 1.5.
    val_ratio
        Fraction of cold-start edges used for validation; default 0.1.
    s0_train_budget
        S0 train cap so its train size matches S1; legacy default 25,690.
    """
    if "drug_a_id" not in ddi_edges.columns or "drug_b_id" not in ddi_edges.columns:
        raise ValueError("ddi_edges must contain `drug_a_id` and `drug_b_id` columns")

    edges = ddi_edges.reset_index(drop=True).copy()
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)

    all_drugs = pd.unique(pd.concat([a, b]))
    g1_list, g2_list = _partition_drugs(list(all_drugs), seed=seed, drug_ratio=drug_ratio)
    g1, g2 = set(g1_list), set(g2_list)

    # _build_s0 returns a holdout-safe G1×G1 train. We use this as the
    # canonical train and reuse it across S1 / S2 — it's already disjoint
    # from every val/test bucket the three settings produce.
    train, val_s0, test_s0 = _build_s0(
        edges, g1, seed=seed, val_ratio=val_ratio, train_budget=s0_train_budget
    )
    _train_s1, val_s1, test_s1 = _build_s1(edges, g1, g2, seed=seed, val_ratio=val_ratio)
    _train_s2, val_s2, test_s2 = _build_s2(edges, g1, g2, seed=seed, val_ratio=val_ratio)

    return SplitFolds(
        train=train,
        val_s0=val_s0,
        val_s1=val_s1,
        val_s2=val_s2,
        test_s0=test_s0,
        test_s1=test_s1,
        test_s2=test_s2,
        g1_drugs=sorted(g1_list),
        g2_drugs=sorted(g2_list),
        seed=seed,
        drug_ratio=drug_ratio,
        val_ratio=val_ratio,
    )


__all__ = ["SplitFolds", "build_splits", "SPLIT_NAMES", "DEFAULT_DRUG_RATIO", "DEFAULT_VAL_RATIO"]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Build cold-start drug-wise S0/S1/S2 splits and (optionally) "
            "the matching static negatives. Writes one parquet per split "
            "plus manifest.json under --out-dir."
        ),
    )
    parser.add_argument(
        "--ddi-edges",
        required=True,
        type=Path,
        help="Input ddi_edges.csv (must contain `drug_a_id`, `drug_b_id`).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory (typically `splits/seed{N}/`).",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--drug-ratio",
        type=float,
        default=DEFAULT_DRUG_RATIO,
        help=f"|G1| = |drugs| // drug_ratio (default {DEFAULT_DRUG_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Fraction of cold-start edges used for validation (default {DEFAULT_VAL_RATIO}).",
    )
    parser.add_argument(
        "--s0-train-budget",
        type=int,
        default=None,
        help="Optional cap on the S0 train size (legacy default 25690).",
    )
    parser.add_argument(
        "--build-negatives",
        action="store_true",
        help="Also generate the six static-negative parquet files under <out>/negatives/.",
    )
    parser.add_argument(
        "--n-train-negative-epochs",
        type=int,
        default=0,
        help=(
            "If > 0, pre-bake N epochs of training negatives under "
            "<out>/train_negatives/epoch_{0..N-1}.parquet. Each epoch uses a "
            "distinct deterministic sub-seed."
        ),
    )
    args = parser.parse_args(argv)

    edges = pd.read_csv(args.ddi_edges)
    splits = build_splits(
        edges,
        seed=args.seed,
        drug_ratio=args.drug_ratio,
        val_ratio=args.val_ratio,
        s0_train_budget=args.s0_train_budget,
    )
    splits.save(args.out_dir)
    print(splits, file=sys.stderr)

    if args.build_negatives:
        # Local import to avoid a circular dependency at module load time.
        from data_utils.negatives import build_static_negatives

        neg_dir = args.out_dir / "negatives"
        neg_dir.mkdir(parents=True, exist_ok=True)
        statics = build_static_negatives(splits, base_seed=args.seed)
        for name, df in statics.items():
            df.to_parquet(neg_dir / f"{name}.parquet", index=False)
            print(f"  negatives/{name}: {len(df):,} rows", file=sys.stderr)

    if args.n_train_negative_epochs > 0:
        # Stream-write each epoch as soon as it's sampled so a long full-
        # scale run shows progress immediately rather than going silent
        # for several minutes.
        from data_utils.negatives import build_train_negatives

        train_neg_dir = args.out_dir / "train_negatives"
        train_neg_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_train_negative_epochs):
            df = build_train_negatives(splits, base_seed=args.seed, epoch=i)
            df.to_parquet(train_neg_dir / f"epoch_{i}.parquet", index=False)
            print(
                f"  train_negatives/epoch_{i}: {len(df):,} rows",
                file=sys.stderr,
                flush=True,
            )

    print(f"Wrote splits to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
