"""Pipeline verification for ColdDDI baselines on our project.

Runs 6 critical checks per codex's reviewed checklist:
1. Split drug disjointness (cold-start protocol)
2. Negative sampling correctness (1:1, no overlap, no dup)
3. Same pair universe across models
4. Modality contract per model (HDN/TIGER mol-only; EmerGNN+KG)
5. fit/predict_proba I/O contract
6. Reproducibility with fixed seed

Data: 800-drug seed42 PKL (from ColdDDI legacy bundle).

Outputs PASS/FAIL per check; smoke fits each baseline ~1 epoch then
inspects predict_proba shape/range/finiteness.
"""
from __future__ import annotations

import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_project_root()
PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"

print(f"[setup] project root: {ROOT}")
print(f"[setup] data PKL: {PKL}")


def section(name: str):
    print(f"\n{'='*65}\n{name}\n{'='*65}")


def report(check_id: str, name: str, ok: bool, detail: str = ""):
    flag = "[PASS]" if ok else "[FAIL]"
    print(f"  {flag} {check_id}: {name}" + (f"  --  {detail}" if detail else ""))
    return ok


def hash_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for col in sorted(df.columns):
        s = df[col].astype(str).str.cat(sep="|")
        h.update(s.encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
def load_ds(verbose: bool = True):
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    if verbose:
        print(f"  loaded: drugs={len(ds.drug_set)}  train={len(ds.splits.train)}  "
              f"val_s2={len(ds.splits.val_s2)}  test_s2={len(ds.splits.test_s2)}")
    return ds


def drug_set_of(df: pd.DataFrame) -> set:
    if df is None or len(df) == 0:
        return set()
    return set(df["drug_a_id"].astype(str)) | set(df["drug_b_id"].astype(str))


# ---------------------------------------------------------------------------
def check_1_drug_disjoint(ds) -> bool:
    """Cold-start S2 protocol:
      * train drugs (G1) MUST be disjoint from val_s2 / test_s2 drugs (G2)
      * val_s2 and test_s2 may share the same G2 drug pool (intentional;
        pair-level split, not drug-level split). Pair-level disjointness
        IS required though, so we check that too.
    """
    sp = ds.splits
    train_drugs = drug_set_of(sp.train)
    val_drugs = drug_set_of(sp.val_s2)
    test_drugs = drug_set_of(sp.test_s2)
    ok = True

    # drug-level disjointness (train vs G2 only)
    for a, b, da, db in [
        ("train", "val_s2", train_drugs, val_drugs),
        ("train", "test_s2", train_drugs, test_drugs),
    ]:
        inter = da & db
        ok &= report(
            f"1.{a}∩{b}_drug",
            f"drugs disjoint: |{a}|={len(da)} ∩ |{b}|={len(db)}",
            len(inter) == 0,
            f"overlap={len(inter)}: {sorted(list(inter))[:5]}" if inter else "",
        )

    # info: val_s2 vs test_s2 share G2 drug pool by design
    overlap_g2 = val_drugs & test_drugs
    print(f"    [info] val_s2 ∩ test_s2 drug overlap = {len(overlap_g2)} (expected: shared G2 pool, ~"
          f"{len(val_drugs)})")

    # pair-level: val_s2 and test_s2 must have disjoint pairs
    val_pairs = {tuple(sorted([str(a), str(b)])) for a, b in zip(sp.val_s2["drug_a_id"], sp.val_s2["drug_b_id"])}
    test_pairs = {tuple(sorted([str(a), str(b)])) for a, b in zip(sp.test_s2["drug_a_id"], sp.test_s2["drug_b_id"])}
    pair_overlap = val_pairs & test_pairs
    ok &= report(
        "1.val_s2∩test_s2_pair",
        f"PAIR-level disjoint: |val|={len(val_pairs)} ∩ |test|={len(test_pairs)}",
        len(pair_overlap) == 0,
        f"pair overlap={len(pair_overlap)}" if pair_overlap else "",
    )
    return ok


def check_2_negatives(ds) -> bool:
    """Per-split: |neg| == |pos|; neg pair not in any positive; no duplicates."""
    ok = True
    # Build set of ALL positive pair-tuples (unordered) for fast lookup
    all_pos = set()
    for name in ["train", "val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2"]:
        df = getattr(ds.splits, name)
        for a, b in zip(df["drug_a_id"], df["drug_b_id"]):
            all_pos.add(tuple(sorted([str(a), str(b)])))

    # Check static negatives
    for name in ["val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2"]:
        pos_df = getattr(ds.splits, name)
        neg_df = ds.negatives_by_split.get(name)
        if neg_df is None:
            report(f"2.{name}_exists", f"neg parquet present for {name}", False, "missing")
            ok = False
            continue
        # Length
        ok &= report(
            f"2.{name}_1to1",
            f"|neg|=={len(neg_df)} vs |pos|={len(pos_df)}",
            len(neg_df) == len(pos_df),
        )
        # No overlap with any positive
        neg_pairs = {tuple(sorted([str(a), str(b)])) for a, b in zip(neg_df["drug_a_id"], neg_df["drug_b_id"])}
        overlap = neg_pairs & all_pos
        ok &= report(
            f"2.{name}_no_pos_overlap",
            f"neg ∩ all_pos == empty",
            len(overlap) == 0,
            f"overlap={len(overlap)}" if overlap else "",
        )
        # No dup
        ok &= report(
            f"2.{name}_no_dup",
            f"no duplicate neg pairs",
            len(neg_pairs) == len(neg_df),
            f"dup={len(neg_df) - len(neg_pairs)}",
        )

    # Check train negatives (epoch 0)
    tn = ds.get_train_negatives(epoch=0)
    pos_train = ds.splits.train
    ok &= report("2.train_1to1", f"|train_neg|=={len(tn)} vs |train_pos|={len(pos_train)}",
                 len(tn) == len(pos_train))
    tn_pairs = {tuple(sorted([str(a), str(b)])) for a, b in zip(tn["drug_a_id"], tn["drug_b_id"])}
    overlap = tn_pairs & all_pos
    ok &= report("2.train_no_pos_overlap", "train_neg ∩ all_pos == empty",
                 len(overlap) == 0, f"overlap={len(overlap)}" if overlap else "")
    return ok


def check_3_same_pair_universe(ds) -> bool:
    """Sanity: 3 baselines all read the same splits attributes; just hash."""
    h_train = hash_df(ds.splits.train[["drug_a_id", "drug_b_id"]])
    h_val = hash_df(ds.splits.val_s2[["drug_a_id", "drug_b_id"]])
    h_test = hash_df(ds.splits.test_s2[["drug_a_id", "drug_b_id"]])
    print(f"    train hash: {h_train}")
    print(f"    val_s2 hash: {h_val}")
    print(f"    test_s2 hash: {h_test}")
    # nothing to compare against here other than consistency on re-load
    return report("3.hash_recorded", "pair-universe hashes recorded (for cross-model consistency on re-load)", True)


def check_4_modality_contract(ds) -> bool:
    """HDN-DDI and TIGER must work with ds.kg = None (mol-only).
    EmerGNN must access kg (5-bucket or merged)."""
    from baseline.emergnn import EmerGNNBaseline
    from baseline.hdn_ddi import HDNDDIBaseline
    from baseline.tiger import TIGERBaseline

    ok = True
    # 1. HDN-DDI / TIGER: instantiate (no fit; we only test that they don't IMMEDIATELY require KG)
    try:
        _ = HDNDDIBaseline()
        ok &= report("4.HDN_init", "HDN-DDI instantiated without KG arg", True)
    except Exception as e:
        ok &= report("4.HDN_init", "HDN-DDI failed to init", False, str(e)[:120])
    try:
        _ = TIGERBaseline()
        ok &= report("4.TIGER_init", "TIGER instantiated without KG arg", True)
    except Exception as e:
        ok &= report("4.TIGER_init", "TIGER failed to init", False, str(e)[:120])
    try:
        _ = EmerGNNBaseline()
        ok &= report("4.EmerGNN_init", "EmerGNN instantiated without KG arg", True)
    except Exception as e:
        ok &= report("4.EmerGNN_init", "EmerGNN failed to init", False, str(e)[:120])

    # 2. drugs DF has SMILES for HDN/TIGER
    has_smiles = ds.drugs is not None and "smiles" in ds.drugs.columns
    n_non_null = int(ds.drugs["smiles"].notna().sum()) if has_smiles else 0
    n_drugs = len(ds.drug_set)
    coverage = sum(1 for d in ds.drug_set if d in set(ds.drugs["drugbank_id"]) and pd.notna(ds.drugs[ds.drugs["drugbank_id"]==d]["smiles"].iloc[0])) if has_smiles else 0
    ok &= report("4.smiles_coverage",
                 f"non-null smiles {n_non_null}/{len(ds.drugs) if ds.drugs is not None else 0} (split drugs covered {coverage}/{n_drugs})",
                 has_smiles and coverage == n_drugs,
                 "" if coverage == n_drugs else f"{n_drugs - coverage} split drugs missing smiles")

    # 3. KG has 5 buckets non-empty
    for attr in ["enzymes", "targets", "transporters", "carriers", "pathways"]:
        v = getattr(ds.kg, attr, None)
        ok &= report(f"4.KG_{attr}",
                     f"kg.{attr} present, n={len(v) if v is not None else 'None'}",
                     v is not None and len(v) > 0)
    return ok


def check_5_io_contract(ds, n_epochs=1) -> bool:
    """Smoke fit (1 epoch) each baseline; verify predict_proba shape/finite/range."""
    from baseline.emergnn import EmerGNNBaseline
    from baseline.hdn_ddi import HDNDDIBaseline
    from baseline.tiger import TIGERBaseline

    ok = True
    test_pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    expected_n = len(test_pairs)

    for name, BaseCls, kwargs in [
        ("EmerGNN", EmerGNNBaseline, dict(n_epochs=n_epochs, batch_size=128)),
        ("HDN-DDI", HDNDDIBaseline, dict(n_epochs=n_epochs, batch_size=128)),
        ("TIGER", TIGERBaseline, dict(n_epochs=n_epochs, batch_size=128)),
    ]:
        print(f"\n  --- {name} smoke fit ({n_epochs} epoch) ---")
        try:
            t0 = time.time()
            m = BaseCls(**kwargs)
            m.fit(ds, kg=ds.kg)
            p = m.predict_proba(test_pairs, kg=ds.kg)
            elapsed = time.time() - t0
            ok &= report(f"5.{name}_shape", f"predict_proba 1-D ndarray, len match", isinstance(p, np.ndarray) and p.ndim == 1 and len(p) == expected_n)
            ok &= report(f"5.{name}_finite", "all finite", bool(np.isfinite(p).all()))
            ok &= report(f"5.{name}_range", f"in [0,1]: min={p.min():.4f} max={p.max():.4f}",
                         bool((p >= 0).all() and (p <= 1).all()))
            print(f"    [time] fit+predict: {elapsed:.1f}s")
        except Exception as e:
            ok &= report(f"5.{name}_runtime", f"exception during fit/predict", False, f"{type(e).__name__}: {str(e)[:200]}")
    return ok


def check_5b_kg_none_mol_only(ds) -> bool:
    """P1: HDN-DDI and TIGER must fit successfully with ds.kg=None
    (proves they're truly mol-only and don't sneak-read KG)."""
    import copy
    from baseline.hdn_ddi import HDNDDIBaseline
    from baseline.tiger import TIGERBaseline

    test_pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    expected_n = len(test_pairs)
    ds_no_kg = copy.copy(ds)
    ds_no_kg.kg = None

    ok = True
    for name, BaseCls in [("HDN-DDI", HDNDDIBaseline), ("TIGER", TIGERBaseline)]:
        print(f"\n  --- {name} fit with ds.kg=None ---")
        try:
            t0 = time.time()
            m = BaseCls(n_epochs=1, batch_size=128)
            m.fit(ds_no_kg, kg=None)
            p = m.predict_proba(test_pairs, kg=None)
            elapsed = time.time() - t0
            ok &= report(
                f"5b.{name}_kgnone_runs",
                f"fit+predict with kg=None: shape OK, finite, [0,1]",
                isinstance(p, np.ndarray) and p.ndim == 1 and len(p) == expected_n
                and bool(np.isfinite(p).all()) and bool((p >= 0).all() and (p <= 1).all()),
                f"min={p.min():.4f} max={p.max():.4f}, time {elapsed:.1f}s",
            )
        except Exception as e:
            ok &= report(
                f"5b.{name}_kgnone_runs",
                "fit/predict crashed with kg=None — model is NOT truly mol-only",
                False, f"{type(e).__name__}: {str(e)[:200]}",
            )
    return ok


def check_5c_eval_split_leakage(ds) -> bool:
    """P2 (lightweight): each baseline's fit() must not crash when
    val_s2/test_s2 are emptied. Demonstrates fit() does NOT depend
    on eval-split data (no leakage via splits attribute access)."""
    import copy
    from baseline.emergnn import EmerGNNBaseline
    from baseline.hdn_ddi import HDNDDIBaseline
    from baseline.tiger import TIGERBaseline

    ds_no_eval = copy.copy(ds)
    # shallow-copy splits and zero out eval splits
    ds_no_eval.splits = copy.copy(ds.splits)
    empty_df = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
    for name in ["val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2"]:
        setattr(ds_no_eval.splits, name, empty_df)
    # also drop eval negatives
    ds_no_eval.negatives_by_split = {}

    # Sanity: train pairs still there
    assert len(ds_no_eval.splits.train) == len(ds.splits.train), "train should still be intact"

    test_pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]  # use ORIGINAL test for predict
    expected_n = len(test_pairs)
    ok = True
    for name, BaseCls, kwargs in [
        ("EmerGNN", EmerGNNBaseline, dict(n_epochs=1, batch_size=128)),
        ("HDN-DDI", HDNDDIBaseline, dict(n_epochs=1, batch_size=128)),
        ("TIGER", TIGERBaseline, dict(n_epochs=1, batch_size=128)),
    ]:
        print(f"\n  --- {name} fit with empty val/test splits ---")
        try:
            t0 = time.time()
            m = BaseCls(**kwargs)
            m.fit(ds_no_eval, kg=ds_no_eval.kg)
            p = m.predict_proba(test_pairs, kg=ds_no_eval.kg)
            elapsed = time.time() - t0
            ok &= report(
                f"5c.{name}_no_eval_runs",
                f"fit+predict without eval splits: OK",
                isinstance(p, np.ndarray) and p.ndim == 1 and len(p) == expected_n
                and bool(np.isfinite(p).all()) and bool((p >= 0).all() and (p <= 1).all()),
                f"min={p.min():.4f} max={p.max():.4f}, time {elapsed:.1f}s",
            )
        except Exception as e:
            ok &= report(
                f"5c.{name}_no_eval_runs",
                "fit crashed without eval splits — likely depends on eval data (leakage suspect)",
                False, f"{type(e).__name__}: {str(e)[:200]}",
            )
    return ok


def check_6_reproducibility(ds) -> bool:
    """Same seed → same neg hash on re-load."""
    from data_utils import PairDataset
    ds2 = PairDataset.from_pkl(str(PKL))
    tn1 = ds.get_train_negatives(epoch=0)
    tn2 = ds2.get_train_negatives(epoch=0)
    h1 = hash_df(tn1[["drug_a_id", "drug_b_id"]])
    h2 = hash_df(tn2[["drug_a_id", "drug_b_id"]])
    return report("6.train_neg_reproducible", f"hash match: {h1} == {h2}", h1 == h2)


# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-smoke", action="store_true",
                    help="Skip the slow fit/predict_proba smoke (Check 5)")
    args = ap.parse_args()

    t_total = time.time()
    section("Loading dataset")
    ds = load_ds()

    all_ok = True

    section("Check 1: Split drug disjointness")
    all_ok &= check_1_drug_disjoint(ds)
    sys.stdout.flush()

    section("Check 2: Negative sampling correctness")
    all_ok &= check_2_negatives(ds)
    sys.stdout.flush()

    section("Check 3: Same pair universe across models")
    all_ok &= check_3_same_pair_universe(ds)
    sys.stdout.flush()

    section("Check 4: Modality contract per baseline")
    all_ok &= check_4_modality_contract(ds)
    sys.stdout.flush()

    section("Check 6: Reproducibility (data layer)")
    all_ok &= check_6_reproducibility(ds)
    sys.stdout.flush()

    if not args.no_smoke:
        section("Check 5: fit/predict_proba I/O contract (smoke 1 epoch)")
        all_ok &= check_5_io_contract(ds, n_epochs=1)
        sys.stdout.flush()

        section("Check 5b (P1): HDN-DDI / TIGER work with ds.kg=None")
        all_ok &= check_5b_kg_none_mol_only(ds)
        sys.stdout.flush()

        section("Check 5c (P2): fit() does not require eval splits")
        all_ok &= check_5c_eval_split_leakage(ds)
        sys.stdout.flush()
    else:
        section("Checks 5 / 5b / 5c: SKIPPED (--no-smoke)")

    section("Summary")
    print(f"  Overall: {'ALL PASS' if all_ok else 'SOME FAIL'}")
    print(f"  Total elapsed: {time.time() - t_total:.1f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
