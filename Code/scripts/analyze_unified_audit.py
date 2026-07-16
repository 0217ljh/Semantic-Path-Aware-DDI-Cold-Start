"""Full audit of Code/data/ddi_unified/ against the benchmark design.

Read-only. For every leaf (task/dataset/regime/split, 5 folds) checks:
  [labels]   multiclass y_cls in [0,K) + closed-set y_cls_train (-1 only unseen-in-train)
             + label_vocab row count == meta n_labels; multilabel y_label_ids in [0,L).
  [drugs]    every split-pair drug is in drugs.parquet.
  [regime]   transductive/S0: eval drugs known (in train) — report frac;
             inductive/S1: every eval POSITIVE pair exactly-one-seen (seen=train-pos drugs);
             inductive/S2: eval drugs disjoint from train drugs; train/test pair-disjoint.
  [binary]   1:1 pos:neg; negatives ∉ global positive set; negatives STRUCTURE-MATCHED
             to the split geometry (S2 both-held, S1 one-seen, S0 any); no dup pairs;
             no canonical pair shared across train/val/test.
  [cv]       cold: each drug is the test group exactly once across the 5 folds.
Prints PASS/FAIL per check; exits nonzero if any FAIL.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "Code/data/ddi_unified"
import json

FAILS: list[str] = []


def chk(cond: bool, msg: str) -> None:
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(msg)
    print(f"    [{tag}] {msg}")


def report(msg: str) -> None:
    """Informational metric (not a pass/fail gate)."""
    print(f"    [INFO] {msg}")


def canon(df):
    a = df.drug_a_id.astype(str); b = df.drug_b_id.astype(str)
    return set((x, y) if x <= y else (y, x) for x, y in zip(a, b))


def drugset(df):
    return set(df.drug_a_id.astype(str)) | set(df.drug_b_id.astype(str))


def global_positives(dataset_dir_for_mc):
    """Union of canonical positive pairs across all folds/splits of a dataset (from the
    multiclass leaves = positives only)."""
    gp = set()
    for st in ("transductive/S0", "inductive/S1", "inductive/S2"):
        d = dataset_dir_for_mc / st
        if not d.is_dir():
            continue
        for fold in sorted(d.glob("fold*")):
            for sp in ("train", "val", "test"):
                f = fold / f"{sp}.parquet"
                if f.is_file():
                    gp |= canon(pd.read_parquet(f))
    return gp


def audit_binary_leaf(leaf, regime, split, global_pos, seen_by_fold):
    for fold in sorted(leaf.glob("fold*")):
        frames = {sp: pd.read_parquet(fold / f"{sp}.parquet") for sp in ("train", "val", "test")}
        name = f"{leaf.relative_to(B)}/{fold.name}"
        # 1:1 ratio
        for sp, df in frames.items():
            npos = int((df.y_bin == 1).sum()); nneg = int((df.y_bin == 0).sum())
            chk(npos == nneg, f"{name}/{sp}: 1:1 pos==neg ({npos} vs {nneg})")
            chk(set(df.y_bin.unique()) <= {0, 1}, f"{name}/{sp}: y_bin in {{0,1}}")
        # negatives ∉ global positives
        for sp, df in frames.items():
            neg = df[df.y_bin == 0]
            chk(len(canon(neg) & global_pos) == 0, f"{name}/{sp}: negatives ∉ global positives")
        # no canonical pair shared across train/val/test (pos+neg)
        cps = {sp: canon(df) for sp, df in frames.items()}
        chk(not (cps["train"] & cps["test"]), f"{name}: train/test pair-disjoint")
        chk(not (cps["train"] & cps["val"]), f"{name}: train/val pair-disjoint")
        chk(not (cps["val"] & cps["test"]), f"{name}: val/test pair-disjoint")
        # no duplicate canonical pairs within a split
        for sp, df in frames.items():
            cc = [tuple(sorted((str(a), str(b)))) for a, b in zip(df.drug_a_id, df.drug_b_id)]
            chk(len(cc) == len(set(cc)), f"{name}/{sp}: no duplicate canonical pairs")
        # structure-matched negatives + regime semantics
        if regime == "inductive":
            ds = pd.read_parquet(fold / "drug_split.parquet")
            test_d = set(ds[ds.role == "test"].drug_id.astype(str))
            val_d = set(ds[ds.role == "val"].drug_id.astype(str))
            seen = seen_by_fold[fold.name]              # actual train-positive drugs
            for sp, held in (("test", test_d), ("val", val_d)):
                df = frames[sp]
                a = df.drug_a_id.astype(str); b = df.drug_b_id.astype(str)
                if split == "S2":
                    ok = (a.isin(held) & b.isin(held)).all()
                    chk(bool(ok), f"{name}/{sp}: ALL pairs both-endpoints-held-out (S2, incl negs)")
                else:  # S1
                    one = (a.isin(held) & b.isin(seen)) | (b.isin(held) & a.isin(seen))
                    chk(bool(one.all()), f"{name}/{sp}: ALL pairs one-held + one-seen (S1, incl negs)")


def audit_mc_leaf(leaf, n_labels):
    for fold in sorted(leaf.glob("fold*")):
        name = f"{leaf.relative_to(B)}/{fold.name}"
        tr = pd.read_parquet(fold / "train.parquet")
        trc = set(tr.y_cls)
        chk((tr.y_cls_train >= 0).all(), f"{name}: train y_cls_train all >=0")
        chk(int(tr.y_cls.max()) < n_labels and int(tr.y_cls.min()) >= 0,
            f"{name}: train y_cls in [0,{n_labels})")
        for sp in ("val", "test"):
            df = pd.read_parquet(fold / f"{sp}.parquet")
            in_tr = df.y_cls.isin(trc)
            chk(bool(((df.y_cls_train >= 0) == in_tr).all()),
                f"{name}/{sp}: y_cls_train==-1 iff class unseen-in-train")
            chk(int(df.y_cls.max()) < n_labels and int(df.y_cls.min()) >= 0,
                f"{name}/{sp}: y_cls in [0,{n_labels})")


def regime_drug_checks(dataset_mc_dir):
    """Known/unknown drug partition from the multiclass POSITIVES (clean, no negatives)."""
    for split, st in (("S0", "transductive/S0"), ("S1", "inductive/S1"), ("S2", "inductive/S2")):
        leaf = dataset_mc_dir / st
        if not leaf.is_dir():
            continue
        test_once = Counter(); val_once = Counter()
        for fold in sorted(leaf.glob("fold*")):
            name = f"{leaf.relative_to(B)}/{fold.name}"
            tr = pd.read_parquet(fold / "train.parquet")
            te = pd.read_parquet(fold / "test.parquet")
            va = pd.read_parquet(fold / "val.parquet")
            trd = drugset(tr); seen = trd
            if split == "S0":
                # transductive metric (NOT a gate): drugbank_latest warm is strict 1.000;
                # deng/ryu warm = official MRCGNN folds, ~0.99 (some drugs eval-only in a
                # given fold, seen in other folds' train). Reported, not failed.
                frac = np.mean([d in trd for d in drugset(te)]) if len(te) else 1.0
                report(f"{name}: S0 eval drugs known/in-train frac={frac:.3f}")
            elif split == "S1":
                for spn, df in (("test", te), ("val", va)):
                    a = df.drug_a_id.astype(str).isin(seen); b = df.drug_b_id.astype(str).isin(seen)
                    chk(bool((a ^ b).all()), f"{name}/{spn}: S1 positives exactly-one-seen")
            else:  # S2
                chk(not (drugset(te) & trd), f"{name}/test: S2 eval drugs disjoint from train")
                chk(not (drugset(va) & trd), f"{name}/val: S2 eval drugs disjoint from train")
                chk(not (canon(tr) & canon(te)), f"{name}: S2 train/test pair-disjoint")
                chk(not (canon(tr) & canon(va)), f"{name}: S2 train/val pair-disjoint")
                chk(not (canon(va) & canon(te)), f"{name}: S2 val/test pair-disjoint")
                ds = pd.read_parquet(fold / "drug_split.parquet")
                for d in ds[ds.role == "test"].drug_id.astype(str):
                    test_once[d] += 1
                for d in ds[ds.role == "val"].drug_id.astype(str):
                    val_once[d] += 1
        if split == "S2" and test_once:
            chk(set(test_once.values()) == {1},
                f"{dataset_mc_dir.name} S2 CV coverage: each drug test-once "
                f"(dist={dict(Counter(test_once.values()))})")
            chk(set(val_once.values()) == {1},
                f"{dataset_mc_dir.name} S2 CV coverage: each drug val-once "
                f"(dist={dict(Counter(val_once.values()))})")


def main():
    print("================ UNIFIED BENCHMARK AUDIT ================")
    for task in ("binary_cls", "multi_cls", "multi_label_cls"):
        for dsdir in sorted((B / task).glob("*")):
            grp = dsdir.name
            mc_dir = B / "multi_cls" / grp
            ml_dir = B / "multi_label_cls" / grp
            print(f"\n#### {task}/{grp}")
            for st in ("transductive/S0", "inductive/S1", "inductive/S2"):
                leaf = dsdir / st
                if not leaf.is_dir():
                    chk(False, f"{task}/{grp}/{st}: MISSING leaf")
                    continue
                meta = json.load(open(leaf / "meta.json"))
                nfold = len(list(leaf.glob("fold*")))
                chk(nfold == 5, f"{task}/{grp}/{st}: 5 folds (got {nfold})")
                regime, split = meta["regime"], meta["split_code"]
                # drugs.parquet covers every split-pair drug
                duni = set(pd.read_parquet(leaf / "drugs.parquet").drug_id.astype(str))
                for fold in sorted(leaf.glob("fold*")):
                    for sp in ("train", "val", "test"):
                        miss = drugset(pd.read_parquet(fold / f"{sp}.parquet")) - duni
                        chk(not miss, f"{leaf.relative_to(B)}/{fold.name}/{sp}: all drugs in drugs.parquet")
                # pair_format metadata consistency: canonical => every stored pair has a<=b
                if meta["pair_format"] == "canonical":
                    bad = 0
                    for fold in sorted(leaf.glob("fold*")):
                        for sp in ("train", "val", "test"):
                            df = pd.read_parquet(fold / f"{sp}.parquet")
                            bad += int((df.drug_a_id.astype(str) > df.drug_b_id.astype(str)).sum())
                    chk(bad == 0, f"{task}/{grp}/{st}: pair_format=canonical => all pairs a<=b ({bad} violations)")
                if task == "binary_cls":
                    gp = global_positives(mc_dir if mc_dir.is_dir() else dsdir)
                    # seen (train-pos drugs) per fold from the mc leaf (positives only)
                    seen_by_fold = {}
                    src = (mc_dir / st)
                    for fold in sorted(leaf.glob("fold*")):
                        mtr = pd.read_parquet(src / fold.name / "train.parquet")
                        seen_by_fold[fold.name] = drugset(mtr)
                    audit_binary_leaf(leaf, regime, split, gp, seen_by_fold)
                elif task == "multi_cls":
                    audit_mc_leaf(leaf, int(meta["labels"]["n_labels"]))
                    lv = pd.read_parquet(leaf / "label_vocab.parquet")
                    chk(len(lv) == int(meta["labels"]["n_labels"]),
                        f"{task}/{grp}/{st}: label_vocab rows == n_labels ({len(lv)})")
                else:  # multilabel
                    L = int(meta["labels"]["n_labels"])
                    lv = pd.read_parquet(leaf / "label_vocab.parquet")
                    chk(len(lv) == L, f"{task}/{grp}/{st}: label_vocab rows == {L}")
                    for fold in sorted(leaf.glob("fold*")):
                        for sp in ("train", "val", "test"):
                            df = pd.read_parquet(fold / f"{sp}.parquet")
                            ll = [list(x) for x in df.y_label_ids]
                            mx = max((max(x) if len(x) else -1) for x in ll)
                            mn = min((min(x) if len(x) else 0) for x in ll)
                            chk(mx < L and mn >= 0, f"{leaf.relative_to(B)}/{fold.name}/{sp}: y_label_ids in [0,{L})")
                            chk(all(len(x) == len(set(x)) for x in ll),
                                f"{leaf.relative_to(B)}/{fold.name}/{sp}: y_label_ids no dup within row")
                            # is_positive agrees with non-empty label list
                            pos_nonempty = df[df.is_positive == 1].y_label_ids.apply(len)
                            chk(bool((pos_nonempty > 0).all()),
                                f"{leaf.relative_to(B)}/{fold.name}/{sp}: positives have >=1 label")
            # regime drug partition (from mc positives, where available)
            if task == "multi_cls":
                regime_drug_checks(mc_dir)
    print("\n================ SUMMARY ================")
    print(f"TOTAL FAILS: {len(FAILS)}")
    for f in FAILS[:40]:
        print("  -", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
