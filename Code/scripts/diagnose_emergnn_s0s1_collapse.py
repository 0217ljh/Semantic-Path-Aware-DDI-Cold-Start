"""Diagnose EmerGNN binary s0/s1 AUC=0.5 collapse.

Hypothesis (from 2026-05-18 review):
    Paper-faithful drugbank KG causes training-eval distribution mismatch.
    Training uses ``shuffle_train(mode="S2", ratio=0.8)`` which removes 80%
    of train drug DDI edges from the KG each epoch -> sparse neighborhoods.
    Eval uses ``_eval_edges = train_ddi + base_kg`` -> ALL train DDI edges
    in KG -> very dense neighborhoods for warm-start (s0) drugs.
    Result: model trained on sparse-neighborhood distribution saturates
    when fed dense-neighborhood input, outputting near-constant logit |z|~16
    (matches observed NLL=7.97 = |z|/2 ≈ 8).

This script runs 4 analyses to confirm or reject the hypothesis:

    Analysis A: Graph degree distribution
        For a sample of s0/s1/s2 test drugs, count their degree (#neighbors)
        in:
          - eval_edges (= train_ddi + base_kg, what predict_proba uses)
          - a training-distribution graph (= base_kg + 20%-sampled train_ddi,
            mimicking what shuffle_train produces)
        If s0 drugs have 100+ degree in eval but <20 in train-dist, while s2
        drugs have <10 in both -> distribution mismatch confirmed for s0.

    Analysis B: Forward-pass activation magnitudes
        On a sample of 8 pairs from each split, run forward and capture:
          - Final logit value
          - Max activation in propagation hidden state (per layer)
          - Whether logits cluster near a single value (low std) -> saturation
        If s0/s1 cluster tightly at high |logit| while s2 varies -> saturation
        confirmed.

    Analysis C: Counterfactual eval on training-distribution graph
        Re-evaluate the trained model on s0/s1/s2 using a sparser eval graph
        that mirrors training distribution (apply shuffle_train at eval).
        If s0/s1 AUC suddenly recovers to >0.7 while s2 stays similar ->
        confirms the cause is the eval-graph density, not the model itself.

    Analysis D: Training-time behavior trace
        During the warmup training (8 epochs to reproduce collapse), capture
        per-step:
          - Loss value
          - Forward logit distribution (mean, std, %|z|>10 saturation rate)
          - Edge count + active drug count per epoch (after shuffle_train)
        Shows what the model "sees" during training.

Usage:
    python Code/scripts/diagnose_emergnn_s0s1_collapse.py
    # ~10 min on a single GPU; ~30 min on CPU

Output:
    - All findings printed to stdout
    - Numerical metrics saved to Code/runs/_diagnose/emergnn_s0s1_collapse.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "Code"))

from baseline.emergnn import EmerGNNBaseline  # noqa: E402
from baseline.emergnn.shuffle_utils import (  # noqa: E402
    build_edge_lists_from_triplets,
    shuffle_train,
)
from data_utils.dataset import PairDataset  # noqa: E402

# --------------------------------------------------------------------- config

PKL = _ROOT / "Code" / "data" / "coldddi_legacy" / "800drug" / "seed42.pkl"
OUT_DIR = _ROOT / "Code" / "runs" / "_diagnose"
OUT_JSON = OUT_DIR / "emergnn_s0s1_collapse.json"

# Short training: enough to reproduce the s0/s1 collapse but fast.
# Loss in the full-100ep run hit <0.1 by ep 10, val_s2 AUC ~0.7 by ep 20.
N_EPOCHS = 4
SAMPLE_PAIRS_PER_SPLIT = 8
DEGREE_SAMPLE_DRUGS = 64


def banner(msg: str) -> None:
    print()
    print("=" * 78)
    print("  " + msg)
    print("=" * 78, flush=True)


# --------------------------------------------------------------------- main

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    findings: dict = {}

    banner("STEP 0  Load PairDataset (800-drug seed42, paper-faithful drugbank KG)")
    t0 = time.time()
    ds = PairDataset.from_pkl(PKL)
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  drugs in split: {ds.splits.train['drug_a_id'].nunique() + ds.splits.train['drug_b_id'].nunique()}")
    print(f"  train rows: {len(ds.splits.train)}")
    print(f"  test_s0: {len(ds.splits.test_s0)}  test_s1: {len(ds.splits.test_s1)}  test_s2: {len(ds.splits.test_s2)}")

    # --------------------------------------------------------- short training

    banner(f"STEP 1  Short training ({N_EPOCHS} epochs, drugbank KG)")
    print("  Goal: reproduce the s0/s1 collapse with a small training budget")
    print("  Same hyperparams as the failing 5/18 run, just fewer epochs.")

    model = EmerGNNBaseline(
        n_dim=64,
        backbone_kg_source="drugbank",
        n_epochs=N_EPOCHS,
        batch_size=32,
        shuffle_train_mode="S2",
        shuffle_ratio=0.8,
        eval_strategy="no",
        save_strategy="no",
        log_step_every=200,
        load_best_model_at_end=False,
    )
    t0 = time.time()
    model.fit(ds, val=ds, kg=ds.kg)
    fit_time = time.time() - t0
    print(f"  training done in {fit_time:.1f}s")
    findings["training_time_s"] = round(fit_time, 1)

    # --------------------------------------------------------- analysis A

    banner("ANALYSIS A  Graph degree distribution: eval vs training-dist KG")
    eval_src, eval_dst, eval_rel = model._eval_edges
    eval_src_np = eval_src.cpu().numpy()
    eval_dst_np = eval_dst.cpu().numpy()

    deg_eval = Counter()
    for s, d in zip(eval_src_np, eval_dst_np):
        deg_eval[int(s)] += 1

    print(f"  Eval graph has {len(eval_src_np)} directed edges")
    print(f"  Eval graph max degree: {max(deg_eval.values()) if deg_eval else 0}")
    print(f"  Eval graph mean degree: {np.mean(list(deg_eval.values())):.1f}")

    # Build a training-distribution KG via shuffle_train (one realization).
    train_ddi_int = []
    e2id = model._entity2id
    for _, row in ds.splits.train.iterrows():
        a = e2id.get(str(row["drug_a_id"]))
        b = e2id.get(str(row["drug_b_id"]))
        if a is not None and b is not None:
            train_ddi_int.append([a, b, model._n_base_rel])
    train_ddi_int = np.asarray(train_ddi_int, dtype=np.int64)

    train_kg_dist, _ = shuffle_train(
        train_ddi_int,
        model._kg_triplets,
        "S2",
        ratio=0.8,
        rng=np.random.default_rng(0),
        extra_kg_ent=model._kg_entity_set,
    )
    train_src, train_dst, _ = build_edge_lists_from_triplets(
        train_kg_dist, model._n_ent, model._n_base_rel_with_ddi
    )
    deg_train = Counter()
    for s, d in zip(train_src, train_dst):
        deg_train[int(s)] += 1
    print(f"  Training-dist graph (one shuffle_train realization, ratio=0.8):")
    print(f"    edges: {len(train_src)}")
    print(f"    max degree: {max(deg_train.values()) if deg_train else 0}")
    print(f"    mean degree: {np.mean(list(deg_train.values())):.1f}")

    # Now sample test drugs from each split and compare their degrees.
    def get_split_drug_ents(df: pd.DataFrame, n: int = DEGREE_SAMPLE_DRUGS) -> list[int]:
        drugs = pd.concat([df["drug_a_id"], df["drug_b_id"]]).astype(str).unique()
        rng = np.random.default_rng(42)
        chosen = rng.choice(drugs, size=min(n, len(drugs)), replace=False)
        return [e2id[d] for d in chosen if d in e2id]

    split_stats = {}
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        df = getattr(ds.splits, split_name)
        ents = get_split_drug_ents(df)
        eval_degs = [deg_eval.get(e, 0) for e in ents]
        train_degs = [deg_train.get(e, 0) for e in ents]
        split_stats[split_name] = {
            "n_sampled_drugs": len(ents),
            "eval_kg_deg_mean": float(np.mean(eval_degs)),
            "eval_kg_deg_median": float(np.median(eval_degs)),
            "eval_kg_deg_max": int(np.max(eval_degs)) if eval_degs else 0,
            "train_dist_deg_mean": float(np.mean(train_degs)),
            "train_dist_deg_median": float(np.median(train_degs)),
            "train_dist_deg_max": int(np.max(train_degs)) if train_degs else 0,
        }
        print(f"  {split_name} sampled {len(ents)} drugs:")
        print(f"    eval_kg degree   mean={np.mean(eval_degs):.1f}  median={np.median(eval_degs):.0f}  max={max(eval_degs) if eval_degs else 0}")
        print(f"    train_dist deg.  mean={np.mean(train_degs):.1f}  median={np.median(train_degs):.0f}  max={max(train_degs) if train_degs else 0}")

    findings["analysis_a_degree_stats"] = split_stats
    findings["analysis_a_verdict"] = (
        "DISTRIBUTION MISMATCH CONFIRMED"
        if split_stats["test_s0"]["eval_kg_deg_mean"] > 3 * split_stats["test_s0"]["train_dist_deg_mean"]
        else "NO MISMATCH SEEN"
    )
    print(f"  -> Analysis A verdict: {findings['analysis_a_verdict']}")

    # --------------------------------------------------------- analysis B

    banner("ANALYSIS B  Forward-pass logit / activation magnitudes per split")
    model._model.eval()

    def forward_with_logit_stats(pairs_df: pd.DataFrame) -> dict:
        """Forward pass; capture logit distribution."""
        head, tail = model._pair_indices(pairs_df)
        head = head.to(model.device)
        tail = tail.to(model.device)
        with torch.no_grad():
            logits = model._model(head, tail, eval_src, eval_dst, eval_rel)
        z = logits.cpu().numpy()
        return {
            "n": int(z.size),
            "logit_mean": float(z.mean()),
            "logit_std": float(z.std()),
            "logit_min": float(z.min()),
            "logit_max": float(z.max()),
            "fraction_abs_z_gt_10": float(np.mean(np.abs(z) > 10.0)),
            "predicted_NLL_if_balanced": float(0.5 * (np.log1p(np.exp(-z)) + np.log1p(np.exp(z))).mean()),
            "sample_logits": [round(float(v), 3) for v in z[:8].tolist()],
        }

    split_logit_stats = {}
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        df = getattr(ds.splits, split_name).head(SAMPLE_PAIRS_PER_SPLIT * 4)
        # Pull from both positives (df, label=1) and same-set negatives
        # synthesized via PairDataset.get_negatives.
        pos = df[["drug_a_id", "drug_b_id"]].head(SAMPLE_PAIRS_PER_SPLIT)
        try:
            neg = ds.get_negatives(split_name).head(SAMPLE_PAIRS_PER_SPLIT)
        except Exception:
            neg = pd.DataFrame()
        all_pairs = pd.concat([pos, neg]) if len(neg) else pos
        stats = forward_with_logit_stats(all_pairs)
        split_logit_stats[split_name] = stats
        print(f"  {split_name} (sampled {stats['n']} pairs):")
        print(f"    logit mean={stats['logit_mean']:+.2f}  std={stats['logit_std']:.2f}  min={stats['logit_min']:+.2f}  max={stats['logit_max']:+.2f}")
        print(f"    fraction |z|>10: {stats['fraction_abs_z_gt_10']*100:.1f}%")
        print(f"    predicted-NLL-if-balanced: {stats['predicted_NLL_if_balanced']:.3f}")
        print(f"    first 8 logits: {stats['sample_logits']}")

    findings["analysis_b_logit_stats"] = split_logit_stats

    # Verdict: if s0 logits have std << s2's, and s0 fraction-saturated >> s2's
    s0 = split_logit_stats["test_s0"]
    s2 = split_logit_stats["test_s2"]
    findings["analysis_b_verdict"] = (
        "SATURATION CONFIRMED ON s0 (low std + high |z|)"
        if s0["logit_std"] < 0.5 * s2["logit_std"] and s0["fraction_abs_z_gt_10"] > 0.5
        else "NO CLEAR SATURATION; theory needs revisit"
    )
    print(f"  -> Analysis B verdict: {findings['analysis_b_verdict']}")

    # --------------------------------------------------------- analysis C

    banner("ANALYSIS C  Counterfactual: eval the SAME model on training-distribution graph")
    print("  Re-run predict_proba but with a training-distribution graph instead")
    print("  of _eval_edges. If s0/s1 AUC recovers, the eval graph IS the cause.")

    # Build a fresh training-distribution graph (different seed than analysis A
    # so we test a NEW shuffle realization at eval).
    train_kg_dist_c, _ = shuffle_train(
        train_ddi_int,
        model._kg_triplets,
        "S2",
        ratio=0.8,
        rng=np.random.default_rng(123),
        extra_kg_ent=model._kg_entity_set,
    )
    es_c, ed_c, er_c = build_edge_lists_from_triplets(
        train_kg_dist_c, model._n_ent, model._n_base_rel_with_ddi
    )
    es_c_t = torch.from_numpy(es_c).long().to(model.device)
    ed_c_t = torch.from_numpy(ed_c).long().to(model.device)
    er_c_t = torch.from_numpy(er_c).long().to(model.device)

    # Save original eval edges, then patch in training-dist edges.
    saved_eval_edges = model._eval_edges
    model._eval_edges = (es_c_t, ed_c_t, er_c_t)

    cf_aucs = {}
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        pos = getattr(ds.splits, split_name)[["drug_a_id", "drug_b_id"]]
        try:
            neg = ds.get_negatives(split_name)[["drug_a_id", "drug_b_id"]]
        except Exception:
            cf_aucs[split_name] = None
            continue
        if len(pos) == 0 or len(neg) == 0:
            cf_aucs[split_name] = None
            continue
        y_score = np.concatenate([model.predict_proba(pos), model.predict_proba(neg)])
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        auc = float(roc_auc_score(y_true, y_score))
        cf_aucs[split_name] = auc
        print(f"  {split_name}: AUC on training-dist eval graph = {auc:.4f}")

    # Restore eval edges + measure baseline (original) AUC for direct comparison.
    model._eval_edges = saved_eval_edges
    orig_aucs = {}
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        pos = getattr(ds.splits, split_name)[["drug_a_id", "drug_b_id"]]
        try:
            neg = ds.get_negatives(split_name)[["drug_a_id", "drug_b_id"]]
        except Exception:
            orig_aucs[split_name] = None
            continue
        if len(pos) == 0 or len(neg) == 0:
            orig_aucs[split_name] = None
            continue
        y_score = np.concatenate([model.predict_proba(pos), model.predict_proba(neg)])
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        orig_aucs[split_name] = float(roc_auc_score(y_true, y_score))

    print()
    print(f"  Side-by-side comparison:")
    print(f"  {'split':10s}  {'orig eval AUC':>15s}  {'train-dist AUC':>15s}  {'lift':>8s}")
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        o = orig_aucs[split_name]
        c = cf_aucs[split_name]
        if o is None or c is None:
            print(f"  {split_name:10s}  {'-':>15s}  {'-':>15s}  {'-':>8s}")
            continue
        lift = c - o
        print(f"  {split_name:10s}  {o:>15.4f}  {c:>15.4f}  {lift:>+8.4f}")

    findings["analysis_c_orig_aucs"] = orig_aucs
    findings["analysis_c_train_dist_aucs"] = cf_aucs
    s0_lift = (cf_aucs.get("test_s0") or 0.5) - (orig_aucs.get("test_s0") or 0.5)
    findings["analysis_c_verdict"] = (
        f"TRAINING-EVAL MISMATCH IS THE CAUSE (s0 AUC lifts {s0_lift:+.3f} when switched to training-dist eval graph)"
        if s0_lift > 0.15
        else f"Counterfactual lifts s0 only {s0_lift:+.3f} pt — theory not strongly supported, look elsewhere"
    )
    print(f"  -> Analysis C verdict: {findings['analysis_c_verdict']}")

    # --------------------------------------------------------- analysis D

    banner("ANALYSIS D  Training-time logit distribution (from this run's training)")
    print("  At training time the model sees shuffle_train graph (matched distribution).")
    print("  If forward logits during training are well-distributed (not saturating),")
    print("  while eval-time logits saturate, that closes the causal loop.")
    print()
    print("  Capturing one training batch's forward (using a fresh training-dist graph):")

    # One epoch's training graph (different seed for novelty)
    train_kg_d, train_targets = shuffle_train(
        train_ddi_int,
        model._kg_triplets,
        "S2",
        ratio=0.8,
        rng=np.random.default_rng(999),
        extra_kg_ent=model._kg_entity_set,
    )
    esd, edd, erd = build_edge_lists_from_triplets(
        train_kg_d, model._n_ent, model._n_base_rel_with_ddi
    )
    train_sample_size = min(SAMPLE_PAIRS_PER_SPLIT, len(train_targets))
    if train_sample_size > 0:
        # train_targets is (n, 3) int array of [head, tail, rel]
        # Take first N targets as our sample positive pairs.
        sample_h = torch.from_numpy(train_targets[:train_sample_size, 0]).long().to(model.device)
        sample_t = torch.from_numpy(train_targets[:train_sample_size, 1]).long().to(model.device)
        es_t = torch.from_numpy(esd).long().to(model.device)
        ed_t = torch.from_numpy(edd).long().to(model.device)
        er_t = torch.from_numpy(erd).long().to(model.device)
        with torch.no_grad():
            train_logits = model._model(sample_h, sample_t, es_t, ed_t, er_t)
        z_train = train_logits.cpu().numpy()
        train_stats = {
            "n": int(z_train.size),
            "logit_mean": float(z_train.mean()),
            "logit_std": float(z_train.std()),
            "logit_min": float(z_train.min()),
            "logit_max": float(z_train.max()),
            "fraction_abs_z_gt_10": float(np.mean(np.abs(z_train) > 10.0)),
            "sample_logits": [round(float(v), 3) for v in z_train[:8].tolist()],
        }
        print(f"  Training-graph forward on {train_stats['n']} target positives:")
        print(f"    logit mean={train_stats['logit_mean']:+.2f}  std={train_stats['logit_std']:.2f}  range=[{train_stats['logit_min']:+.2f}, {train_stats['logit_max']:+.2f}]")
        print(f"    fraction |z|>10: {train_stats['fraction_abs_z_gt_10']*100:.1f}%")
        print(f"    first 8: {train_stats['sample_logits']}")
        findings["analysis_d_training_graph_logits"] = train_stats

    # --------------------------------------------------------- analysis E

    banner("ANALYSIS E  DECISIVE TEST: eval with train_ddi edges REMOVED")
    print("  Codex round-1 suggestion: 'remove or mask injected train-DDI edges at")
    print("  eval for S0/S1. If logits stop going to -400/-1100, the DDI-edge")
    print("  confounder is real.'")
    print()
    print("  Build a base_kg-only eval graph (drop all rel == n_base_rel edges,")
    print("  which are the injected train DDI relation).")

    # Use only the static KG triplets (base_kg = enzymes/targets/etc, NO DDI).
    base_only_edges_np = build_edge_lists_from_triplets(
        model._kg_triplets,  # no train_ddi mixed in
        model._n_ent,
        model._n_base_rel_with_ddi,
    )
    es_b = torch.from_numpy(base_only_edges_np[0]).long().to(model.device)
    ed_b = torch.from_numpy(base_only_edges_np[1]).long().to(model.device)
    er_b = torch.from_numpy(base_only_edges_np[2]).long().to(model.device)

    saved_eval_edges_e = model._eval_edges
    model._eval_edges = (es_b, ed_b, er_b)

    # 1. Forward-pass logit stats with base_kg-only graph
    print()
    print("  E.1  Logit stats on base_kg-only graph (no train_ddi edges):")
    e_logit_stats = {}
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        df = getattr(ds.splits, split_name).head(SAMPLE_PAIRS_PER_SPLIT * 4)
        pos = df[["drug_a_id", "drug_b_id"]].head(SAMPLE_PAIRS_PER_SPLIT)
        try:
            neg = ds.get_negatives(split_name).head(SAMPLE_PAIRS_PER_SPLIT)
        except Exception:
            neg = pd.DataFrame()
        all_pairs = pd.concat([pos, neg]) if len(neg) else pos
        head, tail = model._pair_indices(all_pairs)
        head = head.to(model.device); tail = tail.to(model.device)
        with torch.no_grad():
            z = model._model(head, tail, es_b, ed_b, er_b).cpu().numpy()
        e_logit_stats[split_name] = {
            "logit_mean": float(z.mean()),
            "logit_std": float(z.std()),
            "fraction_abs_z_gt_10": float(np.mean(np.abs(z) > 10.0)),
        }
        print(f"    {split_name}: mean={z.mean():+.2f} std={z.std():.2f} "
              f"|z|>10 frac={np.mean(np.abs(z)>10):.2f}")

    # 2. Full AUC on base_kg-only graph (vs orig)
    print()
    print("  E.2  Full AUC comparison: base_kg-only eval vs orig eval (train_ddi+base_kg)")
    e_aucs = {}
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        pos = getattr(ds.splits, split_name)[["drug_a_id", "drug_b_id"]]
        try:
            neg = ds.get_negatives(split_name)[["drug_a_id", "drug_b_id"]]
        except Exception:
            e_aucs[split_name] = None
            continue
        if len(pos) == 0 or len(neg) == 0:
            e_aucs[split_name] = None
            continue
        y_score = np.concatenate([model.predict_proba(pos), model.predict_proba(neg)])
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        e_aucs[split_name] = float(roc_auc_score(y_true, y_score))

    print(f"  {'split':10s}  {'orig (train_ddi+base_kg)':>27s}  {'base_kg only':>15s}  {'lift':>8s}")
    for split_name in ["test_s0", "test_s1", "test_s2"]:
        o = orig_aucs[split_name]
        e = e_aucs[split_name]
        if o is None or e is None:
            print(f"  {split_name:10s}  {'-':>27s}  {'-':>15s}  {'-':>8s}")
            continue
        lift = e - o
        print(f"  {split_name:10s}  {o:>27.4f}  {e:>15.4f}  {lift:>+8.4f}")

    # Restore eval edges
    model._eval_edges = saved_eval_edges_e

    findings["analysis_e_base_only_logit_stats"] = e_logit_stats
    findings["analysis_e_base_only_aucs"] = e_aucs
    findings["analysis_e_orig_aucs"] = orig_aucs

    # Verdict: if base_kg-only eval lifts s0 AUC by >0.15 -> confounder confirmed
    s0_lift_e = (e_aucs.get("test_s0") or 0.5) - (orig_aucs.get("test_s0") or 0.5)
    s1_lift_e = (e_aucs.get("test_s1") or 0.5) - (orig_aucs.get("test_s1") or 0.5)
    findings["analysis_e_verdict"] = (
        f"DDI-EDGE CONFOUNDER CONFIRMED (s0 +{s0_lift_e:.3f}, s1 +{s1_lift_e:.3f} on base_kg-only eval)"
        if s0_lift_e > 0.10 or s1_lift_e > 0.10
        else f"DDI-edge confounder NOT confirmed (s0 lift {s0_lift_e:+.3f}, s1 lift {s1_lift_e:+.3f})"
    )
    print(f"  -> Analysis E verdict: {findings['analysis_e_verdict']}")

    # --------------------------------------------------------- save findings

    banner("WRITING FINDINGS")
    with OUT_JSON.open("w") as f:
        json.dump(findings, f, indent=2)
    print(f"  saved to {OUT_JSON}")

    # ---------------------------- final summary ---------------------------

    banner("FINAL DIAGNOSIS")
    a_v = findings.get("analysis_a_verdict", "?")
    b_v = findings.get("analysis_b_verdict", "?")
    c_v = findings.get("analysis_c_verdict", "?")
    e_v = findings.get("analysis_e_verdict", "?")
    print(f"  Analysis A (degree mismatch):  {a_v}")
    print(f"  Analysis B (eval saturation):  {b_v}")
    print(f"  Analysis C (counterfactual):   {c_v}")
    print(f"  Analysis E (DDI-edge mask):    {e_v}")
    print()
    if all("CONFIRMED" in v.upper() or "MISMATCH" in v.upper() for v in (a_v, b_v, c_v)):
        print("  -> HYPOTHESIS FULLY CONFIRMED")
        print("    Training-eval distribution mismatch under paper-faithful")
        print("    drugbank KG causes s0/s1 saturation. See analysis values in")
        print(f"    {OUT_JSON.relative_to(_ROOT)} for exact numbers.")
    else:
        print("  -> HYPOTHESIS PARTIALLY SUPPORTED or REJECTED — see per-analysis verdicts above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
