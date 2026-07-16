"""E8.2 — LLM oracle with class-specific probability prompts (codex round-2 fix).

E8.1 used binary YES/NO with a strict "clinically relevant DDI" question. The
LLM defaulted to NO for ~90% of cases (yes_rate < 15%), drowning the PD-effect
signal we wanted to surface. Codex recommended class-specific probability
prompts that align the question with the hypothesized pathway:

  Molecular prompt (PK lens): "estimate probability of shared pharmacokinetic
    pathway via enzymes/transporters/metabolism"
  Effect prompt (PD lens): "estimate probability of shared pharmacodynamic
    risk pathway via additive toxicity / shared organ-system risk"
  Name prompt: "estimate probability of clinical DDI between these drugs"

Evaluate via AUROC (positive vs negative score distributions per prompt-class
cell) instead of accuracy. AUROC is unaffected by base-rate / threshold bias.

Same 90 pairs as E8.1 (deterministic via seed=42). Same KG features.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Reuse the helpers from 08_llm_oracle.py
sys.path.insert(0, str(Path(__file__).parent))
import importlib
e8 = importlib.import_module("08_llm_oracle")

load_env = e8.load_env
build_features = e8.build_features
sample_pairs = e8.sample_pairs
call_anthropic = e8.call_anthropic

ENV_FILE = e8.ENV_FILE
NODES = e8.NODES
EDGES = e8.EDGES
PKPD = e8.PKPD
SPLITS = e8.SPLITS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent
CACHE_FILE = OUT_DIR / "llm_oracle_predictions_prob.jsonl"
RESULTS_FILE = OUT_DIR / "llm_oracle_results_prob.json"

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 60
TEMPERATURE = 0.0


SYSTEM_PROMPT = (
    "You are a clinical pharmacology assistant. You will assess the likelihood "
    "of a specific drug-drug interaction risk pathway. Always reply in EXACTLY "
    "this format: on the first line, a single floating-point number between 0 "
    "and 1 (the probability); on the second line, a one-sentence reason. "
    "Output nothing else."
)


def prompt_A_name(pair: dict, mol_feats, eff_feats) -> str:
    return (
        f"Drug A: {pair['drug_a_name']}\n"
        f"Drug B: {pair['drug_b_name']}\n\n"
        f"Estimate the probability (0 to 1) that these two drugs would have a "
        f"clinically relevant drug-drug interaction when co-administered."
    )


def prompt_B_molecular(pair: dict, mol_feats, eff_feats) -> str:
    m_a = mol_feats.get(pair["drug_a_id"], [])
    m_b = mol_feats.get(pair["drug_b_id"], [])
    return (
        f"Drug A: {pair['drug_a_name']}\n"
        f"Drug A's molecular targets/enzymes/transporters: {', '.join(m_a) if m_a else '(none documented)'}\n"
        f"Drug B: {pair['drug_b_name']}\n"
        f"Drug B's molecular targets/enzymes/transporters: {', '.join(m_b) if m_b else '(none documented)'}\n\n"
        f"Based ONLY on the listed molecular features, estimate the probability "
        f"(0 to 1) that these drugs share a pharmacokinetic interaction pathway "
        f"(e.g., shared CYP isoform substrate/inhibitor pair, transporter "
        f"competition, metabolic enzyme overlap) when co-administered."
    )


def prompt_C_effect(pair: dict, mol_feats, eff_feats) -> str:
    e_a = eff_feats.get(pair["drug_a_id"], [])
    e_b = eff_feats.get(pair["drug_b_id"], [])
    return (
        f"Drug A: {pair['drug_a_name']}\n"
        f"Drug A's documented side effects / phenotypes: {', '.join(e_a) if e_a else '(none documented)'}\n"
        f"Drug B: {pair['drug_b_name']}\n"
        f"Drug B's documented side effects / phenotypes: {', '.join(e_b) if e_b else '(none documented)'}\n\n"
        f"Based ONLY on the listed side-effect/phenotype features, estimate the "
        f"probability (0 to 1) that these drugs share a pharmacodynamic risk "
        f"pathway (e.g., overlapping toxicity, additive adverse effects on a "
        f"shared organ system, common phenotypic risk like bleeding or QT "
        f"prolongation) when co-administered."
    )


PROMPT_BUILDERS = {
    "A_name": prompt_A_name,
    "B_molecular": prompt_B_molecular,
    "C_effect": prompt_C_effect,
}


# Parse first-line float in [0, 1]
_PROB_RE = re.compile(r"(?<![\d.])(0?\.\d+|1(?:\.0+)?|0(?:\.0+)?|1)(?![\d.])")


def parse_probability(text: str) -> float | None:
    if not text:
        return None
    # try first non-empty line first
    for line in text.strip().split("\n"):
        s = line.strip()
        if not s:
            continue
        m = _PROB_RE.search(s)
        if m:
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    return v
            except ValueError:
                pass
        break
    # fallback: full text
    m = _PROB_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            return v if 0.0 <= v <= 1.0 else None
        except ValueError:
            return None
    return None


# Cache I/O
def load_cache(path: Path) -> dict:
    cache = {}
    if not path.exists():
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cache[rec["call_id"]] = rec
            except json.JSONDecodeError:
                continue
    return cache


def append_cache(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def bootstrap_auc_ci(scores: list, labels: list, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    s = np.array(scores)
    y = np.array(labels)
    n = len(s)
    if n < 2 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            aucs.append(roc_auc_score(y[idx], s[idx]))
        except ValueError:
            continue
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def paired_bootstrap_auc_diff(scores_a, scores_b, labels, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    y = np.array(labels)
    n = len(y)
    if n < 2 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        except ValueError:
            continue
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main():
    # 1. Env
    env = load_env(ENV_FILE)
    api_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_API_KEY")
    if not api_key:
        print(f"[E8.2] FATAL: no Claude key found")
        return
    print(f"[E8.2] loaded API key (len {len(api_key)})")

    # 2. KG + features + pairs (reuse E8.1 logic for identical sample)
    print("[E8.2] loading KG ...")
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    id2name = dict(zip(nodes["id"], nodes["name"].fillna("")))
    mol_feats, eff_feats = build_features(edges, nodes, drug_set)
    print(f"[E8.2] avg #mol per drug = {np.mean([len(v) for v in mol_feats.values()]):.1f}; "
          f"avg #eff per drug = {np.mean([len(v) for v in eff_feats.values()]):.1f}")

    pkpd_df = pd.read_csv(PKPD)
    pkpd_labels = dict(zip(pkpd_df["ddi_type"], pkpd_df["pk_pd_label"]))

    pairs = sample_pairs(pkpd_labels, mol_feats, eff_feats, id2name, seed=42)
    n_pd = sum(1 for p in pairs if p["mech_class"] == "PD")
    n_pk = sum(1 for p in pairs if p["mech_class"] == "PK")
    n_neg = sum(1 for p in pairs if p["mech_class"] == "NEG")
    print(f"[E8.2] sampled: PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}  total={len(pairs)}")

    # 3. Preview new prompts
    print("\n[E8.2] === NEW PROMPT PREVIEW (first PD-pos pair) ===")
    sample_p = next(p for p in pairs if p["mech_class"] == "PD")
    for v in ["A_name", "B_molecular", "C_effect"]:
        print(f"\n--- {v} ---")
        print(PROMPT_BUILDERS[v](sample_p, mol_feats, eff_feats))
    print("\n[E8.2] === END PREVIEW ===\n")

    # 4. Call API with cache
    cache = load_cache(CACHE_FILE)
    print(f"[E8.2] cache hits: {len(cache)}")

    variants = ["A_name", "B_molecular", "C_effect"]
    total = len(pairs) * len(variants)
    done = 0
    t0 = time.time()
    for pair in pairs:
        for v in variants:
            call_id = f"{pair['pair_id']}_{v}"
            done += 1
            if call_id in cache:
                continue
            user = PROMPT_BUILDERS[v](pair, mol_feats, eff_feats)
            resp = call_anthropic(api_key, SYSTEM_PROMPT, user, model=MODEL,
                                   max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            text = ""
            score = None
            tok_in = tok_out = 0
            err = None
            if "error" in resp:
                err = resp["error"]
            else:
                try:
                    text = resp["content"][0]["text"]
                    score = parse_probability(text)
                    tok_in = resp.get("usage", {}).get("input_tokens", 0)
                    tok_out = resp.get("usage", {}).get("output_tokens", 0)
                except (KeyError, IndexError, TypeError) as e:
                    text = json.dumps(resp)[:500]
                    err = f"parse: {e}"

            rec = dict(
                call_id=call_id,
                pair_id=pair["pair_id"],
                mech_class=pair["mech_class"],
                label=pair["label"],
                variant=v,
                response_text=text,
                score=score,
                tokens_in=tok_in,
                tokens_out=tok_out,
                error=err,
            )
            append_cache(CACHE_FILE, rec)
            cache[call_id] = rec
            if done % 30 == 0 or done == 1:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}]  pair={pair['pair_id'][:40]}  variant={v}  "
                      f"score={score}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # 5. AUROC per (class, variant)
    print("\n[E8.2] === AUROC by (class, variant) ===")
    results = {}
    for class_label in ["PD", "PK"]:
        for v in variants:
            pos_recs = [cache[f"{p['pair_id']}_{v}"] for p in pairs if p["mech_class"] == class_label]
            neg_recs = [cache[f"{p['pair_id']}_{v}"] for p in pairs if p["mech_class"] == "NEG"]
            recs = pos_recs + neg_recs
            scores = [r["score"] for r in recs if r["score"] is not None]
            labels = [r["label"] for r in recs if r["score"] is not None]
            if len(set(labels)) < 2:
                continue
            auc = float(roc_auc_score(labels, scores))
            auc_m, lo, hi = bootstrap_auc_ci(scores, labels)
            score_pos = [r["score"] for r in pos_recs if r["score"] is not None]
            score_neg = [r["score"] for r in neg_recs if r["score"] is not None]
            key = f"{class_label}_{v}"
            results[key] = {
                "n_pos": len(score_pos), "n_neg": len(score_neg),
                "auc": auc, "auc_boot_mean": auc_m, "auc_ci_95": [lo, hi],
                "mean_score_pos": float(np.mean(score_pos)) if score_pos else None,
                "mean_score_neg": float(np.mean(score_neg)) if score_neg else None,
            }
            print(f"  {class_label:3s}  {v:13s}  AUROC={auc:.3f}  [CI {lo:.3f}, {hi:.3f}]  "
                  f"pos_score={np.mean(score_pos):.2f} neg_score={np.mean(score_neg):.2f}  "
                  f"n_pos={len(score_pos)} n_neg={len(score_neg)}")

    # 6. Paired AUROC differentials (effect - molecular, effect - name) per class
    print("\n[E8.2] === Paired ΔAUROC ===")
    diffs = {}
    for class_label in ["PD", "PK"]:
        pids = [p["pair_id"] for p in pairs if p["mech_class"] in (class_label, "NEG")]
        ys, sA, sB, sC = [], [], [], []
        for pid in pids:
            rA = cache.get(f"{pid}_A_name")
            rB = cache.get(f"{pid}_B_molecular")
            rC = cache.get(f"{pid}_C_effect")
            if not (rA and rB and rC):
                continue
            if rA["score"] is None or rB["score"] is None or rC["score"] is None:
                continue
            ys.append(rA["label"])
            sA.append(rA["score"])
            sB.append(rB["score"])
            sC.append(rC["score"])
        if len(set(ys)) < 2:
            continue
        d_ec_a, lo1, hi1 = paired_bootstrap_auc_diff(sC, sA, ys)
        d_ec_b, lo2, hi2 = paired_bootstrap_auc_diff(sC, sB, ys)
        d_b_a, lo3, hi3 = paired_bootstrap_auc_diff(sB, sA, ys)
        diffs[class_label] = {
            "effect_minus_name": (d_ec_a, [lo1, hi1]),
            "effect_minus_molecular": (d_ec_b, [lo2, hi2]),
            "molecular_minus_name": (d_b_a, [lo3, hi3]),
        }
        print(f"  {class_label}:  effect-name = {d_ec_a:+.3f} [{lo1:+.3f}, {hi1:+.3f}]  "
              f"effect-mol = {d_ec_b:+.3f} [{lo2:+.3f}, {hi2:+.3f}]  "
              f"mol-name = {d_b_a:+.3f} [{lo3:+.3f}, {hi3:+.3f}]")

    # 7. Stats and save
    n_err = sum(1 for r in cache.values() if r.get("error"))
    n_unparsed = sum(1 for r in cache.values() if r.get("score") is None and not r.get("error"))
    tin = sum(r.get("tokens_in", 0) for r in cache.values())
    tout = sum(r.get("tokens_out", 0) for r in cache.values())
    cost = tin * 3 / 1_000_000 + tout * 15 / 1_000_000
    print(f"\n[E8.2] tokens: in={tin}  out={tout}  cost~=${cost:.3f}")
    print(f"[E8.2] errors: {n_err}, unparsed: {n_unparsed}")

    payload = {
        "model": MODEL,
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg},
        "metric": "AUROC",
        "results_by_class_variant": results,
        "paired_diffs": {k: {kk: {"mean": vv[0], "ci": vv[1]} for kk, vv in v.items()}
                          for k, v in diffs.items()},
        "tokens": {"input": tin, "output": tout},
        "cost_usd_est": cost,
        "errors": n_err,
        "unparsed": n_unparsed,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8.2] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
