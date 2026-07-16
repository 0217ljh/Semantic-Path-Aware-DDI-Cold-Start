"""E8.3 — Anonymized LLM oracle (codex round-3 fix).

Round 2 showed that name-only AUROC = 0.76 for both classes — Claude is
memorizing DDIs from drug names regardless of provided features. To test
whether effect features carry PD-specific biological signal independent of
memorization, redact drug names ("Drug A" / "Drug B") and provide ONLY the
KG features.

Same 90 pairs as E8.2 (deterministic via seed=42). Three anonymized variants:
  A_anon_null     : "Drug A" / "Drug B", no features (null floor, should ≈ chance)
  B_anon_molecular: drug names redacted, molecular features only
  C_anon_effect   : drug names redacted, effect features only

If PD-effect-anon > PK-effect-anon AND PK-molecular-anon > PD-molecular-anon,
the asymmetric paradigm-specific reliance is confirmed without name confound.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import importlib
e8 = importlib.import_module("08_llm_oracle")
e8b = importlib.import_module("08b_llm_oracle_prob")

load_env = e8.load_env
build_features = e8.build_features
sample_pairs = e8.sample_pairs
call_anthropic = e8.call_anthropic
parse_probability = e8b.parse_probability
load_cache = e8b.load_cache
append_cache = e8b.append_cache
bootstrap_auc_ci = e8b.bootstrap_auc_ci
paired_bootstrap_auc_diff = e8b.paired_bootstrap_auc_diff

ENV_FILE = e8.ENV_FILE
NODES = e8.NODES
EDGES = e8.EDGES
PKPD = e8.PKPD
SPLITS = e8.SPLITS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent
CACHE_FILE = OUT_DIR / "llm_oracle_predictions_anon.jsonl"
RESULTS_FILE = OUT_DIR / "llm_oracle_results_anon.json"

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 60
TEMPERATURE = 0.0


SYSTEM_PROMPT = (
    "You are a clinical pharmacology assistant. You will assess the likelihood "
    "of a specific drug-drug interaction risk pathway based ONLY on listed "
    "features. The drug identities are redacted. Reply in EXACTLY this format: "
    "on the first line, a single floating-point number between 0 and 1 (the "
    "probability); on the second line, a one-sentence reason. Output nothing else."
)


def prompt_A_null(pair, mol_feats, eff_feats):
    return (
        "Drug A: (identity redacted)\n"
        "Drug B: (identity redacted)\n\n"
        "Estimate the probability (0 to 1) that these two drugs would have a "
        "clinically relevant drug-drug interaction when co-administered. "
        "No features are provided about either drug."
    )


def prompt_B_anon_molecular(pair, mol_feats, eff_feats):
    m_a = mol_feats.get(pair["drug_a_id"], [])
    m_b = mol_feats.get(pair["drug_b_id"], [])
    return (
        f"Drug A: (identity redacted)\n"
        f"Drug A's molecular targets/enzymes/transporters: {', '.join(m_a) if m_a else '(none documented)'}\n"
        f"Drug B: (identity redacted)\n"
        f"Drug B's molecular targets/enzymes/transporters: {', '.join(m_b) if m_b else '(none documented)'}\n\n"
        f"Based ONLY on the listed molecular features (drug identities are "
        f"redacted), estimate the probability (0 to 1) that these drugs share "
        f"a pharmacokinetic interaction pathway (e.g., shared CYP isoform "
        f"substrate/inhibitor pair, transporter competition, metabolic enzyme "
        f"overlap) when co-administered."
    )


def prompt_C_anon_effect(pair, mol_feats, eff_feats):
    e_a = eff_feats.get(pair["drug_a_id"], [])
    e_b = eff_feats.get(pair["drug_b_id"], [])
    return (
        f"Drug A: (identity redacted)\n"
        f"Drug A's documented side effects / phenotypes: {', '.join(e_a) if e_a else '(none documented)'}\n"
        f"Drug B: (identity redacted)\n"
        f"Drug B's documented side effects / phenotypes: {', '.join(e_b) if e_b else '(none documented)'}\n\n"
        f"Based ONLY on the listed side-effect/phenotype features (drug "
        f"identities are redacted), estimate the probability (0 to 1) that "
        f"these drugs share a pharmacodynamic risk pathway (e.g., overlapping "
        f"toxicity, additive adverse effects on a shared organ system, common "
        f"phenotypic risk like bleeding or QT prolongation) when co-administered."
    )


PROMPT_BUILDERS = {
    "A_anon_null": prompt_A_null,
    "B_anon_molecular": prompt_B_anon_molecular,
    "C_anon_effect": prompt_C_anon_effect,
}


def main():
    env = load_env(ENV_FILE)
    api_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[E8.3] FATAL: no Claude key")
        return
    print(f"[E8.3] loaded key (len {len(api_key)})")

    print("[E8.3] loading KG ...")
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    id2name = dict(zip(nodes["id"], nodes["name"].fillna("")))
    mol_feats, eff_feats = build_features(edges, nodes, drug_set)

    pkpd_df = pd.read_csv(PKPD)
    pkpd_labels = dict(zip(pkpd_df["ddi_type"], pkpd_df["pk_pd_label"]))
    pairs = sample_pairs(pkpd_labels, mol_feats, eff_feats, id2name, seed=42)
    n_pd = sum(1 for p in pairs if p["mech_class"] == "PD")
    n_pk = sum(1 for p in pairs if p["mech_class"] == "PK")
    n_neg = sum(1 for p in pairs if p["mech_class"] == "NEG")
    print(f"[E8.3] PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}")

    print("\n[E8.3] === ANON PROMPT PREVIEW (first PD-pos pair) ===")
    sample_p = next(p for p in pairs if p["mech_class"] == "PD")
    for v in ["A_anon_null", "B_anon_molecular", "C_anon_effect"]:
        print(f"\n--- {v} ---")
        print(PROMPT_BUILDERS[v](sample_p, mol_feats, eff_feats))
    print("\n[E8.3] === END PREVIEW ===\n")

    cache = load_cache(CACHE_FILE)
    print(f"[E8.3] cache hits: {len(cache)}")

    variants = ["A_anon_null", "B_anon_molecular", "C_anon_effect"]
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
                call_id=call_id, pair_id=pair["pair_id"],
                mech_class=pair["mech_class"], label=pair["label"],
                variant=v, response_text=text, score=score,
                tokens_in=tok_in, tokens_out=tok_out, error=err,
            )
            append_cache(CACHE_FILE, rec)
            cache[call_id] = rec
            if done % 30 == 0 or done == 1:
                el = time.time() - t0
                eta = el / done * (total - done)
                print(f"  [{done}/{total}]  pair={pair['pair_id'][:35]}  variant={v}  "
                      f"score={score}  elapsed={el:.0f}s ETA={eta:.0f}s")

    # AUROC per (class, variant)
    print("\n[E8.3] === AUROC ===")
    results = {}
    for cl in ["PD", "PK"]:
        for v in variants:
            pos = [cache[f"{p['pair_id']}_{v}"] for p in pairs if p["mech_class"] == cl]
            neg = [cache[f"{p['pair_id']}_{v}"] for p in pairs if p["mech_class"] == "NEG"]
            recs = pos + neg
            sc = [r["score"] for r in recs if r["score"] is not None]
            ys = [r["label"] for r in recs if r["score"] is not None]
            if len(set(ys)) < 2:
                continue
            auc = float(roc_auc_score(ys, sc))
            auc_m, lo, hi = bootstrap_auc_ci(sc, ys)
            sp = [r["score"] for r in pos if r["score"] is not None]
            sn = [r["score"] for r in neg if r["score"] is not None]
            key = f"{cl}_{v}"
            results[key] = {
                "n_pos": len(sp), "n_neg": len(sn),
                "auc": auc, "auc_ci_95": [lo, hi],
                "mean_score_pos": float(np.mean(sp)) if sp else None,
                "mean_score_neg": float(np.mean(sn)) if sn else None,
            }
            print(f"  {cl}  {v:18s}  AUROC={auc:.3f} [CI {lo:.3f},{hi:.3f}]  "
                  f"pos={np.mean(sp):.2f} neg={np.mean(sn):.2f}  n_pos={len(sp)}/n_neg={len(sn)}")

    # Paired ΔAUROC
    print("\n[E8.3] === Paired ΔAUROC ===")
    diffs = {}
    for cl in ["PD", "PK"]:
        ys, sA, sB, sC = [], [], [], []
        for p in pairs:
            if p["mech_class"] not in (cl, "NEG"):
                continue
            rA = cache.get(f"{p['pair_id']}_A_anon_null")
            rB = cache.get(f"{p['pair_id']}_B_anon_molecular")
            rC = cache.get(f"{p['pair_id']}_C_anon_effect")
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
        diffs[cl] = {
            "effect-null": (d_ec_a, [lo1, hi1]),
            "effect-molecular": (d_ec_b, [lo2, hi2]),
            "molecular-null": (d_b_a, [lo3, hi3]),
        }
        print(f"  {cl}:  eff-null = {d_ec_a:+.3f} [{lo1:+.3f},{hi1:+.3f}]  "
              f"eff-mol = {d_ec_b:+.3f} [{lo2:+.3f},{hi2:+.3f}]  "
              f"mol-null = {d_b_a:+.3f} [{lo3:+.3f},{hi3:+.3f}]")

    # Cross-class: PD-effect-anon vs PK-effect-anon
    print("\n[E8.3] === Cross-class biological-signal check ===")
    print(f"  PD + effect-anon AUROC: {results.get('PD_C_anon_effect', {}).get('auc', 'NA')}")
    print(f"  PK + effect-anon AUROC: {results.get('PK_C_anon_effect', {}).get('auc', 'NA')}")
    print(f"  PD + mol-anon   AUROC: {results.get('PD_B_anon_molecular', {}).get('auc', 'NA')}")
    print(f"  PK + mol-anon   AUROC: {results.get('PK_B_anon_molecular', {}).get('auc', 'NA')}")

    tin = sum(r.get("tokens_in", 0) for r in cache.values())
    tout = sum(r.get("tokens_out", 0) for r in cache.values())
    cost = tin * 3 / 1_000_000 + tout * 15 / 1_000_000
    print(f"\n[E8.3] tokens: in={tin} out={tout}  cost~=${cost:.3f}")

    payload = {
        "model": MODEL,
        "anonymized": True,
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg},
        "results_by_class_variant": results,
        "paired_diffs": {k: {kk: {"mean": vv[0], "ci": vv[1]} for kk, vv in v.items()}
                          for k, v in diffs.items()},
        "tokens": {"input": tin, "output": tout},
        "cost_usd_est": cost,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8.3] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
