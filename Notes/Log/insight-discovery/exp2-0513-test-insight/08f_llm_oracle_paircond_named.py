"""E8.6 — Pair-conditional two-stage WITH drug names visible (user request).

User correction: drug names should NEVER be hidden — that's the actual
deployment scenario. E8.5 was anonymized per codex round-3 anti-memorization
recommendation, but the user prefers the realistic setting where the LLM
sees both names and features.

Comparison target: E8.2 (single-stage, named) — PD effect AUROC = 0.652,
PK effect AUROC = 0.510, PK molecular AUROC = 0.693.

If E8.6 PD-effect > 0.652 → pair-conditional selection lifts PD signal
beyond bag-aggregation, validating perception-first hypothesis.
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
e8e = importlib.import_module("08e_llm_oracle_pair_cond")

load_env = e8.load_env
build_features = e8.build_features
sample_pairs = e8.sample_pairs
call_anthropic = e8.call_anthropic
parse_probability = e8b.parse_probability
load_cache = e8b.load_cache
append_cache = e8b.append_cache
bootstrap_auc_ci = e8b.bootstrap_auc_ci
paired_bootstrap_auc_diff = e8b.paired_bootstrap_auc_diff
parse_selector_output = e8e.parse_selector_output
format_pairs_for_reasoner = e8e.format_pairs_for_reasoner

ENV_FILE = e8.ENV_FILE
NODES = e8.NODES
EDGES = e8.EDGES
PKPD = e8.PKPD
SPLITS = e8.SPLITS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent
SELECTOR_CACHE = OUT_DIR / "llm_oracle_paircond_named_selector.jsonl"
REASONER_CACHE = OUT_DIR / "llm_oracle_paircond_named_reasoner.jsonl"
RESULTS_FILE = OUT_DIR / "llm_oracle_results_paircond_named.json"

MODEL = "claude-sonnet-4-5-20250929"
MAX_FEATURES_INPUT = 30
MAX_SEL_OUTPUT_TOKENS = 250
MAX_REASONER_OUTPUT_TOKENS = 80
TEMPERATURE = 0.0


SELECTOR_SYSTEM = (
    "You are a clinical pharmacology assistant. You will see two named drugs "
    "and their documented effect lists. Your task: pick ONE pair of effects "
    "(one from each drug) that best characterizes how these two drugs relate "
    "at the effect level.\n\n"
    "Rule:\n"
    "- If a clinically meaningful composition exists (e.g., shared "
    "physiological endpoint such as both increasing bleeding; downstream "
    "cascade; shared organ-system risk), pick that pair.\n"
    "- Otherwise, pick the most GENERIC / COMMON effect from each drug "
    "(non-mechanism-specific effects like nausea, headache, fatigue, rash, "
    "dizziness). Never refuse to pick.\n\n"
    "Output exactly one line in this format:\n"
    "Pair1: A_effect=\"<effect from drug A>\" | B_effect=\"<effect from drug B>\"\n"
    "Output nothing else (no preamble, no rationale, no second pair)."
)


def selector_prompt_named(name_a: str, name_b: str, eff_a: list[str], eff_b: list[str]) -> str:
    return (
        f"Drug A: {name_a}\n"
        f"Drug A's documented side effects / phenotypes:\n"
        f"{', '.join(eff_a) if eff_a else '(none documented)'}\n\n"
        f"Drug B: {name_b}\n"
        f"Drug B's documented side effects / phenotypes:\n"
        f"{', '.join(eff_b) if eff_b else '(none documented)'}\n\n"
        f"Pick ONE pair of effects (one from drug A, one from drug B). If a "
        f"clinically meaningful composition exists, pick that pair. Otherwise, "
        f"pick the most generic / common effect from each list."
    )


REASONER_SYSTEM = (
    "You are a clinical pharmacology assistant. You will be shown two named "
    "drugs and a few of their documented clinical effects. Estimate the "
    "probability of a clinically significant drug-drug interaction between "
    "them. Reply in EXACTLY: line 1 a single number between 0 and 1; line 2 "
    "a one-sentence reason. Output nothing else."
)


def reasoner_prompt_named(name_a: str, name_b: str, pairs_text: str) -> str:
    # Selector now always picks ONE pair (composable or trivial fallback).
    # Reasoner sees the picked pair as neutral context.
    return (
        f"Drug A: {name_a}\n"
        f"Drug B: {name_b}\n\n"
        f"Some of these drugs' documented clinical effects:\n{pairs_text}\n\n"
        f"Estimate the probability (0 to 1) that drug A and drug B have a "
        f"clinically relevant drug-drug interaction."
    )


def run_selector(api_key: str, pair: dict, eff_feats: dict, cache: dict) -> dict:
    call_id = f"{pair['pair_id']}_pcsel_named"
    if call_id in cache:
        return cache[call_id]
    eff_a = eff_feats.get(pair["drug_a_id"], [])[:MAX_FEATURES_INPUT]
    eff_b = eff_feats.get(pair["drug_b_id"], [])[:MAX_FEATURES_INPUT]
    prompt = selector_prompt_named(pair["drug_a_name"], pair["drug_b_name"], eff_a, eff_b)
    resp = call_anthropic(api_key, SELECTOR_SYSTEM, prompt,
                          model=MODEL, max_tokens=MAX_SEL_OUTPUT_TOKENS,
                          temperature=TEMPERATURE)
    if "error" in resp:
        rec = {"call_id": call_id, "pair_id": pair["pair_id"],
               "drug_a_name": pair["drug_a_name"], "drug_b_name": pair["drug_b_name"],
               "eff_a_input": eff_a, "eff_b_input": eff_b,
               "selected_pairs": [], "raw_text": "", "error": resp["error"],
               "tokens_in": 0, "tokens_out": 0}
    else:
        try:
            text = resp["content"][0]["text"]
            sel = parse_selector_output(text)
            rec = {"call_id": call_id, "pair_id": pair["pair_id"],
                   "drug_a_name": pair["drug_a_name"], "drug_b_name": pair["drug_b_name"],
                   "eff_a_input": eff_a, "eff_b_input": eff_b,
                   "selected_pairs": sel, "raw_text": text, "error": None,
                   "tokens_in": resp.get("usage", {}).get("input_tokens", 0),
                   "tokens_out": resp.get("usage", {}).get("output_tokens", 0)}
        except (KeyError, IndexError, TypeError) as e:
            rec = {"call_id": call_id, "pair_id": pair["pair_id"],
                   "drug_a_name": pair["drug_a_name"], "drug_b_name": pair["drug_b_name"],
                   "eff_a_input": eff_a, "eff_b_input": eff_b,
                   "selected_pairs": [], "raw_text": json.dumps(resp)[:300],
                   "error": f"parse: {e}", "tokens_in": 0, "tokens_out": 0}
    append_cache(SELECTOR_CACHE, rec)
    cache[call_id] = rec
    return rec


def run_reasoner(api_key: str, pair: dict, selected_pairs: list, cache: dict) -> dict:
    call_id = f"{pair['pair_id']}_pcreas_named"
    if call_id in cache:
        return cache[call_id]
    pairs_text = format_pairs_for_reasoner(selected_pairs)
    prompt = reasoner_prompt_named(pair["drug_a_name"], pair["drug_b_name"], pairs_text)
    resp = call_anthropic(api_key, REASONER_SYSTEM, prompt,
                          model=MODEL, max_tokens=MAX_REASONER_OUTPUT_TOKENS,
                          temperature=TEMPERATURE)
    score = None
    text = ""
    tin = tout = 0
    err = None
    if "error" in resp:
        err = resp["error"]
    else:
        try:
            text = resp["content"][0]["text"]
            score = parse_probability(text)
            tin = resp.get("usage", {}).get("input_tokens", 0)
            tout = resp.get("usage", {}).get("output_tokens", 0)
        except (KeyError, IndexError, TypeError) as e:
            text = json.dumps(resp)[:300]
            err = f"parse: {e}"
    rec = {"call_id": call_id, "pair_id": pair["pair_id"],
           "drug_a_name": pair["drug_a_name"], "drug_b_name": pair["drug_b_name"],
           "mech_class": pair["mech_class"], "label": pair["label"],
           "selected_pairs": selected_pairs,
           "response_text": text, "score": score,
           "tokens_in": tin, "tokens_out": tout, "error": err}
    append_cache(REASONER_CACHE, rec)
    cache[call_id] = rec
    return rec


def main():
    env = load_env(ENV_FILE)
    api_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[E8.6] FATAL: no Claude key")
        return
    print(f"[E8.6] loaded key (len {len(api_key)})")

    print("[E8.6] loading KG ...")
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
    print(f"[E8.6] PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}")

    # Preview
    sample_p = next(p for p in pairs if p["mech_class"] == "PD")
    print(f"\n[E8.6] === SELECTOR PROMPT PREVIEW (with names) ===\n")
    print(selector_prompt_named(
        sample_p["drug_a_name"], sample_p["drug_b_name"],
        eff_feats.get(sample_p["drug_a_id"], [])[:MAX_FEATURES_INPUT],
        eff_feats.get(sample_p["drug_b_id"], [])[:MAX_FEATURES_INPUT]
    ))
    print("[E8.6] === END PREVIEW ===\n")

    sel_cache = load_cache(SELECTOR_CACHE)
    sel_cache = {r["call_id"]: r for r in sel_cache.values()} if sel_cache else {}
    print(f"[E8.6] selector cache: {len(sel_cache)}")

    print("[E8.6] === Stage 1: pair-conditional selector WITH names ===")
    t0 = time.time()
    done = 0
    for pair in pairs:
        done += 1
        rec = run_selector(api_key, pair, eff_feats, sel_cache)
        if done % 20 == 0 or done == 1:
            el = time.time() - t0
            eta = el / done * (len(pairs) - done)
            print(f"  [{done}/{len(pairs)}]  pair={pair['pair_id'][:35]}  "
                  f"#sel={len(rec['selected_pairs'])}  elapsed={el:.0f}s ETA={eta:.0f}s")

    # Preview selector outputs
    print("\n[E8.6] === SELECTOR OUTPUTS (first 4 PD-pos) ===")
    cnt = 0
    for p in pairs:
        if p["mech_class"] != "PD":
            continue
        if cnt >= 4:
            break
        rec = sel_cache.get(f"{p['pair_id']}_pcsel_named")
        if not rec:
            continue
        print(f"\n  [{p['pair_id']}]  {p['drug_a_name']} + {p['drug_b_name']}")
        print(f"  Raw LLM output:")
        for ln in rec["raw_text"].split("\n"):
            print(f"    {ln}")
        print(f"  Parsed pairs: {rec['selected_pairs']}")
        cnt += 1

    # Stage 2
    reas_cache = load_cache(REASONER_CACHE)
    reas_cache = {r["call_id"]: r for r in reas_cache.values()} if reas_cache else {}
    print(f"\n[E8.6] reasoner cache: {len(reas_cache)}")

    print("[E8.6] === Stage 2: reasoner WITH names ===")
    t1 = time.time()
    done = 0
    for pair in pairs:
        done += 1
        sel_rec = sel_cache[f"{pair['pair_id']}_pcsel_named"]
        rec = run_reasoner(api_key, pair, sel_rec["selected_pairs"], reas_cache)
        if done % 20 == 0 or done == 1:
            el = time.time() - t1
            eta = el / done * (len(pairs) - done)
            print(f"  [{done}/{len(pairs)}]  pair={pair['pair_id'][:35]}  "
                  f"score={rec.get('score')}  elapsed={el:.0f}s ETA={eta:.0f}s")

    # AUROC
    print("\n[E8.6] === AUROC (pair-conditional two-stage, WITH names, effect) ===")
    pid_to_pair = {p["pair_id"]: p for p in pairs}
    results = {}
    for cl in ["PD", "PK"]:
        pos = [reas_cache[f"{p['pair_id']}_pcreas_named"] for p in pairs
               if p["mech_class"] == cl
               and f"{p['pair_id']}_pcreas_named" in reas_cache]
        neg = [reas_cache[f"{p['pair_id']}_pcreas_named"] for p in pairs
               if p["mech_class"] == "NEG"
               and f"{p['pair_id']}_pcreas_named" in reas_cache]
        sc = [r["score"] for r in (pos + neg) if r.get("score") is not None]
        ys = [pid_to_pair[r["pair_id"]].get("label", 0)
              for r in (pos + neg) if r.get("score") is not None]
        if len(set(ys)) < 2:
            continue
        auc = float(roc_auc_score(ys, sc))
        am, lo, hi = bootstrap_auc_ci(sc, ys)
        sp = [r["score"] for r in pos if r.get("score") is not None]
        sn = [r["score"] for r in neg if r.get("score") is not None]
        results[f"{cl}_effect_paircond_named"] = {
            "n_pos": len(sp), "n_neg": len(sn),
            "auc": auc, "auc_ci_95": [lo, hi],
            "pos_mean": float(np.mean(sp)) if sp else None,
            "neg_mean": float(np.mean(sn)) if sn else None,
        }
        print(f"  {cl}  pair-cond+named effect  AUROC={auc:.3f} "
              f"[CI {lo:.3f},{hi:.3f}]  pos_mean={np.mean(sp):.2f} "
              f"neg_mean={np.mean(sn):.2f}  n_pos={len(sp)}/n_neg={len(sn)}")

    # Compare to E8.2 (single-stage WITH names, C_effect)
    try:
        e8b_res = json.loads((OUT_DIR / "llm_oracle_results_prob.json").read_text())
        print("\n[E8.6] === E8.6 (pair-cond+named) vs E8.2 (single+named) ===")
        for cl in ["PD", "PK"]:
            old = e8b_res["results_by_class_variant"].get(f"{cl}_C_effect", {}).get("auc")
            new = results.get(f"{cl}_effect_paircond_named", {}).get("auc")
            if old is not None and new is not None:
                print(f"  {cl}  single+named={old:.3f}  pair-cond+named={new:.3f}  Δ={new-old:+.3f}")
    except Exception as ex:
        print(f"  (could not load E8.2 results: {ex})")

    # Selection stats
    n_sel = [len(r["selected_pairs"]) for r in sel_cache.values()]
    print(f"\n[E8.6] avg pairs/drug-pair: {np.mean(n_sel):.2f}  "
          f"(0-sel: {sum(1 for n in n_sel if n == 0)}, "
          f"1-sel: {sum(1 for n in n_sel if n == 1)}, "
          f"2-sel: {sum(1 for n in n_sel if n == 2)})")

    tin_sel = sum(r.get("tokens_in", 0) for r in sel_cache.values())
    tout_sel = sum(r.get("tokens_out", 0) for r in sel_cache.values())
    tin_reas = sum(r.get("tokens_in", 0) for r in reas_cache.values())
    tout_reas = sum(r.get("tokens_out", 0) for r in reas_cache.values())
    cost = (tin_sel + tin_reas) * 3 / 1_000_000 + (tout_sel + tout_reas) * 15 / 1_000_000
    print(f"[E8.6] tokens: sel in={tin_sel} out={tout_sel}; reas in={tin_reas} out={tout_reas}")
    print(f"[E8.6] cost~=${cost:.3f}")

    payload = {
        "model": MODEL,
        "design": "pair-conditional composition selector → reasoner, drug names visible",
        "modality": "effect only",
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg},
        "results": results,
        "selection_stats": {"mean": float(np.mean(n_sel)),
                            "zero": sum(1 for n in n_sel if n == 0),
                            "one": sum(1 for n in n_sel if n == 1),
                            "two": sum(1 for n in n_sel if n == 2)},
        "tokens": {"selector_in": tin_sel, "selector_out": tout_sel,
                    "reasoner_in": tin_reas, "reasoner_out": tout_reas},
        "cost_usd_est": cost,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8.6] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
