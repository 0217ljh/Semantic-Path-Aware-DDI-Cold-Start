"""E8.7 — Confirmation run at 60+60+60 with neutralized Stage-2 prompts and
forced-selection selector (no NONE allowed; trivial-fallback if no
composition exists).

Frozen design from E8.6 + two refinements per user feedback:
  1. Stage-2 reasoner prompt is neutral ("documented clinical effects",
     no "composition / interaction risk" framing).
  2. Stage-1 selector ALWAYS returns ONE pair. If no meaningful composition
     exists, it picks the most generic / trivial effect from each drug.

Cache files are NEW (do not reuse E8.6's stale cache).
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import importlib
e8 = importlib.import_module("08_llm_oracle")
e8b = importlib.import_module("08b_llm_oracle_prob")
e8f = importlib.import_module("08f_llm_oracle_paircond_named")  # updated prompts

load_env = e8.load_env
build_features = e8.build_features
sample_pairs = e8.sample_pairs
call_anthropic = e8.call_anthropic
parse_probability = e8b.parse_probability
load_cache = e8b.load_cache
append_cache = e8b.append_cache
bootstrap_auc_ci = e8b.bootstrap_auc_ci
paired_bootstrap_auc_diff = e8b.paired_bootstrap_auc_diff
format_pairs_for_reasoner = e8f.format_pairs_for_reasoner

# Override sample sizes
e8.N_PD_POS = 60
e8.N_PK_POS = 60
e8.N_NEG = 60

ENV_FILE = e8.ENV_FILE
NODES = e8.NODES
EDGES = e8.EDGES
PKPD = e8.PKPD
SPLITS = e8.SPLITS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent
SELECTOR_CACHE = OUT_DIR / "e87_selector_cache.jsonl"
REASONER_CACHE = OUT_DIR / "e87_reasoner_cache.jsonl"
RESULTS_FILE = OUT_DIR / "e87_results.json"

MODEL = "claude-sonnet-4-5-20250929"
MAX_FEATURES_INPUT = 30
MAX_SEL_OUTPUT_TOKENS = 200
MAX_REASONER_OUTPUT_TOKENS = 80
TEMPERATURE = 0.0


# Pair-line regex matching `Pair1: A_effect="..." | B_effect="..."`
_PAIR_RE = re.compile(
    r'Pair\s*\d*:\s*A_effect\s*=\s*"([^"]+)"\s*\|\s*B_effect\s*=\s*"([^"]+)"',
    flags=re.IGNORECASE,
)


def parse_with_fallback(text: str, eff_a: list, eff_b: list) -> list[tuple[str, str]]:
    """Parse selector output; if no pair detected (rare since prompt forbids NONE),
    fall back deterministically to the first effect from each drug's list."""
    if text:
        m = _PAIR_RE.search(text)
        if m:
            return [(m.group(1).strip(), m.group(2).strip())]
    # Fallback: pick first effect from each
    a = eff_a[0] if eff_a else "(no documented effect)"
    b = eff_b[0] if eff_b else "(no documented effect)"
    return [(a, b)]


def run_selector(api_key, pair, eff_feats, cache):
    cid = f"{pair['pair_id']}_pcsel_e87"
    if cid in cache:
        return cache[cid]
    eff_a = eff_feats.get(pair["drug_a_id"], [])[:MAX_FEATURES_INPUT]
    eff_b = eff_feats.get(pair["drug_b_id"], [])[:MAX_FEATURES_INPUT]
    prompt = e8f.selector_prompt_named(pair["drug_a_name"], pair["drug_b_name"], eff_a, eff_b)
    resp = call_anthropic(api_key, e8f.SELECTOR_SYSTEM, prompt,
                          model=MODEL, max_tokens=MAX_SEL_OUTPUT_TOKENS,
                          temperature=TEMPERATURE)
    text = ""
    err = None
    tin = tout = 0
    if "error" in resp:
        err = resp["error"]
        sel = parse_with_fallback("", eff_a, eff_b)  # use fallback
        fallback_used = True
    else:
        try:
            text = resp["content"][0]["text"]
            tin = resp.get("usage", {}).get("input_tokens", 0)
            tout = resp.get("usage", {}).get("output_tokens", 0)
        except (KeyError, IndexError, TypeError) as e:
            err = f"parse: {e}"
        # always one pair (forced)
        pre_parse_text = text
        sel = parse_with_fallback(pre_parse_text, eff_a, eff_b)
        fallback_used = not bool(_PAIR_RE.search(text or ""))
    rec = {"call_id": cid, "pair_id": pair["pair_id"],
           "drug_a_name": pair["drug_a_name"], "drug_b_name": pair["drug_b_name"],
           "eff_a_input": eff_a, "eff_b_input": eff_b,
           "selected_pairs": sel, "raw_text": text, "fallback_used": fallback_used,
           "tokens_in": tin, "tokens_out": tout, "error": err}
    append_cache(SELECTOR_CACHE, rec)
    cache[cid] = rec
    return rec


def run_reasoner(api_key, pair, selected_pairs, cache):
    cid = f"{pair['pair_id']}_pcreas_e87"
    if cid in cache:
        return cache[cid]
    pairs_text = format_pairs_for_reasoner(selected_pairs)
    prompt = e8f.reasoner_prompt_named(pair["drug_a_name"], pair["drug_b_name"], pairs_text)
    resp = call_anthropic(api_key, e8f.REASONER_SYSTEM, prompt,
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
    rec = {"call_id": cid, "pair_id": pair["pair_id"],
           "drug_a_name": pair["drug_a_name"], "drug_b_name": pair["drug_b_name"],
           "mech_class": pair["mech_class"], "label": pair["label"],
           "selected_pairs": selected_pairs,
           "response_text": text, "score": score,
           "tokens_in": tin, "tokens_out": tout, "error": err}
    append_cache(REASONER_CACHE, rec)
    cache[cid] = rec
    return rec


def main():
    env = load_env(ENV_FILE)
    api_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[E8.7] FATAL: no key")
        return
    print(f"[E8.7] loaded key (len {len(api_key)})")

    print("[E8.7] loading KG ...")
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
    print(f"[E8.7] PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}  total={len(pairs)}")

    sel_cache = load_cache(SELECTOR_CACHE)
    sel_cache = {r["call_id"]: r for r in sel_cache.values()} if sel_cache else {}
    print(f"[E8.7] selector cache: {len(sel_cache)}")

    print("[E8.7] === Stage 1: selector (forced-pair, no NONE allowed) ===")
    t0 = time.time()
    done = 0
    fallback_ct = 0
    for pair in pairs:
        done += 1
        rec = run_selector(api_key, pair, eff_feats, sel_cache)
        if rec.get("fallback_used"):
            fallback_ct += 1
        if done % 30 == 0 or done == 1:
            el = time.time() - t0
            eta = el / done * (len(pairs) - done)
            print(f"  [{done}/{len(pairs)}] {pair['pair_id'][:35]} "
                  f"sel={rec['selected_pairs']} fb={rec.get('fallback_used')} "
                  f"elapsed={el:.0f}s ETA={eta:.0f}s")
    print(f"[E8.7] selector fallback used: {fallback_ct}/{len(pairs)}")

    # Preview selector outputs
    print("\n[E8.7] === SELECTOR OUTPUTS preview (5 PD-pos, 3 NEG) ===")
    cnt = {"PD": 0, "NEG": 0}
    for p in pairs:
        cl = p["mech_class"]
        if cl not in cnt:
            continue
        if cnt[cl] >= (5 if cl == "PD" else 3):
            continue
        rec = sel_cache.get(f"{p['pair_id']}_pcsel_e87")
        if not rec:
            continue
        print(f"  [{cl}] {p['drug_a_name']} + {p['drug_b_name']}")
        print(f"    raw: {rec['raw_text'][:200]}")
        print(f"    selected: {rec['selected_pairs']}  fallback={rec.get('fallback_used')}")
        cnt[cl] += 1

    reas_cache = load_cache(REASONER_CACHE)
    reas_cache = {r["call_id"]: r for r in reas_cache.values()} if reas_cache else {}
    print(f"\n[E8.7] reasoner cache: {len(reas_cache)}")

    print("[E8.7] === Stage 2: reasoner (neutral prompt) ===")
    t1 = time.time()
    done = 0
    for pair in pairs:
        done += 1
        sel_rec = sel_cache[f"{pair['pair_id']}_pcsel_e87"]
        rec = run_reasoner(api_key, pair, sel_rec["selected_pairs"], reas_cache)
        if done % 30 == 0 or done == 1:
            el = time.time() - t1
            eta = el / done * (len(pairs) - done)
            print(f"  [{done}/{len(pairs)}] {pair['pair_id'][:35]} "
                  f"score={rec.get('score')} elapsed={el:.0f}s ETA={eta:.0f}s")

    # AUROC per class
    print("\n[E8.7] === AUROC ===")
    pid_to_pair = {p["pair_id"]: p for p in pairs}
    results = {}
    for cl in ["PD", "PK"]:
        pos = [reas_cache[f"{p['pair_id']}_pcreas_e87"] for p in pairs
               if p["mech_class"] == cl
               and f"{p['pair_id']}_pcreas_e87" in reas_cache]
        neg = [reas_cache[f"{p['pair_id']}_pcreas_e87"] for p in pairs
               if p["mech_class"] == "NEG"
               and f"{p['pair_id']}_pcreas_e87" in reas_cache]
        sc = [r["score"] for r in (pos + neg) if r.get("score") is not None]
        ys = [pid_to_pair[r["pair_id"]].get("label", 0)
              for r in (pos + neg) if r.get("score") is not None]
        if len(set(ys)) < 2:
            continue
        auc = float(roc_auc_score(ys, sc))
        am, lo, hi = bootstrap_auc_ci(sc, ys)
        sp = [r["score"] for r in pos if r.get("score") is not None]
        sn = [r["score"] for r in neg if r.get("score") is not None]
        results[f"{cl}_effect_e87"] = {
            "n_pos": len(sp), "n_neg": len(sn),
            "auc": auc, "auc_ci_95": [lo, hi],
            "pos_mean": float(np.mean(sp)) if sp else None,
            "neg_mean": float(np.mean(sn)) if sn else None,
        }
        print(f"  {cl}  AUROC={auc:.3f} [CI {lo:.3f},{hi:.3f}]  "
              f"pos_mean={np.mean(sp):.2f} neg_mean={np.mean(sn):.2f}  "
              f"n_pos={len(sp)}/n_neg={len(sn)}")

    # Compare to E8.6 results if exist
    try:
        e86 = json.loads((OUT_DIR / "llm_oracle_results_paircond_named.json").read_text())
        print("\n[E8.7] === vs E8.6 (n=30/class) ===")
        for cl in ["PD", "PK"]:
            old = e86["results"].get(f"{cl}_effect_paircond_named", {}).get("auc")
            new = results.get(f"{cl}_effect_e87", {}).get("auc")
            if old is not None and new is not None:
                print(f"  {cl}  E8.6={old:.3f}  E8.7={new:.3f}  Δ={new-old:+.3f}")
    except Exception as ex:
        print(f"  (E8.6 results not loaded: {ex})")

    # Cost
    tin_sel = sum(r.get("tokens_in", 0) for r in sel_cache.values())
    tout_sel = sum(r.get("tokens_out", 0) for r in sel_cache.values())
    tin_r = sum(r.get("tokens_in", 0) for r in reas_cache.values())
    tout_r = sum(r.get("tokens_out", 0) for r in reas_cache.values())
    cost = (tin_sel + tin_r) * 3 / 1_000_000 + (tout_sel + tout_r) * 15 / 1_000_000
    print(f"\n[E8.7] tokens: sel in={tin_sel} out={tout_sel}; reas in={tin_r} out={tout_r}")
    print(f"[E8.7] cost~=${cost:.3f}")

    payload = {
        "model": MODEL,
        "design": "pair-conditional selector (forced one pair) + neutral reasoner",
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg},
        "results": results,
        "selector_fallback_count": fallback_ct,
        "tokens": {"sel_in": tin_sel, "sel_out": tout_sel,
                    "reas_in": tin_r, "reas_out": tout_r},
        "cost_usd_est": cost,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8.7] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
