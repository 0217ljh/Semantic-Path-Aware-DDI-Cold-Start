"""E8.7-hetero — Two-model heterogeneous oracle.

Stage 1 (Selector): Claude Opus 4.5
Stage 2 (Reasoner): GPT-5.4 (reasoning model)

Rationale: avoid same-model self-reinforcement between perception (selector)
and scoring (reasoner). Heterogeneous setup gives cross-model robustness
check on the pair-conditional perception hypothesis.

Same neutral prompts as E8.7 (from 08f_llm_oracle_paircond_named.py):
  - Stage 1 selector forces ONE pair (no NONE; trivial-fallback otherwise)
  - Stage 2 reasoner uses neutral "documented clinical effects" framing
Same sample: 60 PD-pos + 60 PK-pos + 60 NEG.

Cost estimate: ~$7-13 (Opus + GPT-5.4 reasoning is much pricier than Sonnet).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import importlib
e8 = importlib.import_module("08_llm_oracle")
e8b = importlib.import_module("08b_llm_oracle_prob")
e8f = importlib.import_module("08f_llm_oracle_paircond_named")

load_env = e8.load_env
build_features = e8.build_features
sample_pairs = e8.sample_pairs
parse_probability = e8b.parse_probability
load_cache = e8b.load_cache
append_cache = e8b.append_cache
bootstrap_auc_ci = e8b.bootstrap_auc_ci
format_pairs_for_reasoner = e8f.format_pairs_for_reasoner

# Override sample sizes (E8.7 confirmation scale)
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
SELECTOR_CACHE = OUT_DIR / "e87_hetero_selector_cache.jsonl"  # Opus 4.5
REASONER_CACHE = OUT_DIR / "e87_hetero_reasoner_cache.jsonl"  # GPT-5.4
RESULTS_FILE = OUT_DIR / "e87_hetero_results.json"

OPUS_MODEL = "claude-opus-4-6"          # latest Opus (verified in registry)
GPT_MODEL = "gpt-4o-2024-08-06"          # non-reasoning per user request
MAX_FEATURES_INPUT = 30
MAX_SEL_OUTPUT_TOKENS = 200
MAX_REASONER_OUTPUT_TOKENS = 80          # non-reasoning, small budget OK
TEMPERATURE = 0.0
GPT_IS_REASONING = False                 # gpt-4o is not a reasoning model

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------
def call_anthropic(api_key, system, user, model, max_tokens, temperature=0.0, max_retries=3):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(ANTHROPIC_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [Anthropic] HTTP {e.code}, retry in {wait}s: {err_body[:200]}")
                time.sleep(wait)
                continue
            return {"error": f"HTTP {e.code}: {err_body}"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"error": f"{type(e).__name__}: {e}"}
    return {"error": "max retries"}


def call_openai(api_key, system, user, model, max_completion_tokens=4096, max_retries=3, is_reasoning=True):
    """OpenAI chat completions. For reasoning models (gpt-5.x, o-series),
    use max_completion_tokens and DO NOT set temperature."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if is_reasoning:
        payload["max_completion_tokens"] = max_completion_tokens
        # no temperature for reasoning models
    else:
        payload["max_tokens"] = max_completion_tokens
        payload["temperature"] = TEMPERATURE
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(OPENAI_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:  # longer timeout for reasoning
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [OpenAI] HTTP {e.code}, retry in {wait}s: {err_body[:200]}")
                time.sleep(wait)
                continue
            return {"error": f"HTTP {e.code}: {err_body}"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"error": f"{type(e).__name__}: {e}"}
    return {"error": "max retries"}


_PAIR_RE = re.compile(
    r'Pair\s*\d*:\s*A_effect\s*=\s*"([^"]+)"\s*\|\s*B_effect\s*=\s*"([^"]+)"',
    flags=re.IGNORECASE,
)


def parse_with_fallback(text, eff_a, eff_b):
    if text:
        m = _PAIR_RE.search(text)
        if m:
            return [(m.group(1).strip(), m.group(2).strip())]
    a = eff_a[0] if eff_a else "(no documented effect)"
    b = eff_b[0] if eff_b else "(no documented effect)"
    return [(a, b)]


def run_selector(opus_key, pair, eff_feats, cache):
    cid = f"{pair['pair_id']}_sel_opus"
    if cid in cache:
        return cache[cid]
    eff_a = eff_feats.get(pair["drug_a_id"], [])[:MAX_FEATURES_INPUT]
    eff_b = eff_feats.get(pair["drug_b_id"], [])[:MAX_FEATURES_INPUT]
    prompt = e8f.selector_prompt_named(pair["drug_a_name"], pair["drug_b_name"], eff_a, eff_b)
    resp = call_anthropic(opus_key, e8f.SELECTOR_SYSTEM, prompt,
                          model=OPUS_MODEL, max_tokens=MAX_SEL_OUTPUT_TOKENS,
                          temperature=TEMPERATURE)
    text = ""
    err = None
    tin = tout = 0
    fallback_used = False
    if "error" in resp:
        err = resp["error"]
        sel = parse_with_fallback("", eff_a, eff_b)
        fallback_used = True
    else:
        try:
            text = resp["content"][0]["text"]
            tin = resp.get("usage", {}).get("input_tokens", 0)
            tout = resp.get("usage", {}).get("output_tokens", 0)
        except (KeyError, IndexError, TypeError) as e:
            err = f"parse: {e}"
        sel = parse_with_fallback(text, eff_a, eff_b)
        fallback_used = not bool(_PAIR_RE.search(text or ""))
    rec = {"call_id": cid, "pair_id": pair["pair_id"],
           "model": OPUS_MODEL,
           "drug_a_name": pair["drug_a_name"], "drug_b_name": pair["drug_b_name"],
           "eff_a_input": eff_a, "eff_b_input": eff_b,
           "selected_pairs": sel, "raw_text": text, "fallback_used": fallback_used,
           "tokens_in": tin, "tokens_out": tout, "error": err}
    append_cache(SELECTOR_CACHE, rec)
    cache[cid] = rec
    return rec


def run_reasoner(openai_key, pair, selected_pairs, cache):
    cid = f"{pair['pair_id']}_reas_gpt54"
    if cid in cache:
        return cache[cid]
    pairs_text = format_pairs_for_reasoner(selected_pairs)
    user_prompt = e8f.reasoner_prompt_named(pair["drug_a_name"], pair["drug_b_name"], pairs_text)
    resp = call_openai(openai_key, e8f.REASONER_SYSTEM, user_prompt,
                       model=GPT_MODEL, max_completion_tokens=MAX_REASONER_OUTPUT_TOKENS,
                       is_reasoning=GPT_IS_REASONING)
    score = None
    text = ""
    tin = tout = 0
    err = None
    if "error" in resp:
        err = resp["error"]
    else:
        try:
            text = resp["choices"][0]["message"]["content"] or ""
            score = parse_probability(text)
            tin = resp.get("usage", {}).get("prompt_tokens", 0)
            tout = resp.get("usage", {}).get("completion_tokens", 0)
        except (KeyError, IndexError, TypeError) as e:
            text = json.dumps(resp)[:300]
            err = f"parse: {e}"
    rec = {"call_id": cid, "pair_id": pair["pair_id"],
           "model": GPT_MODEL,
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
    opus_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY")
    openai_key = env.get("OPENAI_KEY") or env.get("OPENAI_API_KEY")
    if not opus_key or not openai_key:
        print(f"[E8.7-hetero] FATAL: missing key(s). Keys present: {list(env.keys())}")
        return
    print(f"[E8.7-hetero] loaded keys (Claude={len(opus_key)}, OpenAI={len(openai_key)})")
    print(f"[E8.7-hetero] Stage 1 model: {OPUS_MODEL}")
    print(f"[E8.7-hetero] Stage 2 model: {GPT_MODEL}")

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
    print(f"[E8.7-hetero] PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}  total={len(pairs)}")

    # ============ Stage 1 — Opus 4.5 selector ============
    sel_cache = load_cache(SELECTOR_CACHE)
    sel_cache = {r["call_id"]: r for r in sel_cache.values()} if sel_cache else {}
    print(f"\n[Stage 1] Opus 4.5 selector. Cache: {len(sel_cache)}")
    t0 = time.time()
    done = 0
    fb_ct = 0
    err_ct = 0
    for pair in pairs:
        done += 1
        rec = run_selector(opus_key, pair, eff_feats, sel_cache)
        if rec.get("error"):
            err_ct += 1
        if rec.get("fallback_used"):
            fb_ct += 1
        if done % 20 == 0 or done == 1:
            el = time.time() - t0
            eta = el / done * (len(pairs) - done)
            print(f"  [{done}/{len(pairs)}] {pair['pair_id'][:35]} "
                  f"sel={rec['selected_pairs']} fb={rec.get('fallback_used')} "
                  f"err={'Y' if rec.get('error') else 'N'} "
                  f"elapsed={el:.0f}s ETA={eta:.0f}s")
    print(f"[Stage 1] fallback used: {fb_ct}/{len(pairs)}; errors: {err_ct}")

    # ============ Stage 2 — GPT-5.4 reasoner ============
    reas_cache = load_cache(REASONER_CACHE)
    reas_cache = {r["call_id"]: r for r in reas_cache.values()} if reas_cache else {}
    print(f"\n[Stage 2] GPT-5.4 reasoner. Cache: {len(reas_cache)}")
    t1 = time.time()
    done = 0
    err2_ct = 0
    for pair in pairs:
        done += 1
        sel_rec = sel_cache[f"{pair['pair_id']}_sel_opus"]
        rec = run_reasoner(openai_key, pair, sel_rec["selected_pairs"], reas_cache)
        if rec.get("error"):
            err2_ct += 1
        if done % 20 == 0 or done == 1:
            el = time.time() - t1
            eta = el / done * (len(pairs) - done)
            print(f"  [{done}/{len(pairs)}] {pair['pair_id'][:35]} "
                  f"score={rec.get('score')} err={'Y' if rec.get('error') else 'N'} "
                  f"elapsed={el:.0f}s ETA={eta:.0f}s")
    print(f"[Stage 2] errors: {err2_ct}")

    # ============ AUROC ============
    print("\n[E8.7-hetero] === AUROC ===")
    pid_to_pair = {p["pair_id"]: p for p in pairs}
    results = {}
    for cl in ["PD", "PK"]:
        pos = [reas_cache[f"{p['pair_id']}_reas_gpt54"] for p in pairs
               if p["mech_class"] == cl
               and f"{p['pair_id']}_reas_gpt54" in reas_cache]
        neg = [reas_cache[f"{p['pair_id']}_reas_gpt54"] for p in pairs
               if p["mech_class"] == "NEG"
               and f"{p['pair_id']}_reas_gpt54" in reas_cache]
        sc = [r["score"] for r in (pos + neg) if r.get("score") is not None]
        ys = [pid_to_pair[r["pair_id"]].get("label", 0)
              for r in (pos + neg) if r.get("score") is not None]
        if len(set(ys)) < 2:
            continue
        auc = float(roc_auc_score(ys, sc))
        am, lo, hi = bootstrap_auc_ci(sc, ys)
        sp = [r["score"] for r in pos if r.get("score") is not None]
        sn = [r["score"] for r in neg if r.get("score") is not None]
        results[f"{cl}_effect_hetero"] = {
            "n_pos": len(sp), "n_neg": len(sn),
            "auc": auc, "auc_ci_95": [lo, hi],
            "pos_mean": float(np.mean(sp)) if sp else None,
            "neg_mean": float(np.mean(sn)) if sn else None,
        }
        print(f"  {cl}  AUROC={auc:.3f} [CI {lo:.3f},{hi:.3f}]  "
              f"pos_mean={np.mean(sp):.2f} neg_mean={np.mean(sn):.2f}  "
              f"n={len(sp)}/{len(sn)}")

    # Cost
    # Anthropic: claude-opus-4-5 ~ $15/M input, $75/M output (approx)
    tin_s = sum(r.get("tokens_in", 0) for r in sel_cache.values())
    tout_s = sum(r.get("tokens_out", 0) for r in sel_cache.values())
    cost_s = tin_s * 15 / 1_000_000 + tout_s * 75 / 1_000_000
    # OpenAI: gpt-4o ~ $2.5/M input, $10/M output
    tin_r = sum(r.get("tokens_in", 0) for r in reas_cache.values())
    tout_r = sum(r.get("tokens_out", 0) for r in reas_cache.values())
    cost_r = tin_r * 2.5 / 1_000_000 + tout_r * 10 / 1_000_000
    print(f"\n[Cost] Opus sel: in={tin_s} out={tout_s} ~${cost_s:.2f}")
    print(f"[Cost] GPT-5.4 reas: in={tin_r} out={tout_r} ~${cost_r:.2f}")
    print(f"[Cost] Total ~${cost_s + cost_r:.2f}")

    # Compare to E8.6 single-Sonnet baseline
    try:
        e86 = json.loads((OUT_DIR / "llm_oracle_results_paircond_named.json").read_text())
        print("\n[E8.7-hetero] === vs E8.6 single-Sonnet (n=30/cl) ===")
        for cl in ["PD", "PK"]:
            old = e86["results"].get(f"{cl}_effect_paircond_named", {}).get("auc")
            new = results.get(f"{cl}_effect_hetero", {}).get("auc")
            if old is not None and new is not None:
                print(f"  {cl}  E8.6 Sonnet single = {old:.3f}  "
                      f"E8.7-hetero (Opus+GPT5.4) = {new:.3f}  Δ={new-old:+.3f}")
    except Exception as ex:
        print(f"  (E8.6 not loaded: {ex})")

    payload = {
        "design": "Stage 1 Opus 4.5 (selector) + Stage 2 GPT-5.4 (reasoner)",
        "models": {"stage1": OPUS_MODEL, "stage2": GPT_MODEL},
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg},
        "results": results,
        "selector_fallback_count": fb_ct,
        "selector_errors": err_ct,
        "reasoner_errors": err2_ct,
        "tokens": {"sel_in": tin_s, "sel_out": tout_s,
                    "reas_in": tin_r, "reas_out": tout_r},
        "cost_usd_est": {"selector": cost_s, "reasoner": cost_r, "total": cost_s + cost_r},
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8.7-hetero] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
