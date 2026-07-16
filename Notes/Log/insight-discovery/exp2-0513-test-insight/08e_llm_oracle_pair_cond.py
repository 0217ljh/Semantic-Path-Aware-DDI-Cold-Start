"""E8.5 — Pair-conditional two-stage LLM oracle (user-corrected design).

E8.4 (per-drug selector) FAILED because it picked "core effects" for each
drug in isolation, losing pair-relevant signals. The user's original example
("drug A causes respiratory depression × drug B accelerates metabolism →
worse respiratory failure") requires PAIR-CONDITIONAL perception: the
selector must see BOTH drugs' effects simultaneously and identify composable
effect-pairs.

DESIGN (effect modality only — focused PD test):
  Stage 1 (Pair-conditional COMPOSITION selector):
    Input: drug A's full effect list + drug B's full effect list (anonymized)
    Task: identify 1-2 pairs of effects (one from each drug) that could
          compose into a clinically significant DDI risk.
    Output: structured pair selections.

  Stage 2 (Reasoner over selected composition pairs):
    Input: the 1-2 selected effect pairs only.
    Task: estimate probability of DDI based on the composition.
    Output: probability 0-1.

If E8.5 PD-effect AUROC > E8.3 PD-effect (0.686) → pair-conditional perception
is the missing ingredient → strong evidence for relational selection.
If ≤ → bag aggregation already captures all available LLM signal.

Same 90 pairs as previous experiments (seed=42).
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
SELECTOR_CACHE = OUT_DIR / "llm_oracle_pair_cond_selector.jsonl"
REASONER_CACHE = OUT_DIR / "llm_oracle_pair_cond_reasoner.jsonl"
RESULTS_FILE = OUT_DIR / "llm_oracle_results_pair_cond.json"

MODEL = "claude-sonnet-4-5-20250929"
MAX_FEATURES_INPUT = 30
MAX_SEL_OUTPUT_TOKENS = 200
MAX_REASONER_OUTPUT_TOKENS = 80
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Stage 1: PAIR-CONDITIONAL COMPOSITION SELECTOR
# ---------------------------------------------------------------------------
SELECTOR_SYSTEM = (
    "You are a clinical pharmacology assistant. You will see two drugs' "
    "documented effect lists (identities hidden). Your task: identify pairs "
    "of effects (one from each drug) that could COMPOSE into a clinically "
    "significant pharmacodynamic drug-drug interaction risk via causal pathway "
    "interaction. Examples of composition: shared physiological endpoint "
    "(both increase bleeding risk); downstream cascade (one accelerates the "
    "other's effect); shared organ-system risk (both prolong QT).\n\n"
    "Output 1 or 2 composition pairs, ONE PER LINE, in EXACTLY this format:\n"
    "Pair1: A_effect=\"<effect from drug A>\" | B_effect=\"<effect from drug B>\"\n"
    "If no plausible composition exists, output a single line: NONE\n"
    "Output nothing else."
)


def selector_prompt(eff_a: list[str], eff_b: list[str]) -> str:
    return (
        f"Drug A (identity hidden) — documented side effects / phenotypes:\n"
        f"{', '.join(eff_a) if eff_a else '(none documented)'}\n\n"
        f"Drug B (identity hidden) — documented side effects / phenotypes:\n"
        f"{', '.join(eff_b) if eff_b else '(none documented)'}\n\n"
        f"Identify 1 or 2 pairs of effects (one from A, one from B) that could "
        f"compose into a clinically significant pharmacodynamic DDI risk. "
        f"Prefer one pair; output a second pair only if it is mechanistically "
        f"important and independent of the first."
    )


# ---------------------------------------------------------------------------
# Stage 2: REASONER over selected composition pairs
# ---------------------------------------------------------------------------
REASONER_SYSTEM = (
    "You are a clinical pharmacology assistant. Two drugs (identities hidden) "
    "have been analyzed and 1-2 effect-pairs were identified as the most "
    "clinically composable points of interaction risk. Based ONLY on these "
    "selected effect-pairs, estimate the probability of a clinically "
    "significant drug-drug interaction. Reply in EXACTLY: line 1 a single "
    "number 0-1; line 2 a one-sentence reason. Output nothing else."
)


def reasoner_prompt(pairs_text: str) -> str:
    return (
        f"Selected composition pairs:\n{pairs_text}\n\n"
        f"Estimate the probability (0 to 1) that these two drugs have a "
        f"clinically relevant pharmacodynamic drug-drug interaction based on "
        f"the composition of these effect pairs."
    )


# Parse selector output: lines like 'Pair1: A_effect="..." | B_effect="..."'
_PAIR_LINE_RE = re.compile(
    r'Pair\s*\d*:\s*A_effect\s*=\s*"([^"]+)"\s*\|\s*B_effect\s*=\s*"([^"]+)"',
    flags=re.IGNORECASE,
)


def parse_selector_output(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    t = text.strip()
    if t.upper().startswith("NONE"):
        return []
    pairs = []
    for m in _PAIR_LINE_RE.finditer(t):
        pairs.append((m.group(1).strip(), m.group(2).strip()))
    # Fallback: look for any line with "A:" and "B:" pattern if regex above missed
    if not pairs:
        for ln in t.split("\n"):
            if "|" in ln and ("A" in ln.upper() and "B" in ln.upper()):
                parts = ln.split("|", 1)
                a = re.sub(r".*A[^:=]*[:=]\s*\"?", "", parts[0]).strip(' "')
                b = re.sub(r".*B[^:=]*[:=]\s*\"?", "", parts[1]).strip(' "')
                if a and b:
                    pairs.append((a, b))
    return pairs[:2]


def format_pairs_for_reasoner(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "(no composable pairs identified)"
    lines = []
    for i, (a, b) in enumerate(pairs, 1):
        lines.append(f"  Pair {i}: drug A has \"{a}\"; drug B has \"{b}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def run_selector(api_key: str, pair: dict, eff_feats: dict, cache: dict) -> dict:
    call_id = f"{pair['pair_id']}_pair_sel"
    if call_id in cache:
        return cache[call_id]
    eff_a = eff_feats.get(pair["drug_a_id"], [])[:MAX_FEATURES_INPUT]
    eff_b = eff_feats.get(pair["drug_b_id"], [])[:MAX_FEATURES_INPUT]
    prompt = selector_prompt(eff_a, eff_b)
    resp = call_anthropic(api_key, SELECTOR_SYSTEM, prompt,
                          model=MODEL, max_tokens=MAX_SEL_OUTPUT_TOKENS,
                          temperature=TEMPERATURE)
    if "error" in resp:
        rec = {"call_id": call_id, "pair_id": pair["pair_id"],
               "eff_a_input": eff_a, "eff_b_input": eff_b,
               "selected_pairs": [], "raw_text": "", "error": resp["error"],
               "tokens_in": 0, "tokens_out": 0}
    else:
        try:
            text = resp["content"][0]["text"]
            sel = parse_selector_output(text)
            rec = {"call_id": call_id, "pair_id": pair["pair_id"],
                   "eff_a_input": eff_a, "eff_b_input": eff_b,
                   "selected_pairs": sel, "raw_text": text, "error": None,
                   "tokens_in": resp.get("usage", {}).get("input_tokens", 0),
                   "tokens_out": resp.get("usage", {}).get("output_tokens", 0)}
        except (KeyError, IndexError, TypeError) as e:
            rec = {"call_id": call_id, "pair_id": pair["pair_id"],
                   "eff_a_input": eff_a, "eff_b_input": eff_b,
                   "selected_pairs": [], "raw_text": json.dumps(resp)[:300],
                   "error": f"parse: {e}", "tokens_in": 0, "tokens_out": 0}
    append_cache(SELECTOR_CACHE, rec)
    cache[call_id] = rec
    return rec


def run_reasoner(api_key: str, pair: dict, selected_pairs: list, cache: dict) -> dict:
    call_id = f"{pair['pair_id']}_pair_reas"
    if call_id in cache:
        return cache[call_id]
    pairs_text = format_pairs_for_reasoner(selected_pairs)
    prompt = reasoner_prompt(pairs_text)
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
           "mech_class": pair["mech_class"], "label": pair["label"],
           "selected_pairs": selected_pairs,
           "response_text": text, "score": score,
           "tokens_in": tin, "tokens_out": tout, "error": err}
    append_cache(REASONER_CACHE, rec)
    cache[call_id] = rec
    return rec


# ---------------------------------------------------------------------------
def main():
    env = load_env(ENV_FILE)
    api_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[E8.5] FATAL: no Claude key")
        return
    print(f"[E8.5] loaded key (len {len(api_key)})")

    print("[E8.5] loading KG ...")
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
    print(f"[E8.5] PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}")

    # Preview a single selector prompt
    sample_p = next(p for p in pairs if p["mech_class"] == "PD")
    print(f"\n[E8.5] === SELECTOR PROMPT PREVIEW (first PD-pos) ===\n")
    print(selector_prompt(
        eff_feats.get(sample_p["drug_a_id"], [])[:MAX_FEATURES_INPUT],
        eff_feats.get(sample_p["drug_b_id"], [])[:MAX_FEATURES_INPUT]
    ))
    print("[E8.5] === END PREVIEW ===\n")

    sel_cache = load_cache(SELECTOR_CACHE)
    sel_cache = {r["call_id"]: r for r in sel_cache.values()} if sel_cache else {}
    print(f"[E8.5] selector cache: {len(sel_cache)}")

    print("[E8.5] === Stage 1: pair-conditional COMPOSITION selector ===")
    t0 = time.time()
    done = 0
    for pair in pairs:
        done += 1
        rec = run_selector(api_key, pair, eff_feats, sel_cache)
        if done % 20 == 0 or done == 1:
            el = time.time() - t0
            eta = el / done * (len(pairs) - done)
            sel = rec["selected_pairs"]
            print(f"  [{done}/{len(pairs)}]  pair={pair['pair_id'][:35]}  "
                  f"#sel={len(sel)}  elapsed={el:.0f}s ETA={eta:.0f}s")

    # Preview selector outputs
    print("\n[E8.5] === SELECTOR OUTPUTS (first 3 PD-pos) ===")
    cnt = 0
    for p in pairs:
        if p["mech_class"] != "PD":
            continue
        if cnt >= 3:
            break
        rec = sel_cache.get(f"{p['pair_id']}_pair_sel")
        if not rec:
            continue
        print(f"\n  [{p['pair_id']}]  {id2name.get(p['drug_a_id'])} + {id2name.get(p['drug_b_id'])}")
        print(f"  Raw LLM output:")
        for ln in rec["raw_text"].split("\n"):
            print(f"    {ln}")
        print(f"  Parsed pairs: {rec['selected_pairs']}")
        cnt += 1

    # Stage 2
    reas_cache = load_cache(REASONER_CACHE)
    reas_cache = {r["call_id"]: r for r in reas_cache.values()} if reas_cache else {}
    print(f"\n[E8.5] reasoner cache: {len(reas_cache)}")

    print("[E8.5] === Stage 2: reasoner over composition pairs ===")
    t1 = time.time()
    done = 0
    for pair in pairs:
        done += 1
        sel_rec = sel_cache[f"{pair['pair_id']}_pair_sel"]
        rec = run_reasoner(api_key, pair, sel_rec["selected_pairs"], reas_cache)
        if done % 20 == 0 or done == 1:
            el = time.time() - t1
            eta = el / done * (len(pairs) - done)
            print(f"  [{done}/{len(pairs)}]  pair={pair['pair_id'][:35]}  "
                  f"score={rec.get('score')}  elapsed={el:.0f}s ETA={eta:.0f}s")

    # AUROC
    print("\n[E8.5] === AUROC (pair-conditional two-stage, effect modality) ===")
    pid_to_pair = {p["pair_id"]: p for p in pairs}
    results = {}
    for cl in ["PD", "PK"]:
        pos = [reas_cache[f"{p['pair_id']}_pair_reas"] for p in pairs
               if p["mech_class"] == cl
               and f"{p['pair_id']}_pair_reas" in reas_cache]
        neg = [reas_cache[f"{p['pair_id']}_pair_reas"] for p in pairs
               if p["mech_class"] == "NEG"
               and f"{p['pair_id']}_pair_reas" in reas_cache]
        sc = [r["score"] for r in (pos + neg) if r.get("score") is not None]
        ys = [pid_to_pair[r["pair_id"]].get("label", 0)
              for r in (pos + neg) if r.get("score") is not None]
        if len(set(ys)) < 2:
            continue
        auc = float(roc_auc_score(ys, sc))
        am, lo, hi = bootstrap_auc_ci(sc, ys)
        sp = [r["score"] for r in pos if r.get("score") is not None]
        sn = [r["score"] for r in neg if r.get("score") is not None]
        results[f"{cl}_effect_paircond"] = {
            "n_pos": len(sp), "n_neg": len(sn),
            "auc": auc, "auc_ci_95": [lo, hi],
            "pos_mean": float(np.mean(sp)) if sp else None,
            "neg_mean": float(np.mean(sn)) if sn else None,
        }
        print(f"  {cl}  pair-conditional effect  AUROC={auc:.3f} "
              f"[CI {lo:.3f},{hi:.3f}]  pos_mean={np.mean(sp):.2f} "
              f"neg_mean={np.mean(sn):.2f}  n_pos={len(sp)}/n_neg={len(sn)}")

    # Compare to E8.3 single-stage anonymized effect baseline
    try:
        e8c_res = json.loads((OUT_DIR / "llm_oracle_results_anon.json").read_text())
        print("\n[E8.5] === Pair-conditional vs Single-stage anonymized (E8.3 baseline) ===")
        for cl in ["PD", "PK"]:
            old = e8c_res["results_by_class_variant"].get(f"{cl}_C_anon_effect", {}).get("auc")
            new = results.get(f"{cl}_effect_paircond", {}).get("auc")
            if old is not None and new is not None:
                print(f"  {cl}  single-anon={old:.3f}  pair-cond-two-stage={new:.3f}  Δ={new-old:+.3f}")
    except Exception as ex:
        print(f"  (could not load E8.3 results: {ex})")

    # Selection stats
    n_sel = [len(r["selected_pairs"]) for r in sel_cache.values()]
    n_zero = sum(1 for n in n_sel if n == 0)
    print(f"\n[E8.5] avg selected pairs per drug-pair: {np.mean(n_sel):.2f}  "
          f"(0-sel: {n_zero}, 1-sel: {sum(1 for n in n_sel if n == 1)}, "
          f"2-sel: {sum(1 for n in n_sel if n == 2)})")

    tin_sel = sum(r.get("tokens_in", 0) for r in sel_cache.values())
    tout_sel = sum(r.get("tokens_out", 0) for r in sel_cache.values())
    tin_reas = sum(r.get("tokens_in", 0) for r in reas_cache.values())
    tout_reas = sum(r.get("tokens_out", 0) for r in reas_cache.values())
    cost = (tin_sel + tin_reas) * 3 / 1_000_000 + (tout_sel + tout_reas) * 15 / 1_000_000
    print(f"[E8.5] tokens: selector in={tin_sel} out={tout_sel}; reasoner in={tin_reas} out={tout_reas}")
    print(f"[E8.5] cost~=${cost:.3f}")

    payload = {
        "model": MODEL,
        "design": "pair-conditional composition selector → reasoner",
        "modality": "effect only",
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg},
        "results": results,
        "selection_stats": {
            "mean_pairs_per_drug_pair": float(np.mean(n_sel)),
            "zero_sel": int(n_zero),
            "one_sel": int(sum(1 for n in n_sel if n == 1)),
            "two_sel": int(sum(1 for n in n_sel if n == 2)),
        },
        "tokens": {"selector_in": tin_sel, "selector_out": tout_sel,
                    "reasoner_in": tin_reas, "reasoner_out": tout_reas},
        "cost_usd_est": cost,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8.5] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
