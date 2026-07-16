"""E8.4 — Two-stage LLM oracle (codex round-4 fix per user methodological critique).

User observation: E8.2/E8.3 stuffed 20 SEs per drug into the prompt, which is
structurally identical to GNN pooling over a noisy neighborhood — the failure
mode the paper criticizes. To validate "effect signal IS meaningful for PD,
but requires semantic SELECTION", split the oracle into two stages:

  Stage 1 (SELECTOR, name-blind): "From this drug's documented effects/molecular
    features (drug identity HIDDEN), select 1-3 clinically significant primary
    items. Prefer fewer."
  Stage 2 (REASONER, anonymized): "Drug A core: {sel_A}. Drug B core: {sel_B}.
    Could these core items compose into a DDI risk via causal pathway
    interaction?" → probability 0-1.

If two-stage AUROC > one-stage AUROC for PD-effect (E8.2 baseline 0.652) →
SELECTION is the missing ingredient → directly motivates perception-first.

Same 90 pairs as E8.2 (deterministic seed=42).
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
SELECTOR_CACHE = OUT_DIR / "llm_oracle_selector_cache.jsonl"  # per-drug-modality
REASONER_CACHE = OUT_DIR / "llm_oracle_reasoner_cache.jsonl"  # per-pair-modality
RESULTS_FILE = OUT_DIR / "llm_oracle_results_twostage.json"

MODEL = "claude-sonnet-4-5-20250929"
MAX_FEATURES_INPUT = 30
MAX_SEL_OUTPUT_TOKENS = 60
MAX_REASONER_OUTPUT_TOKENS = 80
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Stage-1 SELECTOR prompts (name-blind)
# ---------------------------------------------------------------------------
SELECTOR_SYSTEM = (
    "You are a clinical pharmacology assistant. You will review a drug's "
    "documented features and select the most clinically significant primary "
    "items. The drug's identity is hidden — judge ONLY from the listed "
    "features. Output the selected items as a single comma-separated line "
    "(no quotes, no enumeration). Output nothing else."
)


def selector_prompt_effect(features: list[str]) -> str:
    feat_text = ", ".join(features) if features else "(none documented)"
    return (
        f"A drug (identity hidden) has the following documented side-effect / "
        f"phenotype list:\n{feat_text}\n\n"
        f"Select 1 to 3 effects that you judge most likely to represent "
        f"primary clinical mechanisms (i.e., effects that are major mechanisms "
        f"of clinical concern, NOT minor / generic / nonspecific side effects). "
        f"Prefer fewer; include a third only if mechanistically important.\n\n"
        f"Output ONLY the selected effect names, comma-separated, on one line."
    )


def selector_prompt_molecular(features: list[str]) -> str:
    feat_text = ", ".join(features) if features else "(none documented)"
    return (
        f"A drug (identity hidden) has the following documented molecular "
        f"targets / enzymes / transporters:\n{feat_text}\n\n"
        f"Select 1 to 3 items that you judge most likely to be primary "
        f"clinically significant interaction partners (e.g., enzymes/transporters "
        f"that mediate canonical drug-drug interactions, NOT minor or off-target "
        f"associations). Prefer fewer; include a third only if mechanistically "
        f"important.\n\n"
        f"Output ONLY the selected names, comma-separated, on one line."
    )


# ---------------------------------------------------------------------------
# Stage-2 REASONER prompts (anonymized, sees ONLY core selections)
# ---------------------------------------------------------------------------
REASONER_SYSTEM = (
    "You are a clinical pharmacology assistant. Each drug is represented only "
    "by 1-3 clinically significant core items previously selected from its "
    "documented features. Drug identities are hidden. Based ONLY on these "
    "core items, estimate the probability of a clinically relevant drug-drug "
    "interaction. Reply in EXACTLY this format: line 1 a single number in "
    "[0,1]; line 2 a one-sentence reason. Output nothing else."
)


def reasoner_prompt_effect(core_a: list[str], core_b: list[str]) -> str:
    return (
        f"Drug A's core effects (selected from documented side effects/phenotypes):\n"
        f"{', '.join(core_a) if core_a else '(none selected)'}\n\n"
        f"Drug B's core effects (selected from documented side effects/phenotypes):\n"
        f"{', '.join(core_b) if core_b else '(none selected)'}\n\n"
        f"Could these two drugs' core effects compose into a clinically "
        f"significant pharmacodynamic interaction risk via causal pathway "
        f"interaction (e.g., one drug amplifies the other's risk through a "
        f"shared physiological endpoint, additive toxicity on a shared organ "
        f"system, or downstream-effect cascade)?\n\n"
        f"Estimate probability 0 to 1."
    )


def reasoner_prompt_molecular(core_a: list[str], core_b: list[str]) -> str:
    return (
        f"Drug A's core molecular interaction partners (selected from documented "
        f"targets/enzymes/transporters):\n"
        f"{', '.join(core_a) if core_a else '(none selected)'}\n\n"
        f"Drug B's core molecular interaction partners (selected from documented "
        f"targets/enzymes/transporters):\n"
        f"{', '.join(core_b) if core_b else '(none selected)'}\n\n"
        f"Could these two drugs' core molecular partners compose into a "
        f"clinically significant pharmacokinetic interaction risk (e.g., shared "
        f"CYP isoform substrate/inhibitor pair, transporter competition, "
        f"metabolic enzyme overlap leading to plasma-concentration change)?\n\n"
        f"Estimate probability 0 to 1."
    )


# ---------------------------------------------------------------------------
def select_features(api_key: str, drug_id: str, modality: str, features: list[str], cache: dict) -> tuple[list[str], dict]:
    """Stage-1 selection per (drug, modality). Cached."""
    key = f"{drug_id}_{modality}"
    if key in cache:
        sel = cache[key].get("selected", [])
        return sel, cache[key]
    feats = features[:MAX_FEATURES_INPUT]
    if modality == "effect":
        prompt = selector_prompt_effect(feats)
    elif modality == "molecular":
        prompt = selector_prompt_molecular(feats)
    else:
        raise ValueError(modality)
    resp = call_anthropic(api_key, SELECTOR_SYSTEM, prompt,
                           model=MODEL, max_tokens=MAX_SEL_OUTPUT_TOKENS,
                           temperature=TEMPERATURE)
    if "error" in resp:
        rec = {"drug_id": drug_id, "modality": modality, "input": feats,
               "selected": [], "raw_text": "", "error": resp["error"],
               "tokens_in": 0, "tokens_out": 0}
    else:
        try:
            text = resp["content"][0]["text"].strip()
            # parse first non-empty line, split by comma
            line = next((ln for ln in text.split("\n") if ln.strip()), "")
            sel_raw = [s.strip().strip(".,;:") for s in line.split(",") if s.strip()]
            # cap to 3 selections regardless of LLM output count
            sel = sel_raw[:3]
            rec = {"drug_id": drug_id, "modality": modality, "input": feats,
                   "selected": sel, "raw_text": text, "error": None,
                   "tokens_in": resp.get("usage", {}).get("input_tokens", 0),
                   "tokens_out": resp.get("usage", {}).get("output_tokens", 0)}
        except (KeyError, IndexError, TypeError) as e:
            rec = {"drug_id": drug_id, "modality": modality, "input": feats,
                   "selected": [], "raw_text": json.dumps(resp)[:300],
                   "error": f"parse: {e}", "tokens_in": 0, "tokens_out": 0}
    append_cache(SELECTOR_CACHE, {**rec, "call_id": key})
    cache[key] = {**rec, "call_id": key}
    return rec["selected"], rec


def reason_pair(api_key: str, pair_id: str, modality: str, core_a: list[str], core_b: list[str], cache: dict) -> dict:
    key = f"{pair_id}_{modality}_two"
    if key in cache:
        return cache[key]
    if modality == "effect":
        prompt = reasoner_prompt_effect(core_a, core_b)
    elif modality == "molecular":
        prompt = reasoner_prompt_molecular(core_a, core_b)
    else:
        raise ValueError(modality)
    resp = call_anthropic(api_key, REASONER_SYSTEM, prompt,
                           model=MODEL, max_tokens=MAX_REASONER_OUTPUT_TOKENS,
                           temperature=TEMPERATURE)
    score = None
    text = ""
    err = None
    tin = tout = 0
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
    rec = {"call_id": key, "pair_id": pair_id, "modality": modality,
           "core_a": core_a, "core_b": core_b,
           "response_text": text, "score": score,
           "tokens_in": tin, "tokens_out": tout, "error": err}
    append_cache(REASONER_CACHE, rec)
    cache[key] = rec
    return rec


# ---------------------------------------------------------------------------
def main():
    env = load_env(ENV_FILE)
    api_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[E8.4] FATAL: no Claude key")
        return
    print(f"[E8.4] loaded key (len {len(api_key)})")

    print("[E8.4] loading KG ...")
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
    print(f"[E8.4] PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}")

    unique_drugs = set()
    for p in pairs:
        unique_drugs.add(p["drug_a_id"])
        unique_drugs.add(p["drug_b_id"])
    print(f"[E8.4] unique drugs in sample: {len(unique_drugs)}")

    # --- Stage 1: SELECTOR (name-blind) per (drug, modality) ---
    sel_cache = load_cache(SELECTOR_CACHE)
    # rewrite the cache_id key
    sel_cache = {rec.get("call_id", f"{rec['drug_id']}_{rec['modality']}"): rec
                 for rec in sel_cache.values()} if sel_cache else {}
    print(f"[E8.4] selector cache hits: {len(sel_cache)}")

    print("[E8.4] === Stage 1: running selector (name-blind) ===")
    sel_results: dict = {}  # (drug_id, modality) → selected list
    t0 = time.time()
    done = 0
    for drug_id in sorted(unique_drugs):
        for modality, feats in [("effect", eff_feats.get(drug_id, [])),
                                ("molecular", mol_feats.get(drug_id, []))]:
            done += 1
            if not feats:
                sel_results[(drug_id, modality)] = []
                continue
            sel, rec = select_features(api_key, drug_id, modality, feats, sel_cache)
            sel_results[(drug_id, modality)] = sel
            if done % 50 == 0 or done == 1:
                el = time.time() - t0
                total = 2 * len(unique_drugs)
                eta = el / done * (total - done)
                print(f"  [{done}/{total}]  drug={drug_id} mod={modality} sel={sel}  elapsed={el:.0f}s ETA={eta:.0f}s")

    # Preview a few selections
    print("\n[E8.4] === SELECTOR PREVIEW (first 3 PD-pos drugs) ===")
    seen = set()
    cnt = 0
    for p in pairs:
        if p["mech_class"] != "PD":
            continue
        if cnt >= 3:
            break
        for d in (p["drug_a_id"], p["drug_b_id"]):
            if d in seen:
                continue
            seen.add(d)
            print(f"  {id2name.get(d, d)}:")
            print(f"    eff core: {sel_results.get((d, 'effect'), [])}")
            print(f"    mol core: {sel_results.get((d, 'molecular'), [])}")
            cnt += 1

    # --- Stage 2: REASONER per (pair, modality) ---
    reas_cache = load_cache(REASONER_CACHE)
    reas_cache = {rec["call_id"]: rec for rec in reas_cache.values()} if reas_cache else {}
    print(f"\n[E8.4] reasoner cache hits: {len(reas_cache)}")

    print("[E8.4] === Stage 2: running reasoner ===")
    t1 = time.time()
    done = 0
    total_r = len(pairs) * 2
    for pair in pairs:
        for modality in ["effect", "molecular"]:
            done += 1
            ca = sel_results.get((pair["drug_a_id"], modality), [])
            cb = sel_results.get((pair["drug_b_id"], modality), [])
            rec = reason_pair(api_key, pair["pair_id"], modality, ca, cb, reas_cache)
            # tag with pair meta
            rec["mech_class"] = pair["mech_class"]
            rec["label"] = pair["label"]
            if done % 30 == 0 or done == 1:
                el = time.time() - t1
                eta = el / done * (total_r - done)
                print(f"  [{done}/{total_r}]  pair={pair['pair_id'][:35]} mod={modality}  "
                      f"score={rec.get('score')}  elapsed={el:.0f}s ETA={eta:.0f}s")

    # --- AUROC by (class, modality) for two-stage; compare to E8.2 single-stage ---
    print("\n[E8.4] === Two-stage AUROC ===")
    pid_to_pair = {p["pair_id"]: p for p in pairs}
    results = {}
    for cl in ["PD", "PK"]:
        for modality in ["effect", "molecular"]:
            pos = [reas_cache[f"{p['pair_id']}_{modality}_two"] for p in pairs
                   if p["mech_class"] == cl
                   and f"{p['pair_id']}_{modality}_two" in reas_cache]
            neg = [reas_cache[f"{p['pair_id']}_{modality}_two"] for p in pairs
                   if p["mech_class"] == "NEG"
                   and f"{p['pair_id']}_{modality}_two" in reas_cache]
            sc = [r["score"] for r in (pos + neg) if r.get("score") is not None]
            ys = [(pid_to_pair[r["pair_id"]].get("label", 0)) for r in (pos + neg) if r.get("score") is not None]
            if len(set(ys)) < 2:
                continue
            auc = float(roc_auc_score(ys, sc))
            am, lo, hi = bootstrap_auc_ci(sc, ys)
            sp = [r["score"] for r in pos if r.get("score") is not None]
            sn = [r["score"] for r in neg if r.get("score") is not None]
            results[f"{cl}_{modality}_two"] = {
                "n_pos": len(sp), "n_neg": len(sn),
                "auc": auc, "auc_ci_95": [lo, hi],
                "pos_mean": float(np.mean(sp)) if sp else None,
                "neg_mean": float(np.mean(sn)) if sn else None,
            }
            print(f"  {cl}  {modality:10s} two-stage  AUROC={auc:.3f} "
                  f"[CI {lo:.3f},{hi:.3f}]  pos_mean={np.mean(sp):.2f} neg_mean={np.mean(sn):.2f}  "
                  f"n_pos={len(sp)}/n_neg={len(sn)}")

    # Paired ΔAUROC two-stage effect vs molecular per class
    print("\n[E8.4] === Paired Δ within class (effect_two - molecular_two) ===")
    diffs = {}
    for cl in ["PD", "PK"]:
        ys, se, sm = [], [], []
        for p in pairs:
            if p["mech_class"] not in (cl, "NEG"):
                continue
            re_ = reas_cache.get(f"{p['pair_id']}_effect_two")
            rm = reas_cache.get(f"{p['pair_id']}_molecular_two")
            if not (re_ and rm) or re_.get("score") is None or rm.get("score") is None:
                continue
            ys.append(p["label"])
            se.append(re_["score"])
            sm.append(rm["score"])
        if len(set(ys)) < 2:
            continue
        d_em, lo, hi = paired_bootstrap_auc_diff(se, sm, ys)
        diffs[cl] = {"effect_minus_molecular": (d_em, [lo, hi]), "n": len(ys)}
        print(f"  {cl}  eff_two - mol_two = {d_em:+.3f}  [CI {lo:+.3f}, {hi:+.3f}]  n={len(ys)}")

    # Compare to E8.2 single-stage if results file exists
    try:
        e8b_res = json.loads((OUT_DIR / "llm_oracle_results_prob.json").read_text())
        print("\n[E8.4] === Two-stage vs Single-stage AUROC (E8.2 baseline) ===")
        for cl in ["PD", "PK"]:
            for mod in ["effect", "molecular"]:
                old_key = f"{cl}_C_effect" if mod == "effect" else f"{cl}_B_molecular"
                old = e8b_res["results_by_class_variant"].get(old_key, {}).get("auc")
                new = results.get(f"{cl}_{mod}_two", {}).get("auc")
                if old is not None and new is not None:
                    print(f"  {cl}  {mod:10s}  single={old:.3f}  two-stage={new:.3f}  Δ={new-old:+.3f}")
    except Exception as ex:
        print(f"  (could not load E8.2 results for comparison: {ex})")

    tin_sel = sum(r.get("tokens_in", 0) for r in sel_cache.values())
    tout_sel = sum(r.get("tokens_out", 0) for r in sel_cache.values())
    tin_reas = sum(r.get("tokens_in", 0) for r in reas_cache.values())
    tout_reas = sum(r.get("tokens_out", 0) for r in reas_cache.values())
    cost = (tin_sel + tin_reas) * 3 / 1_000_000 + (tout_sel + tout_reas) * 15 / 1_000_000
    print(f"\n[E8.4] tokens: selector in={tin_sel} out={tout_sel}; reasoner in={tin_reas} out={tout_reas}")
    print(f"[E8.4] cost~=${cost:.3f}")

    # Avg selection counts
    sel_counts = [len(r["selected"]) for r in sel_cache.values()]
    print(f"[E8.4] avg selections per (drug, modality): {np.mean(sel_counts):.2f}  "
          f"(min={min(sel_counts)}, max={max(sel_counts)})")

    payload = {
        "model": MODEL,
        "design": "two-stage (selector name-blind → reasoner anonymized)",
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg, "unique_drugs": len(unique_drugs)},
        "two_stage_results": results,
        "two_stage_paired_diffs": {k: {kk: {"mean": vv[0], "ci": vv[1]} for kk, vv in v.items() if isinstance(vv, tuple)} | {"n": v.get("n")}
                                    for k, v in diffs.items()},
        "tokens": {"selector_in": tin_sel, "selector_out": tout_sel,
                    "reasoner_in": tin_reas, "reasoner_out": tout_reas},
        "cost_usd_est": cost,
        "selection_stats": {"mean": float(np.mean(sel_counts)), "min": min(sel_counts), "max": max(sel_counts)},
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8.4] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
