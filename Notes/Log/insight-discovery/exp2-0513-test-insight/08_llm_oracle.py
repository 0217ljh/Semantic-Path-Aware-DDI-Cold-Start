"""E8 — LLM oracle on effect-vs-molecular feature meaningfulness for PD vs PK.

Supports the post-experiment reframe: E1b proved (topology) that PK pairs
concentrate in molecular-layer intermediates, PD in effect-layer. E4 showed
GCN under effect-only KG hurts PK 12pt but PD only 4pt. To verify the
underlying biology (effect IS meaningful for PD, GCN just fails to use it),
we use a closed-source LLM as biological oracle:

  Three prompts per drug pair:
    (A) Name-only          : just the two drug names  (memorization baseline)
    (B) Molecular features : drug targets / enzymes / transporters
    (C) Effect features    : drug-associated side effects / phenotypes

  Hypothesis: for PD positives, accuracy(C) > accuracy(B) and > accuracy(A).
              for PK positives, accuracy(B) > accuracy(C) (or both similar to A).

Sampling (cost-controlled):
  - 20 PD positives from test_s2 (pk_pd_label='PD' via ddi_pk_pd_labels.csv)
  - 20 PK positives from test_s2 (pk_pd_label='PK')
  - 20 negatives from negatives/test_s2.parquet (shared across PD and PK eval)
  Total 60 pairs × 3 prompts = 180 API calls.
  Estimated cost: ~$0.50 (Claude Sonnet 4.5, temp=0, max_tokens=80).

Cold-start closedness: positives are from test_s2.parquet (S2 split — both
drugs are in G2 = unseen at training time per ColdDDI seed42 manifest).
Negatives from negatives/test_s2.parquet (paired with same constraint).
LLM was trained on biomedical literature so the cold-start constraint is
imperfect — the differential across prompts A/B/C controls for memorization.

Run:
  /home/lakestar_ljh/miniconda3/envs/project_1/bin/python \\
      Notes/Log/insight-discovery/exp2-0513-test-insight/08_llm_oracle.py

Output:
  - llm_oracle_predictions.jsonl : per-call cache (resume-safe)
  - llm_oracle_results.json      : accuracy + bootstrap CI per (class, prompt)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")


PROJECT_ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
PKPD = PROJECT_ROOT / "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
SPLITS = PROJECT_ROOT / "Code/data/KG/drugbank/splits/seed42"

# Env file holds the API keys. Format: KEY [= or =] VALUE (with or without quotes)
ENV_FILE = Path("/mnt/d/.secrets/API.env")

# Sample sizes (cost-controlled, bumped per codex round 2 review)
N_PD_POS = 30
N_PK_POS = 30
N_NEG = 30
SEED = 42

# LLM config
MODEL = "claude-sonnet-4-5-20250929"  # Claude Sonnet 4.5
MAX_TOKENS = 80
TEMPERATURE = 0.0
MAX_FEATURES_PER_DRUG = 20  # cap for SE / molecular lists (rare-by-KG-degree per codex)

# Anthropic API
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

CACHE_FILE = OUT_DIR / "llm_oracle_predictions.jsonl"
RESULTS_FILE = OUT_DIR / "llm_oracle_results.json"


# ---------------------------------------------------------------------------
# Env loader
# ---------------------------------------------------------------------------
def load_env(env_path: Path) -> dict[str, str]:
    """Robust parser for KEY = value, KEY=value, with or without quotes."""
    out: dict[str, str] = {}
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # strip ASCII or smart quotes
            for q in ("'", '"', "‘", "’", "“", "”"):
                if v.startswith(q):
                    v = v[1:]
                if v.endswith(q):
                    v = v[:-1]
            out[k] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Feature extraction from merged KG
# ---------------------------------------------------------------------------
MOLECULAR_RELATIONS = {
    "db:target", "db:enzyme", "db:transporter", "db:carrier",
    "het:CbG", "het:CuG", "het:CdG",  # Hetionet Compound binds/up/down Gene
}
EFFECT_RELATIONS = {
    "het:CcSE",                  # Hetionet Compound causes Side Effect
    "prime:drug_effect",         # PrimeKG drug → effect/phenotype
}


def build_features(edges: pd.DataFrame, nodes: pd.DataFrame, drug_set: set) -> tuple[dict, dict]:
    """Return (drug → list of molecular feature names, drug → list of effect feature names).

    Per codex round 2: ranking by **KG-degree ascending** (rare = more specific).
    Tie-break alphabetically for determinism.
    """
    id2name = dict(zip(nodes["id"], nodes["name"].fillna("")))

    # Compute degree per non-drug node (across all KG edges)
    deg = defaultdict(int)
    for s, d in zip(edges["src"], edges["dst"]):
        deg[s] += 1
        deg[d] += 1

    # Collect per-drug feature node IDs by category
    mol_ids = defaultdict(set)
    eff_ids = defaultdict(set)
    for src, dst, rel, directed in zip(edges["src"], edges["dst"], edges["relation"], edges["directed"]):
        if rel in MOLECULAR_RELATIONS:
            if src in drug_set and dst not in drug_set:
                mol_ids[src].add(dst)
            elif dst in drug_set and src not in drug_set and not directed:
                mol_ids[dst].add(src)
        elif rel in EFFECT_RELATIONS:
            if src in drug_set and dst not in drug_set:
                eff_ids[src].add(dst)
            elif dst in drug_set and src not in drug_set and not directed:
                eff_ids[dst].add(src)

    def pick(node_ids: set) -> list:
        """Rank by (degree asc, name asc), keep only nodes with readable names ≥2 chars."""
        candidates = []
        for nid in node_ids:
            name = id2name.get(nid, "")
            if not name or len(name) < 2:
                continue
            candidates.append((deg[nid], name, nid))
        candidates.sort()
        return [name for _, name, _ in candidates[:MAX_FEATURES_PER_DRUG]]

    mol_out = {d: pick(s) for d, s in mol_ids.items()}
    eff_out = {d: pick(s) for d, s in eff_ids.items()}
    return mol_out, eff_out


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
def sample_pairs(pkpd_labels: dict, mol_feats: dict, eff_feats: dict, id2name: dict, seed: int = SEED):
    """Return list of dicts: each pair has {pair_id, drug_a_id, drug_b_id, drug_a_name, drug_b_name, label, class}.

    Per codex round 2: assert cold-start closedness — both drugs must be in G2.
    """
    rng = np.random.default_rng(seed)

    # Cold-start G2 set (unseen drugs at training time)
    manifest = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    g2_set = set(manifest["g2_drugs"])
    g1_set = set(manifest["g1_drugs"])
    print(f"[E8] cold-start G2 drugs (unseen at train): {len(g2_set)};  G1 (seen): {len(g1_set)}")

    test_pos = pd.read_parquet(SPLITS / "test_s2.parquet")
    test_pos["pk_pd_label"] = test_pos["ddi_type"].map(pkpd_labels)
    test_neg = pd.read_parquet(SPLITS / "negatives/test_s2.parquet")

    # Assert S2 closedness: both drugs of every test pair must be in G2
    def is_s2(da, db):
        return da in g2_set and db in g2_set
    test_pos = test_pos[[is_s2(a, b) for a, b in zip(test_pos["drug_a_id"], test_pos["drug_b_id"])]]
    test_neg = test_neg[[is_s2(a, b) for a, b in zip(test_neg["drug_a_id"], test_neg["drug_b_id"])]]
    print(f"[E8] after S2 closedness filter: positives={len(test_pos)}  negatives={len(test_neg)}")

    # Filter to pairs where BOTH drugs have at least 3 mol AND 3 eff features
    def has_features(da, db):
        return (
            len(mol_feats.get(da, [])) >= 3
            and len(mol_feats.get(db, [])) >= 3
            and len(eff_feats.get(da, [])) >= 3
            and len(eff_feats.get(db, [])) >= 3
        )

    pd_pos = test_pos[test_pos["pk_pd_label"] == "PD"]
    pd_pos = pd_pos[[has_features(a, b) for a, b in zip(pd_pos["drug_a_id"], pd_pos["drug_b_id"])]]
    pk_pos = test_pos[test_pos["pk_pd_label"] == "PK"]
    pk_pos = pk_pos[[has_features(a, b) for a, b in zip(pk_pos["drug_a_id"], pk_pos["drug_b_id"])]]
    neg = test_neg[[has_features(a, b) for a, b in zip(test_neg["drug_a_id"], test_neg["drug_b_id"])]]

    print(f"[E8] feature-filtered pool sizes:  PD-pos={len(pd_pos)}  PK-pos={len(pk_pos)}  neg={len(neg)}")

    pd_pos_s = pd_pos.sample(min(N_PD_POS, len(pd_pos)), random_state=seed).reset_index(drop=True)
    pk_pos_s = pk_pos.sample(min(N_PK_POS, len(pk_pos)), random_state=seed).reset_index(drop=True)
    neg_s = neg.sample(min(N_NEG, len(neg)), random_state=seed).reset_index(drop=True)

    pairs = []
    for _, r in pd_pos_s.iterrows():
        pairs.append(dict(
            pair_id=f"PD-pos-{r['drug_a_id']}_{r['drug_b_id']}",
            drug_a_id=r["drug_a_id"], drug_b_id=r["drug_b_id"],
            drug_a_name=id2name.get(r["drug_a_id"], r["drug_a_id"]),
            drug_b_name=id2name.get(r["drug_b_id"], r["drug_b_id"]),
            label=1, mech_class="PD", ddi_type=r.get("ddi_type", ""),
        ))
    for _, r in pk_pos_s.iterrows():
        pairs.append(dict(
            pair_id=f"PK-pos-{r['drug_a_id']}_{r['drug_b_id']}",
            drug_a_id=r["drug_a_id"], drug_b_id=r["drug_b_id"],
            drug_a_name=id2name.get(r["drug_a_id"], r["drug_a_id"]),
            drug_b_name=id2name.get(r["drug_b_id"], r["drug_b_id"]),
            label=1, mech_class="PK", ddi_type=r.get("ddi_type", ""),
        ))
    for _, r in neg_s.iterrows():
        pairs.append(dict(
            pair_id=f"NEG-{r['drug_a_id']}_{r['drug_b_id']}",
            drug_a_id=r["drug_a_id"], drug_b_id=r["drug_b_id"],
            drug_a_name=id2name.get(r["drug_a_id"], r["drug_a_id"]),
            drug_b_name=id2name.get(r["drug_b_id"], r["drug_b_id"]),
            label=0, mech_class="NEG", ddi_type="",
        ))
    return pairs


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a clinical pharmacology assistant. Given information about two drugs, "
    "you must predict whether they will have a clinically relevant drug-drug interaction (DDI). "
    "Reply on the first line with YES or NO. Then on a second line give a one-sentence reason. "
    "Do not add any other text."
)


def build_user_prompt(pair: dict, variant: str, mol_feats: dict, eff_feats: dict) -> str:
    name_a, name_b = pair["drug_a_name"], pair["drug_b_name"]
    if variant == "A_name":
        body = (
            f"Drug A: {name_a}\n"
            f"Drug B: {name_b}\n"
        )
    elif variant == "B_molecular":
        m_a = mol_feats.get(pair["drug_a_id"], [])
        m_b = mol_feats.get(pair["drug_b_id"], [])
        body = (
            f"Drug A: {name_a}\n"
            f"Drug A's molecular targets/enzymes/transporters: {', '.join(m_a) if m_a else '(none documented)'}\n"
            f"Drug B: {name_b}\n"
            f"Drug B's molecular targets/enzymes/transporters: {', '.join(m_b) if m_b else '(none documented)'}\n"
        )
    elif variant == "C_effect":
        e_a = eff_feats.get(pair["drug_a_id"], [])
        e_b = eff_feats.get(pair["drug_b_id"], [])
        body = (
            f"Drug A: {name_a}\n"
            f"Drug A's documented side effects / phenotypes: {', '.join(e_a) if e_a else '(none documented)'}\n"
            f"Drug B: {name_b}\n"
            f"Drug B's documented side effects / phenotypes: {', '.join(e_b) if e_b else '(none documented)'}\n"
        )
    else:
        raise ValueError(variant)
    body += "\nQuestion: Is there a clinically relevant drug-drug interaction between Drug A and Drug B?"
    return body


# ---------------------------------------------------------------------------
# Anthropic API call (sync via urllib to avoid SDK install)
# ---------------------------------------------------------------------------
def call_anthropic(api_key: str, system: str, user: str, model: str = MODEL,
                   max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE,
                   max_retries: int = 3) -> dict:
    import urllib.request
    import urllib.error
    import json as _json

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
    data = _json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(ANTHROPIC_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                return _json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  HTTP {e.code}, retry in {wait}s: {err_body[:200]}")
                time.sleep(wait)
                continue
            return {"error": f"HTTP {e.code}: {err_body[:500]}"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"error": f"{type(e).__name__}: {e}"}
    return {"error": "max retries exceeded"}


def parse_yes_no(text: str) -> int | None:
    if not text:
        return None
    first_line = text.strip().split("\n")[0].strip().upper()
    if first_line.startswith("YES"):
        return 1
    if first_line.startswith("NO"):
        return 0
    # fallback: scan full text for the first YES/NO token
    m = re.search(r"\b(YES|NO)\b", text.upper())
    if m:
        return 1 if m.group(1) == "YES" else 0
    return None


# ---------------------------------------------------------------------------
# Cache (resume-safe)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def bootstrap_acc_ci(predictions: list, labels: list, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    preds = np.array(predictions)
    labs = np.array(labels)
    n = len(preds)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append((preds[idx] == labs[idx]).mean())
    mean = float(np.mean(accs))
    lo, hi = float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))
    return mean, lo, hi


# ---------------------------------------------------------------------------
def main():
    # 1. Load env (without echoing values)
    env = load_env(ENV_FILE)
    api_key = env.get("CLAUDE_KEY") or env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_API_KEY")
    if not api_key:
        print(f"[E8] FATAL: no Claude/Anthropic key found in {ENV_FILE}. Keys present: {list(env.keys())}")
        return
    print(f"[E8] loaded API key (length {len(api_key)} chars)")

    # 2. Load KG + features + pairs
    print("[E8] loading KG ...")
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    id2name = dict(zip(nodes["id"], nodes["name"].fillna("")))
    print(f"[E8] nodes={len(nodes)}  edges={len(edges)}  drugs={len(drug_set)}")

    mol_feats, eff_feats = build_features(edges, nodes, drug_set)
    print(f"[E8] drugs with mol features: {len(mol_feats)};  with eff features: {len(eff_feats)}")
    # average feature counts (capped)
    avg_mol = np.mean([len(v) for v in mol_feats.values()]) if mol_feats else 0
    avg_eff = np.mean([len(v) for v in eff_feats.values()]) if eff_feats else 0
    print(f"[E8] avg #mol per drug = {avg_mol:.1f};  avg #eff per drug = {avg_eff:.1f}")

    pkpd_df = pd.read_csv(PKPD)
    pkpd_labels = dict(zip(pkpd_df["ddi_type"], pkpd_df["pk_pd_label"]))

    pairs = sample_pairs(pkpd_labels, mol_feats, eff_feats, id2name, seed=SEED)
    n_pd = sum(1 for p in pairs if p["mech_class"] == "PD")
    n_pk = sum(1 for p in pairs if p["mech_class"] == "PK")
    n_neg = sum(1 for p in pairs if p["mech_class"] == "NEG")
    print(f"[E8] sampled: PD-pos={n_pd}  PK-pos={n_pk}  NEG={n_neg}  total={len(pairs)}")

    # 3. Show sample to user before running (for last-mile sanity)
    print("\n[E8] === PROMPT PREVIEW (first PD-pos pair, all three variants) ===")
    sample_p = next(p for p in pairs if p["mech_class"] == "PD")
    for v in ["A_name", "B_molecular", "C_effect"]:
        u = build_user_prompt(sample_p, v, mol_feats, eff_feats)
        print(f"\n--- {v} ---")
        print(u)
    print("\n[E8] === END PREVIEW ===\n")

    # 4. Call API, with cache
    cache = load_cache(CACHE_FILE)
    print(f"[E8] cache hits available: {len(cache)}")

    variants = ["A_name", "B_molecular", "C_effect"]
    total_calls = len(pairs) * len(variants)
    done = 0
    t0 = time.time()
    for pair in pairs:
        for v in variants:
            call_id = f"{pair['pair_id']}_{v}"
            done += 1
            if call_id in cache:
                continue
            user = build_user_prompt(pair, v, mol_feats, eff_feats)
            resp = call_anthropic(api_key, SYSTEM_PROMPT, user)
            text = ""
            if "error" in resp:
                err = resp["error"]
                pred = None
                tokens_in = tokens_out = 0
            else:
                try:
                    text = resp["content"][0]["text"]
                    pred = parse_yes_no(text)
                    tokens_in = resp.get("usage", {}).get("input_tokens", 0)
                    tokens_out = resp.get("usage", {}).get("output_tokens", 0)
                    err = None
                except (KeyError, IndexError, TypeError) as e:
                    text = json.dumps(resp)[:500]
                    pred = None
                    tokens_in = tokens_out = 0
                    err = f"parse: {e}"

            rec = dict(
                call_id=call_id,
                pair_id=pair["pair_id"],
                mech_class=pair["mech_class"],
                label=pair["label"],
                variant=v,
                response_text=text,
                pred=pred,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error=err,
            )
            append_cache(CACHE_FILE, rec)
            cache[call_id] = rec
            if done % 20 == 0 or done == 1:
                elapsed = time.time() - t0
                eta = elapsed / done * (total_calls - done)
                print(f"  [{done}/{total_calls}]  pair={pair['pair_id'][:40]}  variant={v}  pred={pred}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # 5. Aggregate accuracy by (class, variant)
    print("\n[E8] === Results ===")
    results = {}
    for class_label in ["PD", "PK"]:
        # For each class's accuracy, use positives of that class + all negatives
        for v in variants:
            pos_recs = [cache[f"{p['pair_id']}_{v}"] for p in pairs if p["mech_class"] == class_label]
            neg_recs = [cache[f"{p['pair_id']}_{v}"] for p in pairs if p["mech_class"] == "NEG"]
            recs = pos_recs + neg_recs
            preds = [r["pred"] for r in recs if r["pred"] is not None]
            labs = [r["label"] for r in recs if r["pred"] is not None]
            n_parsed = len(preds)
            n_total = len(recs)
            if n_parsed == 0:
                continue
            acc = float(np.mean(np.array(preds) == np.array(labs)))
            yes_rate = float(np.mean(np.array(preds) == 1))
            acc_mean, lo, hi = bootstrap_acc_ci(preds, labs)
            key = f"{class_label}_{v}"
            results[key] = {
                "n_parsed": n_parsed, "n_total": n_total,
                "accuracy": acc, "acc_mean_boot": acc_mean,
                "acc_ci_95": [lo, hi], "yes_rate": yes_rate,
            }
            print(f"  {class_label:3s}  {v:13s}  acc={acc:.3f}  [CI {lo:.3f}, {hi:.3f}]  yes_rate={yes_rate:.2f}  n={n_parsed}/{n_total}")

    # 6. Differentials
    print("\n[E8] === Differentials (effect - name-only, effect - molecular) ===")
    diffs = {}
    for class_label in ["PD", "PK"]:
        eff_recs = {r["pair_id"]: r for r in (cache[f"{p['pair_id']}_C_effect"] for p in pairs if p["mech_class"] in (class_label, "NEG"))}
        name_recs = {r["pair_id"]: r for r in (cache[f"{p['pair_id']}_A_name"] for p in pairs if p["mech_class"] in (class_label, "NEG"))}
        mol_recs = {r["pair_id"]: r for r in (cache[f"{p['pair_id']}_B_molecular"] for p in pairs if p["mech_class"] in (class_label, "NEG"))}

        # paired accuracy
        common = set(eff_recs) & set(name_recs) & set(mol_recs)
        eff_preds = [eff_recs[pid]["pred"] for pid in common if all(eff_recs[pid]["pred"] is not None for pid in [pid])]
        eff_acc, _, _ = bootstrap_acc_ci([eff_recs[pid]["pred"] for pid in common if eff_recs[pid]["pred"] is not None],
                                         [eff_recs[pid]["label"] for pid in common if eff_recs[pid]["pred"] is not None])
        name_acc, _, _ = bootstrap_acc_ci([name_recs[pid]["pred"] for pid in common if name_recs[pid]["pred"] is not None],
                                          [name_recs[pid]["label"] for pid in common if name_recs[pid]["pred"] is not None])
        mol_acc, _, _ = bootstrap_acc_ci([mol_recs[pid]["pred"] for pid in common if mol_recs[pid]["pred"] is not None],
                                         [mol_recs[pid]["label"] for pid in common if mol_recs[pid]["pred"] is not None])
        diffs[class_label] = {
            "name_only_acc": name_acc,
            "molecular_acc": mol_acc,
            "effect_acc": eff_acc,
            "effect_minus_name": eff_acc - name_acc,
            "effect_minus_molecular": eff_acc - mol_acc,
        }
        print(f"  {class_label}: name={name_acc:.3f}  mol={mol_acc:.3f}  eff={eff_acc:.3f}  "
              f"eff-name={eff_acc-name_acc:+.3f}  eff-mol={eff_acc-mol_acc:+.3f}")

    # 7. Save
    n_errors = sum(1 for r in cache.values() if r.get("error"))
    n_unparsed = sum(1 for r in cache.values() if r.get("pred") is None)
    total_in = sum(r.get("tokens_in", 0) for r in cache.values())
    total_out = sum(r.get("tokens_out", 0) for r in cache.values())
    print(f"\n[E8] tokens: in={total_in}  out={total_out}")
    print(f"[E8] estimated cost (Sonnet 4.5): ${total_in * 3 / 1_000_000 + total_out * 15 / 1_000_000:.3f}")
    print(f"[E8] errors: {n_errors}, unparsed: {n_unparsed}")

    payload = {
        "model": MODEL,
        "sample_sizes": {"PD_pos": n_pd, "PK_pos": n_pk, "neg": n_neg},
        "per_class_per_variant": results,
        "differentials": diffs,
        "tokens": {"input": total_in, "output": total_out},
        "errors": n_errors,
        "unparsed": n_unparsed,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E8] saved → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
