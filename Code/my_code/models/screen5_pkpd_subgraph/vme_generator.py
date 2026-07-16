"""Generate Virtual Mediator Embeddings (VME) for structural gaps.

Two pipelines (per first_step_plan.md §4.7):

  S4 variant — GPT-4o text generation:
    For each (node_a, node_c, layer) gap, prompt GPT-4o to name + describe
    the most likely intermediate biological entity. Encode response with
    PubMedBERT [CLS] -> 768d -> project to 64d (shared with Screen 1's
    projection helpers).
    Hard cost cap: $50 USD. Aborts if exceeded.

  S4b variant — Qwen-72B input-embedding lookup (zero-cost fallback):
    Build a prompt string, tokenize with Qwen-72B tokenizer, mean-pool over
    the input-embedding table (no transformer forward). 8192d -> project
    to 64d.
    Zero API cost, fully reproducible.

This module exposes two callable generators that take a list of
(node_a_name, node_c_name, layer) and return a (N, 64) tensor.

Cache:
  Code/data/KG/_merged_kg/_cache/screen5_pkpd_subgraph/vme_gpt4o/{sha}.json
  Code/data/KG/_merged_kg/_cache/screen5_pkpd_subgraph/vme_qwen/{sha}.json

API key loaded at runtime from filename in
  D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/API-KEY/sk-proj-*.txt
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

CACHE_ROOT = PROJECT_ROOT / "Code" / "data" / "KG" / "_merged_kg" / "_cache" / "screen5_pkpd_subgraph"
API_KEY_DIR = PROJECT_ROOT / "API-KEY"
COST_LOG = CACHE_ROOT / "vme_gpt4o" / "_cost_log.csv"
DEFAULT_BUDGET_USD = 50.0

PROMPT_TEMPLATE = (
    "In a {layer} drug-interaction pathway, name and briefly describe (in "
    "one sentence) the most likely intermediate biological entity between "
    "the entity \"{a_name}\" and the entity \"{c_name}\". Respond with: "
    "<NAME>: <DESCRIPTION>. Keep total response under 40 words."
)


def _load_openai_key() -> str:
    """Read API key from filename in API-KEY/ folder (one file per key)."""
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    for f in API_KEY_DIR.glob("sk-proj-*.txt"):
        return f.stem  # filename without .txt
    raise RuntimeError("OPENAI_API_KEY not found in env nor API-KEY/ folder")


def _gap_hash(a_name: str, c_name: str, layer: str) -> str:
    h = hashlib.sha256()
    h.update(a_name.encode()); h.update(b"||")
    h.update(c_name.encode()); h.update(b"||")
    h.update(layer.encode())
    return h.hexdigest()


def _log_cost(model: str, input_tokens: int, output_tokens: int):
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    # GPT-4o pricing per 2026-05: $2.50/M input, $10/M output
    if "4o" in model.lower():
        cost = (input_tokens * 2.50 + output_tokens * 10.00) / 1_000_000
    else:
        cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000  # gpt-4o-mini
    with open(COST_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{model},{input_tokens},{output_tokens},{cost:.6f}\n")
    return cost


def _total_spent() -> float:
    if not COST_LOG.exists():
        return 0.0
    total = 0.0
    with open(COST_LOG) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                try:
                    total += float(parts[-1])
                except ValueError:
                    pass
    return total


def generate_vme_via_gpt4o(
    gaps: list[tuple[str, str, str]],
    *,
    budget_usd: float = DEFAULT_BUDGET_USD,
    model: str = "gpt-4o",
    max_workers: int = 20,
    skip_if_cached: bool = True,
) -> dict[tuple[str, str, str], str]:
    """Generate text responses for each gap. Returns dict (gap -> response text).

    Hard-aborts when running total exceeds `budget_usd`.
    Caches each response by sha256(a_name, c_name, layer).
    """
    cache_dir = CACHE_ROOT / "vme_gpt4o"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Filter cached
    pending = []
    cached = {}
    for gap in gaps:
        a_name, c_name, layer = gap
        h = _gap_hash(a_name, c_name, layer)
        p = cache_dir / f"{h}.json"
        if p.exists() and skip_if_cached:
            cached[gap] = json.loads(p.read_text())["response"]
        else:
            pending.append((gap, p))

    print(f"[vme/gpt4o] {len(cached):,} cached, {len(pending):,} to fetch "
          f"(spent so far: ${_total_spent():.2f})")

    if not pending:
        return cached

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package required for VME gen. pip install openai") from e

    api_key = _load_openai_key()
    client = OpenAI(api_key=api_key)

    spent = _total_spent()
    n_done = 0
    for gap, p in pending:
        if spent >= budget_usd:
            print(f"[vme/gpt4o] BUDGET CAP REACHED (${spent:.2f} >= ${budget_usd:.2f}); aborting.")
            break
        a_name, c_name, layer = gap
        prompt = PROMPT_TEMPLATE.format(layer=layer, a_name=a_name, c_name=c_name)
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            cost = _log_cost(model, r.usage.prompt_tokens, r.usage.completion_tokens)
            spent += cost
            p.write_text(json.dumps({
                "gap": gap, "prompt": prompt, "response": text,
                "model": model, "tokens_in": r.usage.prompt_tokens, "tokens_out": r.usage.completion_tokens,
                "cost_usd": cost,
            }, indent=2))
            cached[gap] = text
            n_done += 1
            if n_done % 50 == 0:
                print(f"[vme/gpt4o] {n_done}/{len(pending)} done, spent=${spent:.2f}")
        except Exception as e:
            print(f"[vme/gpt4o] error on gap {gap[0][:20]}/{gap[1][:20]}: {e}")
            continue

    print(f"[vme/gpt4o] complete: {n_done} fetched, total cached {len(cached)}, "
          f"total spent ${spent:.2f}")
    return cached


def generate_vme_via_qwen_embed(
    gaps: list[tuple[str, str, str]],
    *,
    qwen_model_path: Path | None = None,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Generate VME by Qwen-72B input-embedding mean-pool (zero-cost fallback).

    Loads ONLY model.embed_tokens.weight from safetensors. Does NOT run the
    transformer. Returns (gap -> 8192d tensor).

    If qwen_model_path is None or weights missing, returns empty dict (skip).
    """
    if qwen_model_path is None:
        candidates = [
            Path("/mnt/g/hf_cache/hub/models--Qwen--Qwen2.5-72B"),
            Path("G:/hf_cache/hub/models--Qwen--Qwen2.5-72B"),
        ]
        qwen_model_path = next((p for p in candidates if p.exists()), None)
    if qwen_model_path is None or not qwen_model_path.exists():
        print("[vme/qwen] Qwen-72B not available locally; skipping S4b variant.")
        return {}

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError("transformers + safetensors required for Qwen VME") from e

    print(f"[vme/qwen] loading tokenizer + embed_tokens from {qwen_model_path}")
    tok = AutoTokenizer.from_pretrained(qwen_model_path)
    # Find shard with model.embed_tokens.weight
    embed_tensor = None
    for shard in qwen_model_path.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            if "model.embed_tokens.weight" in keys:
                embed_tensor = f.get_tensor("model.embed_tokens.weight")
                break
    if embed_tensor is None:
        raise RuntimeError("model.embed_tokens.weight not found in any shard")
    print(f"[vme/qwen] embed table: {tuple(embed_tensor.shape)} dtype={embed_tensor.dtype}")
    embed_tensor = embed_tensor.float()

    out = {}
    for gap in gaps:
        a_name, c_name, layer = gap
        prompt = PROMPT_TEMPLATE.format(layer=layer, a_name=a_name, c_name=c_name)
        ids = tok.encode(prompt, add_special_tokens=False)
        if not ids:
            continue
        emb = embed_tensor[ids].mean(dim=0)  # [hidden]
        out[gap] = emb
    print(f"[vme/qwen] generated {len(out)} embeddings, dim={emb.shape[-1]}")
    return out
