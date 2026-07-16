"""Real-KG sanity check on the generated PubMedBERT embedding caches.

Picks a few known drug pairs and prints cosine similarity for D, D-name, E, F.
Expected pattern:
  - D: pharmacologically similar drugs (Aspirin/Ibuprofen NSAIDs) high; unrelated lower
  - D-name: similar (only name matters now, less informative than D)
  - E: shuffled control should give LOWER and noisier sim than D
  - F: kind-only (all "Drug") gives sim = 1.0 within drug nodes
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import pandas as pd

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen1_tag_init import node_text_builder as ntb  # noqa
from my_code.models.screen1_tag_init import init_features as ift     # noqa


# Real drug pairs from the project's drug pool (DrugBank IDs)
TEST_PAIRS = [
    ("DB00945", "DB01050", "Aspirin / Ibuprofen", "BOTH NSAID — HIGH"),
    ("DB00945", "DB01101", "Aspirin / Capecitabine", "antiplatelet/antineoplastic — LOWER"),
    ("DB00564", "DB00475", "Carbamazepine / Chlordiazepoxide", "anticonvulsant/anxiolytic — MIXED"),
    ("DB00564", "DB00829", "Carbamazepine / Diazepam", "BOTH CNS-active — HIGHER"),
    ("DB00006", "DB00007", "Bivalirudin / Leuprolide", "anticoagulant/GnRH agonist — LOW"),
]


def main():
    # Load 4 caches
    caches = {}
    for tag in ["d_full_text", "d_name_only", "e_shuffled_text", "f_typename_only"]:
        path = ntb.KG_ROOT / "_merged_kg" / "_cache" / "screen1_tag_init" / f"{tag}__pubmedbert.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        caches[tag] = payload
        print(f"loaded {tag}: N={len(payload['node_ids']):,}, n_empty={payload.get('n_empty', '?')}")

    nodes_df = pd.read_parquet(ntb.MERGED_NODES)
    id2idx = {nid: i for i, nid in enumerate(caches["d_full_text"]["node_ids"])}

    print(f"\n{'PAIR':<45} {'D-full':>8} {'D-name':>8} {'E-shuf':>8} {'F-type':>8}")
    print("-" * 80)
    for da, db, desc, expect in TEST_PAIRS:
        ia, ib = id2idx.get(da), id2idx.get(db)
        if ia is None or ib is None:
            print(f"SKIP {desc}: drug id not in KG ({da if ia is None else db})")
            continue
        sims = {}
        for tag, payload in caches.items():
            emb = payload["embeddings"].float()
            ea, eb = emb[ia], emb[ib]
            if ea.norm() < 1e-6 or eb.norm() < 1e-6:
                sims[tag] = float("nan")
            else:
                sims[tag] = torch.nn.functional.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        print(f"{desc:<45} "
              f"{sims['d_full_text']:>8.4f} "
              f"{sims['d_name_only']:>8.4f} "
              f"{sims['e_shuffled_text']:>8.4f} "
              f"{sims['f_typename_only']:>8.4f}")
    print(f"\nExpectation column:")
    for _, _, desc, expect in TEST_PAIRS:
        print(f"  {desc}: {expect}")

    # Also test: variant E (shuffled) should have LOWER avg sim than D
    # on a random sample of intra-Drug pairs
    drug_ids = nodes_df[nodes_df["kind"] == "Drug"]["id"].tolist()
    import random
    rng = random.Random(0)
    sample_pairs = [(rng.choice(drug_ids), rng.choice(drug_ids)) for _ in range(200)]
    sample_pairs = [(a, b) for a, b in sample_pairs if a != b][:100]

    avg = {}
    for tag, payload in caches.items():
        emb = payload["embeddings"].float()
        sims = []
        for a, b in sample_pairs:
            ia, ib = id2idx.get(a), id2idx.get(b)
            if ia is None or ib is None:
                continue
            ea, eb = emb[ia], emb[ib]
            if ea.norm() < 1e-6 or eb.norm() < 1e-6:
                continue
            sims.append(torch.nn.functional.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item())
        avg[tag] = sum(sims) / len(sims) if sims else float("nan")
        print(f"\navg cosine over {len(sims):>3} random Drug pairs ({tag}): {avg[tag]:.4f}")

    # Variant E sim should be CLOSE TO variant D sim distribution (same encoder,
    # different texts → expectation that semantic ordering changes, NOT that
    # average magnitude drops)
    print()
    print("Gate sanity:")
    print(f"  variant D avg sim: {avg['d_full_text']:.4f} — should be reasonable (~0.85-0.95)")
    print(f"  variant E avg sim: {avg['e_shuffled_text']:.4f} — close to D but with scrambled identity")
    print(f"  variant F avg sim: {avg['f_typename_only']:.4f} — should be ~1.0 (all same kind='Drug')")


if __name__ == "__main__":
    main()
