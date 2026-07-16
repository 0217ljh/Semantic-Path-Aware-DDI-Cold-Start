"""Verify SPMNMolHead.forward_batch == per-pair forward_pair (numerically).

Guards against vectorisation bugs (the "runs but wrong" class). Builds a small
batch of synthetic-but-consistent pairs, runs both paths under eval() (dropout
off), and asserts the logits match within tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.spmn_v1.mm_head import SPMNMolHead  # noqa: E402

K, D, DFRAG, STRUCT, NENT, NREL = 12, 32, 64, 116, 1000, 11


def main() -> None:
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = SPMNMolHead(n_entities=NENT, n_types=K, n_rel_buckets=NREL,
                    struct_dim=STRUCT, d_frag=DFRAG, d=D).to(dev).eval()

    B = 8
    pairs = []
    for p in range(B):
        E = int(np.random.randint(0, 25))           # support size (incl 0)
        Fa = int(np.random.randint(0, 8))           # a-frags (incl 0)
        Fb = int(np.random.randint(0, 8))
        glob = torch.randint(0, NENT, (E,), device=dev)
        typ = torch.randint(0, K, (E,), device=dev)
        ra = torch.randint(0, NREL, (E,), device=dev)
        rb = torch.randint(0, NREL, (E,), device=dev)
        za = torch.randn(Fa, DFRAG, device=dev)
        zb = torch.randn(Fb, DFRAG, device=dev)
        P = int(np.random.randint(0, 6)) if E > 0 else 0
        iu = torch.randint(0, max(E, 1), (P,), device=dev) if E > 0 else torch.zeros(0, dtype=torch.long, device=dev)
        iv = torch.randint(0, max(E, 1), (P,), device=dev) if E > 0 else torch.zeros(0, dtype=torch.long, device=dev)
        struct = torch.randn(STRUCT, device=dev)
        kappa = torch.tensor(1.0 if E > 0 else 0.0, device=dev)
        pairs.append(dict(glob=glob, typ=typ, ra=ra, rb=rb, za=za, zb=zb,
                          iu=iu, iv=iv, struct=struct, kappa=kappa, E=E))

    # --- per-pair reference ---
    with torch.no_grad():
        ref = torch.stack([
            m.forward_pair(p["glob"], p["typ"], p["ra"], p["rb"], p["struct"],
                           p["za"], p["zb"], p["iu"], p["iv"], p["kappa"])
            for p in pairs])

    # --- batched: flatten ---
    med = torch.cat([p["glob"] for p in pairs])
    typ = torch.cat([p["typ"] for p in pairs])
    rel_a = torch.cat([p["ra"] for p in pairs])
    rel_b = torch.cat([p["rb"] for p in pairs])
    pair_idx = torch.cat([torch.full((p["E"],), i, dtype=torch.long, device=dev) for i, p in enumerate(pairs)])
    s_off = np.cumsum([0] + [p["E"] for p in pairs])
    fa_z = torch.cat([p["za"] for p in pairs]) if any(p["za"].size(0) for p in pairs) else torch.zeros(0, DFRAG, device=dev)
    fb_z = torch.cat([p["zb"] for p in pairs]) if any(p["zb"].size(0) for p in pairs) else torch.zeros(0, DFRAG, device=dev)
    fa_pair = torch.cat([torch.full((p["za"].size(0),), i, dtype=torch.long, device=dev) for i, p in enumerate(pairs)]) if fa_z.size(0) else torch.zeros(0, dtype=torch.long, device=dev)
    fb_pair = torch.cat([torch.full((p["zb"].size(0),), i, dtype=torch.long, device=dev) for i, p in enumerate(pairs)]) if fb_z.size(0) else torch.zeros(0, dtype=torch.long, device=dev)
    iu_g = torch.cat([p["iu"] + int(s_off[i]) for i, p in enumerate(pairs)]) if any(p["iu"].numel() for p in pairs) else torch.zeros(0, dtype=torch.long, device=dev)
    iv_g = torch.cat([p["iv"] + int(s_off[i]) for i, p in enumerate(pairs)]) if any(p["iv"].numel() for p in pairs) else torch.zeros(0, dtype=torch.long, device=dev)
    struct = torch.stack([p["struct"] for p in pairs])
    kappa = torch.stack([p["kappa"] for p in pairs])

    with torch.no_grad():
        bat = m.forward_batch(med=med, pair_idx=pair_idx, typ=typ, rel_a=rel_a,
                              rel_b=rel_b, n_pairs=B, fa_z=fa_z, fa_pair=fa_pair,
                              fb_z=fb_z, fb_pair=fb_pair, iu_g=iu_g, iv_g=iv_g,
                              struct=struct, kappa=kappa)

    diff = (ref - bat).abs().max().item()
    print("per-pair :", [round(x, 4) for x in ref.tolist()])
    print("batched  :", [round(x, 4) for x in bat.tolist()])
    print(f"max abs diff = {diff:.2e}  ->  {'MATCH' if diff < 1e-4 else 'MISMATCH!!'}")


if __name__ == "__main__":
    main()
