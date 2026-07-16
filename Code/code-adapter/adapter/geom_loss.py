"""Unsupervised z_uv neighborhood-preservation loss (design R, part 2). Makes the scorer-visible
fused rep p_bb' preserve the LOCAL neighbor structure of the adapter's own KG-semantic pooled rep
z_uv. NO analysis labels (shared-MBE) are used -> unsupervised; it aligns p_bb' to the method's own
KG geometry, which shared-MBE clustering then validates POST HOC. Disclosed inductive bias, not an
'emergent' claim (codex).

Local multi-positive InfoNCE: for each pair (anchor) in a batch, its top-k neighbors in z_uv are
positives; the loss pulls them closer than the rest in p_bb' cosine space. Local (top-k, in-batch)
so it does not wash out task signal.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def neighborhood_preservation_loss(z_uv: torch.Tensor, p_prime: torch.Tensor,
                                   k: int = 10, temp: float = 0.5) -> torch.Tensor:
    """z_uv [B, d_z], p_prime [B, d] -> scalar. In-batch multi-positive InfoNCE: positives = each
    anchor's top-k z_uv cosine neighbors; encourages those to dominate p_prime's neighbor softmax."""
    B = z_uv.shape[0]
    kk = int(min(k, B - 2))
    if B < 4 or kk < 1:
        return z_uv.new_zeros(())
    zn = F.normalize(z_uv, dim=-1)
    pn = F.normalize(p_prime, dim=-1)
    eye = torch.eye(B, device=z_uv.device, dtype=torch.bool)
    sz = (zn @ zn.t()).masked_fill(eye, float("-inf"))        # z_uv neighbor sims (self masked)
    sp = ((pn @ pn.t()) / temp).masked_fill(eye, float("-inf"))  # p_prime logits (self masked)
    pos = sz.topk(kk, dim=1).indices                          # [B, kk] z_uv top-k neighbor idx
    logden = torch.logsumexp(sp, dim=1)                       # over all j != i
    lognum = torch.logsumexp(torch.gather(sp, 1, pos), dim=1)  # over the kk positives
    return (logden - lognum).mean()                          # = -log(sum_pos exp / sum_all exp)


__all__ = ["neighborhood_preservation_loss"]
