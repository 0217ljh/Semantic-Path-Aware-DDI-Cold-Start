"""P3 — edge-independent InfoNCE alignment of molecular m_u to KG-neighbor k_u.

Core mechanism of README_i1_routing_v2_learned.md (multimodal alignment). Molecular and KG
modalities collapse on cold-start because they only fuse through the DDI edge. Fix: learn a
projection proj_m that pulls each drug's molecular fingerprint m_u toward its KG-neighborhood
embedding k_u, with NO DDI labels and using SEEN (train-graph) drugs only. The mapping then
transfers to unseen drugs by construction.

Leakage safety: trained ONLY on drugs that appear in ds.splits.train (the legacy 800drug
seed42 split, same split MNAH 0.77 used). k_u itself is built from the DDI-masked merged KG.

Outputs Code/data/_cache/molecular_aligned_infonce.npz:
  drug_ids      : (N,) str   — all drugs that have a valid Morgan m_u
  z_m           : (N, D)     — proj_m(m_u), the aligned molecular embedding (D=ALIGN_DIM)
  residual      : (N,)       — ||l2(z_m) - l2(z_k)|| where z_k available (else nan); the
                               exploratory PK-ness self-gating proxy (R6, stop-grad use only)
  is_seen       : (N,) bool  — drug was in the alignment training set
  meta          : dict-ish via separate scalars (align_dim, temp, top1_seen, top1_heldout)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

MU = ROOT / "Code/data/_cache/molecular_mu_morgan.npz"
KU = ROOT / "Code/data/_cache/kg_neighbor_target_pubmedbert.npz"
PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
OUT = ROOT / "Code/data/_cache/molecular_aligned_infonce.npz"


class ProjM(nn.Module):
    """Shallow molecular projector (codex 019e622a patch): Linear->LN->GELU->Drop->Linear."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjK(nn.Module):
    """Linear-only KG projector: stable target, not a co-adapting lookup (codex)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _offdiag_decorr(z: torch.Tensor) -> torch.Tensor:
    """Barlow-style: penalize off-diagonal feature covariance of batch-normalized z."""
    if z.size(0) < 2:
        return z.new_zeros(())
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    cov = (zc.t() @ zc) / (z.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / z.size(1)


def _seen_drugs() -> set[str]:
    from data_utils import PairDataset  # noqa
    ds = PairDataset.from_pkl(str(PKL))
    tr = ds.splits.train
    seen = set(tr["drug_a_id"].astype(str)) | set(tr["drug_b_id"].astype(str))
    print(f"[align] seen (train-graph) drugs from legacy seed42 pkl: {len(seen)}", flush=True)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-dim", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--decorr", type=float, default=0.005)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                    help="fraction of SEEN drugs held out to measure transfer.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[align] device={device} cuda={torch.cuda.is_available()}", flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    mu = np.load(MU, allow_pickle=True)
    ku = np.load(KU, allow_pickle=True)
    m_ids = [str(x) for x in mu["drug_ids"]]
    m_valid = mu["valid"].astype(bool)
    m = mu["m"].astype(np.float32)
    k_ids = [str(x) for x in ku["drug_ids"]]
    k = ku["k"].astype(np.float32)
    k_ncount = ku["n_neighbors"].astype(np.int64)
    k_row = {d: i for i, d in enumerate(k_ids)}

    seen = _seen_drugs()

    # Training pairs: drugs with valid Morgan AND a non-empty k_u AND seen.
    train_rows_m, train_rows_k, train_ids = [], [], []
    for i, d in enumerate(m_ids):
        if not m_valid[i]:
            continue
        ki = k_row.get(d)
        if ki is None or k_ncount[ki] <= 0:
            continue
        if d not in seen:
            continue
        train_rows_m.append(i); train_rows_k.append(ki); train_ids.append(d)
    print(f"[align] aligned training drugs (valid m_u & k_u & seen): {len(train_ids)}", flush=True)

    perm = rng.permutation(len(train_ids))
    n_hold = int(len(perm) * args.holdout_frac)
    hold_idx = set(perm[:n_hold].tolist())
    fit_pos = [j for j in range(len(train_ids)) if j not in hold_idx]
    hold_pos = [j for j in range(len(train_ids)) if j in hold_idx]
    print(f"[align] fit={len(fit_pos)} holdout={len(hold_pos)}", flush=True)

    # Input normalization (codex patch): z-score Morgan by FIT-set mean/std; L2-norm k_u.
    fit_m_rows = np.array([train_rows_m[j] for j in fit_pos], dtype=np.int64)
    m_mean = m[fit_m_rows].mean(axis=0)
    m_std = m[fit_m_rows].std(axis=0) + 1e-6
    m_norm = (m - m_mean[None, :]) / m_std[None, :]
    k_l2 = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-8)

    M = torch.from_numpy(m_norm.astype(np.float32)).to(device)
    K = torch.from_numpy(k_l2.astype(np.float32)).to(device)
    tr_m = torch.tensor([train_rows_m[j] for j in fit_pos], dtype=torch.long, device=device)
    tr_k = torch.tensor([train_rows_k[j] for j in fit_pos], dtype=torch.long, device=device)
    ho_m = torch.tensor([train_rows_m[j] for j in hold_pos], dtype=torch.long, device=device)
    ho_k = torch.tensor([train_rows_k[j] for j in hold_pos], dtype=torch.long, device=device)

    proj_m = ProjM(m.shape[1], args.align_dim, dropout=args.dropout).to(device)
    proj_k = ProjK(k.shape[1], args.align_dim).to(device)
    opt = torch.optim.AdamW(
        list(proj_m.parameters()) + list(proj_k.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    def infonce(zm: torch.Tensor, zk: torch.Tensor) -> torch.Tensor:
        zmn = F.normalize(zm, dim=1)
        zkn = F.normalize(zk, dim=1)
        logits = zmn @ zkn.t() / args.temp
        tgt = torch.arange(zmn.size(0), device=zmn.device)
        nce = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
        return nce + args.decorr * (_offdiag_decorr(zm) + _offdiag_decorr(zk))

    @torch.no_grad()
    def holdout_metrics(m_idx: torch.Tensor, k_idx: torch.Tensor) -> dict:
        """Pairwise retrieval AUC (primary) + top1/top5/top10 + MRR on holdout."""
        proj_m.eval(); proj_k.eval()
        zm = F.normalize(proj_m(M[m_idx]), dim=1)
        zk = F.normalize(proj_k(K[k_idx]), dim=1)
        sim = zm @ zk.t()  # (H, H), positive = diagonal
        H = sim.size(0)
        diag = sim.diag().unsqueeze(1)  # (H,1) positive scores per row
        # rank of positive among all candidates (1 = best)
        ranks = (sim >= diag).sum(dim=1).float()  # >= counts self too
        top1 = (ranks <= 1).float().mean().item()
        top5 = (ranks <= 5).float().mean().item()
        top10 = (ranks <= 10).float().mean().item()
        mrr = (1.0 / ranks).mean().item()
        # pairwise retrieval AUC: frac of negatives scored below the positive
        neg_below = (diag > sim).sum(dim=1).float()  # excludes self (diag==diag false)
        auc = (neg_below / max(H - 1, 1)).mean().item()
        proj_m.train(); proj_k.train()
        return {"auc": auc, "top1": top1, "top5": top5, "top10": top10, "mrr": mrr}

    n = tr_m.size(0)
    best_auc = -1.0
    best_metrics = {}
    best_state = None
    bad = 0
    for ep in range(args.epochs):
        proj_m.train(); proj_k.train()
        order = torch.randperm(n, device=device)
        losses = []
        for s in range(0, n, args.batch_size):
            bi = order[s:s + args.batch_size]
            if bi.numel() < 2:
                continue
            zm = proj_m(M[tr_m[bi]]); zk = proj_k(K[tr_k[bi]])
            loss = infonce(zm, zk)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        hm = holdout_metrics(ho_m, ho_k) if ho_m.numel() > 1 else {"auc": float("nan")}
        improved = hm["auc"] > best_auc + args.min_delta
        if improved:
            best_auc = hm["auc"]; best_metrics = hm; bad = 0
            best_state = ({k_: v.detach().clone() for k_, v in proj_m.state_dict().items()},
                          {k_: v.detach().clone() for k_, v in proj_k.state_dict().items()})
        else:
            bad += 1
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[align] ep {ep+1}/{args.epochs} loss={np.mean(losses):.4f} "
                  f"hold_auc={hm['auc']:.3f} top1={hm.get('top1',float('nan')):.3f} "
                  f"top5={hm.get('top5',float('nan')):.3f} top10={hm.get('top10',float('nan')):.3f} "
                  f"mrr={hm.get('mrr',float('nan')):.3f} (best_auc={best_auc:.3f})", flush=True)
        if bad >= args.patience:
            print(f"[align] early stop at ep {ep+1} (no holdout-AUC gain {args.patience} eps)", flush=True)
            break

    if best_state is not None:
        proj_m.load_state_dict(best_state[0]); proj_k.load_state_dict(best_state[1])
    print(f"[align] BEST holdout: auc={best_auc:.3f} "
          f"top1={best_metrics.get('top1',float('nan')):.3f} "
          f"top5={best_metrics.get('top5',float('nan')):.3f} "
          f"top10={best_metrics.get('top10',float('nan')):.3f} "
          f"mrr={best_metrics.get('mrr',float('nan')):.3f}", flush=True)

    # Emit aligned embedding proj_m(m_u) for ALL valid-Morgan drugs (incl. unseen).
    proj_m.eval(); proj_k.eval()
    with torch.no_grad():
        all_m_rows = [i for i, _ in enumerate(m_ids) if m_valid[i]]
        all_ids = [m_ids[i] for i in all_m_rows]
        idx = torch.tensor(all_m_rows, dtype=torch.long, device=device)
        z_m = F.normalize(proj_m(M[idx]), dim=1).cpu().numpy().astype(np.float32)
        # residual proxy where k_u available
        residual = np.full(len(all_ids), np.nan, dtype=np.float32)
        is_seen = np.zeros(len(all_ids), dtype=bool)
        for j, d in enumerate(all_ids):
            is_seen[j] = d in seen
            ki = k_row.get(d)
            if ki is not None and k_ncount[ki] > 0:
                zk = F.normalize(proj_k(K[ki:ki + 1].to(device)), dim=1).cpu().numpy()[0]
                residual[j] = float(np.linalg.norm(z_m[j] - zk))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT, drug_ids=np.array(all_ids), z_m=z_m, residual=residual, is_seen=is_seen,
        align_dim=args.align_dim, temp=args.temp,
        holdout_auc_best=np.float32(best_auc),
        holdout_top5_best=np.float32(best_metrics.get("top5", float("nan"))),
        holdout_top10_best=np.float32(best_metrics.get("top10", float("nan"))),
        holdout_mrr_best=np.float32(best_metrics.get("mrr", float("nan"))),
    )
    print(f"[align] saved -> {OUT} (z_m {z_m.shape}, best holdout AUC={best_auc:.3f})",
          flush=True)


if __name__ == "__main__":
    main()
