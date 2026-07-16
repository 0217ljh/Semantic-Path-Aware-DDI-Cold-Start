"""Offline TrimNet molecular-feature builder for the MRCGNN baseline.

**Baseline-side builder** (CLAUDE.md §Baseline §1: builder co-located with its
outputs inside ``_data/necessary/``). Ported + adapted from upstream
``Paper/Reference/Original-Code/MRCGNN/codes for MRCGNN/trimnet/train.py`` +
``trimnet/data_preprocessing.py`` collate logic. File-independence: COPY+adapt, no
import of ``Paper/Reference/Original-Code/`` or ``reproductions/``.

CONTRACT (from codex, Step 3):
  * SUPERVISED offline pretrain: train TrimNet as a DDI-event CLASSIFIER on the
    leaf's TRAIN pairs (upstream ``trimnet/train.py:131-213``), then dump per-drug
    128-d embeddings from the FINAL training state via ``TrimNet.get_weight``
    (``trimnet/train.py:179`` + ``models.py:173-192``) — NOT a best-val checkpoint.
  * NO LEAKAGE / COLD-START SAFE: TrimNet sees ONLY train pairs (never val/test).
    Embeddings are produced for ALL leaf drugs (seen + unseen); unseen drugs get a
    valid embedding purely from their molecular structure through the trained encoder
    — this is the cold-start mechanism.
  * OUTPUT: a ``(n_drugs, 128)`` float32 matrix keyed to the drug ordering given by
    the ``--smiles-csv`` row order (the training core uses this SAME ordering).
    Row-normalized (see NORMALIZATION below). Written to ``--out``.

NORMALIZATION decision: the row-normalize (upstream ``utils.normalize`` applied at
``data_preprocess.py:150``) is applied HERE in the builder, so the dumped ``.npy`` is
already the exact tensor the MRCGNN training core consumes as ``data_o.x`` / the skip
features. Rationale: upstream applies it once at load time in ``data_preprocess.py``;
baking it into the artifact keeps the training core's featurization trivial and avoids
double-normalization. Documented so the training core does NOT re-normalize.

Hyperparameters (faithful; upstream file:line):
  * ``TrimNet(55, 10, hidden_dim=64, depth=3, heads=4, dropout=0.2, n_classes=K)``
    — upstream ``trimnet/train.py:250`` ``TrimNet(55, 10, hidden_dim=64, depth=3,
    heads=4, dropout=0.2, outdim=1)``. ``n_classes`` = K_global (upstream fixed 65).
  * epochs 300 (``trimnet/train.py:25``); lr 1e-3, weight_decay 5e-4, batch 256
    (``:24,29,27``); Adam (``:254``); LambdaLR ``0.96**epoch`` (``:255``);
    CrossEntropyLoss (``:253``).
  * FLAGGED (could not pin a file:line default for our use): ``--epochs`` defaults to
    300 per upstream, but the builder exposes ``--epochs`` so a smoke can use fewer;
    a real feature build MUST use 300 to match upstream.

SMALL-FOLD ROBUSTNESS (fix #3): upstream trains with ``drop_last=True``
(``trimnet/train.py:69``) at batch 256. On upstream data (train >> 256) this always
yields many full batches. Under our cold-start folds a leaf's usable train pairs (after
SMILES filtering) can be < 256, and ``drop_last=True`` would then run ZERO optimizer
steps and dump RANDOM-INIT embeddings. We therefore clamp the effective batch to
``min(batch_size, n_pairs)`` and set ``drop_last=False``, guaranteeing >=1 real optimizer
step per epoch. This changes behavior ONLY in the degenerate small-fold case; on
large folds the effective batch is unchanged at 256 (only the final short batch, which
upstream dropped, is now kept — a negligible, non-degenerate difference).

CLI (usually auto-invoked by ``_shared.ensure_trimnet_features``; manual for debug):
    python Code/baseline/mrcgnn/_data/necessary/build_trimnet_features.py \\
        --smiles-csv <drug csv: drug_id,smiles>  --id-col drug_id --smiles-col smiles \\
        --train-pairs-csv <train pairs csv: drug_a_id,drug_b_id,ddi_type> \\
        --n-classes <K_global> --epochs 300 --seed 42 \\
        --out Code/baseline/mrcgnn/_data/necessary/mrcgnn_trimnet_features__<hash>__mine.npy
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

# baseline-local imports (this package), NOT the upstream tree.
# build_trimnet_features.py is at Code/baseline/mrcgnn/_data/necessary/ -> parents[4] = Code/
_PKG_ROOT = Path(__file__).resolve().parents[4]   # -> Code/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from baseline.mrcgnn.mol_features import build_drug_graphs  # noqa: E402
from baseline.mrcgnn.trimnet_model import TrimNet  # noqa: E402


def set_random_seed(seed: int) -> None:
    """Deterministic seeding (upstream ``trimnet/train.py`` seeds; correction: no
    unconditional cuda env)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _row_normalize(mx: np.ndarray) -> np.ndarray:
    """Row-normalize (upstream ``utils.normalize`` data_preprocess.py:22-29,
    applied :150). ``r_inv = 1 / rowsum`` with inf->0, scaled per row."""
    rowsum = np.asarray(mx).sum(axis=1)
    r_inv = np.power(rowsum, -1, where=rowsum != 0)
    r_inv[np.isinf(r_inv)] = 0.0
    r_inv[rowsum == 0] = 0.0
    return mx * r_inv[:, None]


class _PairDataset(Dataset):
    """Upstream ``DrugDataset`` (data_preprocessing.py:190-246): filters to pairs
    whose BOTH drugs have a graph, yields (h, t, r)."""

    def __init__(self, tri_list, graphs: dict) -> None:
        self.graphs = graphs
        self.tri = [(h, t, r) for h, t, r in tri_list
                    if h in graphs and t in graphs]

    def __len__(self) -> int:
        return len(self.tri)

    def __getitem__(self, i):
        return self.tri[i]

    def collate_fn(self, batch):
        pos_rels, h_samples, t_samples = [], [], []
        for h, t, r in batch:
            pos_rels.append(int(r))
            h_samples.append(self.graphs[h].clone())
            t_samples.append(self.graphs[t].clone())
        h_batch = Batch.from_data_list(h_samples)
        t_batch = Batch.from_data_list(t_samples)
        rels = torch.LongTensor(pos_rels)
        return h_batch, t_batch, rels


def _all_drugs_batch(drug_ids: list[str], graphs: dict) -> tuple[Batch, list[str]]:
    """Upstream ``DrugDataset1.collate_fn`` (data_preprocessing.py:325-342): one batch
    of ALL drug graphs, used by ``get_weight`` to dump per-drug embeddings. Preserves
    the ``drug_ids`` ordering so the output rows align to that ordering."""
    kept = [d for d in drug_ids if d in graphs]
    drugs = Batch.from_data_list([graphs[d].clone() for d in kept])
    return drugs, kept


def build_trimnet_features(
    smiles_map: dict[str, str],
    drug_order: list[str],
    train_tri: list[tuple[str, str, int]],
    *,
    n_classes: int,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "auto",
) -> np.ndarray:
    """Train TrimNet on TRAIN pairs, dump final-state per-drug embeddings for ALL
    drugs in ``drug_order``, row-normalize. Returns (len(drug_order), 128) float32.

    Drugs whose SMILES fail to parse get a ZERO row (so the output always has one row
    per ``drug_order`` entry; the training core treats these like SSI-DDI's missing
    drugs). Reported to stderr.
    """
    set_random_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    graphs, missing = build_drug_graphs(smiles_map)
    if missing:
        print(f"[mrcgnn-trimnet] {len(missing)} drugs had unparseable SMILES "
              f"(zero-row embeddings): {missing[:5]}{'...' if len(missing) > 5 else ''}",
              file=sys.stderr)
    if not graphs:
        raise ValueError("build_trimnet_features: zero molecular graphs built.")

    # emb width = 2 * hidden_dim (Set2Set doubling), upstream 128 for hidden 64.
    hidden_dim = 64
    emb_dim = 2 * hidden_dim

    ds = _PairDataset(train_tri, graphs)
    n_pairs = len(ds)
    if n_pairs == 0:
        raise ValueError("build_trimnet_features: zero usable TRAIN pairs "
                         "(both endpoints must have a parseable graph).")
    # Robust small-fold handling (fix #3): upstream used drop_last=True (train.py:69),
    # which would run ZERO optimizer steps — dumping RANDOM-INIT embeddings — whenever
    # n_pairs < batch_size (common under cold-start SMILES filtering). We (a) clamp the
    # effective batch to min(batch_size, n_pairs) so a full batch always exists, and
    # (b) set drop_last=False so no pairs are silently dropped. This guarantees >=1 real
    # optimizer step per epoch. Faithful otherwise (same shuffle, loss, optimizer).
    eff_batch = min(batch_size, n_pairs)
    if n_pairs < batch_size:
        print(f"[mrcgnn-trimnet] SMALL FOLD: usable train pairs {n_pairs} < batch_size "
              f"{batch_size}; clamping effective batch to {eff_batch}, drop_last=False "
              f"(prevents zero-step random-init dump)", file=sys.stderr)
    loader = DataLoader(ds, batch_size=eff_batch, shuffle=True, drop_last=False,
                        collate_fn=ds.collate_fn)

    model = TrimNet(55, 10, hidden_dim=hidden_dim, depth=3, heads=4, dropout=0.2,
                    n_classes=n_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 0.96 ** ep)

    # SUPERVISED pretrain on TRAIN pairs only (no val/test) — upstream train.py:135-213.
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for h_batch, t_batch, rels in loader:
            h_batch = h_batch.to(device)
            t_batch = t_batch.to(device)
            rels = rels.to(device)
            scores, gt = model((h_batch, t_batch, rels))
            loss = loss_fn(scores, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        scheduler.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            mean = running / max(n_batches, 1)
            print(f"[mrcgnn-trimnet] ep {epoch}/{epochs} mean_loss={mean:.4f}",
                  file=sys.stderr, flush=True)

    # FINAL-STATE embedding dump (upstream get_weight, models.py:173-192).
    model.eval()
    drugs_batch, kept = _all_drugs_batch(drug_order, graphs)
    drugs_batch = drugs_batch.to(device)
    with torch.no_grad():
        emb = model.get_weight(drugs_batch, n_drugs=len(kept)).cpu().numpy()

    # scatter into full (len(drug_order), emb_dim); missing drugs -> zero row.
    full = np.zeros((len(drug_order), emb_dim), dtype=np.float32)
    kept_pos = {d: i for i, d in enumerate(kept)}
    for row, d in enumerate(drug_order):
        if d in kept_pos:
            full[row] = emb[kept_pos[d]]

    full = _row_normalize(full).astype(np.float32)
    return full


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline TrimNet molecular-feature builder "
                                            "for the MRCGNN baseline.")
    p.add_argument("--smiles-csv", required=True, type=Path,
                   help="CSV with drug-id + smiles columns (row order = drug ordering).")
    p.add_argument("--id-col", default="drug_id")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--train-pairs-csv", required=True, type=Path,
                   help="CSV with drug_a_id,drug_b_id,ddi_type (TRAIN positives only).")
    p.add_argument("--a-col", default="drug_a_id")
    p.add_argument("--b-col", default="drug_b_id")
    p.add_argument("--type-col", default="ddi_type")
    p.add_argument("--n-classes", required=True, type=int, help="K_global.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    sdf = pd.read_csv(args.smiles_csv)
    drug_order = [str(d) for d in sdf[args.id_col].tolist()]
    smiles_map = {str(d): ("" if pd.isna(s) else str(s))
                  for d, s in zip(sdf[args.id_col], sdf[args.smiles_col])}

    tdf = pd.read_csv(args.train_pairs_csv)
    train_tri = [(str(a), str(b), int(r))
                 for a, b, r in zip(tdf[args.a_col], tdf[args.b_col], tdf[args.type_col])]

    feats = build_trimnet_features(
        smiles_map, drug_order, train_tri, n_classes=args.n_classes,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, seed=args.seed, device=args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feats)
    print(f"[mrcgnn-trimnet] wrote {args.out}  shape={feats.shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
