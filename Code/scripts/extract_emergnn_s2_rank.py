"""EmerGNN OFFICIAL-S2 rank analysis, the path-paradigm twin of
train_rgcn_rank_fact_protocol.py, on the SOTA cold-start method.

Difference from extract_emergnn_fact_protocol_rank.py (the FIXED variant):
- TRAIN via the OFFICIAL S2 protocol (shuffle_train_mode="S2", per-epoch
  reshuffle: ~20% drugs marked emerging each epoch, their DDI edges removed,
  emerging-emerging pairs are the sparse-query supervised targets). This is the
  paper-faithful cold-start model (cold test AUC ~0.70), NOT the dense-query
  FIXED split. Codex 2026-07-05: R-GCN is only a mechanism witness; the thesis
  must be replicated on the official SOTA S2 model.
- SELECT the checkpoint by COLD val (no-warm story: the deployed cold path uses
  only the cold model; warm is dropped as a pillar).
- EXTRACT leak-free on a rebuilt FACT KG (bio + a deterministic 80% fact split of
  the seen-seen train DDI). Both the transfer-probe SOURCE (the held-out 20%
  target positives + train negatives) AND the cold test pairs are scored with
  their OWN edge ABSENT -> the transfer probe is apples-to-apples (codex
  reviewer-kill: never extract on a KG where the query pair's own edge is
  present). Cold/unseen drugs have no DDI edge in ANY KG built from train DDI.

Saves pilot_Z.npz (Z_/raw_/y_/logit_/deg_/deg_ddi_) so the exact same curve code
used for R-GCN runs on it unchanged: transfer (load-bearing) + degree_only +
degree-residual (make-or-break hub control) + within-split (oracle, secondary).

Run (project root, WSL conda env project_1, rspmm backend):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && \
    EMERGNN_BACKEND=rspmm PYTHONUNBUFFERED=1 python -u Code/scripts/extract_emergnn_s2_rank.py --epochs 6"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402
# reuse the codex-reviewed helpers from the FIXED variant (imported, not copied)
from extract_emergnn_fact_protocol_rank import (  # noqa: E402
    _transfer_curve, _embed_logit_on_kg, _rebuild_fact_kg, _LeanCore,
    _load_trainable, _rank_summary, _within_curve, MERGED_EDGES, RANKS_SURF)
from extract_emergnn_multickpt_rank import _pairs  # noqa: E402


def _sub(ia, ib, y, n, seed):
    if n >= len(y):
        return ia, ib, y
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return ia[idx], ib[idx], y[idx]


class _S2Core(_LeanCore):
    """Same lean-checkpoint S2 core, but a COLD-ONLY _validate (no-warm story):
    the parent _LeanCore._validate requires a _warm_val_df; we drop warm entirely
    and track cold_val AUROC (on the fact KG) for best-model selection."""

    def _validate(self, val):
        if getattr(self, "_fact_kg_cache", None) is None:
            self._fact_kg_cache, _ = _rebuild_fact_kg(self, self._tr_df, self.device)
        ia, ib, y = _pairs(self._cold_val_df, self._entity2id)
        _, logit = _embed_logit_on_kg(self, ia, ib, self._fact_kg_cache,
                                      self.batch_size, self.device)
        ca = float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit))))
        print(f"    [val cold] cold_val_auc={ca:.4f}", flush=True)
        return ca


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ddi800")
    ap.add_argument("--split", default="cold_s2")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=20,
                    help="full merged KG (7.1M edges): cold peaks ~ep16 (~6.3min/ep w/ rspmm); "
                         "run ~20 with EMERGNN_EARLY_STOP_PATIENCE=5. cold-best ckpt is selected.")
    ap.add_argument("--shuffle-ratio", type=float, default=0.8,
                    help="S2 keep-ratio (train) AND fact-KG fact fraction for extraction.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--auc-sample", type=int, default=2000)
    ap.add_argument("--train-sample", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda")

    leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), args.dataset, "binary",
                       args.split, args.fold)
    tr = leaf.train.reset_index(drop=True)   # FULL official train (no warm holdout)
    cold_val, cold_test = leaf.val, leaf.test
    print(f"[run] OFFICIAL-S2 | tr(full)={tr.shape} "
          f"cold_val={cold_val.shape} cold_test={cold_test.shape}", flush=True)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__extract_emergnn_s2_rank__{args.dataset}_{args.split}_{args.fold}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    merged_kg = Path(leaf.resources.kg.source) / MERGED_EDGES

    core = _S2Core(   # lean checkpoints (<1MB each); OFFICIAL S2 training protocol
        n_dim=64, length=3, feat="M", learning_rate=1e-3, batch_size=args.batch_size,
        n_epochs=args.epochs, device="cuda", backbone_kg_source="merged",
        merged_kg_path=merged_kg, weight_decay=1e-8, shuffle_train_mode="S2",
        shuffle_ratio=args.shuffle_ratio, log_step_every=50, eval_strategy="epoch",
        save_strategy="epoch", save_steps=500, save_total_limit=args.epochs + 2,
        load_best_model_at_end=True, run_dir=run_dir)

    core._tr_df = tr
    core._cold_val_df = cold_val.sample(min(len(cold_val), args.auc_sample),
                                        random_state=args.seed).reset_index(drop=True)

    ds = make_dataset(tr, cold_val, leaf.resources)
    t0 = time.time()
    core.fit(ds, ds)
    print(f"[fit] OFFICIAL-S2 done in {time.time()-t0:.0f}s (rspmm)", flush=True)

    # -- rebuild the deterministic FACT KG for leak-free extraction ----------
    fact_kg, target_pos = _rebuild_fact_kg(core, tr, device)
    e2i = core._entity2id
    kg_tri = np.asarray(core._kg_triplets)                 # bio KG (no DDI)
    logdeg_msg = np.log1p(np.bincount(
        np.concatenate([kg_tri[:, 0], kg_tri[:, 1]]),
        minlength=int(core._n_ent)).astype(np.float64))
    # DDI degree from ALL seen train positives (=0 for cold/unseen drugs)
    tr_pos = tr[tr["y_bin"] == 1]
    pa = tr_pos["drug_a_id"].astype(str).map(e2i); pb = tr_pos["drug_b_id"].astype(str).map(e2i)
    pok = pa.notna() & pb.notna()
    ddi_deg = np.bincount(
        np.concatenate([pa[pok].to_numpy(np.int64), pb[pok].to_numpy(np.int64)]),
        minlength=int(core._n_ent)).astype(np.float64)
    logdeg_ddi = np.log1p(ddi_deg)
    print(f"[fact-kg] rebuilt; target_pos(clean Z_train positives)={target_pos.shape}", flush=True)

    # clean labeled Z_train = FACT target positives (edge-absent) + train negatives
    tr_neg = tr[tr["y_bin"] == 0]
    na = tr_neg["drug_a_id"].astype(str).map(e2i); nb = tr_neg["drug_b_id"].astype(str).map(e2i)
    nok = na.notna() & nb.notna()
    neg_ia = na[nok].to_numpy(np.int64); neg_ib = nb[nok].to_numpy(np.int64)
    train_ia = np.concatenate([target_pos[:, 0], neg_ia])
    train_ib = np.concatenate([target_pos[:, 1], neg_ib])
    train_y = np.concatenate([np.ones(len(target_pos), np.float32),
                              np.zeros(len(neg_ia), np.float32)])

    cold_pk = _pairs(cold_test, e2i)
    cv_pk = _pairs(cold_val, e2i)

    def _auc_on_fact(ia, ib, y):
        _, logit = _embed_logit_on_kg(core, ia, ib, fact_kg, args.batch_size, device)
        return float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit))))

    # -- PER-EPOCH rank scan; select COLD-best (no-warm) ---------------------
    ckpts = sorted([p for p in (run_dir / "checkpoints").glob("checkpoint-*")
                    if (p / "trainable.pt").is_file()],
                   key=lambda p: int(p.name.rsplit("__n", 1)[-1]))
    cvs = _sub(*cv_pk, args.auc_sample, args.seed)
    tr_s = _sub(train_ia, train_ib, train_y, 4000, args.seed)
    ct_s = _sub(*cold_pk, 2000, args.seed)
    scores = []
    surface = {"epochs": [], "ranks": RANKS_SURF, "transfer_cold": [], "within_cold": []}
    for cp in ckpts:
        _load_trainable(core, cp)
        ca = _auc_on_fact(*cvs)                             # cold_val AUROC on fact KG
        Ztr = _embed_logit_on_kg(core, tr_s[0], tr_s[1], fact_kg, args.batch_size, device)[0]
        Zc = _embed_logit_on_kg(core, ct_s[0], ct_s[1], fact_kg, args.batch_size, device)[0]
        # _transfer_curve wants (Ztr,ytr,Zw,yw,Zc,yc); pass cold as the 'warm' slot too (ignored)
        _, tc = _transfer_curve(Ztr, tr_s[2], Zc, ct_s[2], Zc, ct_s[2], RANKS_SURF)
        wc = _within_curve(Zc, ct_s[2], RANKS_SURF)
        scores.append((cp, ca))
        surface["epochs"].append(cp.name)
        surface["transfer_cold"].append(tc); surface["within_cold"].append(wc)
        sC, wC = _rank_summary(tc, RANKS_SURF), _rank_summary(wc, RANKS_SURF)
        print(f"[epoch-rank] {cp.name}: cold_val={ca:.4f} | "
              f"TRANSFER cold(best={sC['best']} full={sC['full']} r@best={sC['rank_of_best']}) | "
              f"WITHIN cold(best={wC['best']} r90*={wC['r90star']})", flush=True)
        torch.cuda.empty_cache()
    (run_dir / "surface.json").write_text(json.dumps(surface, indent=2))

    # single COLD-best model -> pilot_Z (one controlled encoder)
    if scores:
        best_cp, best_ca = max(scores, key=lambda s: s[1])
        _load_trainable(core, best_cp)
        print(f"[select] cold-best (single model for pilot_Z) = {best_cp.name} "
              f"cold_val={best_ca:.4f}", flush=True)

    # -- extract train/cold on the fact KG (no-warm story: warm dropped) -----
    trp = _sub(train_ia, train_ib, train_y, args.train_sample, args.seed)
    packs = {"train": trp, "cold": cold_pk}
    arrs = {}
    for name, (ia, ib, y) in packs.items():
        emb, logit = _embed_logit_on_kg(core, ia, ib, fact_kg, args.batch_size, device)
        arrs[f"Z_{name}"] = emb; arrs[f"raw_{name}"] = emb
        arrs[f"y_{name}"] = y.astype(np.float32); arrs[f"logit_{name}"] = logit
        arrs[f"deg_{name}"] = np.vstack([logdeg_msg[ia], logdeg_msg[ib]]).T.astype(np.float32)
        arrs[f"deg_ddi_{name}"] = np.vstack([logdeg_ddi[ia], logdeg_ddi[ib]]).T.astype(np.float32)
        a = roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit)))
        print(f"[extract] {name}: n={len(y)} head_auc={a:.4f} emb={emb.shape} "
              f"deg_ddi_mean={arrs[f'deg_ddi_{name}'].mean():.3f}", flush=True)
    np.savez_compressed(run_dir / "pilot_Z.npz", **arrs)

    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "model_label": "EmerGNN-OFFICIAL-S2", "epochs": args.epochs,
        "shuffle_ratio": args.shuffle_ratio, "n_target_pos": int(len(target_pos)),
        "per_epoch_cold_val": [(cp.name, round(c, 4)) for cp, c in scores],
        "selected": (best_cp.name if scores else None),
        "note": "OFFICIAL S2 training (per-epoch reshuffle, sparse-query targets). "
                "Extract on rebuilt FACT KG (bio + 80% seen-DDI facts); Z_train = held-out "
                "fact targets + train negs (edge-absent), cold = S2 unseen (edge-absent). "
                "cold-best checkpoint. deg_=bio degree, deg_ddi_=DDI degree (=0 for cold)."},
        indent=2))
    print(f"[done] {run_dir}\n  pilot_Z.npz saved. Run the SAME curve analysis as R-GCN "
          f"(transfer + degree_only + degree-residual) on this pilot_Z.", flush=True)


if __name__ == "__main__":
    main()
