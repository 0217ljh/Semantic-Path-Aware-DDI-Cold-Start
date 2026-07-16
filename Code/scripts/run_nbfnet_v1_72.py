"""Run NBFNet v1.72 — dual-source signed-bilinear interference (action-aware).

v1.72 = v1.71 backbone + SIGNED fields + ACTION-signed edges + endpoint Hadamard
interference readout. Built-in 2×2 attribution: --activation {relu,signed} ×
--combine {additive,hadamard}; D=signed+hadamard is the interference core. Action
kept ON in all cells (--use-action off is a separate ablation).

Edge signs: het:CuG=+1 / het:CdG=-1 ; db:{target,enzyme,transporter,carrier} via
drug_*.csv action (agonist/activator/inducer=+1 ; antagonist/inhibitor/blocker=-1 ;
other/neutral/conflict=+1). Inverse edges inherit the same sign (model side).

Examples:
  # 2x2 attribution (binary S2), one cell:
  python Code/scripts/run_nbfnet_v1_72.py --epochs 100 --seed 42 \
      --activation signed --combine hadamard --tag v172_D_signed_hadamard
  # signed stability smoke (AMP off, 5 epoch):
  python Code/scripts/run_nbfnet_v1_72.py --epochs 5 --no-amp \
      --activation signed --combine hadamard --tag v172_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError("could not locate project root (Code/data/KG)")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from baseline.emergnn.kg_builder_merged import build_kg_from_merged_parquet  # noqa: E402
from my_code.models.nbfnet_v1_72.nbfnet_trainer import NBFNetTrainerV172  # noqa: E402
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
MERGED_EDGES = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DB_FILTERED = ROOT / "Code/data/KG/drugbank/filtered"
ACTION_PAIRS = ROOT / "Code/data/KG/drugbank/enriched/action_pairs.csv"

# action -> sign (codex recipe; 0 = neutral, becomes +1 in the final sign)
_POS = {"agonist", "partial agonist", "activator", "inducer", "potentiator",
        "stimulator", "positive allosteric modulator"}
_NEG = {"antagonist", "inhibitor", "blocker", "suppressor", "inverse agonist",
        "negative allosteric modulator"}
# relation -> (csv filename, id column)
_DB_REL_CSV = {
    "db:target": ("drug_targets.csv", "target_id"),
    "db:enzyme": ("drug_enzymes.csv", "enzyme_id"),
    "db:transporter": ("drug_transporters.csv", "transporter_id"),
    "db:carrier": ("drug_carriers.csv", "carrier_id"),
}


def _action_to_sign(a) -> int:
    if not isinstance(a, str):
        return 0
    a = a.strip().lower()
    if a in _POS:
        return 1
    if a in _NEG:
        return -1
    return 0


def _pkl_for_seed(seed: int) -> Path:
    p = PKL_DIR / f"seed{int(seed)}.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"seed pkl not found: {p}")
    return p


def _build_kg_inputs(ds):
    """Merged drug-incident KG inputs + action edge signs aligned to triplets."""
    drug_ids: set[str] = set()
    for _name, df in ds.splits.items():
        drug_ids.update(df["drug_a_id"].astype(str)); drug_ids.update(df["drug_b_id"].astype(str))
    if hasattr(ds.kg, "drug_ids"):
        drug_ids.update(ds.kg.drug_ids)
    art = build_kg_from_merged_parquet(MERGED_EDGES, sorted(drug_ids), verbose=True)
    e2i = art["entity2id"]; n_ent = int(art["n_ent"])
    trip = np.asarray(art["triplets"], dtype=np.int64)
    id2ent = art["id2entity"]; id2rel = {v: k for k, v in art["rel2id"].items()}
    n_kg_rel = int(art["n_rel"]); ddi_rel_id = n_kg_rel; n_base_rel = n_kg_rel + 1

    # train DDI [h,t,ddi_rel_id]
    pos_df = ds.splits.train
    a = pos_df["drug_a_id"].astype(str).map(e2i); b = pos_df["drug_b_id"].astype(str).map(e2i)
    valid = a.notna() & b.notna()
    train_ddi = np.stack([a[valid].astype(np.int64).to_numpy(),
                          b[valid].astype(np.int64).to_numpy(),
                          np.full(int(valid.sum()), ddi_rel_id, dtype=np.int64)], axis=1)

    # ---- action edge signs aligned to `trip` order ----
    # (drug_str, dst_str) -> sign, from drug_*.csv with action; conflict -> neutral
    pair_sign: dict[tuple[str, str], int] = {}
    conflict = 0
    for rel, (fname, idcol) in _DB_REL_CSV.items():
        fp = DB_FILTERED / fname
        if not fp.is_file():
            print(f"[v172-signs] skip {fname} (missing)", flush=True); continue
        g = pd.read_csv(fp)
        if idcol not in g.columns or "action" not in g.columns:
            print(f"[v172-signs] skip {fname} (no {idcol}/action)", flush=True); continue
        prefix = rel + ":"
        for drug, ent, act in zip(g["drugbank_id"].astype(str), g[idcol].astype(str), g["action"]):
            s = _action_to_sign(act)
            if s == 0:
                continue
            key = (drug, prefix + ent)
            if key in pair_sign and pair_sign[key] != s:
                pair_sign[key] = 0; conflict += 1   # mixed +/- -> neutral
            elif key not in pair_sign:
                pair_sign[key] = s

    signs = np.ones(len(trip), dtype=np.float32)
    n_signed = 0
    for i in range(len(trip)):
        r = id2rel[int(trip[i, 2])]
        if r == "het:CuG":
            signs[i] = 1.0
        elif r == "het:CdG":
            signs[i] = -1.0; n_signed += 1
        elif r.startswith("db:"):
            s = pair_sign.get((id2ent[int(trip[i, 0])], id2ent[int(trip[i, 1])]), 0)
            if s != 0:
                signs[i] = float(s); n_signed += (s < 0)
    n_neg = int((signs < 0).sum())
    print(f"[v172-signs] base edges={len(trip)} neg_signed={n_neg} "
          f"(CuG +/CdG -; db:* via action; conflicts->neutral={conflict})", flush=True)
    return (trip, train_ddi, e2i, n_ent, n_base_rel, ddi_rel_id, signs)


def _load_pd_pairs() -> set:
    """Canonical-sorted set of PD (convergent/opposing) drug pairs (eval-only)."""
    if not ACTION_PAIRS.is_file():
        return set()
    ap = pd.read_csv(ACTION_PAIRS)
    pd_ = ap[ap["chain_type"].isin(["PD_convergent_target", "PD_opposing_target"])]
    out = set()
    for x, y in zip(pd_["drug_a_id"].astype(str), pd_["drug_b_id"].astype(str)):
        out.add((x, y) if x <= y else (y, x))
    return out


def _canon(x, y):
    x, y = str(x), str(y)
    return (x, y) if x <= y else (y, x)


def _eval_test_s2(trainer, ds, e2i):
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    yp = trainer.predict_proba(pos, e2i); yn = trainer.predict_proba(neg, e2i)
    s = np.concatenate([yp, yn]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    out = {"auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "nll": float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))),
           "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    # PD slice (eval-only): PD positives vs all negatives ; non-PD positives vs all negatives
    pdset = _load_pd_pairs()
    is_pd = np.array([_canon(a, b) in pdset for a, b in
                      zip(pos["drug_a_id"], pos["drug_b_id"])])
    for name, mask in [("PD", is_pd), ("nonPD", ~is_pd)]:
        if mask.sum() >= 10:
            ss = np.concatenate([yp[mask], yn]); yy = np.concatenate([np.ones(int(mask.sum())), np.zeros(len(neg))])
            out[f"auc_{name}"] = float(roc_auc_score(yy, ss)); out[f"n_pos_{name}"] = int(mask.sum())
    return out, (pos, neg, s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-dim", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--neg-ratio", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--shuffle-train-mode", type=str, default="S2")
    p.add_argument("--shuffle-ratio", type=float, default=0.8)
    p.add_argument("--log-step-every", type=int, default=50)
    p.add_argument("--eval-every-n-epochs", type=int, default=1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    # v1.72 knobs
    p.add_argument("--activation", type=str, default="signed", choices=["signed", "relu"])
    p.add_argument("--combine", type=str, default="hadamard", choices=["hadamard", "additive"])
    p.add_argument("--use-action", dest="use_action", action="store_true", default=True)
    p.add_argument("--no-action", dest="use_action", action="store_false")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"nbfnet_v1_72_{args.activation}_{args.combine}_seed{args.seed}"
    with RunLogger(script="run_nbfnet_v1_72", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed, kg_source="merged",
                    activation=args.activation, combine=args.combine, use_action=args.use_action)
        print(f"[nbfnet-v1.72] config: {vars(args)}", flush=True)

        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(_pkl_for_seed(args.seed)))
        (base_kg, train_ddi, e2i, n_ent, n_base_rel, ddi_rel_id, signs) = _build_kg_inputs(ds)
        print(f"[nbfnet-v1.72] KG: n_ent={n_ent} n_base_rel={n_base_rel} ddi_rel_id={ddi_rel_id} "
              f"n_base_edges={len(base_kg)} n_train_ddi={len(train_ddi)}", flush=True)

        trainer = NBFNetTrainerV172(
            d=args.n_dim, n_layers=args.n_layers, mlp_hidden=args.mlp_hidden,
            n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, neg_ratio=args.neg_ratio,
            early_stop_patience=args.early_stop_patience, shuffle_train_mode=args.shuffle_train_mode,
            shuffle_ratio=args.shuffle_ratio, log_step_every=args.log_step_every,
            eval_strategy="epoch", device="cuda", seed=args.seed,
            use_amp=not args.no_amp, amp_dtype=args.amp_dtype, eval_every_n_epochs=args.eval_every_n_epochs,
            activation=args.activation, combine=args.combine, use_action=args.use_action,
        )
        trainer.setup_graph(base_kg, train_ddi, n_nodes=n_ent, n_base_rel=n_base_rel,
                            ddi_rel_id=ddi_rel_id, base_kg_signs=signs)
        trainer.init_model()

        val_pos = ds.splits.val_s2[["drug_a_id", "drug_b_id"]]
        val_neg = ds.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]
        t0 = time.time()
        fit_metrics = trainer.fit(val_pos, val_neg, e2i, rl.run_dir)
        fit_sec = time.time() - t0

        m, (pos, neg, s, y) = _eval_test_s2(trainer, ds, e2i)
        print(f"[nbfnet-v1.72] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} NLL={m['nll']:.4f} "
              f"| PD AUC={m.get('auc_PD')} (n={m.get('n_pos_PD')}) "
              f"nonPD AUC={m.get('auc_nonPD')} (n={m.get('n_pos_nonPD')}) "
              f"| fit={fit_sec/3600:.2f}h (v1.71 ref 0.7943)", flush=True)

        (rl.run_dir / "results.json").write_text(json.dumps(
            {"config": vars(args), "fit_sec": fit_sec, "fit_metrics": fit_metrics,
             "metrics": {"test_s2": m}}, indent=2))
        np.savez(rl.run_dir / "test_s2_scores.npz",
                 pair_a=np.concatenate([pos["drug_a_id"].astype(str).values, neg["drug_a_id"].astype(str).values]),
                 pair_b=np.concatenate([pos["drug_b_id"].astype(str).values, neg["drug_b_id"].astype(str).values]),
                 y_true=y, y_score=s)
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[nbfnet-v1.72] run_dir: {rl.run_dir}", flush=True)


if __name__ == "__main__":
    main()
