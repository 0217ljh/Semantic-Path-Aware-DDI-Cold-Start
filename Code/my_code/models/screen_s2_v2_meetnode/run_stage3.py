"""Run Stage 3 (i1 PK/PD dual-channel) + per-class DiD specialization analysis.

Trains _PerModeEmerGNN_PKPD (counts-only, two channels, two global gates).
Then post-hoc:
  - combined test_s2 AUC (compare to Stage 1 ~0.767)
  - learned gates softplus(b_pk), softplus(b_pd)
  - per-class channel-ablation DiD using ddi_pk_pd_labels.csv (analysis-only;
    labels are weak/keyword-derived -> caveat reported)

DiD logic (codex r17):
  removing PK channel == pd_only ; removing PD channel == pk_only
  PK_drop_on_PK = AUC_full(PKpairs) - AUC_pd_only(PKpairs)
  PD_drop_on_PK = AUC_full(PKpairs) - AUC_pk_only(PKpairs)
  spec_PK = PK_drop_on_PK - PD_drop_on_PK   (>0 => PK channel matters more for PK pairs)
  spec_PD = PD_drop_on_PD - PK_drop_on_PD
  DiD     = spec_PK + spec_PD               (>0 => clean specialization)
Per-class AUC uses class positives vs ALL test_s2 negatives (shared neg pool).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.screen_s2_v2_meetnode.stage3_pkpd_trainer import _PerModeEmerGNN_PKPD  # noqa
from my_code.utils.run_logger import RunLogger  # noqa

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
PKPD_LABELS = ROOT / "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
SPLITS_DIR = ROOT / "Code/data/KG/drugbank/splits/seed42"


def _build_pair2type() -> dict:
    """Canonical-pair -> ddi_type from ALL seed42 parquet splits (the only source
    of ddi_type). The PKL test pairs (model vocab) are looked up here."""
    m: dict[tuple, str] = {}
    for f in glob.glob(str(SPLITS_DIR / "*.parquet")):
        d = pd.read_parquet(f)
        if "ddi_type" not in d.columns:
            continue
        for a, b, t in zip(d["drug_a_id"].astype(str), d["drug_b_id"].astype(str), d["ddi_type"]):
            m[tuple(sorted((a, b)))] = t
    return m


def _auc(model, pos_df, neg_df, channel):
    yp = model.predict_proba_channel(pos_df, channel=channel)
    yn = model.predict_proba_channel(neg_df, channel=channel)
    y = np.concatenate([np.ones(len(yp)), np.zeros(len(yn))])
    s = np.concatenate([yp, yn])
    return float(roc_auc_score(y, s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neutral-to", choices=["both", "pk", "pd", "none"], default="both")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()
    tag = args.tag or f"stage3_pkpd_neutral{args.neutral_to}_seed{args.seed}"

    with RunLogger(script="run_stage3", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed, neutral_to=args.neutral_to)
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))
        trainer = _PerModeEmerGNN_PKPD(
            n_dim=64, length=3, feat="M", learning_rate=1e-3, batch_size=32,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50, eval_strategy="epoch",
            save_strategy="no", load_best_model_at_end=True, run_dir=str(rl.run_dir),
            pkpd_neutral_to=args.neutral_to,
        )
        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0

        # overall combined + branch AUCs
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].reset_index(drop=True)
        neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]].reset_index(drop=True)
        auc_full = _auc(trainer, pos, neg, None)
        auc_noaux = _auc(trainer, pos, neg, "no_aux")
        g_pk, g_pd = trainer._aux_mlp.gates()
        print(f"[s3] combined={auc_full:.4f} emergnn-only={auc_noaux:.4f} "
              f"gates: b_pk={g_pk:.3f} b_pd={g_pd:.3f}", flush=True)

        # ---- per-class DiD ----
        # Label the PKL test positives (model vocab) via global pair->type map.
        lab = pd.read_csv(PKPD_LABELS).set_index("ddi_type")["pk_pd_label"].to_dict()
        pair2type = _build_pair2type()
        tp = ds.splits.test_s2
        tp = tp[tp["label"] == 1][["drug_a_id", "drug_b_id"]].reset_index(drop=True)
        cls = [lab.get(pair2type.get(tuple(sorted((str(a), str(b))))))
               for a, b in zip(tp["drug_a_id"], tp["drug_b_id"])]
        tp = tp.assign(_cls=cls)
        pk_pos = tp[tp["_cls"] == "PK"][["drug_a_id", "drug_b_id"]].reset_index(drop=True)
        pd_pos = tp[tp["_cls"] == "PD"][["drug_a_id", "drug_b_id"]].reset_index(drop=True)
        n_pk, n_pd, n_un = len(pk_pos), len(pd_pos), int(pd.isna(pd.Series(cls)).sum())
        print(f"[s3] test positives by class: PK={n_pk} PD={n_pd} unlabeled={n_un}", flush=True)

        did = {}
        if n_pk >= 30 and n_pd >= 30:
            # per-class AUC vs shared neg pool, under full / pk_only / pd_only
            a_pk_full = _auc(trainer, pk_pos, neg, None)
            a_pk_pkonly = _auc(trainer, pk_pos, neg, "pk_only")   # remove PD
            a_pk_pdonly = _auc(trainer, pk_pos, neg, "pd_only")   # remove PK
            a_pd_full = _auc(trainer, pd_pos, neg, None)
            a_pd_pkonly = _auc(trainer, pd_pos, neg, "pk_only")
            a_pd_pdonly = _auc(trainer, pd_pos, neg, "pd_only")
            PK_drop_on_PK = a_pk_full - a_pk_pdonly   # remove PK channel
            PD_drop_on_PK = a_pk_full - a_pk_pkonly   # remove PD channel
            PK_drop_on_PD = a_pd_full - a_pd_pdonly
            PD_drop_on_PD = a_pd_full - a_pd_pkonly
            spec_PK = PK_drop_on_PK - PD_drop_on_PK
            spec_PD = PD_drop_on_PD - PK_drop_on_PD
            did = {
                "a_pk_full": a_pk_full, "a_pk_pkonly": a_pk_pkonly, "a_pk_pdonly": a_pk_pdonly,
                "a_pd_full": a_pd_full, "a_pd_pkonly": a_pd_pkonly, "a_pd_pdonly": a_pd_pdonly,
                "PK_drop_on_PK": PK_drop_on_PK, "PD_drop_on_PK": PD_drop_on_PK,
                "PK_drop_on_PD": PK_drop_on_PD, "PD_drop_on_PD": PD_drop_on_PD,
                "spec_PK": spec_PK, "spec_PD": spec_PD, "DiD": spec_PK + spec_PD,
            }
            print(f"[s3] DiD specialization = {spec_PK + spec_PD:+.4f} "
                  f"(spec_PK={spec_PK:+.4f}, spec_PD={spec_PD:+.4f})", flush=True)
            print(f"[s3]   PK pairs: full={a_pk_full:.4f} rm_PK={a_pk_pdonly:.4f} rm_PD={a_pk_pkonly:.4f}")
            print(f"[s3]   PD pairs: full={a_pd_full:.4f} rm_PK={a_pd_pdonly:.4f} rm_PD={a_pd_pkonly:.4f}")
        else:
            print(f"[s3] insufficient per-class positives for DiD (PK={n_pk},PD={n_pd})", flush=True)

        payload = {
            "config": vars(args), "fit_sec": fit_sec,
            "metrics": {
                "combined_auc": auc_full, "emergnn_only_auc": auc_noaux,
                "gate_pk": g_pk, "gate_pd": g_pd,
                "n_pk_pos": n_pk, "n_pd_pos": n_pd, "n_unlabeled": n_un,
                "did": did,
            },
            "caveat": "PK/PD labels are weak keyword-derived; analysis-only, never trained on.",
        }
        (rl.run_dir / "results.json").write_text(json.dumps(payload, indent=2))
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=auc_full)
        print(f"[s3] done. results -> {rl.run_dir/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
