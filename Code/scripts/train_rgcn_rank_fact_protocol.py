"""A1 pilot (R-GCN, FACT-protocol): keep SEEN drugs' known DDI edges in the
message graph, to test whether the clean warm/cold rank-transfer result survives
when seen drugs are DDI-enriched (user 2026-07-05: the all-DDI-masked pilot
handicaps warm; realistic cold-start keeps seen DDIs).

Difference vs train_rgcn_rank_pilot.py (all-DDI-masked KG): a SINGLE new symmetric
DDI relation is appended to the biomedical KG, carrying ONLY the `fact_pos` seen-
seen positive edges. Everything supervised/evaluated has its own edge ABSENT from
the graph (codex 2026-07-05 must-fix: a predicted pair whose edge is a message
edge leaks a direct-edge feature into Z_train that vanishes at test and spuriously
breaks the transfer probe).

Protocol (codex-reviewed 2026-07-05):
- Outer S2 split (unchanged): seen-seen labeled pairs (train), unseen-involved
  pairs (cold test). load_data() also carves seen-drug warm_val / warm_test.
- Within the seen-seen TRAIN pool, split POSITIVES into:
    fact_pos   -> DDI message edges in the graph (NOT supervised)
    train_pos  -> supervised positive targets (edge NOT in graph)
  Supervised train = train_pos + all train-pool negatives.
- Graph = bio relations (build_relation_edges) + 1 appended symmetric DDI relation
  containing only fact_pos. No warm_val/warm_test/cold positive, and no cold drug,
  ever appears in a DDI fact edge (asserted).
- Node degree FEATURE stays BIO-ONLY (codex: don't leak DDI context through the
  input feature). deg_* saved for analyze_rank_sufficiency.py is deg_bio; deg_msg_*
  / deg_ddi_* are saved as EXTRA diagnostics to attribute any transfer break.
- Same model / head / full-batch training / pilot_Z schema as the pilot, so both
  analyze_rank_sufficiency.py (transfer) and analyze_rank_within_split.py run
  unchanged.

Run (project root, WSL conda env project_1):
  python Code/scripts/train_rgcn_rank_fact_protocol.py --fold fold0 --epochs 1500 --amp
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("project root (with Code/data/KG) not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
# reuse the pilot's edge builder + model + eval, and the GCN pilot's data helpers.
from train_rgcn_rank_pilot import (  # noqa: E402
    build_relation_edges, RGCNPairModel, _eval_auc)
from train_gcn_rank_pilot import load_data, _pairs_to_idx  # noqa: E402


def _canon_pos_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Canonical (min,max) id set of the POSITIVE pairs in df (for leak checks)."""
    pos = df[df["y_bin"] == 1]
    a = pos["drug_a_id"].astype(str).to_numpy()
    b = pos["drug_b_id"].astype(str).to_numpy()
    return {(x, y) if x <= y else (y, x) for x, y in zip(a, b)}


def _split_fact_target(tr: pd.DataFrame, fact_frac: float, seed: int):
    """Split TRAIN-pool positives into fact_pos (graph edges) + train_pos
    (supervised). Negatives all stay supervised. Returns (fact_pos_df,
    train_target_df)."""
    pos = tr[tr["y_bin"] == 1].reset_index(drop=True)
    neg = tr[tr["y_bin"] == 0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pos))
    n_fact = int(round(fact_frac * len(pos)))
    fact_pos = pos.iloc[perm[:n_fact]].reset_index(drop=True)
    train_pos = pos.iloc[perm[n_fact:]].reset_index(drop=True)
    train_target = pd.concat([train_pos, neg], ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    return fact_pos, train_target, train_pos, neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=1500,
                    help="number of optimizer STEPS (full-batch: 1 step/epoch)")
    ap.add_argument("--fact-frac", type=float, default=0.5,
                    help="fraction of TRAIN-pool positives kept as DDI graph facts "
                         "(rest become supervised train positives)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-bases", type=int, default=16)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg-nodes", default=DEFAULT_NODES_PATH)
    ap.add_argument("--kg-edges", default=DEFAULT_EDGES_PATH)
    ap.add_argument("--tag", default="rgcn_ddi800_s2_factproto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} cuda={torch.cuda.is_available()}", flush=True)

    # -- bio KG -> graph tensors (features BIO-ONLY, same as the pilot) -------
    kg = MergedKG.from_parquet(args.kg_nodes, args.kg_edges)
    n = kg.n_nodes
    id2i = kg.id_to_idx
    logdeg_bio = np.log1p(kg.degree.astype(np.float64))
    x = np.zeros((n, N_TYPES + 1), dtype=np.float32)
    x[np.arange(n), kg.type_id.astype(np.int64)] = 1.0
    x[:, N_TYPES] = ((logdeg_bio - logdeg_bio.mean()) / (logdeg_bio.std() + 1e-8)).astype(np.float32)
    x = torch.from_numpy(x).to(device)

    edge_index, edge_type, num_relations, rel_names, rel_meta = build_relation_edges(
        args.kg_edges, id2i)
    print(f"[kg] nodes={n} bio_num_relations={num_relations} "
          f"bio_edges={edge_index.shape[1]}", flush=True)

    # -- data + fact/target split --------------------------------------------
    tr, warm_val, warm_test, cold_test = load_data(args.fold, args.warm_frac, args.seed)
    fact_pos, train_target, train_pos, train_neg = _split_fact_target(
        tr, args.fact_frac, args.seed)

    # -- leak guards (codex) -------------------------------------------------
    fact_set = _canon_pos_set(fact_pos)
    for name, df in [("train_pos", pd.concat([train_pos]).assign(y_bin=1)),
                     ("warm_val", warm_val), ("warm_test", warm_test),
                     ("cold_test", cold_test)]:
        overlap = fact_set & _canon_pos_set(df if "y_bin" in df else df.assign(y_bin=1))
        assert not overlap, f"LEAK: {len(overlap)} fact_pos edges also in {name} positives"
    seen_drugs = set(pd.concat([tr["drug_a_id"], tr["drug_b_id"]]).astype(str))
    cold_drugs = set(pd.concat([cold_test["drug_a_id"], cold_test["drug_b_id"]]).astype(str)) - seen_drugs
    fact_drugs = set(fact_pos["drug_a_id"].astype(str)) | set(fact_pos["drug_b_id"].astype(str))
    assert not (fact_drugs & cold_drugs), "LEAK: a cold (unseen) drug appears in a DDI fact edge"

    # -- append ONE symmetric DDI relation with fact_pos edges ---------------
    fa = fact_pos["drug_a_id"].astype(str).map(id2i).to_numpy()
    fb = fact_pos["drug_b_id"].astype(str).map(id2i).to_numpy()
    if np.isnan(fa).any() or np.isnan(fb).any():
        n_bad = int(np.isnan(fa).sum() + np.isnan(fb).sum())
        raise ValueError(f"{n_bad} fact_pos endpoints not in KG id map")
    fa = fa.astype(np.int64); fb = fb.astype(np.int64)
    ddi_rel_id = num_relations
    ddi_row = np.concatenate([fa, fb])
    ddi_col = np.concatenate([fb, fa])
    ddi_ety = np.full(ddi_row.shape[0], ddi_rel_id, dtype=np.int64)
    aug_edge_index = torch.cat(
        [edge_index, torch.from_numpy(np.vstack([ddi_row, ddi_col]))], dim=1).to(device)
    aug_edge_type = torch.cat([edge_type, torch.from_numpy(ddi_ety)]).to(device)
    aug_num_relations = num_relations + 1

    # degree diagnostics (node-level -> per-pair at extraction)
    ddi_node_deg = np.bincount(np.concatenate([fa, fb]), minlength=n).astype(np.float64)
    logdeg_ddi = np.log1p(ddi_node_deg)
    logdeg_msg = np.log1p(kg.degree.astype(np.float64) + ddi_node_deg)

    print(f"[split] tr={tr.shape} -> fact_pos={fact_pos.shape} train_target={train_target.shape} "
          f"(train_pos={train_pos.shape} neg={train_neg.shape}); warm_test={warm_test.shape} "
          f"cold_test={cold_test.shape}", flush=True)
    print(f"[kg+ddi] ddi_rel_id={ddi_rel_id} ddi_edges(2-dir)={ddi_row.shape[0]} "
          f"aug_num_relations={aug_num_relations} aug_edges={aug_edge_index.shape[1]}", flush=True)

    ia_tr, ib_tr, y_tr = _pairs_to_idx(train_target, id2i)
    yv_ia, yv_ib, yv = _pairs_to_idx(warm_val, id2i)
    packs = {"train": (ia_tr, ib_tr, y_tr),
             "warm": _pairs_to_idx(warm_test, id2i),
             "cold": _pairs_to_idx(cold_test, id2i)}
    ia_tr_t = torch.as_tensor(ia_tr, device=device)
    ib_tr_t = torch.as_tensor(ib_tr, device=device)
    y_tr_t = torch.as_tensor(y_tr, device=device)

    model = RGCNPairModel(x.shape[1], args.hidden, args.bottleneck, args.dropout,
                          num_relations=aug_num_relations, num_bases=args.num_bases,
                          num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__train_rgcn_rank_fact_protocol__{args.tag}__seed{args.seed}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[protocol] FACT: seen drugs keep {args.fact_frac:.0%} of train positives as DDI "
        f"message edges; every supervised/eval pair has its own edge absent. "
        f"features bio-only. aug_num_relations={aug_num_relations}")

    # -- train (step budget, full-batch; identical to the pilot) -------------
    best_auc, best_state, bad = -1.0, None, 0
    for step in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(x, aug_edge_index, aug_edge_type, ia_tr_t, ib_tr_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr_t)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % args.eval_every == 0 or step == args.epochs:
            va = _eval_auc(model, x, aug_edge_index, aug_edge_type, yv_ia, yv_ib, yv, device)
            log(f"[step {step}/{args.epochs}] loss={loss.item():.4f} warm_val_auc={va:.4f} "
                f"time={time.time()-t0:.2f}s")
            if va > best_auc:
                best_auc, bad = va, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += args.eval_every
                if bad >= args.patience:
                    log(f"[early-stop] no warm_val improvement for ~{args.patience} steps")
                    break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def extract(state, suffix: str) -> dict:
        model.load_state_dict(state)
        model.eval()
        arrs, aucs = {}, {}
        with torch.no_grad():
            h = model.encode(x, aug_edge_index, aug_edge_type)
            for name, (ia, ib, y) in packs.items():
                iat = torch.as_tensor(ia, device=device)
                ibt = torch.as_tensor(ib, device=device)
                hu, hv = h[iat], h[ibt]
                raw = torch.cat([hu * hv, (hu - hv).abs()], dim=-1)
                z = model.pair_mlp(raw)
                logit = model.out(z).squeeze(-1)
                arrs[f"Z_{name}"] = z.cpu().numpy().astype(np.float32)
                arrs[f"raw_{name}"] = raw.cpu().numpy().astype(np.float32)
                arrs[f"y_{name}"] = y.astype(np.float32)
                arrs[f"logit_{name}"] = logit.cpu().numpy().astype(np.float32)
                # deg_* = BIO-ONLY (for analyze_rank_sufficiency degree control)
                arrs[f"deg_{name}"] = np.vstack([logdeg_bio[ia], logdeg_bio[ib]]).T.astype(np.float32)
                # EXTRA diagnostics to attribute a transfer break
                arrs[f"deg_msg_{name}"] = np.vstack([logdeg_msg[ia], logdeg_msg[ib]]).T.astype(np.float32)
                arrs[f"deg_ddi_{name}"] = np.vstack([logdeg_ddi[ia], logdeg_ddi[ib]]).T.astype(np.float32)
                auc = roc_auc_score(y, torch.sigmoid(logit).cpu().numpy())
                aucs[name] = float(auc)
                log(f"[extract{suffix}] {name}: n={len(y)} head_auc={auc:.4f} "
                    f"Z={arrs[f'Z_{name}'].shape} mean_ddi_deg={float(np.expm1(arrs[f'deg_ddi_{name}']).mean()):.2f}")
        np.savez_compressed(run_dir / f"pilot_Z{suffix}.npz", **arrs)
        return aucs

    best_aucs = extract(best_state if best_state is not None else last_state, "")
    last_aucs = extract(last_state, "_last")

    meta = {"run_id": run_id, "fold": args.fold, "seed": args.seed, "fact_frac": args.fact_frac,
            "hidden": args.hidden, "num_layers": args.num_layers, "num_bases": args.num_bases,
            "aug_num_relations": aug_num_relations, "ddi_rel_id": ddi_rel_id,
            "bottleneck": args.bottleneck, "best_warm_val_auc": best_auc, "amp": bool(args.amp),
            "n_fact_pos": int(len(fact_pos)), "n_train_target": int(len(train_target)),
            "setting": "FACT-protocol: seen keep DDI facts, targets edge-absent, bio-only features"}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "results.json").write_text(json.dumps(
        {"run_id": run_id, "best_warm_val_auc": best_auc,
         "head_aucs_best": best_aucs, "head_aucs_last": last_aucs}, indent=2))
    log(f"[done] saved Z (best + last) to {run_dir}")
    print(f"\nNEXT:\n  python Code/scripts/analyze_rank_sufficiency.py --z {run_dir/'pilot_Z.npz'} "
          f"--key raw --title R-GCN-fact-transfer\n  python Code/scripts/analyze_rank_within_split.py "
          f"--z {run_dir/'pilot_Z.npz'} --key raw --title R-GCN-fact-within", flush=True)


if __name__ == "__main__":
    main()
