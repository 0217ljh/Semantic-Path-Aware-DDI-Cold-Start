"""A1 pilot (EmerGNN / PATH paradigm variant): the path-based encoder for the
encoder x task grid, producing EmerGNN's pre-scorer pair representation for the
warm/cold rank-transfer analysis. Sibling of train_gcn_rank_pilot.py (node,
relation-agnostic) and train_rgcn_rank_pilot.py (node, relation-aware).

Design (codex-reviewed 2026-07-04, thread 019f2ef3):
- KNOWLEDGE-ONLY, cold-start-safe EmerGNN variant. EmerGNN natively seeds its
  message passing with molecular Morgan fingerprints (feat='M') or a learned
  per-entity embedding (feat='E'); the first violates our no-molecular-features
  constraint, the second is cold-start-unsafe (an unseen drug's ID embedding is
  untrained noise). We instead seed with the SAME KG-STRUCTURAL entity features
  R-GCN used (12-type one-hot + standardized log-degree = 13-d), fed through the
  EXISTING feat='M' interface (structural matrix passed as `morgan_features`,
  morgan_feat_dim=13). This needs ZERO change to the faithful baseline model:
  feat='M' only linearly projects whatever per-entity feature matrix it is
  given, and the relation-gated bidirectional path propagation (rel_kg +
  attention) — the knowledge mechanism — is untouched. So this is EmerGNN's PATH
  mechanism, knowledge-only, directly comparable to the R-GCN node pilot.
  NOTE this is NOT the faithful EmerGNN baseline (which uses Morgan + its own
  inductive shuffle_train protocol); it is EmerGNN's ARCHITECTURE used as the
  path-paradigm encoder for the diagnosis grid.
- KG scope = drug_incident (1-hop drug neighborhood; the full merged KG is
  infeasible for EmerGNN — OOM / hours per epoch). The KG tensors
  (edge_src/dst/rel with EmerGNN's forward+reverse+self-loop convention),
  entity2id and n_base_rel are built by REUSING the baseline's kg_setup builder
  (baseline.emergnn._shared.ensure_kg_setup_cache); we ignore its morgan_mat.
  A matched-scope R-GCN control (R-GCN re-run on the same drug_incident edges)
  is a separate follow-up, to neutralize any "the divergence is scope-driven"
  objection.
- Pre-scorer pair representation = embed = cat([head_hid, tail_hid]) (2*n_dim),
  extracted BEFORE the Wr scorer. This is EmerGNN's own pre-readout rep; it is a
  DIFFERENT pair-feature map from R-GCN's [h_u*h_v ; |h_u-h_v|] (asymmetric
  concat vs symmetric), so the rank analysis is always "each model's own
  pre-readout representation", not a shared feature map. Saved with the same
  pilot_Z.npz schema (Z_/raw_/y_/logit_/deg_) so analyze_rank_sufficiency.py
  runs unchanged (EmerGNN has no bottleneck MLP, so Z == raw == embed here).
- Data protocol identical to the node pilots (imported load_data): S2-train with
  a warm holdout, cold = S2-test unseen drugs; model selection on warm-val.
- Training: mini-batch (EmerGNN is per-pair-batch, not full-batch), STEP-BUDGET
  with warm-val early stop (avoid the node pilots' undertraining trap). Adam.

Run (from project root, WSL conda env project_1):
  python Code/scripts/train_emergnn_rank_pilot.py --fold fold0 --steps 4000 --batch-size 32
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
    KIND_GROUPS, KIND_ORDER, N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
from baseline.emergnn.model import EmerGNN  # noqa: E402
from baseline.emergnn._shared import ensure_kg_setup_cache  # noqa: E402
# reuse the node pilot's S2 split protocol verbatim (identical data handling)
from train_gcn_rank_pilot import load_data  # noqa: E402

_KIND_TO_GROUP = {raw: grp for grp, raws in KIND_GROUPS.items() for raw in raws}
_OTHER_TYPE = KIND_ORDER.index("other")


class _MiniTrain:
    """Duck-typed stand-in for a PairDataset, enough for ensure_kg_setup_cache
    (it reads .splits for the drug pool and .drugs for SMILES; we pass no SMILES
    since we do not use Morgan features)."""

    def __init__(self, splits: dict[str, pd.DataFrame]):
        self.splits = splits
        self.drugs = None


# --------------------------------------------------------------------------- #
# Structural entity features (SAME definition as the node pilots)
# --------------------------------------------------------------------------- #
def build_structural_features(cache: dict, nodes_path: str):
    """(n_ent, N_TYPES+1) = KG-type one-hot + standardized log-degree, indexed by
    EmerGNN entity id. Type from the merged-KG node `kind`; degree from the
    drug_incident KG itself (the graph EmerGNN actually sees), so the feature is
    self-consistent with this scope and identical in form to the node pilots."""
    entity2id: dict[str, int] = cache["entity2id"]
    n_ent = int(cache["n_ent"])
    trip = np.asarray(cache["kg_triplets"], dtype=np.int64)  # (N,3) (h,t,r)

    # drug_incident-scope DISTINCT-undirected-neighbour degree, matching the node
    # pilots' MergedKG degree definition (de-dup parallel typed edges + direction
    # + self-loops so each unordered neighbour pair counts once for both ends).
    if len(trip):
        h, t = trip[:, 0], trip[:, 1]
        nonloop = h != t
        lo = np.minimum(h[nonloop], t[nonloop])
        hi = np.maximum(h[nonloop], t[nonloop])
        uniq = np.unique(lo.astype(np.int64) * n_ent + hi.astype(np.int64))
        ulo, uhi = uniq // n_ent, uniq % n_ent
        deg = np.bincount(np.concatenate([ulo, uhi]), minlength=n_ent).astype(np.float64)
    else:
        deg = np.zeros(n_ent, dtype=np.float64)
    logdeg = np.log1p(deg)

    # type id per entity from the merged-KG node kinds
    nodes = pd.read_parquet(nodes_path, columns=["id", "kind"])
    kind_map = dict(zip(nodes["id"].astype(str), nodes["kind"].astype(str)))
    type_id = np.full(n_ent, _OTHER_TYPE, dtype=np.int64)
    n_missing_type = 0
    for ent_str, eid in entity2id.items():
        k = kind_map.get(str(ent_str))
        if k is None:
            n_missing_type += 1
            continue
        type_id[eid] = KIND_ORDER.index(_KIND_TO_GROUP.get(k, "other"))

    x = np.zeros((n_ent, N_TYPES + 1), dtype=np.float32)
    x[np.arange(n_ent), type_id] = 1.0
    x[:, N_TYPES] = ((logdeg - logdeg.mean()) / (logdeg.std() + 1e-8)).astype(np.float32)
    return x, logdeg, n_missing_type


def _pairs_to_ent(df: pd.DataFrame, entity2id: dict[str, int]):
    """Map a pair frame to (head_ent, tail_ent, y). FAIL HARD on any drug missing
    from the KG entity set (mirrors the node pilots' _pairs_to_idx)."""
    a = df["drug_a_id"].astype(str).map(entity2id)
    b = df["drug_b_id"].astype(str).map(entity2id)
    bad = ~(a.notna() & b.notna())
    if int(bad.sum()):
        miss = pd.unique(pd.concat([
            df.loc[a.isna(), "drug_a_id"], df.loc[b.isna(), "drug_b_id"]]).astype(str))
        raise ValueError(f"{int(bad.sum())} pairs reference drugs not in EmerGNN KG "
                         f"(e.g. {list(miss[:5])}); aborting to avoid a coverage artifact")
    return a.to_numpy(np.int64), b.to_numpy(np.int64), df["y_bin"].to_numpy(np.float32)


# --------------------------------------------------------------------------- #
# Pre-scorer representation (inline EmerGNN.forward up to `embed`, before Wr)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _pair_embed(model: EmerGNN, head: torch.Tensor, tail: torch.Tensor,
                es, ed, er) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (embed, logit) for a batch. embed = cat([head_hid, tail_hid]) is
    the pre-scorer representation (feat='M' path); logit = Wr(embed)."""
    head_embed = model._entity_embed(head)
    tail_embed = model._entity_embed(tail)
    ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
    hid_uv = model._propagate(head, head_embed, ht_embed, es, ed, er)
    B = head.size(0)
    b_arange = torch.arange(B, device=head.device)
    tail_hid = hid_uv[tail, b_arange]
    hid_vu = model._propagate(tail, tail_embed, ht_embed, es, ed, er)
    head_hid = hid_vu[head, b_arange]
    embed = torch.cat([head_hid, tail_hid], dim=-1)  # (B, 2*n_dim), feat='M'
    logit = model.Wr(embed).squeeze(-1)
    return embed, logit


@torch.no_grad()
def _eval_auc(model, es, ed, er, ia, ib, y, device, batch_size):
    model.eval()
    ps = []
    for s in range(0, len(y), batch_size):
        h = torch.as_tensor(ia[s:s + batch_size], device=device)
        t = torch.as_tensor(ib[s:s + batch_size], device=device)
        _, logit = _pair_embed(model, h, t, es, ed, er)
        ps.append(torch.sigmoid(logit).cpu().numpy())
    return roc_auc_score(y, np.concatenate(ps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--steps", type=int, default=4000, help="total optimizer steps (mini-batch)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-dim", type=int, default=64)
    ap.add_argument("--length", type=int, default=3, help="EmerGNN message-passing depth L")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-8)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=600, help="warm-val early-stop patience in STEPS")
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--use-checkpoint", action="store_true",
                    help="gradient-checkpoint the message passing (saves memory, ~2x slower); "
                         "OFF by default since drug_incident scope is small enough to fit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg-nodes", default=DEFAULT_NODES_PATH)
    ap.add_argument("--kg-edges", default=DEFAULT_EDGES_PATH)
    ap.add_argument("--tag", default="emergnn_ddi800_s2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} cuda={torch.cuda.is_available()}", flush=True)

    # -- data (identical S2 protocol to the node pilots) ---------------------
    tr, warm_val, warm_test, cold_test = load_data(args.fold, args.warm_frac, args.seed)
    pool_train = pd.concat([tr, warm_val, warm_test], ignore_index=True)  # full train-role pool

    # -- EmerGNN drug_incident KG via the baseline builder (reused) ----------
    merged_edges = (ROOT / args.kg_edges).resolve()
    cache = ensure_kg_setup_cache(
        _MiniTrain({"train": pool_train, "test": cold_test}),
        backbone_kg_source="merged", merged_kg_path=merged_edges)
    entity2id = cache["entity2id"]
    n_ent = int(cache["n_ent"])
    n_base_rel = int(cache["n_base_rel"])
    es = torch.as_tensor(cache["edge_src"], dtype=torch.long, device=device)
    ed = torch.as_tensor(cache["edge_dst"], dtype=torch.long, device=device)
    er = torch.as_tensor(cache["edge_rel"], dtype=torch.long, device=device)
    print(f"[kg] scope=drug_incident n_ent={n_ent} n_base_rel={n_base_rel} "
          f"edges(+rev+self)={es.shape[0]}", flush=True)

    # -- structural entity features (knowledge-only seed) --------------------
    x, logdeg, n_missing_type = build_structural_features(cache, args.kg_nodes)
    print(f"[feat] structural entity feats={x.shape} (type-onehot+logdeg); "
          f"{n_missing_type} entities missing merged-KG kind -> 'other'", flush=True)

    packs = {
        "train": _pairs_to_ent(tr, entity2id),
        "warm": _pairs_to_ent(warm_test, entity2id),
        "cold": _pairs_to_ent(cold_test, entity2id),
    }
    ia_tr, ib_tr, y_tr = _pairs_to_ent(tr, entity2id)
    yv_ia, yv_ib, yv = _pairs_to_ent(warm_val, entity2id)

    # -- model (EmerGNN architecture, feat='M' fed KG-structural features) ---
    model = EmerGNN(n_ent=n_ent, n_base_rel=n_base_rel, n_dim=args.n_dim,
                    length=args.length, feat="M",
                    morgan_features=x, morgan_feat_dim=N_TYPES + 1).to(device)
    model.set_chunk_size(args.chunk_size)
    model.set_use_checkpoint(args.use_checkpoint)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -- run dir -------------------------------------------------------------
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__train_emergnn_rank_pilot__{args.tag}__seed{args.seed}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[cfg] steps={args.steps} batch={args.batch_size} n_dim={args.n_dim} "
        f"L={args.length} lr={args.lr} wd={args.weight_decay} patience={args.patience}")

    # -- train (mini-batch, step budget, warm-val early stop) ----------------
    n_train = len(y_tr)
    y_tr_t = torch.as_tensor(y_tr, device=device)
    order = rng.permutation(n_train)
    cursor = 0
    best_auc, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if cursor >= n_train:   # exhausted this permutation -> reshuffle (tail batch consumed)
            order = rng.permutation(n_train)
            cursor = 0
        idx = order[cursor:cursor + args.batch_size]   # may be a partial tail batch
        cursor += args.batch_size
        h = torch.as_tensor(ia_tr[idx], device=device)
        t = torch.as_tensor(ib_tr[idx], device=device)
        yb = y_tr_t[idx]

        model.train()
        opt.zero_grad()
        logits = model(h, t, es, ed, er)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            va = _eval_auc(model, es, ed, er, yv_ia, yv_ib, yv, device, args.batch_size)
            log(f"[step {step}/{args.steps}] loss={loss.item():.4f} warm_val_auc={va:.4f} "
                f"elapsed={time.time()-t0:.0f}s")
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
            for name, (ia, ib, y) in packs.items():
                embs, logits = [], []
                for s in range(0, len(y), args.batch_size):
                    h = torch.as_tensor(ia[s:s + args.batch_size], device=device)
                    t = torch.as_tensor(ib[s:s + args.batch_size], device=device)
                    emb, logit = _pair_embed(model, h, t, es, ed, er)
                    embs.append(emb.cpu().numpy().astype(np.float32))
                    logits.append(logit.cpu().numpy().astype(np.float32))
                emb = np.concatenate(embs); logit = np.concatenate(logits)
                # EmerGNN has no bottleneck MLP: the pre-scorer rep IS embed, so
                # Z == raw == embed (analyze --key raw is the analysis target).
                arrs[f"Z_{name}"] = emb
                arrs[f"raw_{name}"] = emb
                arrs[f"y_{name}"] = y.astype(np.float32)
                arrs[f"logit_{name}"] = logit
                arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
                auc = roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit)))
                aucs[name] = float(auc)
                log(f"[extract{suffix}] {name}: n={len(y)} head_auc={auc:.4f} embed={emb.shape}")
        np.savez_compressed(run_dir / f"pilot_Z{suffix}.npz", **arrs)
        return aucs

    best_aucs = extract(best_state if best_state is not None else last_state, "")
    last_aucs = extract(last_state, "_last")

    meta = {"run_id": run_id, "fold": args.fold, "seed": args.seed, "n_dim": args.n_dim,
            "length": args.length, "n_ent": n_ent, "n_base_rel": n_base_rel,
            "kg_scope": "drug_incident", "best_warm_val_auc": best_auc,
            "feat": "M(structural-13d)", "setting": "label-inductive, graph-transductive"}
    results = {"run_id": run_id, "best_warm_val_auc": best_auc,
               "head_aucs_best": best_aucs, "head_aucs_last": last_aucs}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"[done] saved Z (best + last) + artifacts to {run_dir}")
    print(f"\nNEXT: python Code/scripts/analyze_rank_sufficiency.py "
          f"--z {run_dir/'pilot_Z.npz'} --key raw --title EmerGNN-DDI", flush=True)


if __name__ == "__main__":
    main()
