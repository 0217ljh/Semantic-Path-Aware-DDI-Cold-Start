"""Sub-epoch collapse probe: does GD overshoot a useful config WITHIN epoch 1?

The naked-AND adapter's val AUC peaks at epoch 1 then decays (per-epoch). Two
hypotheses for that collapse:
  (A) distribution-shift ceiling — ep1 is the best reachable on the shifted S2 test
      distribution; slowing optimization only flattens the curve at the same peak.
  (B) optimizer overshoot — GD skips a better/complementary config; since the model
      converges within ONE epoch, any skipped config is most likely INTRA-epoch.

Per-epoch eval is too coarse to see (B). This probe trains the naked-AND Step-0
model with FINE-GRAINED val eval (~25x per epoch) for a couple of epochs, at TWO
learning rates (base and base/5) sharing the SAME init / seed / minibatch order
(paired, codex design), and records (step, train_loss_ewma, val_auc). Read-only on
the cached AND supports (reuses run_spmn_v2_standalone). Writes the paired curves to
json and prints a verdict-oriented summary.

Decision rule (codex):
  (B) if base has a sub-epoch checkpoint beating epoch-1-end val by >= +0.01, OR the
      base/5 run attains a higher best val AFTER matching base's epoch-1-end train
      loss.
  (A) if no sub-epoch point materially beats epoch-1-end and base/5 only time-stretches
      the same peak.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_spmn_v2_standalone import (  # noqa: E402
    CACHE_DIR, ROOT, _evaluate, _gather, _load_frames,
)

sys.path.insert(0, str(ROOT / "Code"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from my_code.models.spmn_v1.retrieval import N_REL_BUCKETS, N_TYPES  # noqa: E402
from my_code.models.spmn_v1.struct_features import symmetric_binary_dim  # noqa: E402
from my_code.models.spmn_v2 import StandaloneHead, StructuralVariableCore  # noqa: E402

KEYS = ("struct", "y", "med", "typ", "rela", "relb", "da", "db", "offsets")


def _load_and_cache(seed, l_max, k_per_type, n_max):
    cache = CACHE_DIR / (f"spmn_v2_supports_and_seed{seed}_lmax{l_max}"
                         f"_kpt{k_per_type}_nmax{n_max}_cp0.npz")
    if not cache.is_file():
        raise SystemExit(f"AND support cache not found: {cache}")
    print(f"[probe] support cache HIT: {cache}", flush=True)
    z = np.load(cache)
    names = sorted({k.split("__")[0] for k in z.files})
    return {name: {k: z[f"{name}__{k}"] for k in KEYS} for name in names}


def _train_curve(data, perms, lr, max_epochs, batch, eval_every, device, args):
    """One paired run at a given lr; returns list of {step, frac, train_loss, val_auc}."""
    core = StructuralVariableCore(
        n_entities=178029, n_types=N_TYPES, n_rel_buckets=N_REL_BUCKETS,
        struct_dim=symmetric_binary_dim(), d=args.d, hidden=args.hidden,
        dropout=args.dropout, use_entity_embed=True, use_asym=False,
        use_absdiff_embed=False, use_dist_attn=False, max_dist=args.l_max - 1)
    model = StandaloneHead(core, hidden=args.hidden, dropout=args.dropout).to(device)
    model.load_state_dict(_INIT_STATE)  # identical init across LRs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train = data["train"]; n_train = len(train["y"])
    recent = deque(maxlen=50)
    curve = []; step = 0
    n_steps_ep = (n_train + batch - 1) // batch
    for epoch in range(max_epochs):
        perm = perms[epoch]
        model.train()
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            b, y = _gather(train, idx, device)
            loss = loss_fn(model(b), y)
            opt.zero_grad(); loss.backward(); opt.step()
            recent.append(loss.item()); step += 1
            if step % eval_every == 0 or (epoch == 0 and s + batch >= n_train):
                val = _evaluate(model, data["val_s2"], device, batch)
                curve.append({"step": step, "epoch_frac": round(step / n_steps_ep, 3),
                              "train_loss": float(np.mean(recent)),
                              "val_auc": round(val["auc"], 5)})
                model.train()
    return curve, n_steps_ep


_INIT_STATE = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--k-per-type", type=int, default=64)
    ap.add_argument("--n-max", type=int, default=400)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--lr-divisor", type=float, default=5.0)
    ap.add_argument("--max-epochs", type=int, default=2)
    ap.add_argument("--evals-per-epoch", type=int, default=25)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    args = ap.parse_args()

    global _INIT_STATE
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _load_frames(args.seed)  # log which frames; cache reuse below
    data = _load_and_cache(args.seed, args.l_max, args.k_per_type, args.n_max)
    mu = data["train"]["struct"].mean(0, keepdims=True)
    sd = data["train"]["struct"].std(0, keepdims=True) + 1e-6
    for name in data:
        data[name]["struct"] = ((data[name]["struct"] - mu) / sd).astype(np.float32)

    n_train = len(data["train"]["y"])
    n_steps_ep = (n_train + args.batch - 1) // args.batch
    eval_every = max(1, n_steps_ep // args.evals_per_epoch)

    # identical init + identical minibatch order for both LRs (paired probe).
    init_core = StructuralVariableCore(
        n_entities=178029, n_types=N_TYPES, n_rel_buckets=N_REL_BUCKETS,
        struct_dim=symmetric_binary_dim(), d=args.d, hidden=args.hidden,
        dropout=args.dropout, use_entity_embed=True, use_asym=False,
        use_absdiff_embed=False, use_dist_attn=False, max_dist=args.l_max - 1)
    init_model = StandaloneHead(init_core, hidden=args.hidden, dropout=args.dropout)
    _INIT_STATE = {k: v.clone() for k, v in init_model.state_dict().items()}
    rng = np.random.RandomState(args.seed)
    perms = [rng.permutation(n_train) for _ in range(args.max_epochs)]

    print(f"[probe] n_train={n_train} steps/epoch={n_steps_ep} eval_every={eval_every} "
          f"(~{n_steps_ep // eval_every}/epoch)", flush=True)
    runs = {}
    for label, lr in (("base", args.base_lr), ("low", args.base_lr / args.lr_divisor)):
        print(f"\n[probe] === {label} lr={lr:.2e} ===", flush=True)
        torch.manual_seed(args.seed)  # same dropout RNG stream per run
        curve, _ = _train_curve(data, perms, lr, args.max_epochs, args.batch,
                                eval_every, device, args)
        runs[label] = curve
        ep1 = [c for c in curve if c["epoch_frac"] <= 1.0]
        ep1_end = ep1[-1] if ep1 else curve[-1]
        peak = max(curve, key=lambda c: c["val_auc"])
        print(f"  ep1-end: step{ep1_end['step']} loss={ep1_end['train_loss']:.4f} "
              f"val={ep1_end['val_auc']:.4f}")
        print(f"  PEAK:    step{peak['step']} loss={peak['train_loss']:.4f} "
              f"val={peak['val_auc']:.4f}  (Δ vs ep1-end {peak['val_auc']-ep1_end['val_auc']:+.4f})")

    # verdict-oriented summary
    base, low = runs["base"], runs["low"]
    base_ep1end = [c for c in base if c["epoch_frac"] <= 1.0][-1]
    base_peak = max(base, key=lambda c: c["val_auc"])
    base_subepoch = max((c for c in base if c["epoch_frac"] < 1.0),
                        key=lambda c: c["val_auc"], default=base_ep1end)
    # low-LR val after it matches base ep1-end train loss
    anchor = base_ep1end["train_loss"]
    low_matched = [c for c in low if c["train_loss"] <= anchor]
    low_after = max(low_matched, key=lambda c: c["val_auc"]) if low_matched else None
    print("\n[probe] ===== VERDICT INPUTS =====")
    print(f"  base ep1-end val           : {base_ep1end['val_auc']:.4f} "
          f"(train_loss {anchor:.4f})")
    print(f"  base best sub-epoch (frac<1): {base_subepoch['val_auc']:.4f} "
          f"@step{base_subepoch['step']} (Δ {base_subepoch['val_auc']-base_ep1end['val_auc']:+.4f})")
    print(f"  base best overall          : {base_peak['val_auc']:.4f} @step{base_peak['step']}")
    if low_after is not None:
        print(f"  low-LR best after matching base ep1 train_loss: "
              f"{low_after['val_auc']:.4f} @step{low_after['step']} "
              f"(Δ vs base ep1-end {low_after['val_auc']-base_ep1end['val_auc']:+.4f})")
    else:
        print(f"  low-LR did NOT reach base ep1-end train_loss ({anchor:.4f}) in "
              f"{args.max_epochs} epochs -> increase --max-epochs for a clean read")
    print("  rule: B (overshoot) if base sub-epoch Δ>=+0.01 OR low-LR-after Δ>=+0.01; "
          "A (shift ceiling) if both ~0 and low only time-stretches.")

    out = ROOT / "Code/runs/spmn_v2_standalone" / f"subepoch_probe_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "base_lr": args.base_lr,
                               "lr_divisor": args.lr_divisor, "runs": runs}, indent=2))
    print(f"\n[probe] curves -> {out}")


if __name__ == "__main__":
    main()
