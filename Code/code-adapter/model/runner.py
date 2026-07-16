"""Stage 5-6 - the training/eval loop (harness-owned).

Owns: the protocol epoch loop, leak-free eval contexts + asserts, cold-val
best-epoch selection (on task.val_monitor), the full report-metrics suite every
eval epoch, checkpointing (mode-aware content delegated to the model; the
CheckpointPolicy controls which/where), and the final cold-test extraction +
results.json / train.log write. The model stays protocol-agnostic.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from model_meta import Protocol
from .contracts import CheckpointPolicy, RankModel
from .metrics import compute_metrics
from .protocol import ColdStartProtocol


def _filter_to_known(data, known: set, log):
    """Drop pairs/edges whose either drug is outside the model's KG coverage."""
    def mask(pairs):
        if len(pairs) == 0:
            return np.zeros(0, bool)
        return np.array([str(a) in known and str(b) in known for a, b in pairs[:, :2]])
    mt, mv, me = mask(data.train_pairs), mask(data.cold_val_pairs), mask(data.cold_test_pairs)
    ms, mn = mask(data.seen_ddi), mask(data.train_neg)
    tp = data.train_pairs[mt]
    log(f"[coverage] dropped (drug outside model KG): train={int((~mt).sum())} "
        f"cold_val={int((~mv).sum())} cold_test={int((~me).sum())} seen_ddi={int((~ms).sum())}")
    return replace(
        data, train_pairs=tp, train_labels=data.train_labels[mt],
        cold_val_pairs=data.cold_val_pairs[mv], cold_val_labels=data.cold_val_labels[mv],
        cold_test_pairs=data.cold_test_pairs[me], cold_test_labels=data.cold_test_labels[me],
        seen_ddi=data.seen_ddi[ms],
        train_neg=(data.train_neg[mn] if len(data.train_neg) else data.train_neg),
        train_drugs=(np.unique(tp.reshape(-1)) if len(tp) else data.train_drugs))


def _require(enc, pairs: np.ndarray) -> None:
    """Order-exact alignment: logits present, one row per input pair, pair_ids
    identical + in the same order (no partial / permutation by the model)."""
    n = len(pairs)
    if enc.logits is None:
        raise ValueError(f"encode_pairs ({enc.repr_kind}): logits required")
    if len(enc.logits) != n or len(enc.pair_ids) != n:
        raise ValueError(f"encode_pairs must return one row per input pair "
                         f"(logits={len(enc.logits)}, pair_ids={len(enc.pair_ids)}, pairs={n})")
    if not np.array_equal(np.asarray(enc.pair_ids).astype(str), np.asarray(pairs).astype(str)):
        raise ValueError("encode_pairs must return pair_ids identical + in input order")
    if enc.valid_mask is not None and (len(enc.valid_mask) != n or not bool(np.all(enc.valid_mask))):
        raise ValueError("partial extraction not supported: valid_mask must be aligned + all-true")


def _save_predictions(run_dir: Path, pairs: np.ndarray, logits, labels, task, log) -> None:
    """Write per-pair cold-test predictions to predictions_cold_test.parquet
    (drug_a, drug_b, label, pred, correct + prob columns). Feeds RQ3 per-sample /
    per-type analysis. Best-effort: never aborts a run."""
    try:
        import pandas as pd
        pr = np.asarray(pairs); y = np.asarray(labels); lg = np.asarray(logits)
        cols = {"drug_a": pr[:, 0].astype(str), "drug_b": pr[:, 1].astype(str),
                "label": y.astype(np.int64)}
        if task.is_binary:
            prob = 1.0 / (1.0 + np.exp(-lg.reshape(-1)))
            cols["pred"] = (prob >= 0.5).astype(np.int64)
            cols["prob_pos"] = prob.astype(np.float32)
        else:
            import scipy.special as sp
            proba = sp.softmax(lg, axis=-1)
            pred = proba.argmax(1)
            cols["pred"] = pred.astype(np.int64)
            cols["p_pred"] = proba[np.arange(len(pred)), pred].astype(np.float32)
            cols["p_true"] = proba[np.arange(len(y)), y.astype(int)].astype(np.float32)
        cols["correct"] = (cols["pred"] == y).astype(np.int64)
        pd.DataFrame(cols).to_parquet(run_dir / "predictions_cold_test.parquet", index=False)
        log(f"[predictions] wrote {len(y)} cold-test rows "
            f"(acc={float(cols['correct'].mean()):.4f}) -> predictions_cold_test.parquet")
    except Exception as e:  # noqa: BLE001 - predictions are best-effort, never abort a run
        log(f"[warn] predictions save failed: {e}")


def _save_repr(run_dir: Path, pairs: np.ndarray, pair_repr, labels, log) -> None:
    """Persist the leak-free cold-test pair representations (step-6 output 'repr') to
    repr_cold_test.npz. The RQ3.1 clustering / RQ3.2 dumbbell analyses (step 8) need the
    extracted embeddings, not just logits, so encode_pairs' pair_repr is dumped here alongside
    _save_predictions' logits. Best-effort: never aborts a run."""
    try:
        if pair_repr is None:
            log("[repr] encode_pairs returned no pair_repr; skipping repr_cold_test.npz")
            return
        pr = np.asarray(pairs)
        rep = np.asarray(pair_repr, dtype=np.float32)
        y = np.asarray(labels)
        if len(rep) != len(pr) or len(y) != len(pr):        # _require only aligns logits/pair_ids;
            log(f"[warn] repr/pairs/labels length mismatch (repr={len(rep)}, pairs={len(pr)}, "
                f"labels={len(y)}); skipping repr_cold_test.npz")      # guard against a misaligned repr
            return
        np.savez_compressed(run_dir / "repr_cold_test.npz",
                            drug_a=pr[:, 0].astype(str), drug_b=pr[:, 1].astype(str),
                            repr=rep, label=y.astype(np.int64))
        log(f"[repr] wrote {rep.shape[0]}x{rep.shape[1] if rep.ndim > 1 else 1} cold-test "
            f"pair reprs -> repr_cold_test.npz")
    except Exception as e:  # noqa: BLE001 - repr dump is best-effort, never abort a run
        log(f"[warn] repr save failed: {e}")


def _save_ckpt(state: dict, meta: dict, path: Path, log) -> None:
    """Write a SELF-DESCRIBING checkpoint bundle {state, **meta} so a later frozen
    run can reconstruct the matching backbone (model class + hp + task) and load it.
    See model.checkpoints.load_checkpoint for the read side."""
    try:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": state, **meta}, path)
    except Exception as e:  # noqa: BLE001 - disk ckpt is best-effort, never abort a run
        log(f"[warn] checkpoint save failed ({path}): {e}")


def run_training(model: RankModel, data, protocol: Protocol, run_dir: Path, hp: dict,
                 *, epochs: int, eval_every: int, patience: int,
                 ckpt: CheckpointPolicy | None = None, emerging_ratio: float = 0.8,
                 seed: int = 42, min_targets: int = 20, resume_from=None, log=print) -> dict:
    task = data.task
    proto = ColdStartProtocol(protocol, task, emerging_ratio, seed,
                              binary_pos_cap=hp.get("binary_pos_cap"),    # None -> full (default)
                              mc_target_cap=hp.get("mc_target_cap"))      # SMOKE-only; None -> full
    ckpt = ckpt or CheckpointPolicy()
    run_dir = Path(run_dir)
    t_all0 = time.perf_counter()

    # -- tee logging to run_dir/train.log + _logs/<run>.log (append per line) --
    _log_paths = []
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = run_dir.parent / "_logs"; logs_dir.mkdir(parents=True, exist_ok=True)
        _log_paths = [run_dir / "train.log", logs_dir / f"{run_dir.name}.log"]
    except OSError as e:
        log(f"[warn] disk logging disabled: {e}")

    def _log(msg):
        log(msg)
        for _p in list(_log_paths):
            try:
                with open(_p, "a", encoding="utf-8") as _f:
                    _f.write(str(msg) + "\n")
            except OSError as e:
                log(f"[warn] disk log write failed ({_p}: {e})"); _log_paths.remove(_p)

    model.setup(task, hp)
    data = _filter_to_known(data, model.known_drugs(), _log)
    #: self-describing metadata baked into every on-disk checkpoint. hp is the
    #: model's RESOLVED config (defaults merged) so a frozen wrapper can rebuild it.
    ckpt_meta = {"model": model.__class__.__name__,
                 "hp": (model.effective_hp() or dict(hp)),
                 "task_kind": task.kind, "n_classes": task.n_classes}

    best, best_state, bad, best_epoch = -1e18, None, 0, -1
    if resume_from is not None:                              # continue a prior (non-converged) run
        from model.checkpoints import load_checkpoint
        _b = load_checkpoint(resume_from)
        model.load_state_dict(_b["state"])
        # Seed `best` with the loaded state's cold_val so best.pt is NOT overwritten by a worse
        # continued epoch (the loop only saves when v > best). best.pt carries cold_best_val; a
        # last.pt bundle does not, so evaluate the loaded state to get a correct floor.
        if "cold_best_val" in _b:
            best = float(_b["cold_best_val"]); best_epoch = int(_b.get("best_epoch", -1))
        else:
            _fctx = proto.fact_context(data)
            _enc = model.encode_pairs(data.cold_val_pairs, _fctx); _require(_enc, data.cold_val_pairs)
            best = float(compute_metrics(_enc.logits, data.cold_val_labels, task)[task.val_monitor])
            best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        _log(f"[resume] loaded {resume_from} (best cold_val={best:.4f} @ep{best_epoch}); "
             f"continuing up to {epochs} more epochs (optimizer state reset -> warm restart)")
    val_curve = []
    t_train0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        rng = np.random.default_rng(seed * 100000 + ep)
        epd = proto.epoch(data, ep, rng)
        if len(epd.target_pairs) < min_targets:
            continue
        t_ep0 = time.perf_counter()
        out = model.train_epoch(epd, rng)
        ep_secs = time.perf_counter() - t_ep0
        rec = {"epoch": ep, "mean_loss": round(float(out.mean_loss), 6), "secs": round(ep_secs, 1)}
        stop = False
        if ep % eval_every == 0 or ep == epochs:
            fctx = proto.fact_context(data)
            proto.assert_pairs_absent(data.cold_val_pairs, fctx)
            enc = model.encode_pairs(data.cold_val_pairs, fctx)
            _require(enc, data.cold_val_pairs)
            vm = compute_metrics(enc.logits, data.cold_val_labels, task)
            rec["cold_val"] = {k: round(v, 6) for k, v in vm.items()}
            v = vm[task.val_monitor]
            _log(f"[ep {ep}/{epochs}] loss={out.mean_loss:.4f} "
                 f"cold_val {task.val_monitor}={v:.4f} | {rec['cold_val']} ({ep_secs:.0f}s)")
            if v > best:
                best, bad, best_epoch = v, 0, ep
                best_state = copy.deepcopy(model.state_dict())
                if ckpt.save_best and ckpt.to_disk and ckpt.save_dir:
                    _save_ckpt(best_state, {**ckpt_meta, "best_epoch": ep,
                                            "cold_best_val": float(v)},
                               Path(ckpt.save_dir) / "best.pt", _log)
            else:
                bad += eval_every
                if bad >= patience:
                    stop = True
        val_curve.append(rec)
        if stop:
            _log(f"[early-stop] no cold_val {task.val_monitor} improvement ~{patience}"); break

    # save final-epoch state BEFORE loading the best (so last.pt is truly the last)
    if ckpt.save_last and ckpt.to_disk and ckpt.save_dir:
        _save_ckpt(model.state_dict(), {**ckpt_meta, "epoch": ep},
                   Path(ckpt.save_dir) / "last.pt", _log)
    if best_state is None:                    # no eval improved (metric NaN every eval)
        _log("[warn] no cold-val selection made (val metric NaN?); using final-epoch state")
        best_state = copy.deepcopy(model.state_dict())
        best, best_epoch = float("nan"), ep   # honest bookkeeping: no valid selection
    model.load_state_dict(best_state)
    _log(f"[select] cold-best {task.val_monitor}={best:.4f} @ep{best_epoch} "
         f"(train {time.perf_counter() - t_train0:.0f}s)")

    # -- final cold-test extraction (leak-free) --------------------------------
    fctx = proto.fact_context(data)
    proto.assert_pairs_absent(data.cold_test_pairs, fctx)
    cold = model.encode_pairs(data.cold_test_pairs, fctx)
    _require(cold, data.cold_test_pairs)
    test_metrics = compute_metrics(cold.logits, data.cold_test_labels, task)
    _save_predictions(run_dir, data.cold_test_pairs, cold.logits, data.cold_test_labels, task, _log)
    _save_repr(run_dir, data.cold_test_pairs, cold.pair_repr, data.cold_test_labels, _log)
    _log(f"[cold-test] {test_metrics}")

    results = {
        "model": model.__class__.__name__, "dataset": data.dataset, "task": task.kind,
        "n_classes": task.n_classes, "fold": data.fold, "protocol": protocol.value,
        "cold_best_val": round(float(best), 6), "best_epoch": best_epoch,
        "cold_test": {k: round(v, 6) for k, v in test_metrics.items()},
        "val_monitor": task.val_monitor, "hp": {k: hp[k] for k in hp},
        "val_curve": val_curve}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"[done] {run_dir.name} (total {time.perf_counter() - t_all0:.0f}s)")
    return results


__all__ = ["run_training"]
