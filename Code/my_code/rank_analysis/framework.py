"""RankModel ABC + S2Protocol + RankHarness + RankRunWriter.

The harness OWNS the S2 epoch loop, leak-free context construction + assertions,
cold-val checkpoint selection, extraction orchestration, and measurement/output.
A model only implements: setup / train_epoch / encode_pairs / state_dict /
load_state_dict, all in DRUG-ID space.
"""
from __future__ import annotations

import abc
import copy
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import measurements as M
from .data import RankData, ref_degree_matrix
from .specs import (PairEncoding, S2ProtocolSpec, ScoringContext, TaskSpec,
                    TrainEpochOutput)


@dataclass
class EpochData:
    """One S2-shuffled epoch of training data handed to model.train_epoch."""
    fact_context: ScoringContext         # graph edges available this epoch (kept-kept DDI)
    target_pairs: np.ndarray             # (n, 2) drug-id emerging-emerging targets
    target_labels: np.ndarray            # (n,) labels (binary 0/1 with negs; multiclass class ids)


# --------------------------------------------------------------------------- #
class RankModel(abc.ABC):
    """Contract a DDI model implements to be rank-analyzable. DRUG-ID space."""

    @abc.abstractmethod
    def setup(self, task: TaskSpec, hp: dict) -> None: ...

    @abc.abstractmethod
    def train_epoch(self, epoch: EpochData, rng: np.random.Generator) -> TrainEpochOutput: ...

    @abc.abstractmethod
    def encode_pairs(self, pairs: np.ndarray, context: ScoringContext) -> PairEncoding:
        """Leak-free per-pair pre-scorer PairEncoding on the given context."""

    @abc.abstractmethod
    def head_from_repr(self, pair_repr: np.ndarray) -> np.ndarray:
        """Apply the model's OWN frozen head to a (possibly rank-truncated) pair_repr,
        returning logits (n,) binary / (n,K) multiclass. Powers the Approach-B curve
        (model-faithful rank truncation); at full rank it must reproduce encode_pairs'
        logits for the same repr."""

    @abc.abstractmethod
    def known_drugs(self) -> set:
        """Drug-ids this model can embed (after setup). The harness filters all data
        to pairs whose BOTH drugs are known, so partial-coverage datasets are safe."""

    @abc.abstractmethod
    def state_dict(self) -> dict: ...

    @abc.abstractmethod
    def load_state_dict(self, state: dict) -> None: ...


# --------------------------------------------------------------------------- #
class S2Protocol:
    """Protocol-2 (both-emerging) shuffle + leak-free context builders + asserts."""

    def __init__(self, spec: S2ProtocolSpec, task: TaskSpec):
        self.spec = spec; self.task = task

    def _emerging(self, drugs, rng):
        n_rm = len(drugs) - int(len(drugs) * self.spec.emerging_ratio)
        return set(rng.choice(drugs, size=n_rm, replace=False).tolist())

    def shuffle_epoch(self, data: RankData, rng) -> EpochData:
        seen = data.seen_ddi
        removed = self._emerging(data.train_drugs, rng)
        a_rm = np.array([str(x) in removed for x in seen[:, 0]])
        b_rm = np.array([str(x) in removed for x in seen[:, 1]])
        fact = seen[(~a_rm) & (~b_rm)]
        tgt_pos = seen[a_rm & b_rm]
        fact_ctx = ScoringContext("fact_kg", fact, absent_guarantee=False)
        empty = EpochData(fact_ctx, np.empty((0, 2), object), np.array([], np.int64))
        if self.task.is_binary:
            # both-emerging positives + both-emerging dataset negatives (S2-consistent)
            neg_all = data.train_neg
            if len(tgt_pos) == 0 or len(neg_all) == 0:
                return empty
            na = np.array([str(x) in removed for x in neg_all[:, 0]])
            nb = np.array([str(x) in removed for x in neg_all[:, 1]])
            emerg_neg = neg_all[na & nb]
            if len(emerg_neg) == 0:
                return empty
            nidx = rng.choice(len(emerg_neg), size=min(len(tgt_pos), len(emerg_neg)), replace=False)
            neg = emerg_neg[nidx]
            m = min(len(tgt_pos), len(neg))
            pairs = np.concatenate([tgt_pos[:m, :2], neg[:m]], axis=0)
            labels = np.concatenate([np.ones(m), np.zeros(m)]).astype(np.int64)
        else:
            pairs = tgt_pos[:, :2]
            labels = tgt_pos[:, 2].astype(np.int64)
        return EpochData(fact_ctx, pairs, labels)

    def fact_context(self, data: RankData) -> ScoringContext:
        """All seen DDI as facts (cold-extraction regime; cold drugs are unseen)."""
        return ScoringContext("fact_kg", data.seen_ddi, absent_guarantee=False)

    def probe_context(self, data: RankData, P: set):
        """G_probe = seen DDI with NEITHER endpoint in P; source = P x P pairs
        (binary: P x P positives + P x P dataset negatives; multiclass: typed pos)."""
        seen = data.seen_ddi
        a_in = np.array([str(x) in P for x in seen[:, 0]])
        b_in = np.array([str(x) in P for x in seen[:, 1]])
        fact = seen[(~a_in) & (~b_in)]
        src_pos = seen[a_in & b_in]
        ctx = ScoringContext("g_probe", fact, absent_guarantee=False)
        if self.task.is_binary:
            neg = data.train_neg
            if len(neg):
                na = np.array([str(x) in P for x in neg[:, 0]])
                nb = np.array([str(x) in P for x in neg[:, 1]])
                src_neg = neg[na & nb]
            else:
                src_neg = np.empty((0, 2), object)
            m = min(len(src_pos), len(src_neg))
            pairs = np.concatenate([src_pos[:m, :2], src_neg[:m]], axis=0)
            labels = np.concatenate([np.ones(m), np.zeros(m)]).astype(np.int64)
            return ctx, pairs, labels
        return ctx, src_pos[:, :2], src_pos[:, 2].astype(np.int64)

    @staticmethod
    def assert_pairs_absent(pairs: np.ndarray, context: ScoringContext):
        """Guarantee no scored pair's own DDI edge is a fact in this context."""
        facts = context.ddi_pair_set()
        for a, b in pairs:
            if tuple(sorted((str(a), str(b)))) in facts:
                raise AssertionError(f"leak: pair ({a},{b}) own DDI edge present in {context.name}")
        context.absent_guarantee = True


def _require(enc: PairEncoding, pairs: np.ndarray):
    """Full order-exact alignment: pair_repr + logits present, one row per input
    pair, and enc.pair_ids identical (same order) to the input pairs (no partial,
    no permutation/substitution by the adapter)."""
    n = len(pairs)
    if enc.pair_repr is None or enc.logits is None:
        raise ValueError(f"encode_pairs ({enc.repr_kind}): both pair_repr and logits required")
    if len(enc.pair_repr) != n or len(enc.logits) != n or len(enc.pair_ids) != n:
        raise ValueError(f"encode_pairs must return one row per input pair "
                         f"(repr={len(enc.pair_repr)}, logits={len(enc.logits)}, "
                         f"pair_ids={len(enc.pair_ids)}, pairs={n})")
    if not np.array_equal(np.asarray(enc.pair_ids).astype(str), np.asarray(pairs).astype(str)):
        raise ValueError("encode_pairs must return pair_ids identical + in the same order as input pairs")
    if enc.valid_mask is not None and (len(enc.valid_mask) != n or not bool(np.all(enc.valid_mask))):
        raise ValueError("partial extraction not supported: valid_mask must be aligned + all-true")


def _filter_to_known(data: RankData, known: set, log) -> RankData:
    """Drop pairs/edges whose either drug is outside the model's KG coverage."""
    def mask(pairs):
        if len(pairs) == 0:
            return np.zeros(0, bool)
        return np.array([str(a) in known and str(b) in known for a, b in pairs[:, :2]])
    mt, mv, me = mask(data.train_pairs), mask(data.cold_val_pairs), mask(data.cold_test_pairs)
    ms, mn = mask(data.seen_ddi), mask(data.train_neg)
    tp = data.train_pairs[mt]
    log(f"[coverage] dropped (drug outside model KG): train={int((~mt).sum())} "
        f"cold_val={int((~mv).sum())} cold_test={int((~me).sum())} "
        f"seen_ddi={int((~ms).sum())}")
    return replace(
        data, train_pairs=tp, train_labels=data.train_labels[mt],
        cold_val_pairs=data.cold_val_pairs[mv], cold_val_labels=data.cold_val_labels[mv],
        cold_test_pairs=data.cold_test_pairs[me], cold_test_labels=data.cold_test_labels[me],
        seen_ddi=data.seen_ddi[ms],
        train_neg=(data.train_neg[mn] if len(data.train_neg) else data.train_neg),
        train_drugs=(np.unique(tp.reshape(-1)) if len(tp) else data.train_drugs))


# --------------------------------------------------------------------------- #
class RankHarness:
    def __init__(self, spec: S2ProtocolSpec):
        self.spec = spec

    def run(self, model: RankModel, data: RankData, run_dir: Path, hp: dict,
            epochs: int, eval_every: int, patience: int, log=print) -> dict:
        task = data.task
        proto = S2Protocol(self.spec, task)
        t_all0 = time.perf_counter()

        # -- tee every log line to run_dir/train.log (+ _logs/<run_id>.log) so the
        #    full loss/val curve lands on disk, never only in the terminal. Open+
        #    append per line (no held handle) => partial log survives a crash. ---
        _log_paths = []
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = run_dir.parent / "_logs"; logs_dir.mkdir(parents=True, exist_ok=True)
            _log_paths = [run_dir / "train.log", logs_dir / f"{run_dir.name}.log"]
        except OSError as e:
            log(f"[warn] disk logging disabled (setup failed: {e})")

        def _log(msg):
            log(msg)                             # stdout is authoritative
            for _p in list(_log_paths):          # tee is best-effort; never abort the run
                try:
                    with open(_p, "a", encoding="utf-8") as _f:
                        _f.write(str(msg) + "\n")
                except OSError as e:
                    log(f"[warn] disk log write failed ({_p}: {e}); disabling that sink")
                    _log_paths.remove(_p)

        model.setup(task, hp)
        data = _filter_to_known(data, model.known_drugs(), _log)

        # -- harness-owned S2 epoch loop + cold-val selection -----------------
        best, best_state, bad, best_epoch = -1e9, None, 0, -1
        val_curve = []                       # per-epoch {epoch, mean_loss, secs, cold_val_<monitor>?}
        t_train0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            rng = np.random.default_rng(self.spec.seed * 100000 + ep)
            epd = proto.shuffle_epoch(data, rng)
            if len(epd.target_pairs) < 20:
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
                v = M.head_metrics(enc.logits, data.cold_val_labels, task)[task.val_monitor]
                rec[f"cold_val_{task.val_monitor}"] = round(float(v), 6)
                _log(f"[ep {ep}/{epochs}] loss={out.mean_loss:.4f} "
                     f"cold_val_{task.val_monitor}={v:.4f} ({ep_secs:.0f}s)")
                if v > best:
                    best, bad, best_epoch = v, 0, ep; best_state = copy.deepcopy(model.state_dict())
                else:
                    bad += eval_every
                    if bad >= patience:
                        stop = True
            val_curve.append(rec)
            if stop:
                _log(f"[early-stop] no cold_val improvement ~{patience}"); break
        if best_state is None:
            raise RuntimeError("no checkpoint selected (all epochs skipped / no eval)")
        model.load_state_dict(best_state)
        _log(f"[select] cold-best {task.val_monitor}={best:.4f} @ep{best_epoch} "
             f"(train {time.perf_counter() - t_train0:.0f}s)")

        # -- cold extraction on the fact KG (leak-free) ----------------------
        t_ext0 = time.perf_counter()
        fctx = proto.fact_context(data)
        proto.assert_pairs_absent(data.cold_test_pairs, fctx)
        cold = model.encode_pairs(data.cold_test_pairs, fctx)
        _require(cold, data.cold_test_pairs)
        head = M.head_metrics(cold.logits, data.cold_test_labels, task)
        deg_c = ref_degree_matrix(data.cold_test_pairs, data.ref_logdeg)
        ranks = list(self.spec.ranks)
        _log(f"[extract] head_cold {task.metric}={head.get(task.metric)}; "
             f"within-cold + degree curves (CPU sklearn, silent for multi-cls) ...")
        wc = M.within_curve(cold.pair_repr, data.cold_test_labels, ranks, task)
        wcr = M.within_curve(cold.pair_repr, data.cold_test_labels, ranks, task, deg=deg_c)
        do = M.degree_within(deg_c, data.cold_test_labels, task)
        # Approach B: model's own head under rank-r truncation (recovers head_cold at full rank)
        mh = M.model_head_curve(cold.pair_repr, data.cold_test_labels, ranks, task, model.head_from_repr)

        # -- sparse-source transfer (multi-draw) -----------------------------
        _log(f"[extract] within/degree done ({time.perf_counter() - t_ext0:.0f}s); "
             f"running {self.spec.transfer_draws} transfer draws ...")
        tc_draws, per_draw = [], []
        for d in range(self.spec.transfer_draws):
            rng = np.random.default_rng(self.spec.seed + 1000 + d)
            P = set(rng.choice(data.train_drugs,
                               size=min(self.spec.transfer_k, len(data.train_drugs)),
                               replace=False).tolist())
            pctx, src_pairs, src_labels = proto.probe_context(data, P)
            if len(src_pairs) < 100 or np.unique(src_labels).size < 2:
                _log(f"[draw {d}] degenerate src={len(src_pairs)}; skip"); continue
            keep = np.ones(len(data.cold_test_pairs), bool)
            if not task.is_binary:
                keep = np.isin(data.cold_test_labels, np.unique(src_labels))
                if keep.sum() == 0 or np.unique(data.cold_test_labels[keep]).size < 2:
                    _log(f"[draw {d}] no cold class overlap; skip"); continue
            proto.assert_pairs_absent(src_pairs, pctx)
            proto.assert_pairs_absent(data.cold_test_pairs[keep], pctx)
            src = model.encode_pairs(src_pairs, pctx); _require(src, src_pairs)
            coldp = model.encode_pairs(data.cold_test_pairs[keep], pctx)
            _require(coldp, data.cold_test_pairs[keep])
            src_deg = ref_degree_matrix(src_pairs, data.ref_logdeg)
            cold_deg = deg_c[keep]
            tc = M.transfer_curve(src.pair_repr, src_labels, coldp.pair_repr,
                                  data.cold_test_labels[keep], ranks, task)
            Rs, Rc = M.residualize(src.pair_repr, src_deg, coldp.pair_repr, cold_deg)
            tcr = M.transfer_curve(Rs, src_labels, Rc, data.cold_test_labels[keep], ranks, task)
            tc_draws.append((tc, tcr))
            per_draw.append({"draw": d, "src": int(len(src_pairs)), "cold_kept": int(keep.sum()),
                             "primary_peak": round(float(np.nanmax(list(tc.values())[0])), 4)})
            _log(f"[draw {d}] src={len(src_pairs)} cold_kept={int(keep.sum())} "
                 f"transfer_peak={per_draw[-1]['primary_peak']}")

        transfer_mean = _mean_curves([t[0] for t in tc_draws])
        transfer_resid_mean = _mean_curves([t[1] for t in tc_draws])
        results = {
            "model": model.__class__.__name__, "dataset": data.dataset, "task": task.kind,
            "n_classes": task.n_classes, "fold": data.fold, "cold_best_val": round(best, 4),
            "best_epoch": best_epoch,
            "repr_kind": cold.repr_kind, "repr_stage": cold.repr_stage,
            "repr_dim": (int(cold.pair_repr.shape[1]) if cold.has_repr() else None),
            "ranks": list(self.spec.ranks), "head_cold": head,
            "within_cold": wc, "within_cold_residual": wcr, "degree_only": do,
            "model_head_curve": mh,
            "transfer_mean": transfer_mean, "transfer_resid_mean": transfer_resid_mean,
            "transfer_draws": len(tc_draws), "per_draw": per_draw,
            "val_curve": val_curve}
        RankRunWriter.write(run_dir, results)
        _log(f"[done] {run_dir.name} (measure {time.perf_counter() - t_ext0:.0f}s, "
             f"total {time.perf_counter() - t_all0:.0f}s)")
        return results


def _mean_curves(dicts):
    """Mean over draws of {metric: [per-rank]} dicts (nan-safe)."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: np.nanmean([d[k] for d in dicts], axis=0).tolist() for k in keys}


class RankRunWriter:
    @staticmethod
    def write(run_dir: Path, results: dict):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
