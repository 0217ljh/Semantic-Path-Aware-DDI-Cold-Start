"""Task-aware, leak-free rank-curve measurements for the S2 rank-analysis pipeline.

All rank curves fit standardize+PCA+probe on TRAIN/SOURCE folds only (no cold-label
leak into the fitted subspace). Binary => AUROC; multiclass => macro-AUROC + top5.
Consolidates the logic previously duplicated across the one-off train_*_s2 scripts.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from .specs import TaskSpec


# --------------------------------------------------------------------------- #
# metric primitives
# --------------------------------------------------------------------------- #
def _macro_auc(y, proba, classes) -> float:
    """One-vs-rest macro AUROC over `classes` (proba columns aligned to classes)."""
    Y = label_binarize(y, classes=list(classes))
    if Y.shape[1] == 1:      # binary label_binarize edge case
        Y = np.hstack([1 - Y, Y])
    aucs = []
    for k in range(len(classes)):
        s = Y[:, k].sum()
        if 0 < s < len(y):
            aucs.append(roc_auc_score(Y[:, k], proba[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _topk_acc(y, proba, classes, k=5) -> float:
    classes = np.asarray(classes)
    kk = min(k, proba.shape[1])
    topk = classes[np.argsort(-proba, axis=1)[:, :kk]]
    return float(np.mean([yi in row for yi, row in zip(y, topk)]))


def _probe_metrics(proba, classes, y_eval, task: TaskSpec) -> dict:
    """Score a fitted probe's predictions on eval labels, per task."""
    if task.is_binary:
        col = list(classes).index(1) if 1 in classes else proba.shape[1] - 1
        return {"auroc": float(roc_auc_score(y_eval, proba[:, col]))}
    return {"macro_auroc": _macro_auc(y_eval, proba, classes),
            "top5": _topk_acc(y_eval, proba, classes, 5)}


def _nan_metrics(task: TaskSpec) -> dict:
    return {"auroc": float("nan")} if task.is_binary else {"macro_auroc": float("nan"), "top5": float("nan")}


def _std_svd(Zfit):
    mu = Zfit.mean(0, keepdims=True); sd = Zfit.std(0, keepdims=True) + 1e-8
    S = (Zfit - mu) / sd
    ctr = S.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(S - ctr, full_matrices=False)
    return mu, sd, ctr, Vt


def _balance_idx(y, seed=0):
    y = y.astype(int); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed); m = min(len(pos), len(neg))
    return np.concatenate([rng.choice(pos, m, False), rng.choice(neg, m, False)])


def _min_class_count(y) -> int:
    _, cc = np.unique(y, return_counts=True)
    return int(cc.min()) if len(cc) else 0


# --------------------------------------------------------------------------- #
# public measurements
# --------------------------------------------------------------------------- #
def head_metrics(logits, y, task: TaskSpec) -> dict:
    """Model's own cold head performance."""
    import scipy.special as sp
    if task.is_binary:
        p = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        return {"auroc": float(roc_auc_score(y, p))}
    proba = sp.softmax(logits, axis=-1)
    pred = proba.argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro",
                                       labels=list(range(task.n_classes)), zero_division=0)),
            "macro_auroc": _macro_auc(y, proba, list(range(proba.shape[1]))),
            "top5_acc": _topk_acc(y, proba, list(range(proba.shape[1])), 5)}


def degree_within(deg, y, task: TaskSpec, n_splits=5, seed=0) -> dict:
    """Hub baseline: CV probe on the two drugs' degrees only (within cold)."""
    y = y.astype(int)
    if not task.is_binary:
        vc = np.bincount(y, minlength=int(y.max()) + 1)
        keep = np.isin(y, np.where(vc >= n_splits)[0])
        deg, y = deg[keep], y[keep]
    if len(y) == 0 or np.unique(y).size < 2:
        return _nan_metrics(task)
    if task.is_binary:
        idx = _balance_idx(y, seed); deg, y = deg[idx], y[idx]
    ns = min(n_splits, _min_class_count(y))
    if ns < 2:
        return _nan_metrics(task)
    accs = {k: [] for k in _nan_metrics(task)}
    for tri, tei in StratifiedKFold(ns, shuffle=True, random_state=seed).split(deg, y):
        clf = LogisticRegression(max_iter=2000).fit(deg[tri], y[tri])
        m = _probe_metrics(clf.predict_proba(deg[tei]), clf.classes_, y[tei], task)
        for k, v in m.items():
            accs[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in accs.items()}


def residualize(Zfit, dfit, Zapply, dapply):
    """Regress Z on [deg,1] using the FIT set, subtract from both."""
    Xf = np.hstack([dfit, np.ones((len(dfit), 1))])
    Xa = np.hstack([dapply, np.ones((len(dapply), 1))])
    B, _, _, _ = np.linalg.lstsq(Xf, Zfit, rcond=None)
    return Zfit - Xf @ B, Zapply - Xa @ B


def within_curve(Z, y, ranks, task: TaskSpec, deg=None, n_splits=5, seed=0) -> dict:
    """Leak-free within-cold (oracle) rank curve: per-fold standardize+PCA on the
    train folds, probe, score held-out fold. If `deg` is given, the degree
    residualization is also fit per-fold (leak-free) => the degree-residual curve.
    Returns {metric: [per-rank]}."""
    nan = {k: [float("nan")] * len(ranks) for k in _nan_metrics(task)}
    y = y.astype(int)
    if not task.is_binary:
        vc = np.bincount(y, minlength=int(y.max()) + 1)
        keep = np.isin(y, np.where(vc >= n_splits)[0])
        Z, y = Z[keep], y[keep]
        if deg is not None:
            deg = deg[keep]
    if len(y) == 0 or np.unique(y).size < 2:
        return nan
    if task.is_binary:
        idx = _balance_idx(y, seed)
        Z, y = Z[idx], y[idx]
        if deg is not None:
            deg = deg[idx]
    ns = min(n_splits, _min_class_count(y))
    if ns < 2:
        return nan
    per = {k: {r: [] for r in ranks} for k in _nan_metrics(task)}
    for tri, tei in StratifiedKFold(ns, shuffle=True, random_state=seed).split(Z, y):
        if deg is not None:                    # per-fold degree residualization (leak-free)
            Ztr, Zte = residualize(Z[tri], deg[tri], Z[tei], deg[tei])
        else:
            Ztr, Zte = Z[tri], Z[tei]
        mu, sd, ctr, Vt = _std_svd(Ztr)
        Ptr = (Ztr - mu) / sd - ctr
        Pte = (Zte - mu) / sd - ctr
        for r in ranks:
            rr = min(r, Vt.shape[0])
            clf = LogisticRegression(max_iter=2000).fit(Ptr @ Vt[:rr].T, y[tri])
            m = _probe_metrics(clf.predict_proba(Pte @ Vt[:rr].T), clf.classes_, y[tei], task)
            for k, v in m.items():
                per[k][r].append(v)
    return {k: [float(np.nanmean(per[k][r])) for r in ranks] for k in per}


def transfer_curve(Zsrc, ysrc, Zc, yc, ranks, task: TaskSpec) -> dict:
    """Deployable transfer: fit standardize+PCA+probe on SOURCE, apply to cold.
    No cold labels used. Returns {metric: [per-rank]}."""
    ysrc = ysrc.astype(int); yc = yc.astype(int)
    if len(Zsrc) == 0 or len(Zc) == 0 or np.unique(ysrc).size < 2 or np.unique(yc).size < 2:
        return {k: [float("nan")] * len(ranks) for k in _nan_metrics(task)}
    mu, sd, ctr, Vt = _std_svd(Zsrc)
    Ssrc = (Zsrc - mu) / sd - ctr
    Sc = (Zc - mu) / sd - ctr
    out = {k: [] for k in _nan_metrics(task)}
    for r in ranks:
        rr = min(r, Vt.shape[0])
        clf = LogisticRegression(max_iter=2000).fit(Ssrc @ Vt[:rr].T, ysrc)
        m = _probe_metrics(clf.predict_proba(Sc @ Vt[:rr].T), clf.classes_, yc, task)
        for k, v in m.items():
            out[k].append(v)
    return out


def model_head_curve(Z, y, ranks, task: TaskSpec, head_fn) -> dict:
    """Approach B (model-faithful): rank-r PCA reconstruction of Z (unsupervised,
    Z_r = (Z-mu) V_r V_r^T + mu) fed through the model's OWN frozen head. At full
    rank it recovers the deployed prediction exactly. Returns {metric: [per-rank]}
    with the same primary keys as transfer/within (binary: auroc; multi: macro_auroc + top5)."""
    y = y.astype(int)
    keys = ["auroc"] if task.is_binary else ["macro_auroc", "top5"]
    Z = np.asarray(Z, dtype=np.float64)
    if len(Z) == 0 or np.unique(y).size < 2:
        return {k: [float("nan")] * len(ranks) for k in keys}
    mu = Z.mean(0, keepdims=True); Zc = Z - mu
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    out = {k: [] for k in keys}
    for r in ranks:
        rr = min(r, Vt.shape[0])
        # full rank -> use Z directly so the endpoint equals the deployed head EXACTLY
        Zr = Z if rr == Vt.shape[0] else (Zc @ Vt[:rr].T) @ Vt[:rr] + mu
        hm = head_metrics(head_fn(Zr.astype(np.float32)), y, task)  # model's own head
        if task.is_binary:
            out["auroc"].append(float(hm["auroc"]))
        else:
            out["macro_auroc"].append(float(hm["macro_auroc"])); out["top5"].append(float(hm["top5_acc"]))
    return out
