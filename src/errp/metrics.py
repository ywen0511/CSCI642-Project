"""Evaluation metrics. Labels: 0 = non-error, 1 = error. Scores are P(error)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    assert y.shape == p.shape and y.ndim == 1
    assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / n_pos if n_pos else float("nan")
    tnr = tn / n_neg if n_neg else float("nan")
    if n_pos and n_neg:
        auroc = float(roc_auc_score(y, p))
        auroc_note = ""
        bacc = 0.5 * (tpr + tnr)
    else:
        auroc = float("nan")
        auroc_note = "undefined: only one class present"
        bacc = float("nan")
    eps = 1e-12
    pc = np.clip(p, eps, 1 - eps)
    return {
        "n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
        "auroc": auroc, "auroc_note": auroc_note,
        "balanced_accuracy": bacc, "sensitivity_tpr": tpr, "specificity_tnr": tnr,
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "ece_10bin": expected_calibration_error(y, p, 10),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "mean_p_error": float(p.mean()),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def select_threshold_youden(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (Youden's J) on the given (validation) data."""
    cand = np.unique(np.concatenate([[0.5], p]))
    best_t, best = 0.5, -1.0
    for t in cand:
        pred = p >= t
        tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
        tnr = (~pred[y == 0]).mean() if (y == 0).any() else 0.0
        j = 0.5 * (tpr + tnr)
        if j > best + 1e-12:
            best, best_t = j, float(t)
    return best_t


def macro_over_subjects(rows: list[dict[str, Any]], keys=("auroc", "balanced_accuracy", "brier")) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
        out[k + "_n_subjects"] = len(vals)
    return out
