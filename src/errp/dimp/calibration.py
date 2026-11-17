"""Calibrators on the error margin z = logit_error - logit_nonerror (handoff Section 8).

identity:    p = expit(z)                      (no parameters)
temperature: p = expit(z / T), T in [0.05, 20], minimise weighted mean NLL
affine:      p = expit(a z + b), a > 0, log a in [-log 20, log 20], b in [-5, 5],
             minimise weighted mean NLL + 1e-3 (log(a)^2 + b^2)
Weights give each participant equal total weight. Method choice: leave-one-validation-participant-out
cross-fitting (fit on s11 -> NLL on s13 and reverse), averaged equally over participants and seeds; ties within
1e-6 prefer identity, then temperature, then affine.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_expit

METHODS = ("identity", "temperature", "affine")
LOG20 = float(np.log(20.0))


def nll(z, y, w=None):
    z, y = np.asarray(z, float), np.asarray(y, float)
    w = np.full(len(z), 1.0 / len(z)) if w is None else np.asarray(w, float) / np.sum(w)
    return float(-np.sum(w * (y * log_expit(z) + (1 - y) * log_expit(-z))))


def participant_weights(subjects):
    subjects = np.asarray(subjects)
    w = np.zeros(len(subjects))
    for s in np.unique(subjects):
        m = subjects == s
        w[m] = 1.0 / m.sum()
    return w / w.sum()


def fit(method, z, y, subjects):
    z, y = np.asarray(z, float), np.asarray(y, float)
    w = participant_weights(subjects)
    if method == "identity":
        return {"method": "identity", "params": {}, "success": True, "objective": nll(z, y, w), "fitted": False}
    if method == "temperature":
        res = minimize_scalar(lambda lt: nll(z / np.exp(lt), y, w), bounds=(np.log(0.05), np.log(20.0)),
                              method="bounded", options={"xatol": 1e-8})
        T = float(np.exp(res.x))
        at_bound = abs(res.x - np.log(0.05)) < 1e-6 or abs(res.x - np.log(20.0)) < 1e-6
        return {"method": "temperature", "params": {"T": T}, "success": bool(res.success), "objective": float(res.fun),
                "at_bound": bool(at_bound), "fitted": True, "optimizer": "scipy minimize_scalar bounded on log T"}
    if method == "affine":
        def obj(v):
            la, b = v
            return nll(np.exp(la) * z + b, y, w) + 1e-3 * (la ** 2 + b ** 2)
        res = minimize(obj, x0=np.zeros(2), method="L-BFGS-B", bounds=[(-LOG20, LOG20), (-5.0, 5.0)])
        la, b = res.x
        at_bound = abs(abs(la) - LOG20) < 1e-6 or abs(abs(b) - 5.0) < 1e-6
        return {"method": "affine", "params": {"a": float(np.exp(la)), "b": float(b)}, "success": bool(res.success),
                "objective": float(res.fun), "at_bound": bool(at_bound), "fitted": True,
                "optimizer": "scipy L-BFGS-B on (log a, b)"}
    raise ValueError(method)


def calibrated_margin(cal, z):
    z = np.asarray(z, float)
    if cal["method"] == "identity":
        return z
    if cal["method"] == "temperature":
        return z / cal["params"]["T"]
    if cal["method"] == "affine":
        return cal["params"]["a"] * z + cal["params"]["b"]
    raise ValueError(cal["method"])


def apply(cal, z):
    return expit(calibrated_margin(cal, z))


def crossfit_choice(per_seed):
    """per_seed: {seed: {"z": array, "y": array, "subjects": array}} on the two validation participants.
    Returns the ranking of methods by mean held-out-participant NLL (equal weight per participant and seed)."""
    scores = {}
    detail = {}
    for m in METHODS:
        vals = []
        for seed, d in per_seed.items():
            subs = sorted(set(d["subjects"]))
            assert len(subs) == 2, subs
            for fit_s, score_s in ((subs[0], subs[1]), (subs[1], subs[0])):
                fm, sm = d["subjects"] == fit_s, d["subjects"] == score_s
                c = fit(m, d["z"][fm], d["y"][fm], d["subjects"][fm])
                if not c["success"]:
                    raise RuntimeError(f"{m} fit failed for seed {seed} on {fit_s}")
                v = nll(calibrated_margin(c, d["z"][sm]), d["y"][sm])
                vals.append(v)
                detail.setdefault(m, []).append({"seed": seed, "fit_on": fit_s, "scored_on": score_s, "nll": v,
                                                 "params": c["params"]})
        scores[m] = float(np.mean(vals))
    order = {m: i for i, m in enumerate(METHODS)}
    best = min(scores.values())
    ranking = sorted(METHODS, key=lambda m: (scores[m] - best > 1e-6, scores[m] if scores[m] - best > 1e-6 else 0, order[m]))
    return {"mean_heldout_participant_nll": scores, "ranking": ranking, "selected": ranking[0], "detail": detail}
