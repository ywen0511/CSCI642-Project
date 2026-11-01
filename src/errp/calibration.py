"""Probability calibration fitted on validation logits only.

Default (declared before formal training): binary temperature scaling on the error
logit margin z = logit_error - logit_nonerror, i.e. p = sigmoid(z / T), T > 0, fitted by
minimising validation NLL. T does not change the ranking, so AUROC is unchanged.
An identity calibrator (T = 1) is used if the fit fails; this is recorded.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(z: np.ndarray, y: np.ndarray, T: float) -> float:
    p = np.clip(_sigmoid(z / T), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(z_val: np.ndarray, y_val: np.ndarray) -> dict:
    z_val = np.asarray(z_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    res = minimize_scalar(lambda logT: nll(z_val, y_val, float(np.exp(logT))),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded",
                          options={"xatol": 1e-6})
    T = float(np.exp(res.x)) if res.success else 1.0
    return {
        "method": "temperature_scaling" if res.success else "identity",
        "temperature": T,
        "fit_split": "validation",
        "val_nll_before": nll(z_val, y_val, 1.0),
        "val_nll_after": nll(z_val, y_val, T),
        "n_val": int(len(y_val)),
        "optimizer": "scipy.optimize.minimize_scalar(bounded, log T in [log 0.05, log 20])",
        "success": bool(res.success),
    }


def apply_calibration(z: np.ndarray, calib: dict) -> np.ndarray:
    T = calib["temperature"] if calib["method"] == "temperature_scaling" else 1.0
    return _sigmoid(np.asarray(z, dtype=np.float64) / T)
