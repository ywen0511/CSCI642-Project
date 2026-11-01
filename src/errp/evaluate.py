"""Calibration fit (validation only), evaluation, and per-trial probability export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .calibration import apply_calibration, fit_temperature
from .data import SplitData, get_split
from .metrics import binary_metrics, macro_over_subjects, select_threshold_youden
from .train import load_model_from_ckpt, predict_margin
from .utils import sha256_file, write_json

PRED_COLUMNS = ["subject_id", "trial_id", "label", "split", "model", "decoder_seed", "p_error_raw",
                "p_error", "checkpoint_id", "preprocessing_id", "calibration_id"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def per_subject_metrics(y, p, subj, threshold=0.5):
    rows = []
    for s in sorted(set(subj)):
        m = subj == s
        r = binary_metrics(y[m], p[m], threshold)
        r["subject_id"] = s
        rows.append(r)
    return rows


def margins_from_ckpt(ckpt_path, prep_dir, splits, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, scaler, st = load_model_from_ckpt(ckpt_path, dev)
    model.eval()  # dropout off, BatchNorm uses stored running statistics
    view = st["config"]["input"]["view"]
    out = {}
    for split in ("validation", "test"):
        d = get_split(prep_dir, view, splits, split)
        out[split] = (d, predict_margin(model, scaler.transform(d.X), dev))
    return out, st


def evaluate_margins(dv: SplitData, zv: np.ndarray, dt: SplitData, zt: np.ndarray, run_dir: Path, run_id: str,
                     model_tag: str, decoder_seed: int, preprocessing_id: str, checkpoint_id: str,
                     extra: dict | None = None) -> dict:
    """zv/zt are error-logit margins (P_raw(error) = sigmoid(z))."""
    run_dir = Path(run_dir)
    calib = fit_temperature(zv, dv.y)
    calib["calibration_id"] = f"{run_id}:temperature"
    pv_raw, pt_raw = _sigmoid(zv), _sigmoid(zt)
    pv, pt = apply_calibration(zv, calib), apply_calibration(zt, calib)
    thr_val = select_threshold_youden(dv.y, pv)
    calib["val_selected_threshold_youden_on_calibrated"] = thr_val
    write_json(calib, run_dir / "calibration.json")

    for split, d, z, praw, p in [("validation", dv, zv, pv_raw, pv), ("test", dt, zt, pt_raw, pt)]:
        assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1
        assert np.all(np.isfinite(praw)) and praw.min() >= 0 and praw.max() <= 1
        pd.DataFrame({
            "subject_id": d.subject_ids, "trial_id": d.trial_ids, "label": d.y, "split": split,
            "model": model_tag, "decoder_seed": decoder_seed, "p_error_raw": praw, "p_error": p,
            "checkpoint_id": checkpoint_id, "preprocessing_id": preprocessing_id,
            "calibration_id": calib["calibration_id"],
        })[PRED_COLUMNS].to_csv(run_dir / f"{split}_predictions.csv", index=False, float_format="%.10g")
        np.save(run_dir / f"{split}_margins.npy", z)

    metrics = {
        "run_id": run_id, "model": model_tag, "decoder_seed": decoder_seed, **(extra or {}),
        "calibration": calib,
        "validation_note": "validation was used for early stopping, checkpoint selection and calibration; "
                           "validation scores are optimistic, not unbiased estimates",
        "validation": {"raw@0.5": binary_metrics(dv.y, pv_raw), "calibrated@0.5": binary_metrics(dv.y, pv)},
        "test_pooled": {"raw@0.5": binary_metrics(dt.y, pt_raw, 0.5),
                        "calibrated@0.5": binary_metrics(dt.y, pt, 0.5),
                        "calibrated@val_youden": binary_metrics(dt.y, pt, thr_val)},
        "test_per_subject_calibrated@0.5": per_subject_metrics(dt.y, pt, dt.subject_ids, 0.5),
        "test_per_subject_raw@0.5": per_subject_metrics(dt.y, pt_raw, dt.subject_ids, 0.5),
    }
    metrics["test_subject_macro_calibrated@0.5"] = macro_over_subjects(metrics["test_per_subject_calibrated@0.5"])
    metrics["test_subject_macro_raw@0.5"] = macro_over_subjects(metrics["test_per_subject_raw@0.5"])
    write_json(metrics, run_dir / "metrics.json")
    return metrics


def evaluate_run(run_dir, prep_dir, splits, run_id, model_tag, decoder_seed, preprocessing_id, device="cuda"):
    run_dir = Path(run_dir)
    best = run_dir / "checkpoints" / "best.ckpt"
    out, st = margins_from_ckpt(best, prep_dir, splits, device)
    (dv, zv), (dt, zt) = out["validation"], out["test"]
    return evaluate_margins(dv, zv, dt, zt, run_dir, run_id, model_tag, decoder_seed, preprocessing_id,
                            checkpoint_id=f"{run_id}:best",
                            extra={"best_epoch": st["best_epoch"], "best_val_metric": st["best_metric"],
                                   "best_ckpt_sha256": sha256_file(best)})
