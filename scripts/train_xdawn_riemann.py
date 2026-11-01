#!/usr/bin/env python
"""Supplementary classical ErrP baseline: xDAWN covariances + Riemannian tangent space +
logistic regression (pyriemann / scikit-learn). Same splits, trials and windows as the
neural decoders (eeg256 view). Fit on the training split only; the validation split is used
only for temperature calibration (no hyperparameter search). Not an iterative model, so
there is no last.ckpt; the fitted pipeline is saved as model.joblib.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from pyriemann.estimation import XdawnCovariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from errp.data import get_split, load_splits  # noqa: E402
from errp.evaluate import evaluate_margins  # noqa: E402
from errp.utils import env_info, git_sha, sha256_file, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--out-root", default="artifacts/runs")
    ap.add_argument("--run-id", default="xdawn_riemann_s42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfilter", type=int, default=4)
    a = ap.parse_args()
    splits = load_splits(a.splits)
    tr = get_split(a.prep_dir, "eeg256", splits, "train")
    va = get_split(a.prep_dir, "eeg256", splits, "validation")
    te = get_split(a.prep_dir, "eeg256", splits, "test")
    run_dir = Path(a.out_root) / a.run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    # decimate 256 -> 64 Hz (signal already low-passed at 20 Hz) to keep covariance estimation cheap
    dec = lambda X: X[:, :, ::4].astype(np.float64)
    pipe = make_pipeline(XdawnCovariances(nfilter=a.nfilter, estimator="lwf"), TangentSpace(metric="riemann"),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=a.seed))
    t0 = time.time()
    pipe.fit(dec(tr.X), tr.y)
    fit_s = time.time() - t0
    zv = pipe.decision_function(dec(va.X))
    zt = pipe.decision_function(dec(te.X))
    path = run_dir / "checkpoints" / "model.joblib"
    joblib.dump(pipe, path)
    prep_cfg = json.loads((Path(a.prep_dir) / "preprocessing_config.json").read_text())
    cfg = {"model": "xDAWN(nfilter=%d, lwf) -> TangentSpace(riemann) -> LogisticRegression(C=1, balanced)" % a.nfilter,
           "input": "eeg256 decimated by 4 (64 Hz), microvolts", "seed": a.seed}
    write_json({"config": cfg, "env": env_info(), "source_sha": git_sha(), "fit_time_s": fit_s,
                "n_train": int(len(tr.y)), "splits_sha256": sha256_file(a.splits)}, run_dir / "run_config.json")
    m = evaluate_margins(va, zv, te, zt, run_dir, a.run_id, "xdawn_riemann", a.seed, prep_cfg["preprocessing_id"],
                         checkpoint_id=f"{a.run_id}:model",
                         extra={"fit_time_s": fit_s, "model_sha256": sha256_file(path)})
    print(json.dumps({"test_auroc": m["test_pooled"]["calibrated@0.5"]["auroc"],
                      "test_bacc": m["test_pooled"]["calibrated@0.5"]["balanced_accuracy"]}))


if __name__ == "__main__":
    main()
