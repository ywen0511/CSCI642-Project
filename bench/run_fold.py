#!/usr/bin/env python
"""ErrP-Bench: train one decoder on one cross-subject fold and predict the held-out subjects.

  python bench/run_fold.py --prep-dir DIR --config CFG.yaml --splits FOLD.json --seed 42 --align none|ea \
      --out RUN_DIR [--device cuda|cpu] [--max-epochs N]

- Neural models: the repository Trainer (early stopping on validation AUROC of the fold's validation subjects),
  best checkpoint reloaded, error-logit margins predicted for validation and test subjects.
- `xdawn_rg` config (model.name): xDAWN covariances (lwf) -> Riemannian tangent space -> logistic regression,
  fitted on train subjects (validation subjects unused).
- --align ea: Euclidean alignment per subject (He & Wu 2019), computed from that subject's own unlabeled trials
  (transductive for test subjects: no labels used), trace-preserving so amplitude conventions of each model hold.
Outputs: predictions.npz (subject, trial, label, margin, split), metrics.json, run artifacts of the Trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from errp.data import SplitData, get_split  # noqa: E402
from errp.utils import load_yaml, write_json  # noqa: E402


def ea_align(d: SplitData) -> SplitData:
    X = d.X
    shp = X.shape
    X2 = X.reshape(shp[0], shp[1], -1).astype(np.float64)
    out = np.empty_like(X2)
    for s in np.unique(d.subject_ids):
        m = d.subject_ids == s
        Xs = X2[m]
        R = np.einsum("nct,ndt->cd", Xs, Xs) / (Xs.shape[0] * Xs.shape[2])
        R = 0.5 * (R + R.T)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, w.max() * 1e-10, None)
        W = (V * (1.0 / np.sqrt(w))) @ V.T
        W *= np.sqrt(np.trace(R) / R.shape[0])  # trace-preserving: keep the subject's mean channel power
        out[m] = np.einsum("cd,ndt->nct", W, Xs)
    return SplitData(out.reshape(shp).astype(np.float32), d.y, d.trial_ids, d.subject_ids)


def auroc(y, z):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, z)) if len(set(y.tolist())) == 2 else float("nan")


def metrics_for(y, z, subj):
    per = {s: auroc(y[subj == s], z[subj == s]) for s in sorted(set(subj))}
    vals = [v for v in per.values() if np.isfinite(v)]
    return {"pooled_auroc": auroc(y, z), "macro_subject_auroc": float(np.mean(vals)) if vals else float("nan"),
            "per_subject_auroc": per, "n": int(len(y)), "n_err": int(y.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--align", choices=["none", "ea"], default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "metrics.json").exists():
        print("already done", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(a.config)
    cfg["seed"] = a.seed
    splits = json.loads(Path(a.splits).read_text())
    prep_cfg = json.loads((Path(a.prep_dir) / "preprocessing_config.json").read_text())
    chans = prep_cfg.get("channels") or prep_cfg["config"]["channels"]
    if isinstance(chans, dict):
        chans = chans.get("order")
    cfg["model"]["channels"] = list(chans)
    view = cfg["input"]["view"]
    t0 = time.time()
    data = {s: get_split(a.prep_dir, view, splits, s) for s in ("train", "validation", "test")}
    if a.align == "ea":
        data = {k: ea_align(v) for k, v in data.items()}
    res = {"config": a.config, "splits": a.splits, "seed": a.seed, "align": a.align, "prep_dir": a.prep_dir,
           "model": cfg["model"]["name"], "n_train": int(len(data["train"].y)), "n_val": int(len(data["validation"].y)),
           "n_test": int(len(data["test"].y))}
    if cfg["model"]["name"] == "xdawn_rg":
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        pipe = make_pipeline(XdawnCovariances(nfilter=cfg["model"].get("nfilter", 4), estimator="lwf"),
                             TangentSpace(metric="riemann"),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=a.seed))
        tr = data["train"]
        pipe.fit(tr.X.astype(np.float64), tr.y)
        margins = {k: pipe.decision_function(data[k].X.astype(np.float64)) for k in ("validation", "test")}
        res["n_params"] = None
        res["best_epoch"] = None
    else:
        import torch
        from errp.train import Trainer, load_model_from_ckpt, predict_margin
        dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
        if dev == "cpu":
            torch.set_num_threads(int(__import__("os").environ.get("EB_THREADS", "4")))
        trainer = Trainer(cfg, out, data["train"], data["validation"], device=dev, meta={"bench": True, "input_shape": list(data["train"].X.shape[1:])})
        res["n_params"] = int(sum(p.numel() for p in trainer.model.parameters()))
        lr = getattr(trainer.model, "pretrained_load_report", None)
        if lr is not None:
            res["pretrained_loaded_tensors"] = lr.get("n_loaded")
        if cfg["model"]["name"] == "cbramod":
            res["pretrained_loaded"] = bool(getattr(trainer.model, "pretrained_loaded", False))
        trainer.try_resume()
        summ = trainer.fit(max_epochs=a.max_epochs)
        res["best_epoch"] = summ.get("best_epoch")
        res["epochs_run"] = summ.get("epochs_run", summ.get("final_epoch"))
        model, scaler, _ = load_model_from_ckpt(out / "checkpoints" / "best.ckpt", torch.device(dev))
        model.eval()
        margins = {k: predict_margin(model, scaler.transform(data[k].X), torch.device(dev)) for k in ("validation", "test")}
        res["device"] = dev
        # keep disk small: only best.ckpt
        for p in (out / "checkpoints").glob("*.ckpt"):
            if p.name != "best.ckpt":
                p.unlink()
    for k in ("validation", "test"):
        d = data[k]
        res[k] = metrics_for(d.y, np.asarray(margins[k], dtype=np.float64), d.subject_ids)
    np.savez_compressed(out / "predictions.npz",
                        **{f"{k}_{f}": v for k in ("validation", "test") for f, v in
                           (("subject", data[k].subject_ids), ("trial", data[k].trial_ids), ("label", data[k].y),
                            ("margin", np.asarray(margins[k], dtype=np.float64)))})
    res["wall_s"] = round(time.time() - t0, 1)
    write_json(res, out / "metrics.json")
    print(json.dumps({k: res[k] for k in ("model", "align", "seed", "wall_s")}),
          "test pooled", round(res["test"]["pooled_auroc"], 4), "macro", round(res["test"]["macro_subject_auroc"], 4))


if __name__ == "__main__":
    main()
