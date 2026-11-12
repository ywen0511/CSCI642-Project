#!/usr/bin/env python
"""Sanity checks for an ErrP-Bench prepared dir: format, counts, NaN, grand-average error-minus-correct
difference wave at FCz/Cz (peaks), and a 5-fold pooled shrinkage-LDA AUROC on decimated Fz/FCz/Cz samples.

Usage: python validate_prepared.py --prep <dir> --neg-win 0.0 0.15 --pos-win 0.15 0.45 [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--neg-win", nargs=2, type=float, required=True)
    ap.add_argument("--pos-win", nargs=2, type=float, required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = Path(a.prep)
    cfg = json.load(open(d / "preprocessing_config.json"))
    man = pd.read_csv(d / "trial_manifest.csv", dtype={"subject_id": str, "trial_id": str})
    X = np.load(d / "epochs_256hz_uV.npy", mmap_mode="r")
    P = np.load(d / "epochs_200hz_patches_uV.npy", mmap_mode="r")
    ids = np.load(d / "epoch_trial_ids.npy").astype(str)
    kept = man[man.retained == 1]
    W = cfg["W"]
    assert len(kept) == len(ids) == X.shape[0] == P.shape[0] and (kept.trial_id.values == ids).all()
    assert X.dtype == np.float32 and P.dtype == np.float32
    assert X.shape[1] == len(cfg["channels"]) and X.shape[2] == 256 * W and P.shape[2:] == (W, 200)
    Xa = np.asarray(X)
    rep = {"shape256": list(X.shape), "shape200": list(P.shape), "nan_or_inf_256": int((~np.isfinite(Xa)).sum()),
           "nan_or_inf_200": int((~np.isfinite(np.asarray(P))).sum()), "abs_uV_p99": float(np.percentile(np.abs(Xa), 99))}
    y = kept.label.values.astype(int)
    subj = kept.subject_id.values
    rep["per_subject"] = kept.groupby("subject_id").label.agg(["count", "sum"]).rename(
        columns={"count": "n", "sum": "n_err"}).to_dict(orient="index")
    t = cfg["window_s"][0] + np.arange(256 * W) / 256.0
    ch = cfg["channels"]
    rep["diff_wave"] = {}
    for c in [x for x in ("FCz", "Cz", "Fz") if x in ch]:
        i = ch.index(c)
        # grand average = mean over subjects of the per-subject (error - correct) average
        dw = np.mean([Xa[(subj == s) & (y == 1), i].mean(0) - Xa[(subj == s) & (y == 0), i].mean(0)
                      for s in np.unique(subj)], axis=0)
        ent = {}
        for nm, (lo, hi), fn in (("neg", a.neg_win, np.argmin), ("pos", a.pos_win, np.argmax)):
            m = (t >= lo) & (t < hi)
            k = fn(dw[m])
            ent[nm] = {"latency_ms": round(1000 * float(t[m][k]), 1), "amp_uV": round(float(dw[m][k]), 2),
                       "search_ms": [lo * 1000, hi * 1000]}
        ent["n_subjects_neg_peak_lt0"] = int(sum(
            (Xa[(subj == s) & (y == 1), i].mean(0) - Xa[(subj == s) & (y == 0), i].mean(0))[
                (t >= a.neg_win[0]) & (t < a.neg_win[1])].min() < 0 for s in np.unique(subj)))
        rep["diff_wave"][c] = ent
    # decodability: Fz/FCz/Cz, decimate 256 Hz -> 32 Hz (every 8th sample of a 4-sample mean)
    picks = [ch.index(c) for c in ("Fz", "FCz", "Cz") if c in ch]
    F = Xa[:, picks].reshape(len(Xa), len(picks), -1, 8).mean(-1).reshape(len(Xa), -1)
    sc = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F, y):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[tr], y[tr])
        sc[te] = lda.decision_function(F[te])
    rep["pooled_5fold_shrinkage_lda_auroc_Fz_FCz_Cz_32Hz"] = round(float(roc_auc_score(y, sc)), 4)
    yp = np.random.default_rng(0).permutation(y)
    sc2 = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F, yp):
        sc2[te] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[tr], yp[tr]).decision_function(F[te])
    rep["permuted_label_control_auroc"] = round(float(roc_auc_score(yp, sc2)), 4)
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
