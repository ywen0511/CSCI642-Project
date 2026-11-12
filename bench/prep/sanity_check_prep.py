#!/usr/bin/env python
"""Sanity checks for an ErrP-Bench prepared dir: shapes, counts, NaN, grand-average difference wave, pooled
5-fold shrinkage-LDA AUROC on decimated fronto-central samples.

  python bench/prep/sanity_check_prep.py --prep <dir> [--ref-prep <dir with same trials>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FC = ["Fz", "FCz", "FC1", "FC2", "Cz", "C1", "C2", "FC3", "FC4", "C3", "C4", "CPz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--ref-prep", default=None)
    a = ap.parse_args()
    d = Path(a.prep)
    cfg = json.load(open(d / "preprocessing_config.json"))
    man = pd.read_csv(d / "trial_manifest.csv", dtype={"subject_id": str, "trial_id": str})
    X = np.load(d / "epochs_256hz_uV.npy")
    P = np.load(d / "epochs_200hz_patches_uV.npy")
    ids = np.load(d / "epoch_trial_ids.npy").astype(str)
    kept = man[man.retained == 1]
    W = int(round(cfg["window_s"][1] - cfg["window_s"][0]))
    ch = cfg["channels"]
    res = {"dir": str(d), "X": list(X.shape), "P": list(P.shape), "dtype": [str(X.dtype), str(P.dtype)],
           "ids_match_manifest": bool((kept.trial_id.values == ids).all() and len(ids) == len(X) == len(P)),
           "finite": bool(np.isfinite(X).all() and np.isfinite(P).all()),
           "shape_ok": bool(X.shape[1:] == (len(ch), 256 * W) and P.shape[1:] == (len(ch), W, 200)),
           "n_subjects": int(kept.subject_id.nunique()), "n_epochs": int(len(ids)),
           "class_counts": kept.label.value_counts().sort_index().to_dict()}
    res["per_subject"] = kept.groupby("subject_id").label.agg(n1=lambda s: int((s == 1).sum()),
                                                              n0=lambda s: int((s == 0).sum())).to_dict("index")
    y = kept.label.values.astype(int)
    t = np.arange(X.shape[-1]) / 256.0 + cfg["window_s"][0]
    # grand average over subjects of per-subject (error - correct)
    subj = kept.subject_id.values
    waves = {}
    sites = {"FCz": ["FCz"] if "FCz" in ch else ["FC1", "FC2"], "Cz": ["Cz"]}
    for name, chans in sites.items():
        idx = [ch.index(c) for c in chans]
        dw = np.mean([X[(subj == s) & (y == 1)][:, idx].mean((0, 1)) - X[(subj == s) & (y == 0)][:, idx].mean((0, 1))
                      for s in np.unique(subj)], axis=0)
        m1 = (t >= 0.1) & (t <= 0.6)
        i_neg = np.where(m1)[0][np.argmin(dw[m1])]
        i_pos = np.where(m1)[0][np.argmax(dw[m1])]
        waves[name] = {"channels_used": chans, "neg_latency_ms": round(t[i_neg] * 1000), "neg_uV": round(float(dw[i_neg]), 2),
                       "pos_latency_ms": round(t[i_pos] * 1000), "pos_uV": round(float(dw[i_pos]), 2)}
    res["difference_wave_0.1-0.6s"] = waves
    fc = [ch.index(c) for c in FC if c in ch]
    Xd = X[:, fc].reshape(len(X), len(fc), -1, 8).mean(-1).reshape(len(X), -1)  # 32 Hz bins
    sc = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, y):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xd[tr], y[tr])
        sc[te] = lda.decision_function(Xd[te])
    res["lda_pooled_5fold_auroc"] = round(float(roc_auc_score(y, sc)), 4)
    res["lda_channels"] = [ch[i] for i in fc]
    if a.ref_prep:
        ref = pd.read_csv(Path(a.ref_prep) / "trial_manifest.csv", dtype={"trial_id": str})
        rk = set(ref[ref.retained == 1].trial_id)
        res["ref_trial_set_equal"] = bool(rk == set(ids))
        res["ref_n"] = len(rk)
        rl = ref[ref.retained == 1].set_index("trial_id").label.astype(int)
        res["ref_labels_equal"] = bool((rl.loc[ids].values == y).all()) if rk == set(ids) else None
    print(json.dumps(res, indent=1, default=str))
    (d / "validation.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
