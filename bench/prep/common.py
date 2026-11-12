"""Shared helpers for ErrP-Bench dataset preparation (see bench/DATA_FORMAT.md)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
from scipy.signal import resample_poly

REQUIRED_COLS = ["subject_id", "trial_id", "label", "retained", "session", "source_file", "event_code", "onset_s"]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def filter_continuous(raw, l_freq=1.0, h_freq=20.0):
    """MNE zero-phase FIR (firwin, hamming, default transition bands) on the continuous recording,
    exactly as scripts/prepare_data.py (prep_v1). Returns a description of the resolved filter."""
    sf = raw.info["sfreq"]
    raw.filter(l_freq, h_freq, method="fir", phase="zero", fir_window="hamming", fir_design="firwin",
               verbose="ERROR")
    h = mne.filter.create_filter(np.zeros((1, max(10000, raw.n_times // 4))), sf, l_freq, h_freq, method="fir",
                                 phase="zero", fir_window="hamming", fir_design="firwin", verbose="ERROR")
    return {"method": "mne.io.Raw.filter FIR zero-phase, firwin, hamming, default ('auto') transition bands, "
                      "pad='reflect_limited'", "l_freq": l_freq, "h_freq": h_freq, "n_taps": int(len(h)),
            "filter_sfreq": sf, "applied_to": "continuous_recording_before_epoching", "mne": mne.__version__}


def to_patches200(seg256: np.ndarray, W: int) -> np.ndarray:
    s = resample_poly(seg256.astype(np.float64), up=25, down=32, axis=-1)
    assert s.shape[-1] == 200 * W, s.shape
    return s.reshape(*s.shape[:-1], W, 200).astype(np.float32)


def check_channels(chs):
    mont = set(mne.channels.make_standard_montage("standard_1020").ch_names)
    bad = [c for c in chs if c not in mont]
    assert not bad, f"channels not in standard_1020: {bad}"


def write_prepared(out: Path, man: pd.DataFrame, ep256: list, ids: list, W: int, meta: dict, readme: str):
    """Write the ErrP-Bench files. `man` holds all candidate rows; retained rows must be in array order."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for c in REQUIRED_COLS:
        assert c in man.columns, c
    assert man["trial_id"].is_unique
    kept = man[man["retained"] == 1]
    assert list(kept["trial_id"]) == list(ids), "manifest order != array order"
    X = np.stack(ep256).astype(np.float32)
    assert X.shape[-1] == 256 * W and np.isfinite(X).all()
    P = np.stack([to_patches200(x, W) for x in X])
    check_channels(meta["channels"])
    assert X.shape[1] == len(meta["channels"])
    np.save(out / "epochs_256hz_uV.npy", X)
    np.save(out / "epochs_200hz_patches_uV.npy", P)
    np.save(out / "epoch_trial_ids.npy", np.array(ids))
    man.to_csv(out / "trial_manifest.csv", index=False)
    lab = kept["label"].astype(int)
    per_subj = {s: {"n_error": int((g["label"] == 1).sum()), "n_correct": int((g["label"] == 0).sum())}
                for s, g in kept.groupby("subject_id")}
    excl = man[man["retained"] == 0].groupby("exclusion_reason").size().to_dict() if "exclusion_reason" in man else {}
    cfg = dict(meta)
    cfg.update({"window_s": cfg.get("window_s"), "band_hz": [1, 20],
                "n_subjects": int(kept["subject_id"].nunique()), "n_epochs": int(len(ids)),
                "class_counts": {"error_1": int((lab == 1).sum()), "correct_0": int((lab == 0).sum())},
                "per_subject_counts": per_subj,
                "exclusions": {"counts_by_reason": {k: int(v) for k, v in excl.items()},
                               **cfg.get("exclusions", {})},
                "shapes": {"epochs_256hz_uV": list(X.shape), "epochs_200hz_patches_uV": list(P.shape)},
                "versions": {"mne": mne.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                             "pandas": pd.__version__}})
    write_json(cfg, out / "preprocessing_config.json")
    (out / "README.md").write_text(readme)
    print("WROTE", out, X.shape, P.shape, cfg["class_counts"], flush=True)
    return cfg
