"""Shared helpers for ErrP-Bench dataset converters (see bench/DATA_FORMAT.md).

Conventions mirrored from scripts/prepare_data.py (ErrP-HRI prep):
  MNE zero-phase FIR (firwin, hamming) 1-20 Hz on the CONTINUOUS recording, microvolts,
  256 Hz epochs, 200 Hz patches = resample_poly(25, 32) of the 256 Hz epoch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
from scipy.signal import resample_poly

FS_OUT = 256
BAND = (1.0, 20.0)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def filter_continuous(data_uV: np.ndarray, sfreq: float):
    """MNE zero-phase FIR band-pass on a (C, T) float64 continuous array (uV). Returns filtered array + filter info."""
    out = mne.filter.filter_data(data_uV.astype(np.float64), sfreq, BAND[0], BAND[1], method="fir", phase="zero",
                                 fir_window="hamming", fir_design="firwin", verbose="ERROR")
    h = mne.filter.create_filter(data_uV[:1, : min(data_uV.shape[1], 200000)], sfreq, BAND[0], BAND[1], method="fir",
                                 phase="zero", fir_window="hamming", fir_design="firwin", verbose="ERROR")
    info = {"l_freq": BAND[0], "h_freq": BAND[1], "method": "fir", "phase": "zero", "fir_window": "hamming",
            "fir_design": "firwin", "n_taps": int(len(h)), "duration_s": len(h) / sfreq, "sfreq": sfreq,
            "pad": "reflect_limited (MNE default)", "mne": mne.__version__}
    return out, info


def to_256(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Integer-factor polyphase downsampling of the filtered continuous signal to 256 Hz."""
    down = int(round(sfreq / FS_OUT))
    assert abs(sfreq / down - FS_OUT) < 1e-9, sfreq
    if down == 1:
        return data
    return resample_poly(data, up=1, down=down, axis=1)


def cut(X256: np.ndarray, onset_samp256: int, n0: int, W: int):
    """Return the (C, 256*W) segment starting at onset+n0, or (None, reason)."""
    a, b = onset_samp256 + n0, onset_samp256 + n0 + FS_OUT * W
    if a < 0 or b > X256.shape[1]:
        return None, "epoch_window_outside_recording"
    seg = X256[:, a:b]
    if not np.all(np.isfinite(seg)):
        return None, "nonfinite_samples"
    return seg, ""


def patches200(seg256: np.ndarray, W: int) -> np.ndarray:
    s = resample_poly(seg256.astype(np.float64), up=25, down=32, axis=1)
    assert s.shape[1] == 200 * W, s.shape
    return s.reshape(seg256.shape[0], W, 200).astype(np.float32)


def check_montage(ch_names):
    std = set(mne.channels.make_standard_montage("standard_1020").ch_names)
    missing = [c for c in ch_names if c not in std]
    assert not missing, f"channels not in standard_1020: {missing}"


def write_outputs(out: Path, rows: list, ep256: list, ep200: list, ids: list, cfg: dict, readme: str):
    out.mkdir(parents=True, exist_ok=True)
    man = pd.DataFrame(rows)
    assert man["trial_id"].is_unique
    kept = man[man["retained"] == 1]
    assert list(kept["trial_id"]) == list(ids)
    X256 = np.stack(ep256).astype(np.float32)
    X200 = np.stack(ep200).astype(np.float32)
    assert np.isfinite(X256).all() and np.isfinite(X200).all()
    np.save(out / "epochs_256hz_uV.npy", X256)
    np.save(out / "epochs_200hz_patches_uV.npy", X200)
    np.save(out / "epoch_trial_ids.npy", np.array(ids))
    man.to_csv(out / "trial_manifest.csv", index=False)
    lab = kept["label"].astype(int)
    cfg["n_subjects"] = int(kept["subject_id"].nunique())
    cfg["n_epochs"] = int(len(kept))
    cfg["class_counts"] = {"0": int((lab == 0).sum()), "1": int((lab == 1).sum())}
    cfg["per_subject_counts"] = {s: {"0": int((g["label"] == 0).sum()), "1": int((g["label"] == 1).sum())}
                                 for s, g in kept.groupby("subject_id")}
    excl = man[man["retained"] == 0]
    cfg.setdefault("exclusions", {})
    cfg["exclusions"]["counts_by_reason"] = {str(k): int(v) for k, v in excl["exclusion_reason"].value_counts().items()}
    cfg["shapes"] = {"eeg256": list(X256.shape), "patch200": list(X200.shape)}
    cfg["versions"] = {"mne": mne.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                       "pandas": pd.__version__}
    with open(out / "preprocessing_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    (out / "README.md").write_text(readme)
    print(man.groupby(["subject_id", "retained", "label"], dropna=False).size().unstack([1, 2]).to_string())
    print("PREP_DONE", cfg["preprocessing_id"], X256.shape, X200.shape, cfg["class_counts"], flush=True)
