"""Loading of prepared epochs + manifest, split handling, and model-specific input views."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import sha256_file


@dataclass
class SplitData:
    X: np.ndarray            # model-ready float32 array
    y: np.ndarray            # int64, 0 = non-error, 1 = error
    trial_ids: np.ndarray    # str
    subject_ids: np.ndarray  # str


def load_splits(path: str | Path) -> dict:
    with open(path) as f:
        sp = json.load(f)
    tr, va, te = set(sp["train"]), set(sp["validation"]), set(sp["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "participant overlap between splits"
    return sp


def load_prepared(prep_dir: str | Path, view: str):
    """Model input views of the SAME retained trials and the SAME [0, 2) s windows:
    'eeg256'   -> (N, C, 512) uV at 256 Hz (stored)
    'patch200' -> (N, C, 2, 200) uV at 200 Hz (stored)
    'flat200'  -> (N, C, 400) uV at 200 Hz (reshape of patch200)
    'eeg250'   -> (N, C, 500) uV at 250 Hz (resample_poly(125, 128) of eeg256, computed on load)
    """
    prep_dir = Path(prep_dir)
    man = pd.read_csv(prep_dir / "trial_manifest.csv", dtype={"subject_id": str, "trial_id": str})
    fn = {"eeg256": "epochs_256hz_uV.npy", "patch200": "epochs_200hz_patches_uV.npy",
          "flat200": "epochs_200hz_patches_uV.npy", "eeg250": "epochs_256hz_uV.npy"}[view]
    X = np.load(prep_dir / fn, mmap_mode="r")
    if view == "flat200":
        X = np.asarray(X).reshape(X.shape[0], X.shape[1], -1)
    elif view == "eeg250":
        from scipy.signal import resample_poly
        X = resample_poly(np.asarray(X, dtype=np.float64), up=125, down=128, axis=-1).astype(np.float32)
    ids = np.load(prep_dir / "epoch_trial_ids.npy", allow_pickle=False).astype(str)
    return man, X, ids


def get_split(prep_dir: str | Path, view: str, splits: dict, split: str) -> SplitData:
    man, X, ids = load_prepared(prep_dir, view)
    kept = man[man["retained"] == 1].set_index("trial_id")
    assert len(kept) == len(ids) and (kept.index.values == ids).all(), "manifest / epoch order mismatch"
    subj = set(splits[split])
    mask = kept["subject_id"].isin(subj).values
    return SplitData(
        X=np.ascontiguousarray(X[mask]).astype(np.float32),
        y=kept["label"].values[mask].astype(np.int64),
        trial_ids=ids[mask],
        subject_ids=kept["subject_id"].values[mask].astype(str),
    )


class InputScaler:
    """Model-specific amplitude scaling. Fitted on the TRAINING split only (if it has
    parameters) and then applied unchanged to validation/test trials."""

    def __init__(self, kind: str, params: dict | None = None):
        self.kind = kind
        self.params = params or {}

    def fit(self, X_train: np.ndarray) -> "InputScaler":
        if self.kind == "train_global_std":
            self.params = {"scale_uV": float(np.std(X_train, dtype=np.float64))}
        elif self.kind == "fixed_divide":
            assert "divisor" in self.params
        elif self.kind == "none":
            pass
        else:
            raise ValueError(self.kind)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "train_global_std":
            return (X / self.params["scale_uV"]).astype(np.float32)
        if self.kind == "fixed_divide":
            return (X / self.params["divisor"]).astype(np.float32)
        return X.astype(np.float32)

    def state(self) -> dict:
        return {"kind": self.kind, "params": self.params}


def file_fingerprint(prep_dir: str | Path) -> dict:
    prep_dir = Path(prep_dir)
    return {p.name: sha256_file(p) for p in sorted(prep_dir.glob("*")) if p.is_file() and p.suffix in {".csv", ".npy", ".json"}}
