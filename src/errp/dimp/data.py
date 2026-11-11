"""Explicit participant views of a prepared band (no split generator, no redraw).

Prepared directory layout (scripts/prepare_data.py): trial_manifest.csv, epoch_trial_ids.npy,
epochs_256hz_uV.npy (N,27,512), epochs_200hz_patches_uV.npy (N,27,2,200), preprocessing_config.json.
Trial order is the manifest order of retained trials and is identical across bands (checked by the diagnostics).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import sha256_file

VIEW_FILE = {"eeg256": "epochs_256hz_uV.npy", "patch200": "epochs_200hz_patches_uV.npy"}


@dataclass
class SubjectData:
    X: np.ndarray
    y: np.ndarray
    trial_ids: np.ndarray
    subject_ids: np.ndarray


class Prepared:
    def __init__(self, prep_dir):
        self.dir = Path(prep_dir)
        self.config = json.loads((self.dir / "preprocessing_config.json").read_text())
        self.prep_id = self.config["preprocessing_id"]
        man = pd.read_csv(self.dir / "trial_manifest.csv", dtype={"subject_id": str, "trial_id": str})
        self.ids = np.load(self.dir / "epoch_trial_ids.npy", allow_pickle=False).astype(str)
        kept = man[man["retained"] == 1].set_index("trial_id")
        if len(kept) != len(self.ids) or not (kept.index.values == self.ids).all():
            raise ValueError("manifest / epoch order mismatch")
        self.y = kept["label"].values.astype(np.int64)
        self.subjects = kept["subject_id"].values.astype(str)

    def input_hashes(self, view) -> dict:
        names = ["trial_manifest.csv", "epoch_trial_ids.npy", "preprocessing_config.json", VIEW_FILE[view]]
        return {"prep_id": self.prep_id, "view": view, "files": {n: sha256_file(self.dir / n) for n in names}}

    def view(self, view, participants) -> SubjectData:
        X = np.load(self.dir / VIEW_FILE[view], mmap_mode="r")
        mask = np.isin(self.subjects, list(participants))
        if not mask.any():
            raise ValueError(f"no trials for {participants}")
        missing = set(participants) - set(self.subjects[mask])
        if missing:
            raise ValueError(f"participants without trials: {missing}")
        return SubjectData(np.ascontiguousarray(X[mask]).astype(np.float32), self.y[mask], self.ids[mask],
                           self.subjects[mask])
