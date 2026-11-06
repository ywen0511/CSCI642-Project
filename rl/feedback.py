"""Validated, participant-specific replay of frozen decoder probabilities."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/rl_feedback.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FeedbackLibrary:
    def __init__(self, directory, split, profile_path=PROFILE):
        self.profile = json.loads(Path(profile_path).read_text())
        if split not in ("validation", "test"):
            raise ValueError("Feedback split must be validation or test")
        self.split = split
        spec = self.profile["tables"][split]
        path = Path(directory) / spec["asset"]
        payload = path.read_bytes()
        self.table_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != spec["bytes"] or self.table_sha256 != spec["sha256"]:
            raise ValueError(f"Frozen feedback checksum/size mismatch: {path}")
        self.rows = {m: {} for m in self.profile["models"]}
        required = {"subject_id", "trial_id", "label", "split", "model", "decoder_seed",
                    "p_error_raw", "p_error", "checkpoint_id", "preprocessing_id", "calibration_id"}
        reader = csv.DictReader(io.StringIO(payload.decode()))
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing feedback columns")
        for row in reader:
            model = row["model"]
            if model not in self.rows:
                raise ValueError(f"Unexpected decoder: {model}")
            if row["split"] != split or int(row["decoder_seed"]) != self.profile["decoder_seed"]:
                raise ValueError("Wrong split or decoder seed")
            if row["subject_id"] not in self.profile["splits"][split]:
                raise ValueError("Participant is not in the declared split")
            row["label"] = int(row["label"])
            if row["label"] not in (0, 1):
                raise ValueError("Invalid label")
            for col in ("p_error", "p_error_raw"):
                row[col] = float(row[col])
                if not math.isfinite(row[col]) or not 0 <= row[col] <= 1:
                    raise ValueError("Invalid probability")
            for col in ("checkpoint_id", "preprocessing_id", "calibration_id"):
                if row[col] != self.profile["models"][model][col]:
                    raise ValueError(f"Wrong {col}")
            key = (row["subject_id"], row["trial_id"])
            if key in self.rows[model]:
                raise ValueError("Duplicate trial")
            self.rows[model][key] = row
        reference = self.rows["eegnet"]
        for model, rows in self.rows.items():
            if len(rows) != spec["trials_per_model"] or rows.keys() != reference.keys():
                raise ValueError("Decoder trial sets/counts differ")
            if any(rows[k]["label"] != reference[k]["label"] for k in reference):
                raise ValueError("Decoder labels differ")
            for subject in self.profile["splits"][split]:
                if {r["label"] for r in rows.values() if r["subject_id"] == subject} != {0, 1}:
                    raise ValueError("Participant lacks a label pool")

    def sample(self, model, subject, label, seed, event_index, independent=False):
        """With replacement; identical keys choose identical trial IDs across decoders.

        Independent control samples from the full participant pool without consulting
        the simulated label. Event-index pairing does not force diverging policies to
        visit the same states or produce the same error labels.
        """
        if model not in self.rows or subject not in self.profile["splits"][self.split]:
            raise ValueError("Unknown decoder or participant")
        if label not in (0, 1) or event_index < 0:
            raise ValueError("Invalid sampling key")
        candidates = [r for _, r in sorted(self.rows[model].items())
                      if r["subject_id"] == subject and (independent or r["label"] == label)]
        key = json.dumps([int(seed), self.split, subject, int(event_index),
                          "independent" if independent else int(label)])
        rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))
        return candidates[rng.randrange(len(candidates))].copy()
