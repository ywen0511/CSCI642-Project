"""Tests for feedback identity, experimental gates, reward accounting and checkpoint failure."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from .checkpoint import validate_snapshot
from .environment import FeedbackReward, error_label
from .feedback import FeedbackLibrary, ROOT, sha256
from .suite import jobs
from .train import read_config


class ToyReach(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    observation_space = gym.spaces.Dict({k: gym.spaces.Box(-10, 10, (3,), dtype=np.float64)
                                       for k in ("achieved_goal", "desired_goal")})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.d, self.t = 1.0, 0
        return self.obs(), {}

    def obs(self):
        return {"achieved_goal": np.array([self.d, 0, 0]), "desired_goal": np.zeros(3)}

    def step(self, action):
        self.d += float(action[0])
        self.t += 1
        return self.obs(), -1.0, False, self.t == 7, {"is_success": False}


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = json.loads((ROOT / "rl/protocol.json").read_text())
        self.profile = json.loads((ROOT / "configs/rl_feedback.json").read_text())
        self.profile["splits"]["validation"] = ["s11"]
        rows = []
        for model in ("eegnet", "cbramod"):
            for i in range(8):
                row = dict(subject_id="s11", trial_id=f"trial{i}", label=i % 2, split="validation",
                           model=model, decoder_seed=42, p_error_raw=0.1 + 0.1 * i, p_error=0.1 + 0.1 * i)
                row.update({k: self.profile["models"][model][k] for k in
                            ("checkpoint_id", "preprocessing_id", "calibration_id")})
                rows.append(row)
        self.rows = rows
        self.write_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def write_fixture(self):
        self.table = self.root / "validation_predictions.csv"
        with self.table.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        self.profile["tables"]["validation"] = {"asset": self.table.name, "bytes": self.table.stat().st_size,
                                                 "sha256": sha256(self.table), "trials_per_model": 8}
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(self.profile))

    def library(self):
        return FeedbackLibrary(self.root, "validation", self.profile_path)

    def test_model_pairing_and_label_conditioning(self):
        lib = self.library()
        for label in (0, 1):
            for event in range(30):
                a = lib.sample("eegnet", "s11", label, 3, event)
                b = lib.sample("cbramod", "s11", label, 3, event)
                self.assertEqual(a["trial_id"], b["trial_id"])
                self.assertEqual(a["label"], label)

    def test_independent_control_does_not_consult_label(self):
        lib = self.library()
        for event in range(20):
            a = lib.sample("eegnet", "s11", 0, 7, event, True)
            b = lib.sample("eegnet", "s11", 1, 7, event, True)
            self.assertEqual(a, b)

    def test_checksum_tampering(self):
        self.table.write_text(self.table.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.library()

    def test_cross_model_label_mismatch(self):
        self.rows[-1]["label"] = 0
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "labels differ"):
            self.library()

    def test_wrong_split_and_checkpoint(self):
        for field, value in [("split", "test"), ("decoder_seed", 99), ("checkpoint_id", "wrong")]:
            with self.subTest(field=field):
                old = self.rows[0][field]
                self.rows[0][field] = value
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.library()
                self.rows[0][field] = old

    def test_invalid_probabilities(self):
        for value in (float("nan"), -0.1, 1.01):
            self.rows[0]["p_error"] = value
            self.write_fixture()
            with self.assertRaisesRegex(ValueError, "probability"):
                self.library()

    def test_duplicate_trial(self):
        self.rows[1]["trial_id"] = self.rows[0]["trial_id"]
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.library()

    def test_participant_mixing_rejected(self):
        with self.assertRaises(ValueError):
            self.library().sample("eegnet", "s02", 0, 1, 1)

    def test_label_boundaries(self):
        self.assertEqual(error_label(0.3, 0.4, 0.001, 0.05), 1)
        self.assertEqual(error_label(0.3, 0.2, 0.001, 0.05), 0)
        self.assertEqual(error_label(0.3, 0.3, 0.001, 0.05), 0)
        self.assertEqual(error_label(0.01, 0.04, 0.001, 0.05), 0)

    def test_feedback_interval_and_partial_terminal_segment(self):
        env = FeedbackReward(ToyReach(), self.cfg, "ideal", 0)
        env.reset()
        events, rewards = [], []
        for step in range(1, 8):
            _, reward, _, _, info = env.step([-0.01])
            if info["feedback_event"]:
                events.append(step)
            rewards.append(reward)
        self.assertEqual(events, [5, 7])
        self.assertAlmostEqual(sum(rewards), -7 + 2 * self.cfg["eeg_weight"])
        env.close()

    def test_dense_accounting_and_sparse_no_feedback(self):
        for condition in ("sparse", "dense"):
            env = FeedbackReward(ToyReach(), self.cfg, condition, 0)
            env.reset()
            _, reward, _, _, info = env.step([-0.1])
            self.assertAlmostEqual(reward, -1 + (1 if condition == "dense" else 0))
            self.assertEqual(info["eeg_reward"], 0)
            env.close()

    def test_eeg_reward_uses_calibrated_probability(self):
        env = FeedbackReward(ToyReach(), self.cfg, "eegnet", 0, self.library(), "s11")
        env.reset()
        for _ in range(5):
            _, reward, _, _, info = env.step([0.01])
        self.assertEqual(info["simulated_label"], 1)
        self.assertEqual(info["source_label"], 1)
        self.assertAlmostEqual(reward, -1 + self.cfg["eeg_weight"] * (1 - info["p_error"]))
        env.close()

    def test_final_requires_validation_evidence(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            read_config(ROOT / "rl/protocol.json", "final")

    def test_frozen_config_cannot_drift(self):
        evidence = self.root / "evidence.json"
        evidence.write_text(json.dumps({"split": "validation", "complete": True, "config": self.cfg}))
        cfg = {**self.cfg, "status": "frozen", "validation_evidence_file": evidence.name,
               "validation_evidence_sha256": sha256(evidence)}
        path = self.root / "frozen.json"
        path.write_text(json.dumps(cfg))
        self.assertEqual(read_config(path, "final")["status"], "frozen")
        cfg["eeg_weight"] += 1
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "differs"):
            read_config(path, "final")

    def test_corrupt_checkpoint_fails_before_loading(self):
        (self.root / "model.zip").write_bytes(b"invalid")
        (self.root / "metadata.json").write_text(json.dumps({"sha256": {"model.zip": "wrong",
            "replay_buffer.pkl": "wrong", "rng.pt": "wrong"}}))
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_snapshot(self.root)

    def test_suite_covers_conditions_and_keeps_shared_controls_single(self):
        profile = json.loads((ROOT / "configs/rl_feedback.json").read_text())
        validation = list(jobs("validation", profile))
        final = list(jobs("final", profile))
        self.assertEqual(len(validation), 33)
        self.assertEqual(len(final), 75)
        self.assertEqual(sum(j["condition"] == "sparse" for j in final), 5)
        self.assertEqual({j["seed"] for j in validation}, {100, 101, 102})


if __name__ == "__main__":
    unittest.main()
