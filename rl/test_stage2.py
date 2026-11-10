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

from .analysis import METRICS, aggregate, auroc, paired_differences
from .checkpoint import validate_snapshot
from .environment import FeedbackReward, error_label
from .feedback import FeedbackLibrary, ROOT, sha256
from .metrics import dedupe, normalized_auc
from .suite import jobs
from .train import check_episode_seed_range, read_config


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
        self.assertEqual(error_label(0.3, 0.4, 0.001), 1)
        self.assertEqual(error_label(0.3, 0.2, 0.001), 0)
        for end in (0.3, 0.3005, 0.2995, 0.301, 0.299):
            with self.subTest(end=end):
                self.assertIsNone(error_label(0.3, end, 0.001))
        self.assertEqual(error_label(0.3, 0.301001, 0.001), 1)
        self.assertEqual(error_label(0.3, 0.298999, 0.001), 0)
        self.assertEqual(error_label(0.01, 0.04, 0.001), 1)
        self.assertIsNone(error_label(0.01, 0.01, 0.001))

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
        self.assertAlmostEqual(sum(rewards), -7 + self.cfg["eeg_weight"])
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
        self.assertAlmostEqual(reward, -1 + self.cfg["eeg_weight"] * (0.5 - info["p_error"]))
        env.close()

    def test_signed_rewards_for_every_feedback_condition(self):
        for condition in ("ideal", "eegnet", "cbramod", "independent_eegnet", "independent_cbramod"):
            for p in (0.0, 0.2, 0.5, 0.8, 1.0):
                if condition == "ideal" and p not in (0.0, 1.0):
                    continue
                with self.subTest(condition=condition, p=p):
                    lib = self.library() if condition != "ideal" else None
                    env = FeedbackReward(ToyReach(), self.cfg, condition, 0, lib,
                                         "s11" if lib else None)
                    env.reset()
                    direction = 0.01 if p >= 0.5 else -0.01
                    with patch.object(FeedbackLibrary, "sample", return_value={
                            "p_error": p, "trial_id": "fixed", "label": int(p >= 0.5)}):
                        for _ in range(5):
                            _, reward, _, _, info = env.step([direction])
                    self.assertAlmostEqual(info["eeg_reward"], self.cfg["eeg_weight"] * (0.5 - p))
                    self.assertAlmostEqual(reward, -1 + info["eeg_reward"])
                    self.assertEqual(info["feedback_event"], 1)
                    env.close()

    def test_neutral_segments_never_sample_or_advance_events(self):
        for condition in ("ideal", "eegnet", "cbramod", "independent_eegnet", "independent_cbramod"):
            with self.subTest(condition=condition):
                lib = self.library() if condition != "ideal" else None
                env = FeedbackReward(ToyReach(), self.cfg, condition, 0, lib,
                                     "s11" if lib else None, self.root / (condition + ".csv"))
                env.reset()
                with patch.object(FeedbackLibrary, "sample", side_effect=AssertionError("Neutral sampled")):
                    for step in range(1, 8):
                        _, reward, _, _, info = env.step([0.0])
                        self.assertEqual(reward, -1)
                        self.assertEqual(info["segment_end"], int(step in (5, 7)))
                        self.assertEqual(info["feedback_eligible"], 0)
                        self.assertEqual(info["feedback_event"], 0)
                        for name in ("simulated_label", "p_error", "trial_id", "source_label", "event_index"):
                            self.assertEqual(info[name], "")
                self.assertEqual(env.counters()["event_count"], 0)
                env.close()
                with (self.root / (condition + ".csv")).open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 7)
                self.assertTrue(all(r["simulated_label"] == "" for r in rows))

    def test_skipped_segment_closes_and_next_partial_segment_uses_new_start(self):
        env = FeedbackReward(ToyReach(), self.cfg, "ideal", 0)
        env.reset()
        for _ in range(5):
            env.step([0.00018])  # Total 0.9 mm: skipped, but start resets.
        self.assertEqual(env.segment_steps, 0)
        self.assertEqual(env.event_count, 0)
        for _ in range(2):
            _, reward, _, _, info = env.step([-0.0006])  # Partial segment improves 1.2 mm.
        self.assertEqual(info["simulated_label"], 0)
        self.assertEqual(info["event_index"], 0)
        self.assertEqual(info["feedback_event"], 1)
        self.assertAlmostEqual(reward, -1 + self.cfg["eeg_weight"] * 0.5)
        self.assertEqual(env.event_count, 1)
        env.close()

    def test_baselines_keep_original_rewards_at_segment_boundaries(self):
        for condition in ("sparse", "dense"):
            env = FeedbackReward(ToyReach(), self.cfg, condition, 0)
            env.reset()
            for _ in range(7):
                _, reward, _, _, info = env.step([-0.01])
                self.assertAlmostEqual(reward, -1 + (0.1 if condition == "dense" else 0))
                self.assertEqual(info["feedback_event"], 0)
                self.assertEqual(info["eeg_reward"], 0)
            env.close()

    def test_legacy_or_mismatched_protocol_rejected(self):
        for key, value in (("schema", "csci642-rl-stage2/v1"), ("reward_center", 1.0),
                           ("feedback_rule", "stalled_is_non_error")):
            cfg = {**self.cfg, key: value}
            path = self.root / "wrong_protocol.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                read_config(path, "pilot")

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
        validation = list(jobs("validation", profile, self.cfg))
        final = list(jobs("final", profile, self.cfg))
        self.assertEqual(len(validation), 33)
        self.assertEqual(len(final), 45)
        self.assertEqual(sum(j["condition"] == "sparse" for j in final), 3)
        self.assertEqual({j["seed"] for j in validation}, {100, 101, 102})
        self.assertEqual({j["seed"] for j in final}, {0, 1, 2})
        five = {**self.cfg, "final_seeds": [0, 1, 2, 3, 4]}
        final5 = list(jobs("final", profile, five))
        self.assertEqual(len(final5), 75)
        self.assertEqual(sum(j["condition"] == "ideal" for j in final5), 5)
        self.assertEqual(sum(j["condition"] == "eegnet" for j in final5), 15)
        self.assertEqual({j["subject"] for j in final5 if j["condition"] == "cbramod"}, {"s02", "s08", "s09"})
        self.assertEqual(len({j["name"] for j in final5}), 75)

    def test_seed_lists_validated(self):
        bad = [("validation_seeds", []), ("final_seeds", [0, 0]), ("final_seeds", [0, -1]),
               ("final_seeds", [0, 1.0]), ("final_seeds", [True]), ("final_seeds", "012"),
               ("validation_seeds", [100, 0]), ("training_episode_seed_stride", 0)]
        for key, value in bad:
            with self.subTest(key=key, value=value):
                path = self.root / "seeds.json"
                path.write_text(json.dumps({**self.cfg, key: value}))
                with self.assertRaises(ValueError):
                    read_config(path, "pilot")
        missing = {k: v for k, v in self.cfg.items() if k != "final_seeds"}
        path = self.root / "missing.json"
        path.write_text(json.dumps(missing))
        with self.assertRaises(ValueError):
            read_config(path, "pilot")

    def test_frozen_identity_includes_seed_lists(self):
        evidence = self.root / "evidence.json"
        evidence.write_text(json.dumps({"split": "validation", "complete": True, "config": self.cfg}))
        cfg = {**self.cfg, "status": "frozen", "validation_evidence_file": evidence.name,
               "validation_evidence_sha256": sha256(evidence), "final_seeds": [0, 1, 2, 3, 4]}
        path = self.root / "frozen.json"
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "differs"):
            read_config(path, "final")

    def test_training_episode_seeds_are_disjoint_across_rl_seeds(self):
        class Recorder(ToyReach):
            def reset(self, *, seed=None, options=None):
                self.seeds = getattr(self, "seeds", []) + [seed]
                return super().reset(seed=seed, options=options)

        used_all = []
        for seed in (0, 1, 2, 100):
            inner = Recorder()
            env = FeedbackReward(inner, self.cfg, "sparse", seed)
            for _ in range(4):
                env.reset(seed=12345)  # an externally supplied seed (as SB3 passes) is ignored
            stride = self.cfg["training_episode_seed_stride"]
            self.assertEqual(inner.seeds, [seed * stride + k for k in range(4)])
            used_all += inner.seeds
            env.close()
        self.assertEqual(len(used_all), len(set(used_all)))

    def test_episode_seed_range_excludes_evaluation_goals(self):
        check_episode_seed_range(self.cfg, 0, 150000)
        check_episode_seed_range(self.cfg, 102, 150000)
        with self.assertRaises(ValueError):
            check_episode_seed_range({**self.cfg, "training_episode_seed_stride": 10}, 0, 150000)
        with self.assertRaises(ValueError):
            check_episode_seed_range({**self.cfg, "eval_seed_base": 500}, 0, 150000)

    def test_normalized_auc_and_duplicate_final_record(self):
        def rec(step, s):
            return {"steps": step, "success_rate": s, "episodes": []}
        curve = dedupe([rec(0, 0.0), rec(50, 1.0), rec(100, 1.0), rec(100, 1.0)])
        self.assertEqual([r["steps"] for r in curve], [0, 50, 100])
        self.assertAlmostEqual(normalized_auc(curve, 100), (25 + 50) / 100)
        self.assertAlmostEqual(normalized_auc(dedupe([rec(0, 0.2), rec(100, 0.2)]), 100), 0.2)
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            dedupe([rec(100, 1.0), rec(100, 0.5)])
        with self.assertRaises(ValueError):
            normalized_auc([rec(10, 0.0), rec(100, 1.0)], 100)  # no step-0 record
        with self.assertRaises(ValueError):
            normalized_auc([rec(0, 0.0), rec(90, 1.0)], 100)  # does not reach the budget
        faster = normalized_auc([rec(0, 0), rec(20, 1), rec(100, 1)], 100)
        slower = normalized_auc([rec(0, 0), rec(80, 1), rec(100, 1)], 100)
        self.assertGreater(faster, slower)

    def test_auroc_ties_and_missing_class(self):
        self.assertAlmostEqual(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)
        self.assertAlmostEqual(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertTrue(np.isnan(auroc([0.1, 0.2], [1, 1])))

    def test_aggregate_macro_average_and_shared_controls(self):
        rows = []
        for seed, s in ((0, 0.2), (1, 0.4)):
            rows.append({"condition": "sparse", "subject": None, "seed": seed, **dict.fromkeys(METRICS, s)})
        for subject, vals in (("s02", (1.0, 1.0)), ("s08", (0.0, 0.0)), ("s09", (0.5, 0.5))):
            for seed, v in zip((0, 1), vals):
                rows.append({"condition": "eegnet", "subject": subject, "seed": seed, **dict.fromkeys(METRICS, v)})
        within, overall = aggregate(rows)
        eeg = next(o for o in overall if o["condition"] == "eegnet")
        sparse = next(o for o in overall if o["condition"] == "sparse")
        self.assertAlmostEqual(eeg["success_rate_mean"], 0.5)
        self.assertEqual(eeg["n_participants"], 3)
        self.assertAlmostEqual(eeg["success_rate_sd"], 0.5)
        self.assertAlmostEqual(sparse["success_rate_mean"], 0.3)
        self.assertEqual(sparse["n_participants"], 0)
        self.assertEqual(sparse["n_runs"], 2)
        diffs = paired_differences(rows, "eegnet", "sparse")
        self.assertEqual(len(diffs), 6)
        self.assertAlmostEqual(next(d for d in diffs if d["subject"] == "s02" and d["seed"] == 1)["difference"], 0.6)

    def test_info_does_not_shadow_sb3_episode_statistics(self):
        env = FeedbackReward(ToyReach(), self.cfg, "ideal", 0)
        env.reset()
        for _ in range(7):
            _, _, _, truncated, info = env.step([-0.01])
            self.assertNotIn("episode", info)
            self.assertEqual(info["episode_index"], 1)
        env.close()

    def test_policy_observation_has_no_feedback_fields(self):
        env = FeedbackReward(ToyReach(), self.cfg, "eegnet", 0, self.library(), "s11")
        obs, _ = env.reset()
        self.assertEqual(set(obs), {"achieved_goal", "desired_goal"})
        for _ in range(5):
            obs, _, _, _, info = env.step([0.01])
        self.assertEqual(info["feedback_event"], 1)
        self.assertEqual(set(obs), {"achieved_goal", "desired_goal"})
        self.assertEqual(env.observation_space, ToyReach.observation_space)
        env.close()


if __name__ == "__main__":
    unittest.main()
