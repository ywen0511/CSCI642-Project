"""Tests for protocol v3 (centered EEG reward) and the route-preference extension (FINAL_V3_HANDOFF.md 6-7)."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .environment import CONDITIONS, FeedbackReward, make_task, reward_center
from .feedback import ROOT, sha256
from .metrics import normalized_auc
from .preference import (Geometry, GoalFilter, PreferenceFeedbackReward, crossings_left, episode_metrics,
                         make_preference_task)
from .test_stage2 import ToyReach
from .train import read_config, training_sources

V3 = ROOT / "configs/final_v3/v3_candidates/v3_w100.json"
PREF = ROOT / "configs/final_v3/pref_candidates/pref_w100.json"


class FixedLibrary:
    """Deterministic library stub: p_error depends only on (label, independent)."""

    def __init__(self, split="validation", subject="s11"):
        self.split, self.profile = split, {"splits": {split: [subject]}}

    def sample(self, model, subject, label, seed, event_index, independent=False):
        p = 0.37 if independent else (0.31 if label == 0 else 0.62)
        return {"p_error": p, "trial_id": f"t{event_index}", "label": label}


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def cfg(self, path=V3):
        return read_config(path, "pilot")

    def write(self, cfg, name="cfg.json"):
        path = self.root / name
        path.write_text(json.dumps(cfg))
        return path

    def test_v2_configs_still_load_unchanged(self):
        for path in sorted((ROOT / "configs/rl_v2").glob("candidate_*.json")) + [ROOT / "rl/protocol.json"]:
            cfg = read_config(path, "pilot")
            self.assertEqual(cfg["reward_center"], 0.5)
            self.assertEqual(training_sources(cfg), ["train.py", "environment.py", "feedback.py", "checkpoint.py"])
        frozen = read_config(ROOT / "configs/rl_v2/frozen/protocol.json", "final")
        self.assertEqual(frozen["eeg_weight"], 1.0)
        # v2 keeps the scalar center 0.5 for every EEG condition
        for c in CONDITIONS:
            self.assertEqual(reward_center(frozen, c), None if c in ("sparse", "dense") else 0.5)

    def test_v2_rejects_object_center_and_v3_rejects_scalar(self):
        v2 = json.loads((ROOT / "configs/rl_v2/candidate_w100.json").read_text())
        v2["reward_center"] = {"eegnet": 0.5, "cbramod": 0.5, "ideal": 0.5}
        with self.assertRaises(ValueError):
            read_config(self.write(v2), "pilot")
        v3 = json.loads(V3.read_text())
        v3["reward_center"] = 0.5
        with self.assertRaises(ValueError):
            read_config(self.write(v3), "pilot")

    def test_centers_come_from_committed_validation_file(self):
        cfg = self.cfg()
        centers = json.loads((ROOT / cfg["centers_file"]).read_text())
        self.assertEqual(centers["split"], "validation")
        self.assertEqual(centers["participants"], ["s11", "s13"])
        self.assertEqual(cfg["reward_center"], centers["centers"])
        for m in ("eegnet", "cbramod"):
            pm = centers["per_model"][m]
            self.assertAlmostEqual(pm["center"], 0.5 * (pm["mean_p_error_label0"] + pm["mean_p_error_label1"]), 15)

    def test_wrong_missing_or_tampered_centers_rejected(self):
        good = json.loads(V3.read_text())
        bad_cases = []
        c = copy.deepcopy(good); c["reward_center"]["eegnet"] += 1e-6; bad_cases.append(c)
        c = copy.deepcopy(good); del c["reward_center"]["cbramod"]; bad_cases.append(c)
        c = copy.deepcopy(good); c["reward_center"]["ideal"] = 0.45; bad_cases.append(c)
        c = copy.deepcopy(good); c["centers_sha256"] = "0" * 64; bad_cases.append(c)
        c = copy.deepcopy(good); del c["centers_file"]; bad_cases.append(c)
        c = copy.deepcopy(good); c["centers_file"] = "configs/final_v3/missing.json"; bad_cases.append(c)
        c = copy.deepcopy(good); c["centers_file"] = "/etc/passwd"; bad_cases.append(c)
        for i, cfg in enumerate(bad_cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                read_config(self.write(cfg, f"bad{i}.json"), "pilot")
        # a file whose content changed after the hash was recorded
        tampered = json.loads((ROOT / good["centers_file"]).read_text())
        tampered["centers"]["eegnet"] = 0.4
        fake_root = self.root / "repo"
        (fake_root / "configs/final_v3").mkdir(parents=True)
        (fake_root / good["centers_file"]).write_text(json.dumps(tampered))
        with patch("rl.train.ROOT", fake_root), self.assertRaises(ValueError):
            read_config(V3, "pilot")

    def test_v3_reward_equals_weight_times_center_minus_p(self):
        cfg = self.cfg()
        w = cfg["eeg_weight"]
        for condition in ("eegnet", "cbramod", "independent_eegnet", "independent_cbramod", "ideal"):
            env = FeedbackReward(ToyReach(), cfg, condition, 0, None if condition == "ideal" else FixedLibrary(),
                                 None if condition == "ideal" else "s11")
            env.reset()
            seen = 0
            for action in [-0.01] * 5 + [0.01] * 5 + [-0.01] * 5:
                _, _, _, trunc, info = env.step(np.array([action]))
                if info["feedback_event"]:
                    seen += 1
                    p = info["p_error"]
                    c = 0.5 if condition == "ideal" else cfg["reward_center"][condition.removeprefix("independent_")]
                    self.assertEqual(info["eeg_reward"], w * (c - p))
                    if condition == "ideal":
                        self.assertEqual(info["eeg_reward"], w / 2 if info["simulated_label"] == 0 else -w / 2)
                if trunc:
                    env.reset()
            self.assertGreaterEqual(seen, 2, condition)

    def test_v2_and_v3_sparse_dense_transitions_bitwise_identical(self):
        v2 = read_config(ROOT / "configs/rl_v2/candidate_w100.json", "pilot")
        v3 = self.cfg()
        for condition in ("sparse", "dense"):
            logs = []
            for name, cfg in (("v2", v2), ("v3", v3)):
                path = self.root / f"{condition}_{name}.csv"
                env = FeedbackReward(make_task(cfg), cfg, condition, 7, log_path=path)
                env.reset()
                env.action_space.seed(3)
                for _ in range(120):
                    _, _, term, trunc, _ = env.step(env.action_space.sample())
                    if term or trunc:
                        env.reset()
                env.close()
                logs.append(path.read_bytes())
            self.assertEqual(logs[0], logs[1], condition)

    def test_v3_training_sources_and_schema(self):
        cfg = self.cfg()
        self.assertEqual(cfg["schema"], "csci642-rl-stage2/v3")
        self.assertEqual(training_sources(cfg), ["train.py", "environment.py", "feedback.py", "checkpoint.py"])
        self.assertEqual(training_sources(read_config(PREF, "pilot"))[-1], "preference.py")


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read_config(PREF, "pilot")
        self.pref = self.cfg["preference"]

    def test_goal_filter_reproducible_and_in_observation(self):
        goals = []
        for _ in range(2):
            env = make_preference_task(self.cfg)
            seq = []
            for seed in list(range(20000, 20030)) + [30000, 30001, 30002, 5, 1000000]:
                obs, info = env.reset(seed=seed)
                g = np.asarray(env.unwrapped.goal)
                np.testing.assert_array_equal(obs["desired_goal"], g)
                self.assertGreaterEqual(np.linalg.norm(g[:2] - env.start[:2]), self.pref["min_goal_distance_xy_m"])
                np.testing.assert_array_equal(obs["achieved_goal"], env.start)
                seq.append((g.tolist(), info["goal_redraws"]))
            env.close()
            goals.append(seq)
        self.assertEqual(goals[0], goals[1])
        self.assertTrue(any(r > 0 for _, r in goals[0]), "the filter never redrew a goal")
        # the filtered goal is what the reward uses
        env = make_preference_task(self.cfg)
        obs, _ = env.reset(seed=20000)
        _, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        d = np.linalg.norm(env.unwrapped.goal - env.unwrapped._get_obs()["achieved_goal"])
        self.assertEqual(bool(info["is_success"]), bool(d < self.cfg["goal_tolerance_m"]))
        env.close()

    def test_geometry_coordinates(self):
        g = Geometry([0, 0, 0], [0.2, 0, 0], self.pref)
        self.assertEqual(g.coords([0.1, 0.03]), (0.5, 0.03))  # left of +x is +y
        self.assertTrue(g.classify([0.1, 0.0])["pillar"])
        self.assertTrue(g.classify([0.1, -0.03])["right_corridor"])
        self.assertFalse(g.classify([0.1, 0.03])["violation"])
        self.assertFalse(g.classify([0.03, -0.02])["violation"])  # a = 0.15 < 0.25
        self.assertTrue(g.classify([0.05, -1e-9])["right_corridor"])  # a = 0.25 inclusive
        np.testing.assert_allclose(g.left_normal(), [0, 1])
        # crossing lateral offset and episode metrics
        left = [[0, 0], [0.05, 0.04], [0.15, 0.04], [0.2, 0]]
        right = [[0, 0], [0.05, -0.04], [0.15, -0.04], [0.2, 0]]
        self.assertAlmostEqual(crossings_left(g, left)[0], 0.04)
        m = episode_metrics(g, left, True)
        self.assertTrue(m["preferred_pass"] and m["violation_free"] and not m["pillar_entry"])
        m = episode_metrics(g, right, True)
        self.assertFalse(m["preferred_pass"] or m["violation_free"])
        self.assertTrue(m["right_corridor_entry"])
        self.assertFalse(episode_metrics(g, left, False)["preferred_pass"])

    def test_violation_segment_labeled_error_even_with_progress(self):
        env = PreferenceFeedbackReward(make_preference_task(self.cfg), self.cfg, "ideal", 0)
        obs, _ = env.reset()
        geo = env.geometry
        target = np.append(geo.m, obs["desired_goal"][2])  # straight at the pillar centre: progress + violation
        labels = []
        for _ in range(15):
            delta = target - obs["achieved_goal"]
            action = np.append(np.clip(delta / 0.05, -0.4, 0.4), 0).astype(np.float32)
            obs, _, _, _, info = env.step(action)
            if info["segment_end"]:
                labels.append((info["segment_violation"], info["simulated_label"], info["segment_progress"]))
        env.close()
        viol = [x for x in labels if x[0]]
        self.assertTrue(viol, "never entered the pillar")
        for _, label, progress in viol:
            self.assertEqual(label, 1)
        self.assertTrue(any(p > 0.001 for _, _, p in viol), "no violating segment with progress")

    def test_preference_wrapper_needs_filtered_task(self):
        with self.assertRaises(ValueError):
            PreferenceFeedbackReward(make_task(self.cfg), self.cfg, "ideal", 0)

    def test_invalid_preference_geometry_rejected(self):
        for change in ({"pillar_radius_m": 0.03}, {"preferred_side": "right"}, {"corridor_a": [0.6, 0.75]},
                       {"extra": 1}):
            cfg = json.loads(PREF.read_text())
            cfg["preference"].update(change)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "c.json"
                path.write_text(json.dumps(cfg))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_config(path, "pilot")

    def test_preferred_pass_auc_metric(self):
        curve = [{"steps": 0, "preferred_pass_rate": 0.0}, {"steps": 10, "preferred_pass_rate": 1.0}]
        self.assertEqual(normalized_auc(curve, 10, "preferred_pass_rate"), 0.5)


if __name__ == "__main__":
    unittest.main()
