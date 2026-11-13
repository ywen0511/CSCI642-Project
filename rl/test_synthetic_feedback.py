"""Tests for the controlled-AUROC synthetic feedback source.

Run: python -m unittest rl.test_synthetic_feedback -v
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

from .synthetic_feedback import (AUROC_LEVELS, DEFAULT_BASE_RATE, SyntheticFeedbackSource,
                                 calibrated_p, dprime_for_auroc, keyed_normal)

N_DRAWS = 100_000


def auroc(labels, scores):
    """Mann-Whitney AUROC with average ranks for ties."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def draw(source, seed, n=N_DRAWS, base_rate=DEFAULT_BASE_RATE):
    labels = (np.random.default_rng(12345 + seed).random(n) < base_rate).astype(int)
    p = np.array([source.p_error(int(y), seed, i) for i, y in enumerate(labels)])
    return labels, p


class SyntheticFeedbackTest(unittest.TestCase):
    def test_dprime_and_logit(self):
        self.assertAlmostEqual(dprime_for_auroc(0.5), 0.0)
        self.assertAlmostEqual(dprime_for_auroc(0.76025), 1.0, places=3)  # Phi(1/sqrt2)
        self.assertAlmostEqual(calibrated_p(0.0, 2.0, 0.35), 0.35)
        with self.assertRaises(ValueError):
            dprime_for_auroc(1.0)

    def test_keyed_normal_deterministic_and_standard(self):
        self.assertEqual(keyed_normal(3, 17), keyed_normal(3, 17))
        self.assertNotEqual(keyed_normal(3, 17), keyed_normal(4, 17))
        x = np.array([keyed_normal(0, i) for i in range(N_DRAWS)])
        self.assertLess(abs(x.mean()), 0.01)
        self.assertLess(abs(x.std() - 1.0), 0.01)

    def test_auroc_and_calibration_all_levels(self):
        report = {}
        for k, target in enumerate(a for a in AUROC_LEVELS if a < 1.0):
            src = SyntheticFeedbackSource(target)
            labels, p = draw(src, seed=k)
            emp = auroc(labels, p)
            self.assertLess(abs(emp - target), 0.005, f"AUROC {target}: empirical {emp:.4f}")
            # Calibration-in-the-large and reliability (deciles of p).
            self.assertLess(abs(p.mean() - labels.mean()), 0.005)
            edges = np.quantile(p, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
            worst = 0.0
            for b in range(10):
                m = idx == b
                if m.sum() >= 1000:
                    worst = max(worst, abs(p[m].mean() - labels[m].mean()))
            self.assertLess(worst, 0.02, f"AUROC {target}: worst decile calibration gap {worst:.4f}")
            report[target] = {"empirical_auroc": round(emp, 4), "mean_p": round(float(p.mean()), 4),
                              "error_rate": round(float(labels.mean()), 4),
                              "worst_decile_gap": round(worst, 4)}
        out = Path(__file__).resolve().parents[1] / "bench/rl_frontier/unit_test_report.json"
        if out.parent.is_dir() and os.environ.get("ERRP_WRITE_TEST_REPORT") == "1":
            out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    def test_common_random_numbers_and_sample_api(self):
        a, b = SyntheticFeedbackSource(0.6), SyntheticFeedbackSource(0.9)
        self.assertEqual(a.margin(1, 0, 5) - a.dprime / 2, b.margin(1, 0, 5) - b.dprime / 2)
        row = a.sample("eegnet", a.name, 1, 7, 3)
        self.assertEqual(set(row), {"p_error", "trial_id", "label"})
        self.assertEqual(row, a.sample("eegnet", a.name, 1, 7, 3))
        self.assertTrue(0 < row["p_error"] < 1)
        with self.assertRaises(ValueError):
            a.sample("eegnet", a.name, 1, 7, 3, independent=True)

    def test_reward_wrapper_integration(self):
        from .environment import FeedbackReward, make_task
        cfg = json.loads((Path(__file__).resolve().parents[1] / "bench/rl_frontier/protocol_push.json").read_text())
        src = SyntheticFeedbackSource(0.8)
        env = FeedbackReward(make_task(cfg), cfg, "eegnet", 0, src, src.name)
        env.reset()
        rng = np.random.default_rng(0)
        events = 0
        for _ in range(500):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
            if info["feedback_event"]:
                events += 1
                self.assertAlmostEqual(info["eeg_reward"], cfg["eeg_weight"] * (0.5 - info["p_error"]))
            if term or trunc:
                env.reset()
        env.close()
        self.assertEqual(events, env.event_count)


if __name__ == "__main__":
    unittest.main()
