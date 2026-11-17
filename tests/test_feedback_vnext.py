"""CPU interface checks of the new (vnext) feedback profile with the unchanged RL feedback code.

Run in the RL environment:  VNEXT_DIR=<dir with vnext tables + manifest>  HIST_DIR=<dir with historical tables>
  python -m unittest tests.test_feedback_vnext -v
"""
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VNEXT = ROOT / "configs" / "decoder_improvement" / "rl_feedback_vnext.json"
HIST = ROOT / "configs" / "rl_feedback.json"


@unittest.skipUnless(os.environ.get("VNEXT_DIR"), "VNEXT_DIR not set")
class VNextFeedback(unittest.TestCase):
    def setUp(self):
        from rl.feedback import FeedbackLibrary
        self.FL = FeedbackLibrary
        self.prof = json.loads(VNEXT.read_text())

    def test_profile_is_separate_and_historical_unchanged(self):
        self.assertNotEqual(VNEXT.resolve(), HIST.resolve())
        hist = json.loads(HIST.read_text())
        self.assertEqual(hist["release_tag"], "eeg-stage1-20261101-bccbb8a")
        self.assertIn("calibration_reuse", self.prof)

    def test_tables_load_with_identity_checks(self):
        for split in ("validation", "test"):
            lib = self.FL(os.environ["VNEXT_DIR"], split, VNEXT)
            for model in lib.rows:
                rows = lib.rows[model].values()
                self.assertEqual(len(rows), self.prof["tables"][split]["trials_per_model"])
                self.assertEqual({r["subject_id"] for r in rows}, set(self.prof["splits"][split]))
                self.assertTrue(all(0 <= r["p_error"] <= 1 for r in rows))

    def test_paired_and_independent_sampling_and_reward(self):
        from rl.environment import FeedbackReward  # noqa: F401  (import check of the unchanged reward code)
        lib = self.FL(os.environ["VNEXT_DIR"], "test", VNEXT)
        models = sorted(lib.rows)
        for subject in self.prof["splits"]["test"]:
            for label in (0, 1):
                for event in range(25):
                    rows = [lib.sample(m, subject, label, 7, event) for m in models]
                    self.assertEqual(len({r["trial_id"] for r in rows}), 1)  # paired across decoders
                    self.assertTrue(all(r["label"] == label for r in rows))
                    self.assertEqual(lib.sample(models[0], subject, label, 7, event),
                                     lib.sample(models[0], subject, label, 7, event))  # deterministic
            a = lib.sample(models[0], subject, 0, 3, 5, independent=True)
            b = lib.sample(models[0], subject, 1, 3, 5, independent=True)
            self.assertEqual(a, b)  # independent control ignores the label
        r = lib.sample(models[0], self.prof["splits"]["test"][0], 1, 1, 0)
        w = 1.0
        self.assertAlmostEqual(w * (0.5 - r["p_error"]), 0.5 - r["p_error"])

    def test_tampering_rejected(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            for f in Path(os.environ["VNEXT_DIR"]).iterdir():
                shutil.copy(f, t)
            p = Path(t) / self.prof["tables"]["test"]["asset"]
            p.write_text(p.read_text() + "\n")
            with self.assertRaises(ValueError):
                self.FL(t, "test", VNEXT)


@unittest.skipUnless(os.environ.get("HIST_DIR"), "HIST_DIR not set")
class HistoricalFeedbackStillValid(unittest.TestCase):
    def test_default_profile_loads(self):
        from rl.feedback import FeedbackLibrary
        for split in ("validation", "test"):
            lib = FeedbackLibrary(os.environ["HIST_DIR"], split)
            self.assertEqual(sorted(lib.rows), ["cbramod", "eegnet"])


if __name__ == "__main__":
    unittest.main()
