"""Unit tests for decoder-improvement metrics, selection, calibration and identities (CPU, no data needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "decoder_improvement"))

from errp.dimp import calibration as cal  # noqa: E402
from errp.dimp.fit import per_participant_auroc  # noqa: E402
from errp.dimp.spec import (cbramod_recipe, eegnet_recipe, fit_spec, folds, load_study, participant_macro_auroc,  # noqa: E402
                            recipe_id, selection_score)


class Metrics(unittest.TestCase):
    def test_class_direction(self):
        s = np.array(["a"] * 4)
        y = np.array([0, 0, 1, 1])
        self.assertEqual(per_participant_auroc(s, y, np.array([0.1, 0.2, 0.8, 0.9]))["a"], 1.0)
        self.assertEqual(per_participant_auroc(s, y, np.array([0.9, 0.8, 0.2, 0.1]))["a"], 0.0)

    def test_single_class_undefined_is_protocol_failure(self):
        pp = per_participant_auroc(np.array(["a", "a", "b", "b"]), np.array([0, 1, 1, 1]), np.array([.1, .9, .5, .6]))
        self.assertTrue(np.isnan(pp["b"]))
        with self.assertRaises(ValueError):
            participant_macro_auroc(pp)

    def test_selection_weights_participants_not_folds(self):
        # fold 1 has 3 participants, fold 2 has 2: equal per-participant weight
        rows = [{"seed": 42, "per_participant_auroc": {"p1": 1.0, "p2": 1.0, "p3": 1.0}},
                {"seed": 42, "per_participant_auroc": {"p4": 0.5, "p5": 0.5}}]
        self.assertAlmostEqual(selection_score(rows)["score"], (3 * 1.0 + 2 * 0.5) / 5)
        # seed mean first, then participant mean
        rows = [{"seed": 42, "per_participant_auroc": {"p1": 0.6}}, {"seed": 43, "per_participant_auroc": {"p1": 0.8}},
                {"seed": 42, "per_participant_auroc": {"p2": 0.5}}, {"seed": 43, "per_participant_auroc": {"p2": 0.5}}]
        self.assertAlmostEqual(selection_score(rows)["score"], (0.7 + 0.5) / 2)


class Calibration(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 2, 400)
        self.z = 6.0 * (self.y - 0.5) + rng.normal(0, 3, 400)  # overconfident-ish margins
        self.s = np.array(["s11"] * 200 + ["s13"] * 200)

    def test_temperature_preserves_ranking_and_finite(self):
        c = cal.fit("temperature", self.z, self.y, self.s)
        p = cal.apply(c, self.z)
        self.assertTrue(np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)))
        self.assertTrue(np.array_equal(np.argsort(p, kind="stable"), np.argsort(self.z, kind="stable")))
        self.assertTrue(0.05 <= c["params"]["T"] <= 20)

    def test_affine_positive_slope_and_bounds(self):
        c = cal.fit("affine", self.z, self.y, self.s)
        self.assertGreater(c["params"]["a"], 0)
        self.assertTrue(-5 <= c["params"]["b"] <= 5)
        # extreme margins stay finite
        self.assertTrue(np.all(np.isfinite(cal.apply(c, np.array([-1e4, 1e4])))))

    def test_identity_preferred_on_ties(self):
        # perfectly calibrated logits: all methods ~equal held-out NLL -> identity within tolerance? enforce tie rule
        r = cal.crossfit_choice({42: {"z": self.z, "y": self.y, "subjects": self.s}})
        self.assertEqual(set(r["ranking"]), set(cal.METHODS))
        best = min(r["mean_heldout_participant_nll"].values())
        within = [m for m in cal.METHODS if r["mean_heldout_participant_nll"][m] - best <= 1e-6]
        self.assertEqual(r["selected"], [m for m in cal.METHODS if m in within][0])

    def test_equal_participant_weighting(self):
        w = cal.participant_weights(np.array(["a"] * 10 + ["b"] * 30))
        self.assertAlmostEqual(w[:10].sum(), 0.5)
        self.assertAlmostEqual(w[10:].sum(), 0.5)


class Spec(unittest.TestCase):
    def test_folds_never_touch_outer_validation_or_test(self):
        st = load_study()
        banned = set(st["outer_split"]["validation"]) | set(st["outer_split"]["test"])
        for k, f in folds(st).items():
            self.assertFalse((set(f["train"]) | set(f["validation"])) & banned, k)
        heldout = sorted(p for f in st["inner_folds"] for p in f["validation"])
        self.assertEqual(heldout, sorted(st["outer_split"]["train"]))

    def test_identity_and_ids(self):
        st = load_study()
        r1 = cbramod_recipe(st, "temporal_mean_linear", "last2", 3e-5)
        r2 = cbramod_recipe(st, "temporal_mean_linear", "last2", 1e-5)
        self.assertNotEqual(recipe_id(r1), recipe_id(r2))
        self.assertEqual(recipe_id(r1), recipe_id(dict(r1)))
        f = fit_spec(st, r1, "cv0", 42, "prep-x")
        self.assertEqual(f["fit_id"], fit_spec(st, r1, "cv0", 42, "prep-x")["fit_id"])
        self.assertNotEqual(f["identity_hash"], fit_spec(st, r1, "cv0", 42, "prep-y")["identity_hash"])
        with self.assertRaises(AssertionError):
            fit_spec(st, r1, "outer", 42, "prep-x")  # final needs mode='final' + epochs
        g = fit_spec(st, eegnet_recipe(st, 1e-3, 0.5), "outer", 42, "prep-x", mode="final", epochs=120)
        self.assertTrue(g["fit_id"].endswith("__e120"))

    def test_frozen_has_no_backbone_lr(self):
        st = load_study()
        self.assertIsNone(cbramod_recipe(st, "historical_flatten_mlp", "frozen", 3e-5)["backbone_lr"])
        with self.assertRaises(ValueError):
            cbramod_recipe(st, "historical_flatten_mlp", "full", None)


class Choose(unittest.TestCase):
    def test_tie_rules(self):
        from study import choose
        t = [{"candidate_id": "b", "score": 0.6000005, "params_trainable": 10, "recipe": {}, "recipe_id": "rb"},
             {"candidate_id": "a", "score": 0.6, "params_trainable": 10, "recipe": {}, "recipe_id": "ra"},
             {"candidate_id": "c", "score": 0.6, "params_trainable": 5, "recipe": {}, "recipe_id": "rc"}]
        self.assertEqual(choose(t)["candidate_id"], "c")  # tie -> fewer params
        t[2]["params_trainable"] = 10
        self.assertEqual(choose(t)["candidate_id"], "a")  # tie -> lexicographic
        t[0]["score"] = 0.61
        self.assertEqual(choose(t)["candidate_id"], "b")


if __name__ == "__main__":
    unittest.main()
