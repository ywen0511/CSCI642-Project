#!/usr/bin/env python
"""Compute the per-decoder reward centers c_m of protocol v3 (FINAL_V3_HANDOFF.md, Section 6.1).

c_m = 0.5 * (mean calibrated p_error over label-0 trials + mean over label-1 trials), pooled over the
VALIDATION participants (s11, s13) of the frozen feedback table, decoder seed 42. Test participants are never
read: the validation table is loaded through rl.feedback.FeedbackLibrary, which checks its size and SHA-256
against configs/rl_feedback.json and rejects any row outside the validation split.

Usage: python scripts/final_v3_compute_centers.py --feedback-dir artifacts/rl_feedback --out configs/final_v3/centers.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.feedback import PROFILE, FeedbackLibrary, sha256  # noqa: E402

SCHEMA = "csci642-rl-centers/v1"


def compute(feedback_dir):
    lib = FeedbackLibrary(feedback_dir, "validation")
    subjects = lib.profile["splits"]["validation"]
    out = {"schema": SCHEMA,
           "definition": "c_m = 0.5 * (mean p_error | label 0 + mean p_error | label 1), calibrated p_error, "
                         "pooled over the validation participants, decoder seed "
                         f"{lib.profile['decoder_seed']}; ideal keeps p = label with center 0.5",
           "split": "validation", "participants": subjects, "decoder_seed": lib.profile["decoder_seed"],
           "feedback_table": lib.profile["tables"]["validation"]["asset"],
           "feedback_table_sha256": lib.table_sha256, "feedback_profile_sha256": sha256(PROFILE),
           "centers": {}, "per_model": {}}
    for model, rows in sorted(lib.rows.items()):
        by_label = {lab: [r["p_error"] for r in rows.values() if r["label"] == lab] for lab in (0, 1)}
        mean0, mean1 = statistics.fmean(by_label[0]), statistics.fmean(by_label[1])
        center = 0.5 * (mean0 + mean1)
        per_subject = {}
        for s in subjects:
            pool = [r["p_error"] for r in rows.values() if r["subject_id"] == s]
            per_subject[s] = {"n": len(pool), "mean_p_error": statistics.fmean(pool),
                              "error_label_fraction": statistics.fmean(
                                  r["label"] for r in rows.values() if r["subject_id"] == s),
                              # label-independent control samples uniformly from this pool
                              "independent_expected_reward_per_unit_weight": center - statistics.fmean(pool)}
        out["centers"][model] = center
        out["per_model"][model] = {"n_label0": len(by_label[0]), "n_label1": len(by_label[1]),
                                   "mean_p_error_label0": mean0, "mean_p_error_label1": mean1,
                                   "center": center, "per_participant": per_subject}
    out["centers"]["ideal"] = 0.5
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feedback-dir", default="artifacts/rl_feedback")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = compute(a.feedback_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"centers": result["centers"], "sha256": sha256(out)}, indent=2))
