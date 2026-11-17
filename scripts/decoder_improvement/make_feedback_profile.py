#!/usr/bin/env python
"""Write configs/decoder_improvement/rl_feedback_vnext.json for the new seed-42 selected-candidate probabilities.

Same schema as configs/rl_feedback.json (which stays unchanged), so rl.feedback.FeedbackLibrary(..., profile_path=...)
and scripts/download_feedback.py --profile accept it explicitly.

Usage: python scripts/decoder_improvement/make_feedback_profile.py --pred-dir DIR --freeze FREEZE --calibration CAL \
          --release-tag TAG --out configs/decoder_improvement/rl_feedback_vnext.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--release-tag", required=True)
    ap.add_argument("--repo", default="ywen0511/CSCI642-Project")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.pred_dir)
    freeze = json.loads(Path(a.freeze).read_text())
    cal = json.loads(Path(a.calibration).read_text())
    prof = {"schema": "csci642-rl-feedback/v1", "profile_version": "vnext-decoder-improvement-v1",
            "status": "experimental follow-up; NOT the historical feedback (configs/rl_feedback.json is unchanged); "
                      "no RL experiment has used these probabilities",
            "release_url": f"https://github.com/{a.repo}/releases/tag/{a.release_tag}", "release_tag": a.release_tag,
            "decoder_seed": 42, "decoder_seed_rule": "seed 42 designated in advance for each selected candidate",
            "label_mapping": {"0": "non-error", "1": "machine error"}, "probability_column": "p_error",
            "splits": {"train": ["s03", "s04", "s05", "s06", "s07", "s10", "s12"],
                       "validation": ["s11", "s13"], "test": ["s02", "s08", "s09"]},
            "tables": {}, "models": {},
            "calibration_reuse": "validation (s11, s13) probabilities come from calibrators fitted on those same "
                                 "participants: calibration-development outputs, not unbiased held-out feedback quality; "
                                 "a future RL tuning study on validation feedback must account for this reuse",
            "freeze_sha256": sha(a.freeze), "calibration_sha256": sha(a.calibration)}
    for split in ("validation", "test"):
        f = d / f"vnext_{split}_predictions.csv"
        df = pd.read_csv(f, dtype={"subject_id": str, "trial_id": str})
        per_model = df.groupby("model").size()
        assert per_model.nunique() == 1
        prof["tables"][split] = {"asset": f.name, "sha256": sha(f), "bytes": f.stat().st_size,
                                 "trials_per_model": int(per_model.iloc[0])}
        for model, g in df.groupby("model"):
            ids = {k: g[k].unique().tolist() for k in ("checkpoint_id", "preprocessing_id", "calibration_id", "recipe_id")}
            assert all(len(v) == 1 for v in ids.values()), ids
            fid = ids["checkpoint_id"][0].rsplit(":", 1)[0]
            dep = cal["deployment"][model]
            prof["models"].setdefault(model, {**{k: v[0] for k, v in ids.items()},
                                              "calibration": {"method": dep["method"], **dep["per_model"][fid]}})
    prof["manifest"] = {"asset": "predictions_manifest.json", "sha256": sha(d / "predictions_manifest.json"),
                        "bytes": (d / "predictions_manifest.json").stat().st_size}
    Path(a.out).write_text(json.dumps(prof, indent=2) + "\n")
    print(json.dumps({"models": list(prof["models"]), "tables": prof["tables"]}, indent=1))


if __name__ == "__main__":
    main()
