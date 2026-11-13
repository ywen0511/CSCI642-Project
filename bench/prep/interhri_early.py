#!/usr/bin/env python
"""IntEr-HRI early-window control (dataset_id interhri_early): the first 0.5 s (128 samples at 256 Hz) of the prepared
IntEr-HRI epochs, before almost all ball-squeeze responses (S 80). Same trials, labels and manifest.

  python bench/prep/interhri_early.py --src $ERRP_ROOT/data/prep/interhri --out $ERRP_ROOT/data/prep/interhri_early
Only the 256 Hz view is written (the control is run with xDAWN+RG and EEGNet).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=128)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(src / "epochs_256hz_uV.npy", mmap_mode="r")
    np.save(out / "epochs_256hz_uV.npy", np.ascontiguousarray(X[:, :, :a.samples]))
    for f in ("epoch_trial_ids.npy", "trial_manifest.csv"):
        shutil.copy2(src / f, out / f)
    cfg = json.loads((src / "preprocessing_config.json").read_text())
    cfg.update(preprocessing_id="interhri_early", window_s=[0.0, a.samples / 256.0],
               shapes={"epochs_256hz_uV": [int(X.shape[0]), int(X.shape[1]), a.samples]},
               derived_from=f"interhri: first {a.samples} samples (0-{a.samples / 256:.1f} s) of the 256 Hz epochs; "
                            "control for the ball-squeeze response")
    (out / "preprocessing_config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print("wrote", out, X.shape[0], "epochs x", a.samples, "samples")


if __name__ == "__main__":
    main()
