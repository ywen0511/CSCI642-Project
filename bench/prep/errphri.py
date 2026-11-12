#!/usr/bin/env python
"""ErrP-HRI (Ehrlich & Cheng 2019) -> ErrP-Bench format, one condition per run.

Reuses the event parsing / label-consistency logic of scripts/prepare_data.py (prep_v1) unchanged; only the
condition, the window (from the yaml) and the output format differ.  Wrapper: errphri_cursor.py (robot: use prep_v1).

  python bench/prep/errphri.py --config bench/prep/errphri_cursor.yaml --raw <dataset-ErrP-HRI> --out <prep dir>
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from common import filter_continuous, sha256_file, write_prepared  # noqa: E402
from prepare_data import condition_from_set, outcome_consistent, parse_trials  # noqa: E402

REPO_URL = "https://github.com/stefan-ehrlich/dataset-ErrP-HRI"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    cond = cfg["dataset"]["condition"]
    pat = cfg["dataset"]["condition_evidence_pattern"]
    raw_root = Path(args.raw)
    commit = cfg["dataset"]["commit"]

    used, inventory = [], []
    for f in sorted(glob.glob(str(raw_root / "**" / "*.set"), recursive=True)):
        f = Path(f)
        comments, *_ = condition_from_set(f)
        c = "robot" if pat["robot"] in comments else ("cursor" if pat["cursor"] in comments else "unknown")
        if c == cond:
            used.append(f)
            for p in (f, f.with_suffix(".fdt")):
                rel = str(p.relative_to(raw_root))
                inventory.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                                  "url": f"{REPO_URL}/raw/{commit}/{rel}", "content_evidence": comments})
    subjects = [f.name.split("_")[0] for f in used]
    assert len(set(subjects)) == len(subjects), subjects
    print(cond, "files:", [str(f.relative_to(raw_root)) for f in used], flush=True)

    ch_order = cfg["channels"]["order"]
    t0, t1 = cfg["epoch"]["tmin_s"], cfg["epoch"]["tmax_s"]
    W = int(round(t1 - t0))
    rows, ep256, ids, filt = [], [], [], None
    timing = {}
    for f in used:
        subj = f.name.split("_")[0]
        raw = mne.io.read_raw_eeglab(f, preload=True, verbose="ERROR")
        sf = raw.info["sfreq"]
        assert sf == 256.0, sf
        assert set(ch_order) | set(cfg["channels"]["drop"]) == set(raw.ch_names), raw.ch_names
        ref = str(getattr(sio_ref(f), "ref", ""))
        trials = parse_trials(raw, cfg["events"])
        d_s7 = [t["s7_sample"] - t["fb_sample"] for t in trials if "s7_sample" in t and "fb_sample" in t]
        d_rsp = [t["fb_sample"] - t["resp_sample"] for t in trials if "resp_sample" in t and "fb_sample" in t]
        timing[subj] = {"fb_to_S7_ms_median": float(np.median(d_s7) / sf * 1000) if d_s7 else None,
                        "resp_to_fb_ms_median": float(np.median(d_rsp) / sf * 1000) if d_rsp else None,
                        "eeglab_ref": ref}
        raw.pick(ch_order)
        raw.reorder_channels(ch_order)
        filt = filter_continuous(raw, cfg["filter"]["l_freq"], cfg["filter"]["h_freq"])
        X = raw.get_data(units="uV")
        n0, n1 = int(round(t0 * sf)), int(round(t1 * sf))
        for k, t in enumerate(trials):
            outcome = t.get("outcome")
            fb = t.get("fb_sample")
            tid = f"{subj}_{cond}_{f.stem}_ev{int(t['fb_event_index']) if 'fb_event_index' in t else -1:04d}"
            row = {"subject_id": subj, "trial_id": tid, "session": cond, "source_file": str(f.relative_to(raw_root)),
                   "event_code": t.get("fb_code"), "onset_s": (fb / sf) if fb is not None else None,
                   "outcome_event_code": t.get("outcome_code"), "outcome": outcome,
                   "trial_index_in_file": k, "block_index": k // 50, "stimulus": t.get("stim"),
                   "response": t.get("resp"), "feedback_direction": t.get("fb"),
                   "fb_event_index": t.get("fb_event_index"), "fb_sample": fb}
            reason = ""
            if fb is None or outcome is None:
                reason = "incomplete_trial_events"
            elif not outcome_consistent(t):
                reason = "outcome_marker_inconsistent_with_directions"
            elif outcome == "human_error":
                reason = "human_error_trial_S5_excluded_by_label_policy"
            elif fb + n1 > X.shape[1] or fb + n0 < 0:
                reason = "epoch_window_outside_recording"
            if not reason:
                seg = X[:, fb + n0: fb + n1]
                if not np.all(np.isfinite(seg)):
                    reason = "nonfinite_samples"
            row["label"] = int(cfg["events"]["label_map"][outcome]) if not reason else -1
            row["retained"] = int(not reason)
            row["exclusion_reason"] = reason
            if not reason:
                ep256.append(seg.astype(np.float32))
                ids.append(tid)
            rows.append(row)
        print(subj, len(trials), "trials", timing[subj], flush=True)

    man = pd.DataFrame(rows)
    did = cfg["dataset_id"]
    meta = {
        "preprocessing_id": did,
        "dataset": {"name": f"ErrP-HRI ({cond} condition)",
                    "citation": "Ehrlich, S. K., & Cheng, G. (2019). A feasibility study for validating robot actions "
                                "using EEG-based error-related potentials. Int. J. Social Robotics 11(2), 271-283.",
                    "doi": "10.1007/s12369-018-0501-8", "url": REPO_URL, "commit": commit,
                    "license": "not stated in the repository (no LICENSE file); public research dataset, cite the paper",
                    "raw_files": inventory},
        "paradigm": ("cursor scenario: participant presses arrow key towards a target, an on-screen square moves; "
                     if cond == "cursor" else
                     "robot scenario: participant presses arrow key towards a target, NAO robot turns its head; ")
                    + "the machine executes the opposite/wrong direction with pErr 20% or 50% per block (S6).",
        "event_locking": "time 0 = FB1/FB2/FB3 marker (onset of the feedback movement; documentation p. 3)",
        "event_mapping": {"label_1": "S6 'Feedback machine error'", "label_0": "S4 'Feedback no error'",
                          "excluded": "S5 'Feedback human error' (participant pressed the wrong key)",
                          "source": "documentation_dataset-ErrP-HRI.pdf p. 3 (Trigger/Marker information)",
                          "markers": cfg["events"]},
        "window_s": [t0, t1], "W": W, "sfreq": {"eeg256": 256, "patch200": 200},
        "channels": ch_order, "channel_rename": {}, "dropped_channels": cfg["channels"]["drop"],
        "filter": filt, "reference": "average of mastoids TP9/TP10 (dataset authors); kept, no re-referencing",
        "units": "microvolt",
        "exclusions": {"policy": "no amplitude rejection; human-error trials (S5) excluded by label policy; "
                                 "incomplete / inconsistent-marker / out-of-recording / non-finite epochs dropped",
                       "files_by_content": "condition decided from EEGLAB comments field; robot/s12_cursor.set is the "
                                           "s12 cursor recording stored in robot/ (used for cursor, not robot)"},
        "timing_audit": timing,
        "trial_id_format": "{subject_id}_{condition}_{file_stem}_ev{fb_event_index:04d} (same as prep_v1)",
        "config": cfg,
    }
    readme = f"""# {did}

ErrP-HRI (Ehrlich & Cheng 2019), **{cond}** condition, ErrP-Bench format (bench/DATA_FORMAT.md).
Label 1 = S6 machine error, 0 = S4 no error, S5 human error excluded. Epoch [{t0}, {t1}) s after the
FB1/FB2/FB3 feedback-onset marker; 27 scalp channels (mastoid reference), MNE FIR 1-20 Hz on the continuous
recording, 256 Hz (native) + 200 Hz patches, microvolts. Event parsing reused from scripts/prepare_data.py.

Regenerate:

    $B/env/bin/python bench/prep/errphri_{cond}.py --raw <dataset-ErrP-HRI @ {commit[:7]}> --out $B/data/prep/{did}
"""
    write_prepared(Path(args.out), man, ep256, ids, W, meta, readme)


def sio_ref(f):
    import scipy.io as sio
    return sio.loadmat(f, squeeze_me=True, struct_as_record=False, variable_names=["EEG"])["EEG"]


if __name__ == "__main__":
    main()
