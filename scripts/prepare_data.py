#!/usr/bin/env python
"""Data audit + epoching for ErrP-HRI (robot condition).

Outputs (to --out-dir):
  raw_files.csv                 every raw file with SHA-256, size, condition evidence, use decision
  trial_manifest.csv            one row per trial (retained and excluded) with labels/split placeholders
  epochs_256hz_uV.npy           (N, 27, 512) float32, retained trials, 1-20 Hz, microvolts
  epochs_200hz_patches_uV.npy   (N, 27, 2, 200) float32, same windows resampled to 200 Hz
  epoch_trial_ids.npy           trial_id per epoch row (same order in both arrays)
  preprocessing_config.json     resolved config + preprocessing_id + versions
  audit.json                    timing / label-consistency statistics used by reports/data_audit.md
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import scipy.io as sio  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from errp.utils import load_yaml, sha256_file, write_json  # noqa: E402

DIR = {"left": 1, "up": 2, "right": 3}


def condition_from_set(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["EEG"])
    E = m["EEG"]
    return str(E.comments), str(E.setname), int(E.srate), int(E.nbchan), str(getattr(E, "ref", ""))


def parse_trials(raw, ev_cfg):
    """Group the flat event list into trials, each starting at a stimulus marker."""
    desc = [d.strip() for d in raw.annotations.description]
    onset_samp = np.round(raw.annotations.onset * raw.info["sfreq"]).astype(int)
    norm = lambda s: s.replace(" ", "")
    stim = {norm(k): v for k, v in ev_cfg["stimulus"].items()}
    resp = {norm(k): v for k, v in ev_cfg["response"].items()}
    fbd = {norm(k): v for k, v in ev_cfg["feedback_direction"].items()}
    outc = {norm(k): v for k, v in ev_cfg["outcome"].items()}
    trials, cur = [], None
    for i, (d, s) in enumerate(zip(desc, onset_samp)):
        d = norm(d)
        if d in stim:
            if cur is not None:
                trials.append(cur)
            cur = {"events": [], "stim": stim[d], "stim_sample": s}
        if cur is None:
            continue
        cur["events"].append((i, d, s))
        if d in resp:
            cur.setdefault("resp", resp[d]); cur.setdefault("resp_sample", s)
        if d in fbd:
            cur.setdefault("fb", fbd[d]); cur.setdefault("fb_sample", s); cur.setdefault("fb_event_index", i)
            cur.setdefault("fb_code", d)
        if d in outc:
            cur.setdefault("outcome", outc[d]); cur.setdefault("outcome_sample", s); cur.setdefault("outcome_code", d)
        if d == "S7":
            cur.setdefault("s7_sample", s)
        if d == "S9":
            cur.setdefault("s9_sample", s)
    if cur is not None:
        trials.append(cur)
    return trials


def outcome_consistent(t) -> bool:
    """Check outcome markers against the direction markers.

    Evidence from the files (reports/data_audit.md): the FB marker always equals the key press
    (it records the commanded direction), so a machine error is NOT visible in the direction
    markers. What can be checked: S5 (human error) <=> response != stimulus, and S4/S6 require
    response == stimulus == FB. S6 is additionally validated by its per-block rate (~20% / ~50%,
    matching the documented pErr design)."""
    if t.get("resp") is None or t.get("fb") is None or t.get("outcome") is None:
        return False
    if t["fb"] != t["resp"]:
        return False
    if t["outcome"] == "human_error":
        return t["resp"] != t["stim"]
    return t["resp"] == t["stim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--config", default="configs/preprocessing.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root)
    pattern = cfg["dataset"]["condition_evidence_pattern"]

    # ---------------------------------------------------------------- raw file inventory
    inv = []
    for f in sorted(glob.glob(str(raw_root / "**" / "*.set"), recursive=True)):
        f = Path(f)
        comments, setname, srate, nbchan, ref = condition_from_set(f)
        cond = "robot" if pattern["robot"] in comments else ("cursor" if pattern["cursor"] in comments else "unknown")
        fdt = f.with_suffix(".fdt")
        for p in [f, fdt]:
            inv.append({"path": str(p.relative_to(raw_root)), "directory": p.parent.name, "file": p.name,
                        "bytes": p.stat().st_size if p.exists() else None,
                        "sha256": sha256_file(p) if p.exists() else None,
                        "subject_id": f.name.split("_")[0], "condition_by_name": f.stem.split("_")[1],
                        "condition_by_content": cond, "content_evidence": comments, "setname": setname,
                        "srate": srate, "nbchan": nbchan, "eeglab_ref_field": ref,
                        "used": bool(cond == "robot" and f.stem.endswith("_robot"))})
    for p in sorted(raw_root.glob("*.pdf")):
        inv.append({"path": p.name, "file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p), "used": False})
    inv_df = pd.DataFrame(inv)
    inv_df.to_csv(out / "raw_files.csv", index=False)
    robot_sets = sorted({r["path"] for r in inv if r.get("used") and r["path"].endswith(".set")})
    print("robot files:", robot_sets, flush=True)

    ch_order = cfg["channels"]["order"]
    t0, t1 = cfg["epoch"]["tmin_s"], cfg["epoch"]["tmax_s"]
    rows, ep256, ep200, ids = [], [], [], []
    audit = {"per_file": {}}
    for rel in robot_sets:
        f = raw_root / rel
        subj = f.name.split("_")[0]
        raw = mne.io.read_raw_eeglab(f, preload=True, verbose="ERROR")
        sf = raw.info["sfreq"]
        assert sf == 256.0, sf
        assert set(cfg["channels"]["drop"]) <= set(raw.ch_names)
        assert set(ch_order) | set(cfg["channels"]["drop"]) == set(raw.ch_names), raw.ch_names
        trials = parse_trials(raw, cfg["events"])
        # --- timing / consistency audit
        fa = collections.defaultdict(list)
        n_mismatch = 0
        for t in trials:
            if "fb_sample" in t and "resp_sample" in t:
                fa[f"resp_to_fb_samples_{t['fb']}"].append(int(t["fb_sample"] - t["resp_sample"]))
            if "outcome_sample" in t and "resp_sample" in t:
                fa["resp_to_outcome_samples"].append(int(t["outcome_sample"] - t["resp_sample"]))
            if "outcome_sample" in t and "fb_sample" in t:
                fa["fb_to_outcome_samples"].append(int(t["outcome_sample"] - t["fb_sample"]))
            if "s7_sample" in t and "fb_sample" in t:
                fa["fb_to_s7_samples"].append(int(t["s7_sample"] - t["fb_sample"]))
            if "fb_sample" in t:
                fa["fb_to_next_stimulus_samples"]  # filled below
            if not outcome_consistent(t):
                n_mismatch += 1
        for a, b in zip(trials[:-1], trials[1:]):
            if "fb_sample" in a:
                fa["fb_to_next_stimulus_samples"].append(int(b["stim_sample"] - a["fb_sample"]))
        audit["per_file"][rel] = {
            "n_trials": len(trials), "label_rule_mismatches": n_mismatch,
            "timing_samples_256Hz": {k: {"min": int(np.min(v)), "median": float(np.median(v)), "max": int(np.max(v)),
                                         "n": len(v)} for k, v in fa.items() if v},
            "n_samples": int(raw.n_times), "duration_s": raw.n_times / sf,
        }
        # --- preprocessing on the continuous recording
        raw.pick(ch_order)
        raw.reorder_channels(ch_order)
        raw.filter(cfg["filter"]["l_freq"], cfg["filter"]["h_freq"], method="fir", phase="zero",
                   fir_window="hamming", fir_design="firwin", verbose="ERROR")
        X = raw.get_data(units="uV")  # (27, n_times)
        n0, n1 = int(round(t0 * sf)), int(round(t1 * sf))
        for k, t in enumerate(trials):
            outcome = t.get("outcome")
            row = {"subject_id": subj, "condition": "robot", "source_file": f.name, "trial_index_in_file": k,
                   "block_index": k // 50, "stimulus": t.get("stim"), "response": t.get("resp"),
                   "feedback_direction": t.get("fb"), "fb_event_code": t.get("fb_code"),
                   "fb_event_index": t.get("fb_event_index"), "fb_sample": t.get("fb_sample"),
                   "fb_time_s": (t["fb_sample"] / sf) if "fb_sample" in t else None,
                   "outcome_event_code": t.get("outcome_code"), "outcome": outcome,
                   "outcome_consistent_with_directions": outcome_consistent(t)}
            reason = ""
            if "fb_sample" not in t or outcome is None:
                reason = "incomplete_trial_events"
            elif not row["outcome_consistent_with_directions"]:
                reason = "outcome_marker_inconsistent_with_directions"
            elif outcome == "human_error":
                reason = "human_error_trial_S5_excluded_by_label_policy"
            elif t["fb_sample"] + n1 > X.shape[1] or t["fb_sample"] + n0 < 0:
                reason = "epoch_window_outside_recording"
            tid = f"{subj}_robot_{f.stem}_ev{int(t['fb_event_index']) if 'fb_event_index' in t else -1:04d}"
            row["trial_id"] = tid
            if not reason:
                seg = X[:, t["fb_sample"] + n0: t["fb_sample"] + n1]
                if not np.all(np.isfinite(seg)):
                    reason = "nonfinite_samples"
            row["label"] = cfg["events"]["label_map"].get(outcome) if not reason else None
            row["retained"] = int(not reason)
            row["exclusion_reason"] = reason
            if not reason:
                row["peak_to_peak_max_uV"] = float(np.ptp(seg, axis=1).max())
                ep256.append(seg.astype(np.float32))
                seg200 = resample_poly(seg.astype(np.float64), up=25, down=32, axis=1)
                assert seg200.shape[1] == 400
                ep200.append(seg200.reshape(len(ch_order), 2, 200).astype(np.float32))
                ids.append(tid)
            rows.append(row)
        print(subj, audit["per_file"][rel]["n_trials"], "mismatch", n_mismatch, flush=True)

    man = pd.DataFrame(rows)
    assert man["trial_id"].is_unique
    man["split"] = "unassigned"  # filled by scripts/make_splits.py
    np.save(out / "epochs_256hz_uV.npy", np.stack(ep256))
    np.save(out / "epochs_200hz_patches_uV.npy", np.stack(ep200))
    np.save(out / "epoch_trial_ids.npy", np.array(ids))
    man.to_csv(out / "trial_manifest.csv", index=False)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    resolved = {"preprocessing_id": f"{cfg['version']}-{cfg_hash}", "config": cfg,
                "versions": {"mne": mne.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
                "n_epochs": len(ids), "shapes": {"eeg256": list(np.stack(ep256[:1]).shape[1:]),
                                                 "patch200": list(np.stack(ep200[:1]).shape[1:])}}
    write_json(resolved, out / "preprocessing_config.json")
    # summary counts
    summ = (man.groupby(["subject_id", "outcome", "exclusion_reason"], dropna=False).size()
            .rename("n").reset_index())
    audit["counts"] = summ.to_dict(orient="records")
    write_json(audit, out / "audit.json")
    print(man.groupby(["subject_id", "retained", "label"], dropna=False).size().unstack([1, 2]))
    print("PREP_DONE", resolved["preprocessing_id"], len(ids))


if __name__ == "__main__":
    main()
