#!/usr/bin/env python
"""ErrP-Bench converter: ERP CORE, ERN (flanker) task -- dataset_id `erpcore_ern`.

Raw input: the OSF "ERN Raw Data BIDS-Compatible" folder (osf.io/q6gwp), sub-001..sub-040, EEGLAB .set/.fdt
at 1024 Hz, Biosemi ActiveTwo, EEGReference = CMS (i.e. no usable reference in the file).

Event mapping (task-ERN_events.json in the BIDS package; response code = <response side><compatibility><target side>,
1 = left, 2 = right): correct = {111, 121, 212, 222}, incorrect = {112, 122, 211, 221}.
Trial inclusion follows the official ERP CORE bin descriptor ERN/EEG_ERP_Processing/BDF_ERN.txt
(github.com/lucklab/ERP_CORE): `{t<200-1000>11;12;21;22}.{response}` = the response must be the event immediately
following a stimulus (11/12/21/22) and occur 200-1000 ms after it, where stimulus codes are first shifted
+26 ms (pop_erplabShiftEventCodes, monitor delay; 1_Import_Raw_EEG_Shift_DS_Reref_Hpfilt.m).

Reference: re-referenced to the average of P9/P10 exactly as the official ERP CORE pipeline
(Rereference_Add_Uncorrected_Bipolars_ERN.txt), then P9, P10 and the 3 EOG channels are dropped -> 28 channels.

Usage: python erpcore_ern.py --raw <raw dir> --out <B>/data/prep/erpcore_ern
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import FS_OUT, check_montage, cut, filter_continuous, patches200, sha256_file, to_256, write_outputs  # noqa: E402

DATASET_ID = "erpcore_ern"
STIM = {11, 12, 21, 22}
CORRECT = {111, 121, 212, 222}
INCORRECT = {112, 122, 211, 221}
STIM_SHIFT_S = 0.026
RT_MIN_S, RT_MAX_S = 0.200, 1.000
REF = ["P9", "P10"]
RENAME = {"FP1": "Fp1", "FP2": "Fp2"}
TMIN_S, W = -0.2, 1
N0 = int(round(TMIN_S * FS_OUT))  # -51 samples at 256 Hz -> [-0.19922, 0.80078) s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="", help="comma list for quick tests, e.g. sub-001")
    args = ap.parse_args()
    raw_root, out = Path(args.raw), Path(args.out)
    subs = sorted(p.name for p in raw_root.glob("sub-*") if p.is_dir())
    if args.subjects:
        subs = [s for s in subs if s in args.subjects.split(",")]
    print("subjects:", len(subs), flush=True)

    sha_list = raw_root / "SHA256SUMS.txt"
    raw_files = []
    if sha_list.exists():
        for line in sha_list.read_text().splitlines():
            h, p = line.split(None, 1)
            raw_files.append({"path": p.strip(), "sha256": h})
    urls = {}
    if (raw_root / "osf_file_list.json").exists():
        urls = {o["path"]: o["url"] for o in json.load(open(raw_root / "osf_file_list.json"))}
    for r in raw_files:
        r["url"] = urls.get(r["path"])

    ch_out, rows, ep256, ep200, ids = None, [], [], [], []
    filt_info, per_sub_notes = None, {}
    for sub in subs:
        d = raw_root / sub / "eeg"
        setf = d / f"{sub}_task-ERN_eeg.set"
        raw = mne.io.read_raw_eeglab(setf, preload=True, verbose="ERROR")
        sf = raw.info["sfreq"]
        assert sf == 1024.0, sf
        chans = pd.read_csv(d / f"{sub}_task-ERN_channels.tsv", sep="\t")
        eeg = list(chans.loc[chans["type"] == "EEG", "name"])
        assert len(eeg) == 30 and set(eeg) <= set(raw.ch_names), (eeg, raw.ch_names)
        keep = [c for c in eeg if c not in REF]
        X = raw.get_data(picks=eeg, units="uV")
        idx = {c: i for i, c in enumerate(eeg)}
        refsig = X[[idx[r] for r in REF]].mean(0, keepdims=True)
        X = X[[idx[c] for c in keep]] - refsig
        names = [RENAME.get(c, c) for c in keep]
        if ch_out is None:
            ch_out = names
            check_montage(ch_out)
        assert names == ch_out, (sub, names)
        Xf, finfo = filter_continuous(X, sf)
        filt_info = filt_info or finfo
        X256 = to_256(Xf, sf)
        del X, Xf

        ev = pd.read_csv(d / f"{sub}_task-ERN_events.tsv", sep="\t")
        ev = ev.sort_values("onset", kind="stable").reset_index(drop=True)
        # cross-check BIDS onsets against the EEGLAB event latencies read by MNE
        ann = np.sort(raw.annotations.onset)
        j = np.clip(np.searchsorted(ann, ev["onset"].values), 1, len(ann) - 1)
        dmin = np.minimum(np.abs(ann[j] - ev["onset"].values), np.abs(ann[j - 1] - ev["onset"].values))
        assert dmin.max() < 2e-3, (sub, "events.tsv onsets do not match EEGLAB events", dmin.max())
        vals = ev["value"].astype(int).values
        ons = ev["onset"].astype(float).values
        n_resp = 0
        for k in range(len(ev)):
            v = int(vals[k])
            if v not in CORRECT | INCORRECT:
                continue
            n_resp += 1
            label = 1 if v in INCORRECT else 0
            prev = int(vals[k - 1]) if k > 0 else None
            rt = (ons[k] - (ons[k - 1] + STIM_SHIFT_S)) if prev in STIM else None
            reason = ""
            if prev not in STIM:
                reason = "response_not_immediately_after_stimulus_(official_BDF)"
            elif not (RT_MIN_S <= rt <= RT_MAX_S):
                reason = "rt_outside_official_200_1000ms_window_(official_BDF)"
            onset256 = int(round(ons[k] * FS_OUT))
            seg = None
            if not reason:
                seg, reason = cut(X256, onset256, N0, W)
            tid = f"{sub.replace('sub-', 'S')}_ERN_ev{k:04d}"
            rows.append({"subject_id": sub.replace("sub-", "S"), "trial_id": tid, "label": label,
                         "retained": int(not reason), "exclusion_reason": reason, "session": "1",
                         "source_file": setf.name, "event_code": str(v), "onset_s": float(ons[k]),
                         "event_index_in_file": k, "preceding_stimulus_code": prev if prev in STIM else None,
                         "rt_s_shifted_stim": rt,
                         "compatibility": "compatible" if (v // 10) % 10 == 1 else "incompatible",
                         "response_side": "left" if v // 100 == 1 else "right",
                         "target_side": "left" if v % 10 == 1 else "right"})
            if not reason:
                ep256.append(seg.astype(np.float32))
                ep200.append(patches200(seg, W))
                ids.append(tid)
        per_sub_notes[sub] = {"n_response_events": n_resp, "n_samples_1024": int(raw.n_times)}
        print(sub, "responses", n_resp, "kept", sum(r["retained"] for r in rows if r["subject_id"] == sub.replace("sub-", "S")),
              flush=True)

    cfg = {
        "preprocessing_id": DATASET_ID,
        "dataset": {
            "name": "ERP CORE -- ERN (Eriksen flanker task), raw BIDS data",
            "citation": "Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., & Luck, S. J. (2021). ERP CORE: "
                        "An open resource for human event-related potential research. NeuroImage, 225, 117465.",
            "doi": "10.1016/j.neuroimage.2020.117465", "resource_doi": "10.18115/D5JW4R",
            "url": "https://osf.io/thsqg (ERN component https://osf.io/q6gwp, folder 'ERN Raw Data BIDS-Compatible')",
            "license": "CC BY-SA 4.0 (LICENSE file in the BIDS package)",
            "raw_files": raw_files,
        },
        "paradigm": "Eriksen flanker task, participant's own erroneous vs correct button-press responses (ERN)",
        "event_locking": "time 0 = button-press response event (BIDS events.tsv onset)",
        "window_s": [N0 / FS_OUT, N0 / FS_OUT + W], "window_s_nominal": [TMIN_S, TMIN_S + W], "W": W,
        "event_mapping": {
            "source": "task-ERN_events.json (BIDS package) + official bin descriptor BDF_ERN.txt "
                      "(github.com/lucklab/ERP_CORE/ERN/EEG_ERP_Processing)",
            "label_1_incorrect": sorted(INCORRECT), "label_0_correct": sorted(CORRECT), "stimulus_codes": sorted(STIM),
            "inclusion": "response is the next event after a stimulus and RT in [200, 1000] ms after the stimulus "
                         "shifted +26 ms (official ERP CORE binning)",
        },
        "channels": ch_out, "channel_rename": RENAME,
        "dropped_channels": ["P9", "P10 (reference)", "HEOG_left", "HEOG_right", "VEOG_lower (EOG)"],
        "band_hz": list((filt_info["l_freq"], filt_info["h_freq"])),
        "filter": {**filt_info, "applied": "on the continuous 1024 Hz recording before epoching, after re-referencing"},
        "resampling": "continuous 1024 -> 256 Hz scipy.signal.resample_poly(up=1, down=4) after filtering; "
                      "200 Hz patches = resample_poly(25, 32) of each 256 Hz epoch",
        "reference": "average of P9/P10 (official ERP CORE reference; raw files are CMS-referenced Biosemi data, "
                     "EEGReference=CMS, which is not a usable reference)",
        "baseline": "none",
        "units": "microvolts",
        "exclusions": {"policy": "no amplitude rejection; responses outside the official ERP CORE bins (not "
                                 "immediately preceded by a stimulus, or RT outside 200-1000 ms) are kept in the "
                                 "manifest with retained=0; epochs outside the recording / non-finite dropped",
                       "subjects_excluded": []},
        "per_subject_notes": per_sub_notes,
    }
    readme = f"""# erpcore_ern -- ERP CORE ERN (flanker), all 40 participants

Response-locked epochs ([-51, 205) samples at 256 Hz = [-0.199, 0.801) s around the button press) from the raw
BIDS data of the ERP CORE ERN task (Kappenman et al., 2021, NeuroImage 225:117465; doi 10.18115/D5JW4R; OSF
osf.io/q6gwp). Label 1 = incorrect response (codes 112/122/211/221), 0 = correct response (111/121/212/222), from
task-ERN_events.json; trials restricted to the official ERP CORE bins (BDF_ERN.txt: response directly after the
stimulus, RT 200-1000 ms with the +26 ms stimulus shift). Re-referenced to mean(P9, P10) as in the official pipeline,
P9/P10/EOG dropped (28 ch), MNE zero-phase FIR 1-20 Hz on the continuous 1024 Hz data, resample_poly to 256 Hz,
200 Hz patches via resample_poly(25, 32). No amplitude rejection, no baseline correction. License CC BY-SA 4.0.

Regenerate (cluster):

    $B/env/bin/python bench/prep/erpcore_ern.py --raw $B/data/raw/erpcore_ern --out $B/data/prep/erpcore_ern
"""
    write_outputs(out, rows, ep256, ep200, ids, cfg, readme)


if __name__ == "__main__":
    main()
