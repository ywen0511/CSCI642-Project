#!/usr/bin/env python
"""IntEr-HRI (Kueper et al. 2024, Zenodo 8345429) -> ErrP-Bench format (dataset_id interhri).

  python bench/prep/interhri.py --raw $B/data/raw/interhri --out $B/data/prep/interhri

--raw holds EEG.zip (+ Metadata.zip, zenodo_record_8345429.json); EEG.zip is extracted to <raw>/EEG_extracted once.

Events (Kueper et al. 2024, Sec. 2.4.3 'Recorded events', Fig. 1C): S 96 = error introduced (orthosis reverses
direction for 250 ms) -> label 1; S 48 = 'correct event', placed in error-free trials around the mean error position
-> label 0 (the authors' own matched non-error event).  S 64 / S 32 = flexion / extension start, S 80 = ball squeeze
(subject's response), S 1 = EMG sync: unused.  Epoch [0, 1) s after the S96/S48 marker.
Runs used: every error run (`*_multi_setN`) of every subject: the 10 runs in <subj>/data/ plus the 3 runs in
<subj>/data/additional sets/ (AQ59D set12, AY63D set11, BS34D set9: extra runs recorded after Bluetooth sample loss,
which was later recovered from the SD card; flagged additional_set=1).  Impedance files (imp/) unused.  Baseline runs (no errors introduced) are a
different condition and excluded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from common import filter_continuous, sha256_file, write_prepared  # noqa: E402

ZENODO = "https://zenodo.org/records/8345429"
EV = {"S 96": 1, "S 48": 0}
NON_EEG = {"X_DIR", "Y_DIR", "Z_DIR"}  # LiveAmp accelerometer channels (unit mg); the other 64 are scalp EEG


def norm_desc(d: str) -> str:
    d = d.split("/")[-1]
    return re.sub(r"\s+", " ", d.strip())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    raw_root = Path(args.raw)
    rec = json.load(open(raw_root / "zenodo_record_8345429.json"))
    md5 = {f["key"]: f["checksum"] for f in rec["files"]}
    zips = []
    for k in ("EEG.zip", "Metadata.zip"):
        p = raw_root / k
        zips.append({"file": k, "bytes": p.stat().st_size, "sha256": sha256_file(p), "zenodo_md5": md5[k],
                     "url": f"https://zenodo.org/api/records/8345429/files/{k}/content"})
    ex = raw_root / "EEG_extracted"
    if not (ex / ".done").exists():
        with zipfile.ZipFile(raw_root / "EEG.zip") as z:
            z.extractall(ex)
        (ex / ".done").write_text("ok")
    vhdrs = sorted(ex.rglob("*.vhdr"))
    runs = [v for v in vhdrs if re.search(r"_multi_set\d+\.vhdr$", v.name)]
    skipped = [str(v.relative_to(ex)) for v in vhdrs if v not in runs]
    print(len(vhdrs), "vhdr;", len(runs), "error runs; skipped:", skipped, flush=True)

    mont_list = mne.channels.make_standard_montage("standard_1020").ch_names
    mont = set(mont_list)
    mont_ci = {c.lower(): c for c in mont}
    rows, ep256, ids = [], [], []
    ch_final, rename_map, ref_info, filt, per_run = None, {}, None, None, {}
    for v in runs:
        subj = re.search(r"_([A-Z]{2}\d{2}D)_", v.name).group(1)
        setn = int(re.search(r"_set(\d+)\.vhdr$", v.name).group(1))
        raw = mne.io.read_raw_brainvision(v, preload=True, verbose="ERROR")
        sf = raw.info["sfreq"]
        assert sf == 500.0, sf
        # --- channels: scalp EEG only, names mapped onto standard_1020
        eeg = [c for c in raw.ch_names if raw.get_channel_types([c])[0] == "eeg" and c.upper() not in NON_EEG]
        ren = {c: mont_ci[c.lower()] for c in eeg if c.lower() in mont_ci and mont_ci[c.lower()] != c}
        bad = [c for c in eeg if c.lower() not in mont_ci]
        assert not bad, f"{v.name}: channels not in standard_1020: {bad}"
        assert len(eeg) == 64, (v.name, len(eeg))
        raw.pick(eeg)
        raw.rename_channels(ren)
        rename_map.update(ren)
        # --- reference: recorded against FCz (not stored); restore FCz as zeros and re-reference to the average
        assert "FCz" not in raw.ch_names
        raw = mne.add_reference_channels(raw, "FCz", copy=False)
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        order = sorted(raw.ch_names, key=mont_list.index)
        raw.reorder_channels(order)
        if ch_final is None:
            ch_final = order
        assert order == ch_final, (v.name, order)
        # --- events
        desc = [norm_desc(d) for d in raw.annotations.description]
        on = raw.annotations.onset
        segs = [o for d, o in zip(desc, on) if "New Segment" in d or d == "Boundary"]
        counts = pd.Series(desc).value_counts().to_dict()
        per_run[v.name] = {"marker_counts": counts, "n_new_segment": len(segs), "duration_s": raw.n_times / sf}
        # --- filter at native rate, then polyphase resample 500 -> 256 Hz (64/125), continuous
        filt = filter_continuous(raw, 1.0, 20.0)
        X = resample_poly(raw.get_data(units="uV"), up=64, down=125, axis=1)
        for k, (d, o) in enumerate(zip(desc, on)):
            if d not in EV:
                continue
            s = int(round(o * 256.0))
            tid = f"{subj}_set{setn:02d}_ev{k:04d}"
            row = {"subject_id": subj, "trial_id": tid, "label": EV[d], "session": f"set{setn:02d}",
                   "additional_set": int("additional sets" in str(v)),
                   "source_file": str(v.relative_to(ex)), "event_code": d, "onset_s": float(o)}
            # subject's ball-squeeze response (S 80) following an error, if within 1.5 s (behavioural info only)
            nxt = [oo for dd, oo in zip(desc[k + 1:], on[k + 1:]) if dd == "S 80" and oo - o < 1.5] if d == "S 96" else []
            row["response_S80_latency_s"] = float(nxt[0] - o) if nxt else float("nan")
            reason = ""
            if s < 0 or s + 256 > X.shape[1]:
                reason = "epoch_window_outside_recording"
            elif any(o < b < o + 1.0 for b in segs):
                reason = "recording_discontinuity_in_window"
            else:
                seg = X[:, s:s + 256]
                if not np.isfinite(seg).all():
                    reason = "nonfinite_samples"
            row["retained"] = int(not reason)
            row["exclusion_reason"] = reason
            if not reason:
                ep256.append(seg.astype(np.float32))
                ids.append(tid)
            rows.append(row)
        print(subj, setn, {e: counts.get(e, 0) for e in ("S 96", "S 48", "S 64", "S 32", "S 80", "S 1")},
              "segments", len(segs), flush=True)
    man = pd.DataFrame(rows)
    meta = {
        "preprocessing_id": "interhri",
        "dataset": {"name": "IntEr-HRI: EEG and EMG dataset for the detection of errors introduced by an active "
                            "orthosis device (IJCAI'23 CC6 competition), complete version",
                    "citation": "Kueper, N., Chari, K., Buetefuer, J., Habenicht, J., Rossol, T., Kim, S. K., Tabie, M., "
                                "Kirchner, F., & Kirchner, E. A. (2024). Front. Hum. Neurosci. 18:1304311.",
                    "doi": "10.3389/fnhum.2024.1304311", "zenodo": ZENODO, "license": "CC-BY-4.0 (Zenodo record)",
                    "raw_archives": zips,
                    "raw_files": [{"path": str(p.relative_to(ex)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                                  for v in runs for p in (v, v.with_suffix(".vmrk"), v.with_suffix(".eeg"))]},
        "paradigm": "active orthosis moves the right elbow (flexion/extension, 30 trials per run); in 6 random trials "
                    "per run the orthosis reverses direction for 250 ms (introduced error); subjects squeeze a ball "
                    "with the left hand when they notice an error. 10 error runs per subject.",
        "event_locking": "time 0 = S 96 (error introduced) or S 48 (matched correct event in an error-free trial, "
                         "around the mean error position)",
        "event_mapping": {"label_1": "S 96", "label_0": "S 48",
                          "unused": {"S 64": "flexion start", "S 32": "extension start", "S 80": "ball squeeze",
                                     "S 1": "EMG sync"},
                          "source": "Kueper et al. 2024 Sec. 2.4.3 'Recorded events' and Fig. 1C ('correct event (S48) "
                                    "and incorrect event (S96)')"},
        "window_s": [0.0, 1.0], "W": 1, "sfreq": {"native": 500, "eeg256": 256, "patch200": 200},
        "resample": "filtered continuous 500 Hz -> 256 Hz with scipy.signal.resample_poly(up=64, down=125); event "
                    "sample = round(onset_s * 256)",
        "channels": ch_final, "channel_rename": rename_map,
        "filter": filt,
        "reference": "recorded against FCz (online reference, not stored; GND AFz). FCz restored as a zero channel "
                     "(mne.add_reference_channels) and the data re-referenced to the common average of all 64 scalp "
                     "channels, as in the authors' ERP analysis (Sec. 3.2); this makes FCz, the ErrP maximum, available.",
        "units": "microvolt",
        "exclusions": {"policy": "no amplitude rejection; only incomplete / non-finite / discontinuous windows dropped",
                       "runs_not_used": skipped,
                       "runs_not_used_reason": "baseline runs (no introduced errors) are a different condition; only "
                                               "the multimodal error runs (*_multi_setN) are used, including the 3 "
                                               "'additional sets' (flag additional_set=1 in the manifest)",
                       "behaviour": "label = introduced error regardless of whether the subject noticed it (ball "
                                    "squeeze S80 not used for labelling)"},
        "confound_note": "error epochs contain the subject's left-hand ball squeeze (S 80) inside [0, 1) s in ~90% of "
                         "cases (median 0.72 s after S 96, manifest column response_S80_latency_s); correct (S 48) "
                         "epochs have no response. Late-window decoding may partly reflect motor/EMG activity.",
        "per_run_audit": per_run,
    }
    readme = """# interhri

IntEr-HRI (Kueper et al. 2024; Zenodo 8345429, CC-BY-4.0), ErrP-Bench format (bench/DATA_FORMAT.md).
8 subjects x 10 error runs. Label 1 = S 96 (orthosis error introduced), 0 = S 48 (authors' matched correct event
in error-free trials). Epoch [0, 1) s after the marker. 64 scalp channels (63 recorded + restored FCz),
common-average reference, MNE FIR 1-20 Hz at 500 Hz on the continuous run, resample_poly 500->256 Hz, 200 Hz
patches, microvolts. Baseline runs excluded.

Regenerate:

    $B/env/bin/python bench/prep/interhri.py --raw $B/data/raw/interhri --out $B/data/prep/interhri
"""
    write_prepared(Path(args.out), man, ep256, ids, 1, meta, readme)


if __name__ == "__main__":
    main()
