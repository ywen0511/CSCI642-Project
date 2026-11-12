#!/usr/bin/env python
"""ErrP-Bench converter: BNCI Horizon 2020 dataset 013-2015 "Monitoring error-related potentials"
(Chavarriaga & Millan 2010, IEEE TNSRE 18:381-388, doi 10.1109/TNSRE.2010.2053387) -- dataset_id `bnci2015_013`.

Raw input: Subject0{1..6}_s{1,2}.mat + description.pdf from https://bnci-horizon-2020.eu/database/data-sets/013-2015/
Each .mat holds `run` (10 runs); run{i}.eeg = (n_samples, 65) [64 EEG + 'Status' trigger], run{i}.header with
SampleRate 512, Label, EVENT.TYP / EVENT.POS (positions in samples, MATLAB 1-based).

Event mapping (description.pdf, "Triggers were coded bitwise ... bit0 target left, bit1 target right, bit2 cursor
movement left, bit3 cursor movement right ... events '5' and '10' mark correct movements, while erroneous movements
are marked with events '6' and '9'"): label 0 = {5, 10}, label 1 = {6, 9}. Other codes (1/2 = target only, 4/8 =
cursor movement with no target bit set, undocumented as correct/error) are NOT epoched (recorded as counts).

Reference: the files are Biosemi CMS-referenced (raw, multi-mV DC offsets); re-referenced to the common average of
the 64 EEG channels, the spatial filter used by the dataset authors' protocol (Chavarriaga et al., IDIAP-RR 07-37 and
EMBC 2010: CAR + band-pass). Each run is a separate continuous recording and is filtered on its own.

Usage: python bnci2015_013.py --raw <raw dir> --out <B>/data/prep/bnci2015_013
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

from _common import FS_OUT, check_montage, cut, filter_continuous, patches200, to_256, write_outputs  # noqa: E402

DATASET_ID = "bnci2015_013"
URL = "https://bnci-horizon-2020.eu/database/data-sets/013-2015/"
CORRECT, ERROR = {5, 10}, {6, 9}
TMIN_S, W = 0.0, 1
N0 = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="", help="comma list of file stems for quick tests")
    args = ap.parse_args()
    raw_root, out = Path(args.raw), Path(args.out)
    files = sorted(raw_root.glob("Subject*_s*.mat"))
    if args.subjects:
        files = [f for f in files if f.stem in args.subjects.split(",")]
    assert files, raw_root

    raw_files = []
    sha = raw_root / "SHA256SUMS.txt"
    if sha.exists():
        for line in sha.read_text().splitlines():
            h, p = line.split(None, 1)
            raw_files.append({"path": p.strip(), "sha256": h, "url": URL + p.strip()})

    ch_out, rows, ep256, ep200, ids = None, [], [], [], []
    filt_info, other_codes, run_notes = None, collections.Counter(), {}
    for f in files:
        subj = "S%02d" % int(f.stem.split("_")[0].replace("Subject", ""))
        sess = f.stem.split("_")[1].replace("s", "")
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False, variable_names=["run"])
        runs = np.atleast_1d(m["run"])
        for ri, run in enumerate(runs, start=1):
            h = run.header
            sf = float(h.SampleRate)
            assert sf == 512.0, sf
            assert int(h.Subject) == int(subj[1:]) and int(h.Session) == int(sess), (f, h.Subject, h.Session)
            labels = [str(x) for x in np.atleast_1d(h.Label)]
            eeg_names = [c for c in labels if c != "Status"]
            assert len(eeg_names) == 64 and len(labels) == run.eeg.shape[1], (labels, run.eeg.shape)
            if ch_out is None:
                ch_out = eeg_names
                check_montage(ch_out)
            assert eeg_names == ch_out, (f, ri)
            X = np.asarray(run.eeg[:, [labels.index(c) for c in ch_out]], dtype=np.float64).T  # (64, T) uV
            X = X - X.mean(0, keepdims=True)  # common average reference
            X = X - X.mean(1, keepdims=True)  # remove DC before filtering (no effect on the band-passed output)
            Xf, finfo = filter_continuous(X, sf)
            filt_info = filt_info or finfo
            X256 = to_256(Xf, sf)
            typ = np.atleast_1d(h.EVENT.TYP).astype(int)
            pos = np.atleast_1d(h.EVENT.POS).astype(np.int64) - 1  # MATLAB 1-based -> 0-based
            run_notes[f"{f.stem}_r{ri:02d}"] = {"n_samples_512": int(X.shape[1]),
                                                 "codes": {str(k): int(v) for k, v in collections.Counter(typ.tolist()).items()}}
            for k, (t, p) in enumerate(zip(typ, pos)):
                if t not in CORRECT | ERROR:
                    other_codes[int(t)] += 1
                    continue
                label = int(t in ERROR)
                onset256 = int(round(p * FS_OUT / sf))
                seg, reason = cut(X256, onset256, N0, W)
                tid = f"{subj}_s{sess}_r{ri:02d}_ev{k:04d}"
                rows.append({"subject_id": subj, "trial_id": tid, "label": label, "retained": int(not reason),
                             "exclusion_reason": reason, "session": sess, "run": ri, "source_file": f.name,
                             "event_code": str(int(t)), "onset_s": float(p / sf), "event_index_in_run": k,
                             "target_side": "left" if t & 1 else "right",
                             "cursor_move": "left" if t & 4 else "right"})
                if not reason:
                    ep256.append(seg.astype(np.float32))
                    ep200.append(patches200(seg, W))
                    ids.append(tid)
            del X, Xf, X256
        print(f.name, "runs", len(runs), "rows so far", len(rows), flush=True)

    cfg = {
        "preprocessing_id": DATASET_ID,
        "dataset": {
            "name": "BNCI Horizon 2020 013-2015: Monitoring error-related potentials",
            "citation": "Chavarriaga, R., & Millan, J. d. R. (2010). Learning from EEG error-related potentials in "
                        "noninvasive brain-computer interfaces. IEEE Trans. Neural Syst. Rehabil. Eng., 18(4), 381-388.",
            "doi": "10.1109/TNSRE.2010.2053387", "url": URL,
            "license": "CC BY-NC-ND 4.0 (BNCI Horizon 2020 database)",
            "raw_files": raw_files,
        },
        "paradigm": "passive monitoring of an autonomous cursor moving towards a target; ~20% of the cursor steps go "
                    "the wrong way (observation ErrP)",
        "event_locking": "time 0 = cursor-movement trigger (EVENT.POS of codes 5/6/9/10)",
        "window_s": [TMIN_S, TMIN_S + W], "W": W,
        "event_mapping": {"source": "description.pdf (BNCI 013-2015): bitwise triggers; '5' and '10' = correct "
                                    "movements, '6' and '9' = erroneous movements",
                          "label_1_error": sorted(ERROR), "label_0_correct": sorted(CORRECT),
                          "not_epoched_codes_counts": {str(k): v for k, v in sorted(other_codes.items())},
                          "not_epoched_codes_meaning": "1/2 = new target (left/right) without movement; 4/8 = "
                                                       "cursor movement left/right with no target bit (not documented "
                                                       "as correct/erroneous) -> not labelled"},
        "channels": ch_out, "channel_rename": {},
        "dropped_channels": ["Status (trigger)"],
        "band_hz": [filt_info["l_freq"], filt_info["h_freq"]],
        "filter": {**filt_info, "applied": "per run, on the continuous 512 Hz CAR signal before epoching"},
        "resampling": "continuous 512 -> 256 Hz scipy.signal.resample_poly(up=1, down=2) after filtering; "
                      "200 Hz patches = resample_poly(25, 32) of each 256 Hz epoch",
        "reference": "common average of the 64 EEG channels (raw files are Biosemi CMS; CAR as in the authors' "
                     "protocol: Chavarriaga et al. IDIAP-RR 07-37 / EMBC 2010)",
        "baseline": "none", "units": "microvolts (Biosemi native uV)",
        "event_pos_convention": "EVENT.POS treated as MATLAB 1-based sample index (0-based = POS - 1)",
        "exclusions": {"policy": "no amplitude rejection; only epochs outside the recording / non-finite dropped",
                       "subjects_excluded": []},
        "per_run_notes": run_notes,
    }
    readme = f"""# bnci2015_013 -- BNCI Horizon 2020 013-2015 "Monitoring error-related potentials"

6 subjects x 2 sessions x 10 runs (Chavarriaga & Millan 2010, IEEE TNSRE, doi 10.1109/TNSRE.2010.2053387; data
{URL}; license CC BY-NC-ND 4.0). Epochs [0, 1) s after each cursor-movement trigger; label 1 = erroneous movement
(codes 6, 9), 0 = correct movement (codes 5, 10), from the dataset description.pdf. 64 EEG channels (Status dropped),
common-average re-referenced, MNE zero-phase FIR 1-20 Hz per continuous run at 512 Hz, resample_poly to 256 Hz,
200 Hz patches via resample_poly(25, 32). No amplitude rejection, no baseline correction.

Regenerate (cluster):

    $B/env/bin/python bench/prep/bnci2015_013.py --raw $B/data/raw/bnci2015_013 --out $B/data/prep/bnci2015_013
"""
    write_outputs(out, rows, ep256, ep200, ids, cfg, readme)


if __name__ == "__main__":
    main()
