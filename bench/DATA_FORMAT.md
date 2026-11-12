# ErrP-Bench prepared-dataset format

Every dataset (or dataset condition) is converted ONCE into one directory
`$B/data/prep/<dataset_id>/` (B=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench) that the existing
loader `src/errp/data.py::load_prepared` reads unchanged.

Files:
- `trial_manifest.csv` — one row per candidate epoch, columns at least:
  `subject_id` (str, e.g. `S01`), `trial_id` (str, unique), `label` (int: 1 = error, 0 = correct/non-error),
  `retained` (1 if the epoch is in the arrays, else 0), `session` (str), `source_file` (str), `event_code` (str),
  `onset_s` (float). Rows with retained == 1 appear in the SAME ORDER as the arrays.
- `epochs_256hz_uV.npy` — float32 (N, C, 256*W) microvolts at 256 Hz.
- `epochs_200hz_patches_uV.npy` — float32 (N, C, W, 200) microvolts at 200 Hz (`scipy.signal.resample_poly`
  from the 256 Hz array is fine: up=25, down=32), i.e. W one-second patches.
- `epoch_trial_ids.npy` — `<U` string array (N,) of trial_ids of the retained epochs, same order.
- `preprocessing_config.json` — at least: `preprocessing_id` (= dataset_id), `dataset` (name, citation, DOI/URL,
  license, exact download URLs + sha256 of every raw file used), `paradigm`, `event_locking`
  (what time 0 is), `window_s` ([tmin, tmax), integer length W seconds), `channels` (ordered list of names that
  exist in MNE `standard_1020` montage — rename e.g. `FP1`->`Fp1`, `T3`->`T7` if needed and record the mapping),
  `band_hz` [1, 20], `filter` (MNE zero-phase FIR on the continuous recording BEFORE epoching), `reference`
  (keep the dataset's own reference unless stated; record it), `n_subjects`, `n_epochs`, `class_counts`,
  `per_subject_counts`, `exclusions` (what was dropped and why).
- `README.md` — short human description + how to regenerate (the exact command).

Rules:
- Real, full public datasets only: ALL subjects/sessions that the dataset provides for the paradigm.
- Labels from the dataset's own event documentation; record the event-code mapping.
- No amplitude rejection; drop only epochs that are incomplete / non-finite (record counts).
- Channels: scalp EEG only (drop EOG/EMG/reference/trigger channels).
- Window: 1 s ([0, 1) s after the error-eliciting event) unless the dataset needs otherwise; for response-locked
  ERN use [-0.2, 0.8) s around the response (still W = 1).
- Preprocessing script lives in `bench/prep/<dataset_id>.py` and is runnable on the cluster with
  `$B/env/bin/python bench/prep/<dataset_id>.py --raw <raw dir> --out $B/data/prep/<dataset_id>`.
- Raw downloads go to `$B/data/raw/<dataset_id>/` (never under /home).
