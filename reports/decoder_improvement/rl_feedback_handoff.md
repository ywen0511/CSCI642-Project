# New decoder probabilities for RL feedback (vnext): handoff

**Status: experimental follow-up. No RL experiment has used these probabilities, and no robot-performance result
follows from them.** The historical feedback (`configs/rl_feedback.json`, Release `eeg-stage1-20261101-bccbb8a`,
tables `validation_predictions.csv` / `test_predictions.csv`) is unchanged and remains the default.

## What is new
- Profile: `configs/decoder_improvement/rl_feedback_vnext.json` (same schema as the historical profile), pointing to
  Release `decoder-v1-final-20261111-f8031ec` assets `vnext_validation_predictions.csv`, `vnext_test_predictions.csv` and
  `predictions_manifest.json` (sizes and SHA-256 pinned in the profile).
- Decoders (selected candidates of the decoder-improvement study, decoder seed **42 designated in advance**, even
  where another seed scored higher):
  - `cbramod`: `cb-flat-frozen-bbnone-hl5e-4-b1_20-ce-3aba4899` (historical flatten head, frozen backbone, 1-20 Hz, unweighted CE,
    5 epochs on all seven training participants); calibration `affine`
    a=0.3322, b=-0.5548.
  - `eegnet`: `eeg-lr1e-3-do0.5-b1_20-wce-ad833d47` (the historical EEGNet recipe, retrained for 47 fixed epochs on all seven
    training participants); calibration `affine`
    a=0.5530, b=-0.5535.
- Columns: historical schema (`subject_id, trial_id, label, split, model, decoder_seed, p_error_raw, p_error,
  checkpoint_id, preprocessing_id, calibration_id`) plus `recipe_id, p_error_temperature, margin`.
  `p_error` = affine-calibrated P(error); `p_error_raw` = sigmoid(logit_error - logit_nonerror).
- Label direction unchanged: 0 = non-error (S4), 1 = machine error (S6).
- All seeds, raw/temperature/deployment probabilities and both recipes per architecture are in the Release asset
  tables `dimp_validation_predictions.csv` / `dimp_test_predictions.csv`.

## Semantics and caveats
- **Calibration reuse:** the validation participants s11/s13 were used to choose and fit the calibrators, so the
  validation rows are calibration-development outputs, not unbiased held-out feedback quality. A future RL tuning
  study on validation feedback must account for this reuse (declared in the profile field `calibration_reuse`).
- The test participants s02/s08/s09 were evaluated in the historical study; these are follow-up probabilities.
- Decoder quality did not improve: on the test participants the new CBraMod has pooled AUROC 0.564 (seed mean;
  historical 0.573 at seed 42) and the new EEGNet 0.599 (historical 0.637 at seed 42). The new probabilities are
  better calibrated (lower Brier), which changes the EEG reward scale `w (0.5 - p_error)`: mean p_error moves
  toward the error prevalence, so a feedback event is less often a positive bonus than with the historical tables.

## Use
```python
from rl.feedback import FeedbackLibrary
lib = FeedbackLibrary("artifacts/rl_feedback_vnext", "validation",
                      profile_path="configs/decoder_improvement/rl_feedback_vnext.json")
```
Download and validate the tables with
`python scripts/download_feedback.py --profile configs/decoder_improvement/rl_feedback_vnext.json --out-dir artifacts/rl_feedback_vnext`.
Interface checks (`tests/test_feedback_vnext.py`, run in the RL environment): identity/hash/participant checks,
paired sampling across decoders, label-independent sampling, determinism, tamper rejection, and the historical
profile still loading; together with `rl.test_stage2`, 35 tests pass.
