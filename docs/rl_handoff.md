# Frozen EEG feedback for the RL stage

Use the original [EEG Release](https://github.com/ywen0511/CSCI642-Project/releases/tag/eeg-stage1-20261101-bccbb8a).
The authoritative machine-readable profile is [`configs/rl_feedback.json`](../configs/rl_feedback.json).
This closeout adds diagnostics and verification fixes; it does not replace any trained checkpoint,
calibrator, split, or prediction table.

## Fixed inputs

- Release tag: `eeg-stage1-20261101-bccbb8a`.
- Release source commit: `bccbb8a2918c0a08381852feb1c6b1eb81e19548`.
- Checkpoint training commit: `3b9564966ecdaf9555e1bbcf5b8cad2013e2700c`.
- Main decoder seed: 42, fixed before these diagnostics. Seeds 43–46 remain separate robustness runs.
- EEGNet checkpoint: `eegnet_lr1e-3_s42:best`; calibration temperature 1.392144737298234.
- CBraMod checkpoint: `cbramod_bblr1e-4_s42:best`; calibration temperature 9.90420611301174.
- Preprocessing: `prep_v1-13fe09dfbe64`.
- Validation: s11, s13; 959 trials per model.
- Test: s02, s08, s09; 1,429 trials per model.
- Labels: 0 = non-error; 1 = robot/machine error. Human-error S5 trials were excluded.
- Feedback column: `p_error`, the frozen calibrated P(error). Preserve `p_error_raw` for diagnostics.

## Download and validate the feedback library without a GPU

The two CSVs already contain inference outputs. Reading them does not require the EEG data, model
weights, a training environment, or a GPU. The following standard-library Python command validates
sizes, SHA-256, schema, seed, model identities, probability ranges, trial uniqueness, participant
membership and matched trial labels between models against the pinned profile:

```bash
python3 scripts/download_feedback.py --out-dir artifacts/rl_feedback
```

This downloads `test_predictions.csv`, `validation_predictions.csv` and `manifest.json` directly
from the original Release. It writes `feedback_download_verification.json` beside them. It does
not claim checkpoint verification. Full-checkpoint verification and resumption commands remain in
[`reports/artifact_index.md`](../reports/artifact_index.md).

## How to use the library

1. Fix the decoder model and seed for an experiment. Filter the table explicitly; never mix two
   models' rows into one feedback stream.
2. Use validation participants to choose RL feedback weights and other RL hyperparameters. These
   participants were also used for decoder selection/calibration, so validation is a tuning resource.
3. After fixing the RL protocol, evaluate using only the test feedback library. Do not fit the
   decoder, recalibrate probabilities, select a decoder seed, or tune RL settings using test results.
4. Define the simulated action-to-error-label mapping in the RL protocol. If sampling conditional on
   a simulated label, select rows with that label and retrieve `p_error`. This is offline replay,
   not EEG recorded from a user observing that simulated action. Preserve the distinction in reports.
5. Declare the participant sampling scheme, sampling with/without replacement, feedback frequency,
   feedback random seed and RL training seed. Participant-specific experiments should use only that
   participant's library. Do not automatically pool participants with different feedback quality.
6. For paired decoder comparisons, use the same sampled participant/trial IDs and simulated labels
   for both models, then join by `(subject_id, trial_id, label, split, decoder_seed)`. Only the model's
   probability changes. If replacement is used, repeated trial IDs are repeated samples, not new EEG.
7. Apply reward transforms and weights inside the RL experiment code. Do not overwrite the EEG CSVs
   with `1-p`, thresholded labels, weighted rewards, or other transformations.
8. Record the Release tag, table hashes, checkpoint/calibration IDs and all RL/sampling settings in
   every RL run. `configs/rl_feedback.json` provides the frozen identifiers and hashes.

## What the decoder results do and do not establish

The [feedback diagnostic report](../reports/feedback_closeout/README.md) adds a constant-probability
baseline fitted only to training labels and label-conditional distributions. Both main decoders
have modest ranking ability. Their probability quality is limited: EEGNet seed 42 barely improves
test Brier over the constant baseline and has worse test NLL; CBraMod is worse on both scores.
These observations do not establish whether either feedback source improves RL.

For the next stage, prespecify controls such as no EEG feedback, ideal simulated-label feedback,
and label-independent feedback. Choose their precise implementation and any weights using validation
only. The constant train-prevalence predictor is available as a diagnostic/control, not a substitute
chosen from test performance. No RL experiment is executed in this closeout.

## Remaining checks in the original training environment

The original agent reported successful download, inference and continuation checks. Their narrative
is preserved in `reports/artifact_index.md`. This closeout checked the actual downloaded prediction
and metadata files on CPU; it did not repeat the original GPU checks.

The original agent should run the updated fail-on-error scripts on fresh directories, retaining the
JSON outputs, exact commands, exit codes, dependency versions and job/stdout logs. Use the pinned
original Release and report both main decoders. Preserve any original machine-readable logs too.
Publish these additional evidence files under a new supplement Release tag; do not overwrite the
original training Release or move its tag. A later evidence report should state its own code commit.
