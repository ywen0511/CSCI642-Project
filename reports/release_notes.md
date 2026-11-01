# EEG stage 1: EEGNet vs CBraMod on ErrP-HRI (robot condition)

Source commit (training, evaluation, export): `bccbb8a`. Training code is identical to `3b95649`,
which is the `source_sha` recorded inside every checkpoint. Tag: `eeg-stage1-20261101-bccbb8a`.

## Contents
- `run_<run_id>.tar.gz` for every run: `best.ckpt`, `last.ckpt` and the retained periodic checkpoints
  (full model + optimizer + scheduler + RNG + sampler state), `train_log.jsonl`, run config,
  calibration parameters, metrics, validation/test predictions and margins.
  Main runs: `eegnet_lr1e-3_s42`, `cbramod_bblr1e-4_s42`. Non-selected candidates: `eegnet_lr3e-4_s42`,
  `cbramod_bblr3e-5_s42`. Extra decoder seeds 43-46. Supplementary baselines: `bd_*`.
- `test_predictions.csv` / `validation_predictions.csv`: the RL feedback library (seed 42, both models,
  schema per README Section 9). `p_error` = temperature-calibrated P(error).
- `manifest.json`: resolves `checkpoint_id`, `preprocessing_id` and `calibration_id` to files, configs,
  versions and SHA-256. `metrics_summary.json`: all metrics.
- `shared_metadata.tar.gz`: configs, split, trial manifest, preprocessing config, audit, selection,
  all prediction tables (main, extra seeds, supplementary), requirements.
- `SHA256SUMS`.

## Split
Test s02, s08, s09; validation s11, s13; train s03, s04, s05, s06, s07, s10, s12
(`configs/splits.json`, SHA-256 `7d34b0b9...`).

## Results (test, 1,429 trials, P(error) >= 0.5, calibrated)
| model | AUROC | bal. acc | Brier raw -> cal. | 5-seed AUROC |
|---|---|---|---|---|
| EEGNet | 0.637 | 0.590 | 0.228 -> 0.227 | 0.613 ± 0.016 |
| CBraMod | 0.573 | 0.559 | 0.363 -> 0.244 | 0.568 ± 0.007 |

## Hardware and duration
1× A100-SXM4-40GB per job (GM Warren gc175), driver 580.105.08, torch 2.15.0+cu132.
About 35 GPU-minutes in total.

## Known limitations
3 test participants; no significance testing; epoch origin = FB marker (documented feedback-direction
event; robot movement onset may be up to 130 ms earlier); shared 1-20 Hz input (CBraMod was pretrained
at 0.3-75 Hz); class weighting leaves residual probability shift after temperature scaling.
Raw EEG and the original pretrained weights are not redistributed. See `reports/` in the repository.
