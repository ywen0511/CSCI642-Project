# Decoder-improvement study: final refits, calibration, predictions and reports

Scope: **final deliverable** of the CBraMod / EEGNet improvement study (branch `decoder-improvement-v1-experiments`).
Tag commit `f8031ec` holds the frozen recipes (`reports/decoder_improvement/decisions/freeze.json`) and the
calibration decision committed before any test evaluation.

Result in one line: no tested change improved held-out decoding. On the previously evaluated test participants
(s02, s08, s09; mean ± SD over five refits) the selected CBraMod (historical flatten head, frozen backbone, 1-20 Hz,
unweighted CE) reached pooled AUROC 0.564 ± 0.005 vs 0.563 ± 0.006 for a controlled refit of the historical CBraMod
recipe; EEGNet's search returned its historical recipe, 0.599 ± 0.014. Affine calibration (chosen on s11/s13 only)
improved Brier scores, not ranking. Full report: `reports/decoder_improvement/final_report.md` (also inside
`evidence.tar`).

Assets:
- `fit_<fit_id>.tar` (15, split into 50 MiB parts where large): the five final refits (seeds 42-46) of each recipe
  on all seven training participants with fixed epoch counts (CBraMod selected 5, CBraMod reference 17, EEGNet 47):
  `final.ckpt` (the evaluation model), `last.ckpt`, every periodic checkpoint, all with full resumable state, plus
  logs, status and probes.
- `dimp_validation_predictions.csv`, `dimp_test_predictions.csv`: every final model and seed; columns subject_id,
  trial_id, label, split, model, recipe_id, role, decoder_seed, logits, margin, p_error_raw, p_error_temperature,
  p_error (selected calibration for candidates, temperature for the reference), checkpoint_id, preprocessing_id,
  scaler_id, calibration_id, temperature_calibration_id. Validation rows are calibration-development outputs.
- `vnext_validation_predictions.csv`, `vnext_test_predictions.csv`, `predictions_manifest.json`: the seed-42
  selected-candidate tables for `configs/decoder_improvement/rl_feedback_vnext.json` (experimental; the historical
  feedback is unchanged; no RL experiment has used them).
- `evidence.tar`: reports, plans, decisions (incl. freeze and calibration), decision log, ledger, sanitized logs,
  validation/test margins with the calibration hash recorded at test access, test metrics, generated tables, figures
  and their source data, configs.
- `source_<sha>.tar`: `git archive` of the delivered source (reproduction bundle).
- `stage_index.json`, `payload_manifest.json` (every logical asset and archive member with size and SHA-256),
  `SHA256SUMS`.

Raw EEG and pretrained weights are external (no redistribution permission established); fetch them with
`scripts/decoder_improvement/acquire_inputs.py`. Verify this Release with
`python scripts/decoder_improvement/release_tools.py fetch --tag decoder-v1-final-20261111-f8031ec --out <new dir>`.
