# EEG stage closeout status

## Completed on CPU

- Downloaded the original Release prediction tables, manifest, metrics and shared metadata; their
  published SHA-256 checks matched. A separate fresh-directory download of the RL feedback files
  also passed the new downloader's hash, size, schema, identity, label and trial-alignment checks.
- Recomputed seed-42 probability metrics and the constant training-prevalence baseline on validation
  and test participants. Saved per-participant and pooled statistics, class-conditional distribution
  plots (PNG/PDF), and exact numerical comparisons with the original reported metrics.
- Pinned the original Release, checkpoint IDs/hashes, temperatures, preprocessing ID, participant
  lists and feedback CSV hashes in `configs/rl_feedback.json`; documented RL usage in `docs/rl_handoff.md`.
- Corrected README environment activation; added a separate CPU analysis dependency lock.
- Made asset/inference verification fail with a nonzero process exit code, stop before extraction on
  bad checksums, require both main decoders for inference, and check participant/split/trial alignment.
- Made continuation/equivalence failures return a nonzero exit code. Existing scratch directories
  are refused rather than deleted; training checkpoints and training code were not changed.
- Ran 20 CPU regression tests, syntax compilation and `git diff --check`. Reviewed both generated plots.
  The tests use synthetic fixtures/mocked inference to exercise acceptance success/failure gates;
  they are not evidence of a fresh neural-network inference or training run.

Evidence: `reports/feedback_closeout/verification.json`, `feedback_download_verification.json`,
`cpu_regression_tests.txt`, and the generated report and CSVs in the same directory.
The recorded CPU environment is Python 3.12, separate from the original Python 3.11 GPU environment.

## Remaining in the original GPU environment

The original training agent has already reported successful re-download, inference and resumption
in `reports/artifact_index.md`. This closeout does not retract or independently repeat those claims.
The agent should retain/upload the original machine-readable logs and rerun the updated verification
and continuation entry points on fresh directories, saving JSON, stdout/stderr, commands, exit codes,
job IDs and environment versions. This is a verification rerun, not new decoder training or tuning.
A new supplemental Release can carry that evidence; preserve the original training Release and tag.

No calibration experiment, new participant split, seed selection, LOSO or RL experiment was performed.
