# RL protocol v2: pilot runs (validation participants only; not a suite)

Scope: **pilot / diagnostic runs; NOT a complete validation or final suite.** All pilots used validation
participants s11/s13 (`phase=pilot`), EEG weight 0.5, dense weight 10.0, pilot seeds 900-902 (disjoint from the
validation seeds 100-102 and final seeds 0-4). Training commit `50888a9`.

- `pilots__*`: 9 runs of 50,000 steps (seed 900: sparse, dense, ideal, EEGNet s11/s13, CBraMod s11/s13,
  independent EEGNet s11, independent CBraMod s13), 5 budget-check runs of 150,000 steps (seed 901: sparse, dense,
  ideal, EEGNet s11, CBraMod s13) and a 20,000-step CPU throughput benchmark (dense, seed 902). These runs predate
  the fix that skips the pre-update periodic evaluation at the final step, so their `evaluations.jsonl` holds two
  records at the final step; `reports/rl_v2_pilots/pilot_report.json` uses the final policy's evaluation.
- `pilots_attempt1_failed__*`: the first pilot batch (commit `3113b6b`), which failed after 4 episodes because of
  the SB3 `info["episode"]` collision. Kept as a failure record; archives contain the partial outputs.

Each run archive contains every periodic and final snapshot (model, optimizer/entropy state inside `model.zip`,
replay buffer, RNG state, inference probe, metadata), `transitions.csv`, `evaluations.jsonl`, SB3 progress logs,
`run.json`, `environment.json`, `status.json` and the job stdout/stderr. `SHA256SUMS` covers every asset.
Pilot summary and the budget decision: `reports/rl_v2_experiment_plan.md` Section 7.

Split asset: `pilots__dense_shared_s901_150k.tar.gz` repeatedly failed to upload (TLS errors) and is published
as lossless parts `.part-00..02`; reassemble with
`cat pilots__dense_shared_s901_150k.tar.gz.part-* > pilots__dense_shared_s901_150k.tar.gz` and check it
against `SHA256SUMS.full-files` (also listed in `pilots__manifest.json`).
