# Weight sensitivity of the v2 result (w = 0.1, 0.5, 1.0)

**Status: post-hoc sensitivity analysis, requested after the v2 final results were seen. The v2 primary result
(w = 1.0, selected by the declared rule) is unchanged. These suites are not a re-selection.**

Assignment: [FINAL_V3_HANDOFF.md](../../FINAL_V3_HANDOFF.md), Section 5. All tables, with every number traced to
its source file: [`generated/sensitivity_tables.md`](generated/sensitivity_tables.md) (audit:
[`generated/sensitivity_tables_audit.json`](generated/sensitivity_tables_audit.json)). Analyses:
[`sensitivity_w010_final/`](sensitivity_w010_final/), [`sensitivity_w050_final/`](sensitivity_w050_final/), and for
w = 1.0 the v2 final analysis [`../rl_v2_final/`](../rl_v2_final/).

## What was run

- The two candidates the v2 rule did not select, `candidate_w010` and `candidate_w050`, were frozen with `rl.freeze`
  from their existing, complete v2 validation suites. These suites were restored from the Releases
  `rl-v2-validation-w010/-w050`, with every archive and file hash checked. Validation was not rerun. The frozen
  protocols are [`configs/final_v3/sensitivity_w010/`](../../configs/final_v3/sensitivity_w010/) and
  [`sensitivity_w050/`](../../configs/final_v3/sensitivity_w050/). They were committed (`f63dc24`) before any of
  these final runs.
- Each weight was run as a complete 75-run final suite: test participants s02/s08/s09, RL seeds 0-4, 150,000
  steps, 100 evaluation goals. The suites ran from commit `f63dc24`, whose RL training sources are byte-identical
  to the v2 final commit `25bb0f4`. All 150 runs completed, with no failures and no resumption, and
  `rl.summarize` passed for both suites.
- **Determinism check.** Sparse and dense runs do not use the EEG weight, so they were rerun in each suite. All
  10 sparse/dense runs of each suite reproduce the v2 final runs exactly: `evaluation.json`, the full evaluation
  history, the transition log and the final inference probe are identical
  ([`sensitivity_determinism.json`](sensitivity_determinism.json)). The sparse and dense rows are therefore the same
  numbers in all three weight tables.

## Results

Final success is near ceiling for every condition and weight. The effect of the weight shows up in learning speed
(normalized success AUC):

- **w = 0.5 is the only weight at which both decoders are ahead of sparse in AUC.**
  - Paired over 15 (participant, seed) pairs:
    - EEGNet - sparse = +0.060 (10 better / 5 worse).
    - CBraMod - sparse = +0.039 (7 / 7 / 1 equal).
  - Both decoders are also ahead of their label-independent controls:
    - EEGNet: +0.092, 11 / 4.
    - CBraMod: +0.084, 11 / 4.
- **At w = 0.1 the EEG reward is tiny** (about +90 to +120 per run), and every condition is close to sparse in AUC.
- **At w = 1.0 (the v2 primary result)**, EEGNet is slightly below sparse and CBraMod is clearly below it
  (-0.017 and -0.087).
- **The label-independent controls are below sparse at w = 0.5 and 1.0.** Their reward carries no label
  information but grows with w (positive bonus per run: +577 to +600 at w = 0.5, +1,172 to +1,197 at w = 1.0). This
  matches the reward-bias explanation that motivates protocol v3 ([`v3_results.md`](v3_results.md)).

What the validation rules would have chosen:
- The declared v2 rule ranks by final success, which is saturated (0.980 / 0.986 / 0.997). It chose **w = 1.0**.
- An AUC-based rule applied to the same v2 validation evidence would have chosen **w = 0.1** (AUC 0.762 vs 0.748
  and 0.731). On the test participants, w = 0.5 shows the largest decoder advantage.
- Neither alternative is promoted post hoc. The three validation weights also do not order the same way on the
  validation and test participants. This is one more sign that differences between these settings are small
  relative to seed and participant variation.

Interpretation limits (as in v2):
- 3 test participants × 5 RL seeds.
- Means, SDs and sign counts only; no population-level significance claims.
- The test participants were already used for the v2 result.
- Feedback is replayed, not live.
