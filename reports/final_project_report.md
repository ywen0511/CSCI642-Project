# CSCI642 final project report: recorded EEG error feedback for robot reaching

Repository `ywen0511/CSCI642-Project`. This is the single final document for the course. It was first written for
the v2 delivery (branch `rl-v2-experiments`, 2026-11-10) and updated in place for the final round
(branch [`final-v3-experiments`](https://github.com/ywen0511/CSCI642-Project/tree/final-v3-experiments),
2026-11-18, assignment [`FINAL_V3_HANDOFF.md`](../FINAL_V3_HANDOFF.md)). The v2 text of Sections 1-6 is kept; the
final round adds Sections 3.2-3.4, 7 (milestone follow-up), 9 (numbers across documents) and the new Releases in
Section 8.

Detailed documents:
- EEG stage: [`eeg_results.md`](eeg_results.md).
- RL v2: plan [`rl_v2_experiment_plan.md`](rl_v2_experiment_plan.md), execution record
  [`rl_v2_execution.md`](rl_v2_execution.md), results [`rl_v2_results.md`](rl_v2_results.md).
- Decoder study (merged into this branch): [`decoder_improvement/final_report.md`](decoder_improvement/final_report.md).
- Final round: [`final_v3/sensitivity_results.md`](final_v3/sensitivity_results.md),
  [`final_v3/v3_results.md`](final_v3/v3_results.md), [`final_v3/preference_results.md`](final_v3/preference_results.md),
  [`final_v3/execution.md`](final_v3/execution.md); status [`final_v3/status.json`](final_v3/status.json), decision log
  [`final_v3/decisions.jsonl`](final_v3/decisions.jsonl).

**Status labels used below.**

| study | status |
|---|---|
| RL v2, w = 1.0 (Section 3.1) | **declared primary result** (plan, validation selection and freeze before the final suite) |
| weight sensitivity w = 0.1, 0.5 (Section 3.2) | **post-hoc sensitivity analysis** (run after the v2 results were seen; not a re-selection) |
| protocol v3, centered reward (Section 3.3) | **post-hoc follow-up with its own declared validation selection and freeze** |
| route preference (Section 3.4) | **extension** (declared plan, pilot gate, validation selection and freeze; reused test participants) |
| decoder improvement (Section 7) | completed separate study, negative; merged, not repeated |

## 1. Question

Can error probabilities decoded from recorded EEG improve robot reaching, and how do EEGNet and CBraMod differ as
feedback sources? A stronger decoder was not assumed to give a stronger policy.

## 2. Methods

**Stage 1-2 (EEG, completed earlier, unchanged here).** ErrP-HRI robot condition, 12 participants, participant
split with seed 42 (train 7, validation s11/s13, test s02/s08/s09). EEGNet trained from scratch and CBraMod
(official pretrained weights, full fine-tuning), both with validation temperature calibration. Test AUROC
0.637 (EEGNet) vs 0.573 (CBraMod); balanced accuracy 0.590 vs 0.559; calibrated Brier 0.227 vs 0.244. Per-trial
calibrated error probabilities for validation and test participants were frozen in Release
`eeg-stage1-20261101-bccbb8a`; they were not retrained, recalibrated or re-selected for RL.

**Stage 3 (EEG-assisted RL).** FetchReach-v4 (Gymnasium-Robotics) with its sparse task reward and 5 cm tolerance,
learned by SAC (Stable-Baselines3) for 150,000 environment steps per run. Every 5 simulator steps, the change in
gripper-goal distance labels the segment: decrease beyond 1 mm = non-error (0), increase beyond 1 mm = error (1),
otherwise no label and no feedback. For a labeled segment, a recorded EEG trial with that label is drawn (with
replacement) from one participant's pool and its calibrated probability adds `w (0.5 - p_error)` to the reward.
Seven conditions: sparse, dense (`10 x` distance decrease), ideal feedback (`p = label`), EEGNet, CBraMod, and two
label-independent controls that draw trials regardless of the label. Policies observe only the standard
FetchReach observation.

**Protocol.** A plan with candidates, seeds, budget, metrics and selection rule was committed before validation.
Three EEG weights (0.1, 0.5, 1.0) were each run as a complete 33-run validation suite on s11/s13 (RL seeds
100-102); the declared rule (mean final success of the EEGNet and CBraMod runs; ties by final distance) selected
w = 1.0, which was frozen with its evidence. The final suite (75 runs) used test participants s02/s08/s09 and RL
seeds 0-4. Evaluation: 100 fixed goals, deterministic policy, sparse reward only, every 10k steps. Metrics:
final-step success (primary), ever-success, final distance, sparse return, and normalized AUC of the success
curve (learning speed). EEG conditions are macro-averaged over participants; shared controls are counted once.

## 3. Results

### 3.1 RL v2 (declared primary result, w = 1.0, unchanged)

| condition | runs | final success | final distance (m) | normalized AUC |
|---|---|---|---|---|
| sparse | 5 | 0.942 ± 0.083 | 0.0301 | 0.659 ± 0.131 |
| dense | 5 | 1.000 ± 0.000 | 0.0214 | 0.915 ± 0.033 |
| EEGNet | 15 | 0.968 ± 0.034 | 0.0233 | 0.641 ± 0.041 |
| CBraMod | 15 | 0.952 ± 0.063 | 0.0229 | 0.571 ± 0.115 |
| ideal | 5 | 0.986 ± 0.031 | 0.0213 | 0.764 ± 0.097 |
| independent EEGNet | 15 | 0.960 ± 0.049 | 0.0241 | 0.595 ± 0.036 |
| independent CBraMod | 15 | 0.940 ± 0.078 | 0.0385 | 0.602 ± 0.038 |

(± = SD over RL seeds for shared controls, SD over the three participants' seed means for EEG conditions.)

- All conditions solved the task by the end of training; final success is near its ceiling.
- Learning speed: dense > ideal > sparse, consistently across seeds (5/5 and 4/5 pairs). Decoded EEG feedback did
  not beat sparse (EEGNet 8/15 pairs, CBraMod 6/15).
- EEGNet vs CBraMod: EEGNet faster in 10/15 pairs (mean AUC +0.070), equal final success; within the seed spread.
- EEGNet beat its label-independent control in 12/15 pairs (AUC +0.046); CBraMod did not (9/15, -0.031).
- During RL, `p_error` separated retreat from progress with AUROC 0.650 (EEGNet) and 0.574 (CBraMod), equal to the
  decoders' test quality. The mean probability gap between error and non-error events was only 0.07 and 0.04,
  against 1.0 for ideal feedback, and calibrated probabilities below 0.5 made the EEG reward mostly a bonus.

**Answer.** With these decoders, replayed EEG error probabilities did not improve robot learning over the sparse
task reward. The better decoder (EEGNet) produced a slightly better learning signal than CBraMod and than its
own label-independent control, so decoder quality matters in the expected direction, but the information content
of single-trial ErrP probabilities at AUROC 0.57-0.65 was too small to help on this task. Noise-free error
feedback (ideal) did help, which locates the bottleneck in decoder quality rather than in the feedback mechanism.

The final round (Sections 3.2-3.4) qualifies the second half of this answer: at w = 1.0 the EEG reward was mostly
a label-free positive bonus, and removing that bonus (smaller weight or a centered reward) puts both decoders
slightly ahead of sparse in learning speed. The v2 primary result itself is unchanged.

### 3.2 Weight sensitivity, w = 0.1 and 0.5 (post-hoc sensitivity analysis)

Details and generated tables: [`final_v3/sensitivity_results.md`](final_v3/sensitivity_results.md). The two weights
the v2 rule did not select were frozen from their existing v2 validation suites (not rerun) and each run as a full
75-run final suite from training sources identical to v2. All 10 sparse/dense runs of each suite reproduce the v2
runs exactly ([`final_v3/sensitivity_determinism.json`](final_v3/sensitivity_determinism.json)).

| condition | AUC, w = 0.1 | AUC, w = 0.5 | AUC, w = 1.0 (v2) |
|---|---|---|---|
| sparse | 0.659 ± 0.131 | 0.659 ± 0.131 | 0.659 ± 0.131 |
| dense | 0.915 ± 0.033 | 0.915 ± 0.033 | 0.915 ± 0.033 |
| EEGNet | 0.664 ± 0.015 | 0.719 ± 0.010 | 0.641 ± 0.041 |
| CBraMod | 0.672 ± 0.064 | 0.698 ± 0.036 | 0.571 ± 0.115 |
| ideal | 0.606 ± 0.173 | 0.762 ± 0.089 | 0.764 ± 0.097 |
| independent EEGNet | 0.665 ± 0.031 | 0.627 ± 0.079 | 0.595 ± 0.036 |
| independent CBraMod | 0.609 ± 0.009 | 0.614 ± 0.020 | 0.602 ± 0.038 |

- w = 0.5 is the only weight at which both decoders lead sparse: paired AUC differences +0.060 (EEGNet, 10/15
  pairs better) and +0.039 (CBraMod, 7/15 better, 1 equal); both also lead their label-independent controls
  (+0.092, +0.084).
- The label-independent controls fall further behind sparse as w (and with it their label-free bonus) grows.
- The v2 rule (final success, saturated) chose w = 1.0; an AUC rule on the same validation evidence would have chosen
  w = 0.1. Neither alternative is promoted post hoc.

### 3.3 Protocol v3, centered EEG reward (post-hoc follow-up with its own validation selection)

Details and generated tables: [`final_v3/v3_results.md`](final_v3/v3_results.md). The reward becomes
`w (c_m - p_error)`, with the center `c_m` computed from the validation participants only (c_EEGNet = 0.4793,
c_CBraMod = 0.4665, [`configs/final_v3/centers.json`](../configs/final_v3/centers.json), committed before any v3
run). Three weights were run as complete validation suites; the declared AUC rule selected w = 0.5 (validation AUC
0.755 vs 0.713 and 0.642), which was frozen before the 75-run final suite.

| condition | runs | final success | normalized AUC |
|---|---|---|---|
| sparse | 5 | 0.942 ± 0.083 | 0.659 ± 0.131 |
| dense | 5 | 1.000 ± 0.000 | 0.915 ± 0.033 |
| EEGNet | 15 | 0.988 ± 0.009 | 0.688 ± 0.026 |
| CBraMod | 15 | 0.967 ± 0.029 | 0.690 ± 0.010 |
| ideal | 5 | 0.908 ± 0.169 | 0.762 ± 0.089 |
| independent EEGNet | 15 | 0.969 ± 0.030 | 0.624 ± 0.064 |
| independent CBraMod | 15 | 0.978 ± 0.019 | 0.642 ± 0.048 |

- The bias is removed: mean EEG reward per run falls from +871 / +979 (v2) to +153 / +39.
- Both decoders lead sparse in AUC (+0.029, 9/15 pairs; +0.031, 9/15) and their independent controls (+0.064,
  10/15; +0.048, 11/15). EEGNet and CBraMod are now equal (-0.002).
- Against v2 on the same participants and seeds, CBraMod's AUC rises by +0.119 (12/15 pairs) and EEGNet's by
  +0.047.
- Because v3 also lowers the weight, centering and weight effects are not separable: v2 at w = 0.5 (Section 3.2)
  gives larger decoder-vs-sparse gaps than v3.

### 3.4 Route-preference extension (extension)

<!--PREF_RESULTS-->

## 4. Proposal deviations

- **Task and simulator.** Gymnasium-Robotics FetchReach-v4 instead of Kim's Kinova pick-and-place; no claim of
  replicating Kim. Feedback is recorded, replayed EEG attached to simulated distance-change labels, not live human
  evaluation; SAC learns online with offline recorded feedback (not offline RL).
- **Conditions.** Ideal and label-independent controls were added to the proposal's four main conditions.
- **Budget and seeds.** 150k steps per run (pilot-confirmed), 3 validation seeds, 5 final seeds.
- **Evaluation precision.** 100 evaluation goals instead of the protocol default of 20 (declared before
  validation; evaluation only).
- **Inherited runner fixes before any formal run.** Disjoint training goal sequences per RL seed; SB3 `info`
  key collision; duplicate evaluation of different policies at the final step (details in the execution record).
- **Compute.** All RL runs on CPU (one core each); the GPU was not needed for this small SAC model (Section 6).
- **Feedback timing** (every 0.2 s of simulated time vs 2 s EEG epochs) is an abstract schedule, not a model of
  acquisition latency.

- **Final round (2026-11-18).** Post-hoc sensitivity suites, a centered-reward protocol (v3) and the route-preference
  extension were added by the final-round assignment after the v2 results; every rule, freeze and deviation is in
  [`final_v3/decisions.jsonl`](final_v3/decisions.jsonl). The preference pilot gate failed in every declared variant;
  the declared fallback (corridor a in [0.10, 0.90], 300k steps) was used ([`final_v3/preference_plan.md`](final_v3/preference_plan.md)).

## 5. Limitations

- **Ceiling effect.** Final success is near 1 for every condition; differences are mainly in learning speed (AUC).
- **Small samples.** Three test participants and five RL seeds: means, SDs, paired differences and sign counts
  only, no population-level significance. Decoder-score vs RL relations are descriptive, not causal.
- **Repeated use of the test participants.** s02/s08/s09 were used for the v2 primary result and again for the
  sensitivity, v3 and preference suites. Only v2 is a confirmatory test; the later suites are follow-up
  evaluations on reused participants, and v3/preference results should be read as such.
- **Replayed, not live, feedback.** Recorded EEG trials are drawn with replacement and attached to simulated
  labels; they are not independent human responses to the robot's behaviour.
- **Privileged geometric labels.** Labels come from the true gripper-goal distance and, for the preference task,
  from the true pillar/corridor geometry; a real user's error perception is noisier and delayed.
- **Abstract feedback timing.** One feedback event per 5 simulator steps (0.2 s) versus 2 s EEG epochs; not a model
  of acquisition latency.
- **Selection metric.** The v2 rule ranked by saturated final success and chose the weight with the lowest
  validation AUC (Section 3.2); v3 and the preference study declared non-saturated metrics before validation.
- **Weight and centering confounded.** v3 changes both the center and the selected weight (Section 3.3).
- **Preference task.** Even noise-free (ideal) preference feedback failed the declared pilot gate; the preference
  result therefore measures a task in which the feedback mechanism itself is weak (Section 3.4).

## 6. Reproducibility and compute

- Environment: Python 3.12.3, gymnasium 1.4.0, gymnasium-robotics 1.4.2, mujoco 3.11.0, stable-baselines3 2.9.2,
  torch 2.15.0, numpy 2.4.6 (full lock `rl/requirements-lock.cluster.txt`).
- Hardware: Intel Xeon Gold 6242 CPU nodes (GM Warren LSF `cpu_ubuntu_wrn`), one core and one torch thread per
  run. 189 SAC runs in total (15 pilots, 99 validation, 75 final) = 167 core-hours; ~56 min per 150k run.
- Training commits: pilots `50888a9`, validation `cfe0305`, final `25bb0f4` (training sources of validation and
  final identical; the summaries enforce identical training-source hashes within a suite).
- Commands (from the repository root, environment `venv/rl`):

```bash
python scripts/download_feedback.py --out-dir artifacts/rl_feedback
python -m unittest rl.test_stage2 -v
python -m rl.smoke --out artifacts/rl_v2_smoke
# validation candidate (repeat for w010, w050, w100); --execute runs serially, --prepare/--run-job per job
python -m rl.suite --phase validation --config configs/rl_v2/candidate_w100.json --out artifacts/rl_v2_validation/candidate_w100 --execute
python scripts/rl_select_candidate.py artifacts/rl_v2_validation/candidate_w0{10,50} artifacts/rl_v2_validation/candidate_w100 --out selection.json --markdown selection.md
python -m rl.freeze --validation-suite artifacts/rl_v2_validation/candidate_w100 --rationale "<from selection.json>" --out artifacts/rl_v2_frozen
python -m rl.suite --phase final --config configs/rl_v2/frozen/protocol.json --out artifacts/rl_v2_final --execute
python -m rl.analysis artifacts/rl_v2_final --out reports/rl_v2_final
python scripts/rl_write_results_tables.py --final reports/rl_v2_final --validation reports/rl_v2_validation --out reports/rl_v2_final/results_tables.md
python -m rl.trajectories --run sparse=<final checkpoint dir> ... --seeds 30000 30001 30002 --out <dir>
python -m rl.render_rollout record <checkpoint> --out a.json && python -m rl.render_rollout render a.json --out a.gif
python -m rl.verify_checkpoint <snapshot dir> --out verify.json
```

## 7. Milestone follow-up

Every commitment of the milestone report, with its final status and evidence.

| milestone commitment | status | evidence |
|---|---|---|
| sparse and dense task-reward baselines | done (v2 final, 5 seeds, 100 evaluation goals) | Section 3.1; [`rl_v2_results.md`](rl_v2_results.md) |
| train and calibrate EEGNet and CBraMod on ErrP-HRI | done (participant split, validation temperature calibration) | Section 2; [`eeg_results.md`](eeg_results.md); Release `eeg-stage1-20261101-bccbb8a` |
| integrate decoded EEG feedback into RL and compare with the baselines | done: v2 primary (EEG ≈ sparse), plus post-hoc sensitivity and centered-reward follow-up (both decoders slightly ahead of sparse in AUC) | Sections 3.1-3.3 |
| relate decoder quality to learning | done, descriptive (EEGNet > CBraMod in v2; equal under v3; per-participant decoder-vs-RL tables) | [`rl_v2_final/decoder_vs_rl.csv`](rl_v2_final/decoder_vs_rl.csv), [`final_v3/v3_final/decoder_vs_rl.csv`](final_v3/v3_final/decoder_vs_rl.csv) |
| freeze the CBraMod backbone / fine-tune only its last layers | done in the decoder study, negative (no change improved held-out decoding: CBraMod 0.564 ± 0.005, EEGNet 0.599 ± 0.014 over five refits); not repeated here | [`decoder_improvement/final_report.md`](decoder_improvement/final_report.md) |
| "investigate whether using EEG signals would allow us to learn the user's preference, in addition to learning the task" | done as the route-preference extension (previously "not performed") | Section 3.4; [`final_v3/preference_results.md`](final_v3/preference_results.md) |

## 8. Artifacts

Code and reports: branch [`rl-v2-experiments`](https://github.com/ywen0511/CSCI642-Project/tree/rl-v2-experiments).
Frozen EEG inputs: Release [`eeg-stage1-20261101-bccbb8a`](https://github.com/ywen0511/CSCI642-Project/releases/tag/eeg-stage1-20261101-bccbb8a) (unchanged).

| Release | scope | training commit |
|---|---|---|
| [`rl-v2-smoke-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-smoke-20261110-cfe0305) | v2 integration smoke (+ superseded smoke runs), all snapshots | `cfe0305` |
| [`rl-v2-pilots-20261110-50888a9`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-pilots-20261110-50888a9) | 15 pilot runs + failed first attempt, all snapshots | `50888a9` |
| [`rl-v2-validation-w010-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-validation-w010-20261110-cfe0305) | validation candidate w = 0.1 (33 runs), final snapshots + all logs | `cfe0305` |
| [`rl-v2-validation-w050-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-validation-w050-20261110-cfe0305) | validation candidate w = 0.5 (33 runs), all snapshots | `cfe0305` |
| [`rl-v2-validation-w100-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-validation-w100-20261110-cfe0305) | validation candidate w = 1.0, selected (33 runs), all snapshots | `cfe0305` |
| [`rl-v2-final-20261110-25bb0f4`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-final-20261110-25bb0f4) | final comparison (75 runs), final snapshots + all logs; analysis data, trajectories and GIFs | `25bb0f4` |

**Checkpoint scope.** At the user's instruction to upload only the important checkpoints, the final-suite and
w = 0.1 Releases contain each run's final snapshot (model with optimizer and entropy state, replay buffer, RNG
state, inference probe, metadata) plus every log, evaluation record, transition log and config; their periodic
10k-step snapshots are listed with SHA-256 in each manifest and retained on the project filer, not uploaded. The
smoke, pilot and w = 0.5 / w = 1.0 validation Releases (uploaded before that instruction) contain every snapshot.

**Fresh-download verification** (`scripts/rl_verify_release.py`, run on a cluster CPU node: list assets through
the GitHub API, download everything into a new directory, check names/sizes against the API and every SHA-256,
reassemble split archives and check full-archive hashes against the package manifest, extract into a new
directory, validate the hash of every file of every snapshot, reload checkpoints on CPU and reproduce their
stored inference probes; reports in [`reports/rl_v2_release_verification/`](rl_v2_release_verification/)):

| Release | assets | GB | inventory + every SHA-256 | split archives / full-archive hashes checked | snapshots hash-validated | inference verified (final / periodic) | max action diff | ok |
|---|---|---|---|---|---|---|---|---|
| [`rl-v2-final-20261110-25bb0f4`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-final-20261110-25bb0f4) | 89 | 1.90 | True | 0 / 76 | 75 | 75 / 0 | 0.0e+00 | **True** |
| [`rl-v2-pilots-20261110-50888a9`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-pilots-20261110-50888a9) | 40 | 1.74 | True | 1 / 32 | 137 | 15 / 7 | 0.0e+00 | **True** |
| [`rl-v2-smoke-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-smoke-20261110-cfe0305) | 45 | 0.01 | True | 0 / 36 | 102 | 32 / 7 | 0.0e+00 | **True** |
| [`rl-v2-validation-w010-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-validation-w010-20261110-cfe0305) | 37 | 0.82 | True | 0 / 34 | 33 | 33 / 0 | 0.0e+00 | **True** |
| [`rl-v2-validation-w050-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-validation-w050-20261110-cfe0305) | 103 | 8.24 | True | 33 / 34 | 528 | 33 / 7 | 0.0e+00 | **True** |
| [`rl-v2-validation-w100-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-validation-w100-20261110-cfe0305) | 103 | 8.24 | True | 33 / 34 | 528 | 33 / 7 | 0.0e+00 | **True** |

All checkpoints were trained on CPU, so there is no GPU-to-CPU difference to measure; CPU reload reproduced every
stored probe exactly (tolerance 1e-6).

**Continuation from downloaded checkpoints** (`scripts/rl_continuation_check.py`, new output directories, not part
of any performance summary): the smoke EEGNet snapshot at step 80 continued to step 120 (updates 59 -> 99, replay
80 -> 120) on the current code; the pilot EEGNet s11 snapshot at step 40,000 continued for 2,000 steps (updates
34,999 -> 36,999, replay 40,000 -> 42,000) in a clean checkout of its training commit `50888a9`, as the
source-hash gate requires. In both cases the restored replay transitions were unchanged, steps, updates and
replay size increased, and the new final snapshot verified.

## 9. Numbers across documents

The milestone report, this report and the decoder study quote different numbers for the same decoders and
baselines. All are correct for what they measure:

| quantity | milestone report | final report (this document) | decoder study | why they differ |
|---|---|---|---|---|
| EEGNet test AUROC (s02/s08/s09) | 0.613 ± 0.016 (five training seeds, mean ± SD) | 0.637 (seed 42) | 0.599 ± 0.014 (five controlled refits) | the milestone averages the five historical seeds; this report uses the seed-42 model, whose calibrated probabilities were frozen for RL; the decoder study retrains the recipe on all 7 training participants with fixed epochs (no validation-based selection), which is a different pipeline, not a reproduction |
| CBraMod test AUROC | 0.568 ± 0.007 (five seeds) | 0.573 (seed 42) | 0.564 ± 0.005 (selected frozen-backbone candidate; historical recipe refit 0.563 ± 0.006) | same as above |
| RL sparse, normalized success AUC | 0.58 ± 0.11 (3 seeds, 30 evaluation episodes) | 0.659 ± 0.131 (5 seeds, 100 evaluation goals) | - | the milestone figures come from the milestone's preliminary baseline runs (3 seeds, 30 evaluation episodes, earlier code); the final figures come from the frozen v2 final suite (5 seeds, 100 fixed evaluation goals, corrected runner of Section 4), so seed and evaluation noise and code differ |
| RL dense, normalized success AUC | 0.89 ± 0.08 (3 seeds, 30 episodes) | 0.915 ± 0.033 (5 seeds, 100 goals) | - | same as above |

**What the RL used.** Every RL run (v2, sensitivity, v3, preference) replays the **seed-42 calibrated
probabilities** from Release `eeg-stage1-20261101-bccbb8a` (profile `configs/rl_feedback.json`, byte-identical to
the historical profile). The five-seed means and the decoder-study refits were never used as RL feedback, and
`configs/rl_feedback_vnext.json` is an explicitly unused experimental profile. Source of the milestone values:
the milestone report as quoted in [`FINAL_V3_HANDOFF.md`](../FINAL_V3_HANDOFF.md), Section 2.6; seed-42 and five-seed
decoder values: [`eeg_results.md`](eeg_results.md); refits: [`decoder_improvement/final_report.md`](decoder_improvement/final_report.md),
Section 6.
