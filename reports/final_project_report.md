# CSCI642 final project report: recorded EEG error feedback for robot reaching

Repository `ywen0511/CSCI642-Project`, execution branch `rl-v2-experiments` (from `rl-proposal-v2` at
`450205e`). Date 2026-11-10. Detailed documents: EEG stage [`eeg_results.md`](eeg_results.md); RL plan
[`rl_v2_experiment_plan.md`](rl_v2_experiment_plan.md), execution record [`rl_v2_execution.md`](rl_v2_execution.md),
results [`rl_v2_results.md`](rl_v2_results.md); artifact links in Section 8.

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

## 5. Limitations

Ceiling effect on final success (differences mainly in learning speed); three test participants and five RL
seeds (descriptive results only, no population-level significance); privileged distance-based labels; feedback
trials replayed with replacement (not independent human measurements); the near-ceiling selection metric chose
the weight with the lowest validation AUC, and lower weights were evaluated on validation participants only.

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

## 7. Optional route-preference extension

**Not performed.** The proposal makes it conditional on time after the main study. It would require a new
environment with an obstacle, left/right routes, collision detection and a new label definition, which do not
exist in FetchReach. The main study, its validation selection and the full Release delivery were completed
instead; no claim about route preference is made.

## 8. Artifacts

ARTIFACTS_PLACEHOLDER
