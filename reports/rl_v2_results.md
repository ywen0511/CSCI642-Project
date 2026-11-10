# RL protocol v2 results: does replayed EEG error feedback speed up robot reaching?

Date: 2026-11-10. Branch `rl-v2-experiments`. All numbers come from executed runs; full tables are generated in
[`rl_v2_final/results_tables.md`](rl_v2_final/results_tables.md) from the CSV/JSON analysis outputs in
[`rl_v2_final/`](rl_v2_final/) and [`rl_v2_validation/`](rl_v2_validation/). Plan declared before validation:
[`rl_v2_experiment_plan.md`](rl_v2_experiment_plan.md). Execution details: [`rl_v2_execution.md`](rl_v2_execution.md).

## 1. Short answer

On FetchReach with SAC, replaying the frozen EEG decoders' error probabilities as an extra reward **did not make
learning faster or more reliable than the sparse task reward alone**, at the frozen EEG weight of 1.0:

- Every condition reached a high final success rate (0.94-1.00), so final-step success sits near its ceiling.
- Learning speed (normalized AUC of the success curve) separated the conditions: dense shaping 0.915 and ideal
  (noise-free) error feedback 0.764 were clearly faster than sparse 0.659; EEGNet feedback 0.641 and CBraMod
  feedback 0.571 were not.
- EEGNet feedback was faster than CBraMod feedback in 10 of 15 participant-seed pairs (mean AUC +0.070), in line
  with EEGNet's better decoding (test AUROC 0.637 vs 0.573), but the difference is within the seed spread and
  final success is equal (0.968 vs 0.952).
- EEGNet feedback beat its own label-independent control in 12 of 15 pairs (AUC +0.046): the label information
  in EEGNet's probabilities helped relative to an equally gated, equally biased random replay, but not enough
  to beat plain sparse reward. CBraMod feedback did not beat its control (AUC -0.031, 9 of 15 pairs).

A stronger decoder gave a weakly better policy-learning signal; neither decoder was good enough to help.

## 2. Setup actually executed

| item | value |
|---|---|
| task | Gymnasium-Robotics `FetchReach-v4`, sparse reward (-1 outside / 0 inside 5 cm), 50-step episodes |
| agent | SB3 2.9.2 SAC, MLP 256x256, lr 3e-4, batch 256, buffer 100k, 5k random warm-up steps, automatic entropy |
| budget | 150,000 environment steps per run, every run from scratch, no resumption |
| feedback | every 5 steps (+ final partial segment); label = sign of the segment's distance change beyond a 1 mm deadband (0 progress, 1 retreat, none inside the deadband); EEG reward `w (0.5 - p_error)` only on labeled segments |
| EEG source | frozen calibrated `p_error`, decoder seed 42, Release `eeg-stage1-20261101-bccbb8a`; one participant per policy; trials sampled with replacement from that participant's label pool |
| conditions | sparse, dense (`10 x` distance decrease), ideal (`p = label`), EEGNet, CBraMod, and label-independent EEGNet/CBraMod controls (same gate and transform, trial drawn without regard to the label) |
| validation | participants s11/s13, RL seeds 100-102, EEG weights 0.1/0.5/1.0 (3 x 33 runs) |
| final | participants s02/s08/s09, RL seeds 0-4, frozen EEG weight 1.0 (75 runs) |
| evaluation | 100 fixed goals (seeds 20000-20099), deterministic policy, sparse reward only, every 10k steps |
| compute | 189 SAC runs, 167 CPU core-hours (one core per run, ~56 min per 150k run) |

"Held out" has two meanings here. The final participants' EEG trials were never used to train or calibrate the
EEG decoders, and their feedback pools were not used for RL protocol selection. They **are** used as feedback
during final RL training, as intended. Robot evaluation uses fresh goals and no EEG or dense shaping.

## 3. Validation and selection (validation participants only)

Selection rule (declared before validation): mean final success over the 12 EEGNet + CBraMod runs; ties by
final distance, then smaller weight.

| candidate | EEG weight | selection success | final distance (m) | AUC |
|---|---|---|---|---|
| `candidate_w010` | 0.1 | 0.980 | 0.0278 | 0.762 |
| `candidate_w050` | 0.5 | 0.986 | 0.0254 | 0.748 |
| `candidate_w100` | 1.0 | **0.997** | 0.0230 | 0.731 |

`candidate_w100` was selected and frozen (`configs/rl_v2/frozen/`). The differences in the selection metric are
small (0.98-1.00, all near ceiling), and the selected weight had the *lowest* AUC of the three; the rule was
applied as declared and is reported as such. In validation, sparse and dense runs (which do not use the EEG
weight) were bit-for-bit identical across the three candidate suites, a check of run-to-run determinism on the
CPU nodes. Validation already showed the pattern seen in the final runs: dense (AUC 0.904) and ideal
(0.81-0.86) learn fastest; decoded EEG (0.70-0.77) is close to sparse (0.748); at w = 1.0 the label-independent
controls were slowest (0.64-0.66).

## 4. Final comparison (test participants)

| condition | runs | final success | ever success | final distance (m) | sparse return | normalized AUC |
|---|---|---|---|---|---|---|
| sparse | 5 | 0.942 ± 0.083 | 0.980 | 0.0301 ± 0.0112 | -4.44 | 0.659 ± 0.131 |
| dense | 5 | 1.000 ± 0.000 | 1.000 | 0.0214 ± 0.0078 | -1.77 | 0.915 ± 0.033 |
| eegnet | 15 | 0.968 ± 0.034 | 0.995 | 0.0233 ± 0.0026 | -3.21 | 0.641 ± 0.041 |
| cbramod | 15 | 0.952 ± 0.063 | 0.983 | 0.0229 ± 0.0050 | -4.04 | 0.571 ± 0.115 |
| ideal | 5 | 0.986 ± 0.031 | 1.000 | 0.0213 ± 0.0080 | -2.65 | 0.764 ± 0.097 |
| independent_eegnet | 15 | 0.960 ± 0.049 | 0.985 | 0.0241 ± 0.0066 | -3.55 | 0.595 ± 0.036 |
| independent_cbramod | 15 | 0.940 ± 0.078 | 0.972 | 0.0385 ± 0.0280 | -4.54 | 0.602 ± 0.038 |

Per-participant results: see `rl_v2_final/results_tables.md` and `by_condition_participant.csv`.

Figures: [`curves_success.png`](rl_v2_final/curves_success.png) (main and diagnostic conditions),
[`curves_success_by_participant.png`](rl_v2_final/curves_success_by_participant.png),
[`curves_distance.png`](rl_v2_final/curves_distance.png),
[`final_success_runs.png`](rl_v2_final/final_success_runs.png) (every run as a dot),
[`feedback_availability.png`](rl_v2_final/feedback_availability.png).

### Paired comparisons

| comparison | metric | pairs | mean difference | SD | first better / worse / equal |
|---|---|---|---|---|---|
| eegnet - sparse | success_rate | 15 | +0.026 | 0.120 | 8 / 4 / 3 |
| eegnet - sparse | normalized_auc_success | 15 | -0.017 | 0.125 | 8 / 7 / 0 |
| cbramod - sparse | success_rate | 15 | +0.010 | 0.126 | 7 / 3 / 5 |
| cbramod - sparse | normalized_auc_success | 15 | -0.087 | 0.182 | 6 / 9 / 0 |
| eegnet - cbramod | success_rate | 15 | +0.016 | 0.143 | 3 / 4 / 8 |
| eegnet - cbramod | normalized_auc_success | 15 | +0.070 | 0.204 | 10 / 5 / 0 |
| dense - sparse | success_rate | 5 | +0.058 | 0.083 | 3 / 0 / 2 |
| dense - sparse | normalized_auc_success | 5 | +0.256 | 0.151 | 5 / 0 / 0 |
| ideal - sparse | success_rate | 5 | +0.044 | 0.099 | 3 / 1 / 1 |
| ideal - sparse | normalized_auc_success | 5 | +0.105 | 0.211 | 4 / 1 / 0 |
| eegnet - independent_eegnet | success_rate | 15 | +0.008 | 0.115 | 4 / 3 / 8 |
| eegnet - independent_eegnet | normalized_auc_success | 15 | +0.046 | 0.100 | 12 / 3 / 0 |
| cbramod - independent_cbramod | success_rate | 15 | +0.012 | 0.114 | 5 / 2 / 8 |
| cbramod - independent_cbramod | normalized_auc_success | 15 | -0.031 | 0.192 | 9 / 6 / 0 |

How to read it: with 3 participants and 5 RL seeds, a consistent sign across most pairs is the strongest evidence
available. Dense > sparse (5/5) and ideal > sparse (4/5) are consistent. EEGNet vs sparse (8/15) and CBraMod vs
sparse (6/15) are not. No population-level significance test is claimed.

## 5. Why the EEG feedback did not help: diagnostics

| condition | neutral fraction | error-label fraction | feedback events / episode | EEG reward per run | mean p_error (label 0 / 1) | AUROC of p_error vs label |
|---|---|---|---|---|---|---|
| sparse | 0.076 | 0.514 | 0.00 | 0 | n/a / n/a | n/a |
| dense | 0.089 | 0.446 | 0.00 | 0 | n/a / n/a | n/a |
| eegnet | 0.072 | 0.512 | 9.28 | 871 | 0.434 / 0.503 | 0.650 |
| cbramod | 0.068 | 0.548 | 9.32 | 979 | 0.443 / 0.484 | 0.574 |
| ideal | 0.084 | 0.448 | 9.16 | 1441 | 0.000 / 1.000 | 1.000 |
| independent_eegnet | 0.069 | 0.527 | 9.31 | 1172 | 0.459 / 0.458 | 0.499 |
| independent_cbramod | 0.069 | 0.532 | 9.31 | 1197 | 0.457 / 0.457 | 0.500 |

- **Feedback is available and correctly gated.** About 9.3 labeled feedback events per 50-step episode; 7-9% of
  segment ends fall in the deadband and give no feedback; error labels are 45-55% of labeled segments.
- **The RL agent receives exactly the decoders' signal quality.** Within RL feedback events, `p_error` separates
  retreat from progress with AUROC 0.650 (EEGNet) and 0.574 (CBraMod), matching the decoders' fixed test AUROC
  (0.637 and 0.573 pooled). Ideal feedback has AUROC 1.0; the independent controls 0.50, as designed.
- **The signal is weak relative to the reward scale.** With w = 1, the expected EEG reward difference between a
  retreat and a progress segment is only `p1 - p0` = 0.07 (EEGNet) and 0.04 (CBraMod), against 1.0 for ideal
  feedback; the per-event noise (`p_error` SD about 0.13) is larger than that gap.
- **Calibrated probabilities are below 0.5 on average**, so EEG reward is mostly a positive bonus at labeled
  segments (about +900 to +1,200 per run). The independent controls receive the same kind of bonus without label
  information; they were slower than sparse (AUC about 0.60), which suggests the bonus itself slows learning a
  little and the label information in EEGNet's probabilities partly compensates.

## 6. Relating decoder quality to robot learning

| condition | participant | decoder AUROC | decoder balanced accuracy | decoder Brier | RL final success | RL AUC |
|---|---|---|---|---|---|---|
| cbramod | s02 | 0.553 | 0.557 | 0.242 | 0.980 | 0.446 |
| independent_cbramod | s02 | 0.553 | 0.557 | 0.242 | 1.000 | 0.646 |
| cbramod | s08 | 0.536 | 0.519 | 0.249 | 0.996 | 0.673 |
| independent_cbramod | s08 | 0.536 | 0.519 | 0.249 | 0.968 | 0.577 |
| cbramod | s09 | 0.632 | 0.601 | 0.242 | 0.880 | 0.595 |
| independent_cbramod | s09 | 0.632 | 0.601 | 0.242 | 0.852 | 0.584 |
| eegnet | s02 | 0.639 | 0.590 | 0.211 | 0.978 | 0.594 |
| independent_eegnet | s02 | 0.639 | 0.590 | 0.211 | 1.000 | 0.555 |
| eegnet | s08 | 0.614 | 0.557 | 0.231 | 0.930 | 0.672 |
| independent_eegnet | s08 | 0.614 | 0.557 | 0.231 | 0.974 | 0.608 |
| eegnet | s09 | 0.694 | 0.619 | 0.240 | 0.996 | 0.658 |
| independent_eegnet | s09 | 0.694 | 0.619 | 0.240 | 0.906 | 0.623 |

Descriptively, the better decoder (EEGNet) gave slightly faster RL learning on average, and per participant the
RL outcome does not track decoder AUROC in a simple way (e.g. CBraMod s09 has CBraMod's best decoder AUROC, 0.632,
but its lowest final success). With three participants this cannot establish a relationship, and decoder-score
correlations are not causal evidence.

## 7. Trajectories

Selection rule (declared in the plan before any final run): RL seed 0 for every condition, participant s02 for
EEG conditions, goal seeds 30000-30002 (outside the training and evaluation goal ranges).
[`rl_v2_media/trajectories_representative/trajectories.png`](rl_v2_media/trajectories_representative/trajectories.png)
shows the gripper paths (x-y and x-z projections, 5 cm goal region dotted); all seven representative policies
reach all three goals. GIFs of the same rollouts (3 episodes each) are Release assets (`gif__*.gif` in the final Release); they
were rendered by replaying the recorded actions in a separate process, and every replay matched the recorded
gripper positions exactly (maximum deviation 0.0 m).

To avoid relying on examples, every final policy (75 runs) was rolled out on the same three goals
([`trajectories_all_final_runs/trajectories.json`](rl_v2_media/trajectories_all_final_runs/trajectories.json),
summary by `scripts/rl_trajectory_table.py`):

| condition | episodes (runs x 3 goals) | final success | final distance (m) | mean path length (m) | median path / straight-line distance |
|---|---|---|---|---|---|
| sparse | 15 | 1.000 | 0.0279 | 0.273 | 1.18 |
| dense | 15 | 1.000 | 0.0170 | 0.528 | 1.59 |
| eegnet | 45 | 1.000 | 0.0190 | 0.302 | 1.19 |
| cbramod | 45 | 0.978 | 0.0218 | 0.201 | 1.24 |
| ideal | 15 | 1.000 | 0.0164 | 0.152 | 1.09 |
| independent_eegnet | 45 | 0.956 | 0.0219 | 0.275 | 1.17 |
| independent_cbramod | 45 | 0.956 | 0.0289 | 0.187 | 1.21 |

Ideal feedback, which penalizes every retreat, produced the most direct paths (path 1.09 x the straight-line
distance); dense shaping produced the longest paths, mostly from hovering and oscillating around the goal after
arrival; decoded EEG paths were similar to sparse ones.

## 8. Limitations

- **Ceiling effect.** FetchReach is easy for SAC at 150k steps; final success saturates, and differences show up
  mainly in learning speed. A harder task would be needed to test whether EEG feedback can change final
  performance.
- **Replayed, not live, feedback.** Recorded EEG probabilities from a different task (the original robot dataset)
  are attached to simulated distance-change labels. This is neither live human evaluation nor offline RL: SAC
  interacts with the simulator online and receives offline recorded feedback. Replayed trials are not new human
  measurements, and the feedback timing (every 0.2 s of simulated time, whereas an EEG epoch spans 2 s) is an
  abstract schedule, not a model of acquisition latency.
- **Small samples.** 3 test participants, 5 RL seeds; results are descriptive.
- **Selection on a near-ceiling metric.** The declared primary metric could barely separate the candidates; the
  frozen weight (1.0) had the lowest validation AUC. Results at weights 0.1/0.5 are only available for validation
  participants.
- **Privileged labels.** The error label is a distance-based proxy computed from the simulator; it is not a
  measured human judgement.
- **Not a Kim replication.** Different robot, task, reward schedule and decoders.
- The optional route-preference extension was not performed (Section 9 of the final project report).
