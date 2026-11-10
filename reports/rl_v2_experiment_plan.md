# RL protocol v2: experiment plan (declared before validation sweeps)

Status: **declared before any validation candidate was run and before any test-participant (s02/s08/s09)
feedback was used.** Pilot evidence that informed it is summarized in Section 7 and in
`reports/rl_v2_execution.md`. Changes after this point are listed as deviations in the final report.

## 1. Fixed protocol (protocol v2, unchanged semantics)

| item | value |
|---|---|
| schema / feedback rule | `csci642-rl-stage2/v2`, `distance_change_deadband`, `reward_center` 0.5 |
| task | Gymnasium-Robotics `FetchReach-v4`, sparse task reward, 50-step episodes, 5 cm goal tolerance |
| feedback | every 5 simulator steps (+ final partial segment); deadband epsilon 1 mm; EEG reward only on feedback events: `w_eeg * (0.5 - p_error)` |
| EEG source | frozen Release `eeg-stage1-20261101-bccbb8a`, decoder seed 42, calibrated `p_error`; validation s11/s13, final s02/s08/s09 |
| SAC | SB3 2.9.2 `MultiInputPolicy`, lr 3e-4, batch 256, buffer 100,000, learning_starts 5,000, gamma 0.99, tau 0.005, train_freq 1, gradient_steps 1, automatic entropy, MLP [256, 256], 1 torch thread |
| evaluation | **100** deterministic episodes on fixed goal seeds 20000-20099 (protocol default was 20; see Section 2) with sparse reward only (no EEG/dense shaping), at step 0, every 10,000 steps, and once for the final policy |
| snapshots | every 10,000 steps plus final: `model.zip`, `replay_buffer.pkl`, `rng.pt`, `probe.npz`, `metadata.json` |

## 2. Execution changes made before any formal run

1. **Configurable seed lists** (`validation_seeds`, `final_seeds`) are part of the protocol and therefore of the
   frozen-config identity. Empty, duplicate, negative, non-integer and overlapping lists are rejected; `rl.train`
   rejects validation/final seeds that are not declared. The suite planner, summary and freeze use these lists.
2. **Disjoint training goal sequences per RL seed.** The inherited wrapper seeded training episode *k* of RL seed
   *s* with `s + k`, so RL seeds 0/1/2 trained on the same goal sequence shifted by one episode, and SB3's own
   first-reset seed was honoured. Episode *k* now uses `s * 1,000,000 + k` (`training_episode_seed_stride`), the
   wrapper ignores externally supplied reset seeds, and the planner rejects any training seed range that would
   reach the evaluation goals (20000-20019) or the trajectory/render goals (30000-30999).
3. `rl.suite --prepare` / `--run-job NAME` run each planned job as its own cluster job (same commands, same output
   layout); `--execute` (serial) still works. The summary requires identical training-source hashes in every run.
4. Normalized AUC, analysis, trajectory and pilot-packaging tools were added (Sections 5-6).
5. Two inherited runner bugs found by the pilots were fixed: (a) the wrapper exposed its episode counter as
   `info["episode"]`, the key SB3 reserves for episode statistics, which crashed SB3 logging after 4 episodes
   (the 160-step smoke never reached it; the smoke now runs 250 steps); (b) at the last step SB3 evaluated the
   policy inside the rollout callback, then made one more gradient update, then the runner evaluated again, so
   two different policies were recorded at the same final step. The periodic evaluation at the final boundary
   is now skipped and every evaluation records `n_updates`.
6. **Evaluation episodes 20 -> 100** (goal seeds 20000-20099) for every validation and final run. Reason
   (pilot evidence, Section 7): final success saturates near 1.0 by 30-50k steps, a single evaluation episode is
   5 percentage points at 20 episodes, and evaluations one gradient update apart differed by up to 10 points.
   This changes evaluation precision only, not training.

No reward, label, sampling, SAC hyperparameter, evaluation or EEG-input semantics were changed.

## 3. Seeds and budget

- Validation RL seeds: **100, 101, 102** (all candidates).
- Final RL seeds: **0, 1, 2, 3, 4** (five seeds; chosen now because CPU capacity allows it; see Section 7).
- Common budget for every validation and final run: **150,000** environment steps (kept; Section 7).
- Run counts: validation candidate = 33 runs (sparse/dense/ideal x 3 seeds + 4 EEG conditions x 2 participants x
  3 seeds); final = 75 runs (3 shared conditions x 5 seeds + 4 EEG conditions x 3 participants x 5 seeds).

## 4. Validation candidates and selection rule

Candidates differ only in the EEG weight; dense weight is fixed at 10.0 (dense is a baseline, not tuned here).

| candidate | eeg_weight | dense_weight |
|---|---|---|
| `candidate_w010` | 0.1 | 10.0 |
| `candidate_w050` | 0.5 | 10.0 |
| `candidate_w100` | 1.0 | 10.0 |

One common EEG weight is used for EEGNet, CBraMod, both label-independent controls and ideal feedback.

Selection (validation only): highest mean **final-step success rate** over both main decoders (EEGNet and
CBraMod), both validation participants (s11, s13) and all three validation seeds (12 runs per candidate, equally
weighted). Ties (equal to 1e-9): lower mean final distance over the same 12 runs, then the smaller EEG weight.
All three candidates are run to completion, preserved and reported whichever is selected. Sparse, dense, ideal and
independent controls are reported for every candidate but do not enter the selection.

## 5. Metrics and aggregation

Per run (from the fixed-budget final policy, never the best checkpoint):
final-step success rate (primary), ever-success rate, mean final distance, mean sparse task return, and normalized
AUC of the success-rate curve: trapezoidal integration over all evaluation points from step 0 to the final step,
divided by the full budget (identical duplicate final records are removed; inconsistent duplicates are an error).

Aggregation: RL-seed mean and SD within each (condition, participant); for EEG-assisted conditions, unweighted
macro-average over participants of those seed means (SD over participants). Shared controls are trained once per
seed and are reported as mean/SD over seeds; they are not three participant observations. Paired differences are
reported per (participant, seed); shared controls are matched by seed. With three EEG participants no
population-level significance test is claimed; differences are described with their seed and participant spread.

Diagnostics from transition logs: neutral (deadband) fraction of segment ends, error-label fraction, feedback
events per episode, mean `p_error` and EEG reward by simulated label, AUROC of `p_error` vs simulated label within
feedback events, reward-component sums, all overall and per 10,000-step bin. Fixed decoder quality per participant
(AUROC, balanced accuracy at 0.5, Brier on calibrated `p_error`) is related descriptively to RL outcomes.

## 6. Curves, trajectories and examples (selection rule fixed in advance)

- Learning curves show every condition; EEG curves are participant macro-averages with participant-level panels;
  every run is also shown individually in the final-success plot.
- Representative trajectories and GIFs: for each condition, the final policy of RL seed **0**; EEG conditions use
  participant **s02** (first listed test participant). Goal seeds **30000, 30001, 30002** for all conditions.
  These examples are chosen by this rule, not by outcome; the complete per-run evaluation data are reported.

## 7. Pilot evidence used for this plan

All pilots used validation participants only (s11/s13), `phase=pilot`, EEG weight 0.5, and the pilot seeds
900/901 (disjoint from validation and final seeds). Machine-readable report: `reports/rl_v2_pilots/pilot_report.json`.

50k-step pilots (seed 900), final-step success / normalized AUC: sparse 0.40 / 0.34; dense 1.00 / 0.66;
ideal 1.00 / 0.55; EEGNet s11 0.50 / 0.25, s13 1.00 / 0.53; CBraMod s11 0.50 / 0.36, s13 0.95 / 0.51;
independent EEGNet s11 0.95 / 0.12; independent CBraMod s13 0.40 / 0.24. Single seeds; differences at this
budget are dominated by seed noise.

150k-step budget checks (seed 901): final success sparse 0.95, dense 1.00, ideal 1.00, EEGNet s11 1.00,
CBraMod s13 1.00; normalized AUC 0.875 / 0.913 / 0.903 / 0.880 / 0.847. Every condition reached >= 0.95 success
by 30-50k steps and stayed near the plateau, with occasional transient dips (CBraMod 0.45 at 120k, dense 0.70
at 140k). **Budget decision: keep 150,000 steps** (learning completes well inside the budget; the remaining
steps test plateau stability). Because final success saturates, learning speed (normalized AUC) and final
distance are expected to carry most of the between-condition differences; the selection rule itself is the one
specified in the handoff and is not changed.

Feedback mechanics (all pilots): 9.1-9.7 feedback events per 50-step episode; neutral (deadband) fraction
3-9 % of segment ends (rising as policies converge and hold still); error-label fraction 0.42-0.64. The AUROC
of replayed `p_error` against the simulated label within RL feedback events was 0.63-0.66 for EEGNet/CBraMod
(matching the decoders' validation AUROC), 0.49-0.50 for the label-independent controls and 1.0 for ideal.
Mean calibrated `p_error` was about 0.42-0.46 for non-error and 0.48-0.53 for error events, so the EEG reward
`w (0.5 - p)` is positive on average: part of its effect is a bonus at labeled (non-neutral) segments, which the
label-independent controls are designed to expose.

Throughput: 50-56 environment steps/s per CPU core (Intel Xeon Gold 6242, one torch thread), i.e. about
50 minutes per 150k-step run including evaluation. A 1-GPU benchmark of the same 20k-step dense run was
submitted to gc175 but could not start while this plan was written (all gc175 GPUs were in use by other jobs);
SAC with a 256x256 MLP and batch 256 is not expected to benefit, and CPU execution is used (Section 8).

## 8. Compute plan

One single-core LSF job per run on the CPU queue `cpu_ubuntu_wrn` (Ubuntu 24.04, Xeon Gold 6242), each in its
own output directory, launched through `rl.suite --prepare` / `--run-job`. Validation: 3 candidates x 33 runs =
99 runs, 14.85 M environment steps. Final: 75 runs, 11.25 M steps. Storage per 150k run is about
500 MB (measured on the 150k pilots: 16 snapshots of ~31 MB each, mostly the replay buffer, plus the transition
log), about 90 GB for validation and final together, within the remaining project quota (~270 GB free).
