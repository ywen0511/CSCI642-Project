# Route-preference extension: declared plan (committed before validation)

Status: **extension** (the milestone item "investigate whether using EEG signals would allow us to learn the
user's preference, in addition to learning the task"). Assignment: [FINAL_V3_HANDOFF.md](../../FINAL_V3_HANDOFF.md),
Section 7. Code: [`rl/preference.py`](../../rl/preference.py) (schema `csci642-rl-pref/v1`).

## 1. Task and geometry

- `FetchReach-v4`, sparse task reward (success only, no preference term), 50-step episodes, 5 cm success tolerance.
  Physics and the policy observation (`observation`, `achieved_goal`, `desired_goal`) are unchanged.
- Goal filter: after the seeded reset the goal is redrawn with the environment's own sampler and RNG until the
  horizontal start-goal distance `D >= 0.15 m`; the observation is recomputed by the
  environment's observation function. Used for training, evaluation (goal seeds 20000-20099) and trajectory
  goals (30000-30002).
- Virtual pillar (no collision): radius `R = 0.025 m`, infinite in z, centred at the x-y midpoint of
  start and goal.
- Simulated user preference "pass the pillar on the left": a step is a violation if at its end the gripper is
  inside the pillar or in the right-hand corridor (`0.10 <= a <= 0.90` and `l < 0`).
- Feedback labels (5-step segments, 1 mm deadband): any violation in the segment -> label 1; otherwise the v2
  distance-change rule. Sampling of replayed EEG trials, independent controls and probabilities exactly as in
  the main study (frozen profile `configs/rl_feedback.json`, decoder seed 42).
- Reward: centered EEG reward of protocol v3, `w (c_m - p_error)` with `configs/final_v3/centers.json`
  (validation participants only); ideal `p = label` with center 0.5 (rewards +-w/2). Sparse and dense
  (`10 x` distance decrease) are task-only baselines without preference information.

Pre-run geometry check (no learning): [`preference/geometry_corridor10_90/geometry_check.json`](preference/geometry_corridor10_90/geometry_check.json) (protocol geometry; the round-1 geometry check is in `preference/geometry_corridor25_75/`).

## 2. Budget and pilots

- Budget: 300,000 environment steps per run (raised from 150,000 by the declared pilot-gate fallback), SAC hyperparameters, evaluation every 10,000 steps on 100
  evaluation goals, checkpoints every 10,000 steps (all as in v2).
- Pilots (validation participants only, pilot seeds 900/901, w = 1.0): sparse, dense, ideal, EEGNet s11,
  CBraMod s13. Gate: ideal must reach a clearly higher final preferred-pass rate than sparse. Outcome and any
  gate change: [`preference/pilot_report.json`](preference/pilot_report.json) and `decisions.jsonl`.

**Pilot gate outcome (declared rule: ideal - sparse mean final preferred-pass >= 0.20 and ideal > sparse on both
seeds).** Round 1 (corridor a in [0.25, 0.75], 150k steps) failed: ideal 0.03 vs sparse 0.03. Before any round-2
result, three variants were declared in order of preference (G1 original geometry at 300k; G2 corridor
[0.10, 0.90] at 150k; G3 corridor [0.10, 0.90] at 300k), with the rule that if none passes, G3 is used and the gate
failure is reported. **None passed** (ideal ≈ 0.06-0.09 vs sparse ≈ 0.03-0.05, see the pilot report), so the
declared fallback G3 is the preference protocol: corridor a in [0.10, 0.90], 300,000 steps, pillar R = 0.025 m.
The radius was not changed: 0.025 m is the largest value that keeps the pillar edge at least 5 cm from start and
goal at the minimum distance 0.15 m. EEG pilots were never consulted. **Consequence declared here:** because even
noise-free (ideal) preference feedback did not teach the preference in pilots, a positive EEG result is not
expected; validation and final suites are run as declared to measure this, and the outcome is reported as is.

## 3. Candidates and selection rule (declared before any validation run)

| candidate | eeg_weight | dense_weight | config |
|---|---|---|---|
| `pref_w100` | 1.0 | 10.0 | `configs/final_v3/pref_candidates/pref_w100.json` |
| `pref_w200` | 2.0 | 10.0 | `configs/final_v3/pref_candidates/pref_w200.json` |
| `pref_w400` | 4.0 | 10.0 | `configs/final_v3/pref_candidates/pref_w400.json` |

- Validation: participants s11/s13, RL seeds 100-102, 33 runs per candidate (7 conditions as in v2).
- **Selection: highest mean final preferred-pass rate over the 12 EEGNet + CBraMod validation runs**; ties
  (|difference| <= 1e-9): higher mean task (final) success over the same runs, then the smaller weight.
  Implemented by `scripts/rl_select_candidate.py --metric preferred_pass` (committed before validation).
- Freeze with `rl.freeze`; commit the frozen protocol and evidence under `configs/final_v3/pref_frozen/`
  before the final suite; final: 75 runs (s02/s08/s09, RL seeds 0-4). All candidates are reported.

## 4. Metrics (evaluation without feedback or shaping)

Per evaluation episode: task success (final distance < 5 cm); **preferred pass** = success, no violation at any
step end, and every crossing of `a = 0.5` (linear interpolation between step-end positions, from the reset
position on) has `l > 0` (at least one crossing); violation-free rate; pillar-entry rate; right-corridor rate;
path length. Primary metric: final-policy preferred-pass rate; also its normalized AUC over the evaluation
curve. Secondary: task success, final distance, task-success AUC.

Analysis: same aggregation as v2 (RL-seed mean/SD within participant; participant macro-average for EEG
conditions; shared controls counted once). Paired differences per participant and seed: EEGNet - sparse,
CBraMod - sparse, EEGNet - CBraMod, decoder - independent control. Means, SDs, paired differences and sign
counts only; no population-level significance claims. Test participants s02/s08/s09 were already used in v2
and v3, so this is an extension evaluated on reused participants, not an untouched confirmatory test.

## 5. Cuts if compute becomes binding (Section 10 of the handoff)

Recorded in `decisions.jsonl` before dependent runs, in this order: preference validation reduced to
w = 1.0, 2.0; then (v3) validation reduced to w = 0.5, 1.0; then preference final seeds reduced to 0-2.
