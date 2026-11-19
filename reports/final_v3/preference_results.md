# Route-preference extension: can replayed ErrP feedback teach a preference the task reward lacks?

**Status: extension** (the milestone item "investigate whether using EEG signals would allow us to learn the
user's preference, in addition to learning the task"), with its own geometry check, pilots and declared gate, a
declared plan committed before validation, a validation selection and a freeze. The test participants s02/s08/s09
were already used in v2 and v3, so the final suite is an evaluation on reused participants, not an untouched
confirmatory test.

Assignment: [FINAL_V3_HANDOFF.md](../../FINAL_V3_HANDOFF.md), Section 7. Declared plan:
[`preference_plan.md`](preference_plan.md). Code: [`rl/preference.py`](../../rl/preference.py). All tables, with every
number traced to its source: [`generated/preference_tables.md`](generated/preference_tables.md) (audit
[`generated/preference_tables_audit.json`](generated/preference_tables_audit.json)). Analyses:
[`preference/validation/`](preference/validation/), [`preference/final/`](preference/final/). Media:
[`preference/media/`](preference/media/); the GIFs are Release assets.

## Task

- FetchReach-v4 with its sparse task reward: success only, with no preference term.
- Goals are filtered to at least 15 cm horizontal start-goal distance.
- A virtual pillar (radius 2.5 cm, no collision) sits at the start-goal midpoint. The simulated user wants the gripper
  to pass it **on the left**.
- A step is a violation if the gripper is inside the pillar or in the right-hand corridor (along-axis fraction
  `a` in [0.10, 0.90], lateral offset `l < 0`).
- Any violation in a 5-step feedback segment labels the segment as an error. Otherwise the v2 distance-change rule
  applies.
- EEG trials, the independent controls and the centered reward `w (c_m - p_error)` are as in protocol v3.
- Sparse and dense are task-only baselines with no preference information.
- **Preferred pass** (primary metric, final policy, 100 evaluation goals) = task success, with no violation at any
  step, and every crossing of `a = 0.5` made on the left.

The geometry check (no learning) passed on all 100 evaluation and 3 trajectory goals: the straight-line controller
violates on every goal, and the left-detour controller succeeds with no violation
([`preference/geometry_corridor10_90/geometry_check.json`](preference/geometry_corridor10_90/geometry_check.json)).
The smoke passed ([`preference/smoke/`](preference/smoke/)).

## Pilots and the declared gate

The gate was declared before any pilot result: ideal − sparse mean final preferred pass ≥ 0.20, and ideal > sparse on
both pilot seeds. Only the ideal and sparse pilots were used for it.
- Round 1 (corridor [0.25, 0.75], 150k steps) failed.
- Three round-2 variants were declared in order before any round-2 result. None passed:
  - G1, original geometry at 300k steps: ideal 0.09 vs sparse 0.06.
  - G2, corridor [0.10, 0.90] at 150k steps: 0.06 vs 0.03.
  - G3, corridor [0.10, 0.90] at 300k steps: 0.065 vs 0.035.
- The declared fallback, G3, became the protocol. Pilot tables:
  [`generated/preference_tables.md`](generated/preference_tables.md); raw data:
  [`preference/pilot_report.json`](preference/pilot_report.json).

At w = 1.0 even noise-free feedback did not teach the preference in pilots. The plan therefore stated in advance
that a positive EEG result was not expected.

## Validation and selection (declared before validation)

Three candidates were each run as a complete 33-run suite from `ae633ea`: s11/s13, seeds 100-102, 300k steps.
- w = 1.0
- w = 2.0
- w = 4.0

The first attempt was killed externally and rerun unchanged; see [`execution.md`](execution.md).

Rule: the highest mean final preferred pass over the 12 EEGNet + CBraMod runs.

| candidate | preferred pass (12 runs) | task success (12 runs) | success AUC (12 runs) |
|---|---|---|---|
| w = 1.0 | 0.0517 | 0.9792 | 0.5052 |
| w = 2.0 | **0.0533** | 0.9867 | 0.4371 |
| w = 4.0 | 0.0483 | 0.8325 | 0.2569 |

**w = 2.0 was selected.** The three candidates differ by at most 0.005 preferred pass, which is well within seed
noise, so the rule effectively picked among equivalent candidates. The selected candidate was frozen with its
evidence ([`configs/final_v3/pref_frozen/`](../../configs/final_v3/pref_frozen/)) and committed (`1aa86a3`) before the
final suite.

**What the validation suites show** (per-condition tables in the generated file):
- **About 5 % is the floor.** Every EEG and control condition, and sparse and dense too, stays at about 0.04-0.07
  preferred pass. Pillar entry is about 0.8-0.9. The policies learn the straight path to the goal, through the pillar.
- **The ideal oracle learns the preference only at the largest weight.**
  - At w = 4.0, ideal reaches **0.503 ± 0.172** preferred pass (pillar entry falls to 0.48, right corridor to 0.20),
    against 0.057 for sparse.
  - At w = 1.0 and 2.0, ideal stays at 0.06-0.07.
  - So the preference is learnable from this feedback mechanism, but only with error-free labels and a feedback
    weight large enough to outweigh the extra steps of the detour.
- **Decoded EEG feedback did not teach the preference at any weight.** EEGNet and CBraMod stay at 0.048-0.057.
- **Larger weights hurt the task.** At w = 4.0, EEG task success falls to 0.80-0.87 and AUC to about 0.26.

## Final suite (w = 2.0, s02/s08/s09, RL seeds 0-4, 300k steps)

All 75 runs completed from `1aa86a3`, with no failures, and `rl.summarize` passed.

| condition | runs | preferred pass | violation-free | pillar entry | right corridor | task success | success AUC |
|---|---|---|---|---|---|---|---|
| sparse | 5 | 0.044 ± 0.026 | 0.056 | 0.848 | 0.650 | 0.838 ± 0.183 | 0.349 ± 0.196 |
| dense | 5 | 0.052 ± 0.024 | 0.052 | 0.838 | 0.626 | 0.998 ± 0.004 | 0.936 ± 0.021 |
| EEGNet | 15 | 0.048 ± 0.010 | 0.091 | 0.833 | 0.655 | 0.931 ± 0.049 | 0.380 ± 0.113 |
| CBraMod | 15 | 0.045 ± 0.015 | 0.071 | 0.824 | 0.687 | 0.867 ± 0.111 | 0.281 ± 0.133 |
| ideal | 5 | 0.082 ± 0.049 | 0.082 | 0.888 | 0.518 | 0.996 ± 0.009 | 0.601 ± 0.056 |
| independent EEGNet | 15 | 0.048 ± 0.012 | 0.155 | 0.735 | 0.610 | 0.749 ± 0.203 | 0.243 ± 0.101 |
| independent CBraMod | 15 | 0.045 ± 0.013 | 0.074 | 0.859 | 0.627 | 0.903 ± 0.047 | 0.318 ± 0.071 |

Paired differences in final preferred pass over 15 (participant, seed) pairs (better / worse / equal):

| comparison | preferred pass | preferred-pass AUC | task success | success AUC |
|---|---|---|---|---|
| EEGNet - sparse | +0.004 (7 / 5 / 3) | +0.001 (7 / 8) | +0.093 (8 / 4 / 3) | +0.031 (6 / 9) |
| CBraMod - sparse | +0.001 (5 / 8 / 2) | -0.005 (3 / 12) | +0.029 (10 / 4 / 1) | -0.068 (5 / 10) |
| EEGNet - CBraMod | +0.003 (10 / 4 / 1) | +0.007 (12 / 3) | +0.064 (6 / 7 / 2) | +0.099 (10 / 5) |
| EEGNet - independent control | -0.000 (6 / 6 / 3) | +0.008 (10 / 5) | +0.183 (7 / 4 / 4) | +0.136 (12 / 3) |
| CBraMod - independent control | +0.001 (7 / 7 / 1) | -0.001 (4 / 11) | -0.036 (5 / 7 / 3) | -0.037 (5 / 10) |

**Answer.** With these decoders, replayed ErrP feedback did **not** teach the route preference.
- Every EEG condition and both label-independent controls end at the same preferred-pass rate as sparse
  (0.044-0.048, the rate at which an unconstrained policy happens to pass on the left).
- The paired differences are within ±0.005.
- The policies keep entering the pillar (about 0.83).
- Noise-free feedback at the selected weight (ideal, w = 2.0) also stays low: 0.082, ahead of sparse in 3 of 5 seeds.
- In validation, only ideal at w = 4.0 learned the preference (0.50).
- So two things block the EEG conditions:
  - the noise of single-trial decoding: event AUROC 0.65 for EEGNet and 0.57 for CBraMod, against 1.0 for ideal;
  - the fact that, at these weights, the per-event preference signal is weaker than the task reward's push toward
    the short straight path.

Secondary observations (descriptive):
- **Penalty mode.** Violations make most labels errors (error-label fraction 0.74-0.78 for the EEG conditions), so
  the centered EEG reward is mostly a penalty here: about -620 per run for EEGNet and -770 for CBraMod. The
  label-independent controls draw from the whole test pool and so receive a bonus (+2,510 and +1,084), because their
  mean `p_error` (0.458 / 0.457) is below `c_m`.
- **Task learning.** It is slower than in the main task for every condition except dense (300k steps; sparse AUC
  0.349), because the goal filter only keeps distant goals.
- **EEGNet vs CBraMod.** EEGNet learned the task faster than CBraMod (AUC +0.099, 10 / 5) and faster than its own
  independent control (+0.136, 12 / 3). The ordering matches the decoders' event AUROC, as in v2.

**Trajectories** (RL seed 0, EEG participant s02, goals 30000-30002, pillar and right corridor drawn on the x-y plots
and the pillar drawn in the GIFs; all 75 final policies on the same goals:
[`preference/media/trajectory_table.md`](preference/media/trajectory_table.md)):
- Of the 21 representative episodes, only one is a preferred pass (CBraMod, goal 30002).
- Across all 75 policies on these 3 goals, preferred pass is 0.09-0.20 per condition, with no condition standing
  out.

Interpretation limits:
- 3 test participants × 5 RL seeds.
- Means, SDs and sign counts only.
- Reused participants.
- Replayed feedback and privileged geometric labels.
- A virtual pillar with no collision.
- The declared pilot gate failed, so this task is one in which even noise-free preference feedback is weak at the
  declared weights.
