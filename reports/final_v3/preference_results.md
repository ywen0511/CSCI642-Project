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

<!--PREF_FINAL-->
