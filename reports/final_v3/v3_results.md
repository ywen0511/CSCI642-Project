# Protocol v3: centered EEG reward

**Status: post-hoc follow-up, with its own declared validation selection and freeze.** It was motivated by the v2
diagnostics. The test participants (s02/s08/s09) were already used in v2, so the v3 final suite is a follow-up
evaluation, not an untouched confirmatory test. The v2 primary result is unchanged.

Assignment: [FINAL_V3_HANDOFF.md](../../FINAL_V3_HANDOFF.md), Section 6. All tables, with every number traced to its
source: [`generated/v3_tables.md`](generated/v3_tables.md) (audit
[`generated/v3_tables_audit.json`](generated/v3_tables_audit.json)). Analyses: [`v3_validation/`](v3_validation/),
[`v3_final/`](v3_final/). Media: [`v3_media/`](v3_media/); the GIFs are Release assets.

## Definition

At labeled feedback events, `r_total = r_task + w (c_m - p_error)`.
- `c_m` is computed from the validation participants only. It is the midpoint between the class-conditional mean
  calibrated `p_error` over s11 + s13, decoder seed 42 ([`scripts/final_v3_compute_centers.py`](../../scripts/final_v3_compute_centers.py)
  → [`configs/final_v3/centers.json`](../../configs/final_v3/centers.json), committed `33e86d8` before any v3 run):
  - c_EEGNet = 0.4793
  - c_CBraMod = 0.4665
- `independent_<m>` uses `c_m`.
- `ideal` keeps `p = label` with center 0.5.

Everything else is identical to v2 (deadband labels, 5-step segments, sampling, seeds, SAC, evaluation).
Implementation: schema `csci642-rl-stage2/v3`, with `reward_center` per decoder and `centers_file` +
`centers_sha256` checked by `rl.train`. The v2 code path and its `reward_center == 0.5` check are unchanged.
Tests ([`rl/test_final_v3.py`](../../rl/test_final_v3.py)) cover:
- the reward equals `w (c_m - p)` for every EEG and independent condition, and ideal gives ±w/2;
- sparse/dense transition logs are bitwise identical between v2 and v3 configs;
- wrong, missing or tampered centers are rejected;
- v2 configs still load.

The v3 smoke passed ([`v3_smoke/smoke_report.json`](v3_smoke/smoke_report.json)).

## Selection (declared before validation)

Three candidates were run as complete 33-run validation suites (s11/s13, seeds 100-102), all from commit `a09817f`:
- w = 0.5
- w = 1.0
- w = 2.0

Rule: highest mean normalized success AUC over the 12 EEGNet + CBraMod runs (`scripts/rl_select_candidate.py
--metric auc`, committed in `a09817f`).

Result: AUC 0.755 for w = 0.5, 0.713 for w = 1.0 and 0.642 for w = 2.0, so **w = 0.5 was selected**. It was frozen with
its evidence ([`configs/final_v3/v3_frozen/`](../../configs/final_v3/v3_frozen/)) and committed (`41109cb`) before the
75-run final suite. All 174 v3 runs completed with no failures, and `rl.summarize` passed for every suite.

## Results (final suite, w = 0.5)

**1. The bonus is removed.**
- Mean EEG reward per run:
  - EEGNet: +153 in v3, versus +871 in v2.
  - CBraMod: +39 in v3, versus +979 in v2.
- Per feedback event it is now +0.005 and +0.001.
- The label-independent controls keep a small residual bonus (+291 and +131 per run). The residual exists because
  they sample the whole test-participant pool, whose mean `p_error` (0.458 / 0.457 in the diagnostics) is below
  `c_m`. The handoff anticipated a residual of this kind (about 0.15 × (μ1 − μ0)).
- The ideal condition's mean reward is positive because progress segments outnumber retreat segments (error-label
  fraction 0.455), not because of a bias.

**2. Both decoders are now ahead of sparse in learning speed.** Normalized success AUC, paired over 15
(participant, seed) pairs:

| comparison | AUC difference | better / worse |
|---|---|---|
| EEGNet - sparse | +0.029 | 9 / 6 |
| CBraMod - sparse | +0.031 | 9 / 6 |
| EEGNet - independent control | +0.064 | 10 / 5 |
| CBraMod - independent control | +0.048 | 11 / 4 |

- Final success is at ceiling: 0.988 for EEGNet and 0.967 for CBraMod, versus 0.942 for sparse.
- **EEGNet and CBraMod are now equal** (AUC difference -0.002, 7 / 8). In v2 EEGNet was ahead (+0.070), mainly
  because CBraMod received the largest bias.

**3. Direct comparison with v2 (w = 1.0), same participants and seeds.**
- **CBraMod's AUC rises from 0.571 to 0.690** (+0.119, 12 of 15 pairs better).
- **EEGNet's AUC rises from 0.641 to 0.688** (+0.047, 9 / 6).
- Sparse and dense are identical in both suites, as expected.

**4. The independent controls and sparse.**
- In v2 the independent controls were behind sparse in AUC: -0.063 (4 / 11) and -0.056 (5 / 10).
- Under v3 they are much closer: -0.034 (8 / 7) and -0.016 (7 / 8).

This fits the hypothesis that the label-free bonus slowed learning in v2. It is not a causal proof, because v3 also
changed the weight (0.5 instead of 1.0).

**5. Two qualifications.**
- **Weight.** In the post-hoc sensitivity suites, v2 at w = 0.5 already gave larger decoder-vs-sparse AUC gaps
  (+0.060 and +0.039) than v3 does. The two effects (centering and a smaller weight) are therefore not separable
  with these suites. See [`sensitivity_results.md`](sensitivity_results.md).
- **Ideal.** v3 `ideal` uses p = label with center 0.5, the same as v2. At w = 0.5 its runs are identical to the v2
  sensitivity w = 0.5 `ideal` runs (AUC 0.762). The ideal oracle is still below dense shaping (0.915).

**Trajectories** (RL seed 0, EEG participant s02, goals 30000-30002; all 75 final policies on the same goals:
[`v3_media/trajectory_table.md`](v3_media/trajectory_table.md)):
- Every condition reaches the goals.
- Dense policies take long, curved paths (median path / straight line 1.59).
- The EEG conditions are between sparse and dense (1.23 for EEGNet, 1.34 for CBraMod).

Interpretation limits:
- 3 test participants × 5 RL seeds.
- Means, SDs and sign counts only; no population-level significance claims.
- Replayed, not live, feedback.
- Participants reused from v2.
- Decoder quality vs RL outcome relations (`v3_final/decoder_vs_rl.csv`) are descriptive.
