# Execute: final-round follow-up (weight sensitivity, reward centering, route preference) and complete delivery

## 1. Assignment, authority and finish line

Work in **ywen0511/CSCI642-Project**. This document is the complete assignment for the executing agent. The user only
needs to provide its link. Carry the work through implementation, experiments, analysis, reports, code push, Release
upload and fresh-download verification. Do not stop after preparing scripts or a run plan, and do not hand routine work
(cloning, downloading inputs, running commands, uploading) back to the user. All written deliverables are in English.

Entry points (resolve every relative path in this document against the planning branch):

- Planning branch (this document): https://github.com/ywen0511/CSCI642-Project/tree/final-v3
- RL v2 execution branch (base): https://github.com/ywen0511/CSCI642-Project/tree/rl-v2-experiments
- Decoder-improvement execution branch (to merge): https://github.com/ywen0511/CSCI642-Project/tree/decoder-improvement-v1-experiments
- Frozen EEG feedback: Release `eeg-stage1-20261101-bccbb8a`, profile `configs/rl_feedback.json` (unchanged).
- Existing RL Releases: `rl-v2-{smoke,pilots,validation-w010,validation-w050,validation-w100,final}-20261110-*`
  (listed in `reports/final_project_report.md`, Section 8).

**Scope rule.** This round *adjusts and completes* the existing study. It reuses the existing code, frozen EEG
probabilities, participants, seeds, budget and evaluation protocol. It does **not** retrain or re-select decoders,
does not change the participant split, and does not add new research directions beyond Sections 5-8. Items explicitly
out of scope are listed in Section 11.

**Finish line.** The task is complete only when every item of the checklist in Section 13 is ticked: all runs done,
reports written, code pushed to the execution branch, all checkpoints/logs uploaded as new Releases, and every new
Release verified from a fresh download. A negative scientific result is an acceptable outcome; an incomplete delivery
is not. If something is genuinely impossible (missing permission, hardware, quota), preserve progress, push what
exists, and report the exact blocker; never claim an upload or run that did not complete.

## 2. Why this round exists (what the previous rounds found)

Read these before starting: `reports/rl_v2_results.md`, `reports/final_project_report.md`,
`reports/rl_v2_experiment_plan.md`, `reports/rl_v2_execution.md`, and on the decoder branch
`reports/decoder_improvement/final_report.md`.

1. **Main RL result (v2, frozen EEG weight w = 1.0, 75 final runs).** Final success is at ceiling (0.94-1.00) for
   every condition. Normalized success AUC: dense 0.915 > ideal 0.764 > sparse 0.659 ≈ EEGNet 0.641 > CBraMod 0.571.
   EEG feedback did not beat sparse.
2. **Weight selection weakness.** The declared rule selected by final success, which was at ceiling in validation
   (0.980 / 0.986 / 0.997 for w = 0.1 / 0.5 / 1.0). The selected w = 1.0 had the *lowest* validation AUC
   (0.731 vs 0.762 for w = 0.1). Weights 0.1 and 0.5 were never run on the final participants.
3. **Reward-bias confound.** The EEG reward is `w (0.5 - p_error)`. Calibrated `p_error` is below 0.5 on average
   (RL feedback events: EEGNet 0.434 / 0.503, CBraMod 0.443 / 0.484 for label 0 / 1), so EEG reward is mostly a
   positive bonus (+900 to +1,200 per run). The label-independent controls received the same bonus without label
   information and were slower than sparse (AUC ≈ 0.60). The bonus itself is a plausible cause of slower learning.
4. **Milestone commitment not done.** The milestone report promised to "investigate whether using EEG signals would
   allow us to learn the user's preference, in addition to learning the task". The route-preference extension was
   marked "not performed" (`reports/final_project_report.md`, Section 7).
5. **Decoder improvement is finished and negative.** The milestone's plan to freeze the CBraMod backbone or fine-tune
   only its final layers was executed on `decoder-improvement-v1-experiments`: no change improved held-out decoding
   (CBraMod ≈ 0.564, EEGNet 0.599 test AUROC over five refits). Do not repeat it. Its results must be merged and
   reported (Section 4).
6. **Report numbers are inconsistent across documents.** The milestone report gives decoder results as the five-seed
   mean (EEGNet 0.613 ± 0.016, CBraMod 0.568 ± 0.007) and RL baselines from 3 seeds / 30 evaluation episodes
   (sparse AUC 0.58 ± 0.11, dense 0.89 ± 0.08); the final report uses decoder seed 42 (0.637 / 0.573) and
   5 seeds / 100 episodes (sparse 0.659, dense 0.915); the decoder branch adds controlled refits (0.599 / 0.564).
   All three are legitimate but must be reconciled in one table (Section 9).

## 3. Workspace, environment and preflight

- Reuse the existing project workspace and cluster environment where available (historical hints, verify rather than
  assume: LSF; CPU queue `cpu_ubuntu_wrn`, Intel Xeon Gold 6242, one core per run; GPU queue `gpu_large_shared`;
  RL environment `venv/rl` with the lock `rl/requirements-lock.cluster.txt`). If no workspace exists, obtain the source
  through GitHub yourself; attach it to the real commit history (no disconnected history).
- **Compute.** All RL work in this round is CPU-only (SAC with a 256x256 MLP; prior runs: ~56 min per 150k-step run on
  one core, ~50-56 env steps/s). A GPU is available to you but is not required; do not spend effort on GPU RL unless a
  measured benchmark shows a benefit, and if you use it, record it and keep every suite on one device type.
- **Use the exact historical RL environment** (same lock) for Sections 5 and 6 so that new runs are comparable with
  the existing v2 runs. Record Python/package versions, hardware, scheduler and wall times.
- **Storage.** A 150k-step run with all periodic snapshots is ~500 MB. This round plans ~520 runs (Section 10), i.e.
  ~260 GB with every periodic snapshot. Check durable quota first. Snapshot policy for this round (same as the user's
  earlier instruction for the v2 final suite): **every run keeps its final snapshot, all logs, transition logs,
  evaluation records and configs**; periodic snapshots are retained on durable storage while space allows and are
  always listed with SHA-256 in the package manifest (`rl.package --final-snapshots-only`). If quota forces deletion,
  keep at least the 50k and 100k snapshots of every run and record exactly which files were deleted, with hashes.
- Verify repository write access and Release upload ability at preflight. Use existing authenticated tools; never
  print or commit credentials.

**Branches.** Create the execution branch **`final-v3-experiments`** from `final-v3` (which is `rl-v2-experiments`
plus this document). Preserve all existing branches, tags and Releases unchanged. Push progress after every major
stage. Merging into `main` is not required.

**Status tracking.** Maintain `reports/final_v3/status.json` (phase, planned / completed / failed / pending runs per
suite, commit SHAs, Release status, blockers, UTC timestamps) and an append-only `reports/final_v3/decisions.jsonl`
(every declared rule, freeze and deviation, with timestamp and commit). Update and push them after each stage.

## 4. Stage 0: merge the decoder-improvement branch

`decoder-improvement-v1-experiments` and `rl-v2-experiments` diverged at `f6decb7`. Merge
`origin/decoder-improvement-v1-experiments` into `final-v3-experiments` with a real merge commit (do not rebase or
rewrite published history). Resolve conflicts so that:

- every report, config and script from both branches is present;
- the RL v2 delivery files that exist only on `rl-v2-experiments` (release verification reports, continuation reports,
  execution record updates, final report Section 8) are kept;
- `README.md` gets one status section describing both completed studies and this round, with links;
- `configs/rl_feedback.json` stays byte-identical to the historical profile; `rl_feedback_vnext.json` stays a separate,
  explicitly unused experimental profile.

Run `python -m unittest rl.test_stage2 -v` and the decoder branch's test suite after the merge; both must pass before
any new run. Commit and push.

## 5. Stage 1: weight-sensitivity final suites (w = 0.1 and w = 0.5), no code change

**Purpose.** Complete the existing protocol: run the two non-selected validation candidates on the final participants
so the final report can say whether the conclusion depends on the selected weight.

**Status of this analysis (must be stated in every report):** post-hoc sensitivity analysis requested after the v2
final results were seen. The v2 primary result (w = 1.0) is unchanged. These suites are not a re-selection.

Procedure:

1. For each of `candidate_w010` and `candidate_w050`, freeze from its **existing complete validation suite** (do not
   rerun validation):
   ```bash
   python -m rl.freeze --validation-suite <existing>/rl_v2_validation/candidate_w010 \
     --rationale "Post-hoc weight-sensitivity suite (not selected by the declared v2 rule; selected protocol remains w=1.0). Requested after v2 final results." \
     --out artifacts/final_v3/sensitivity_w010_frozen
   ```
   If the validation suite directories are no longer on durable storage, restore them from the validation Releases
   (verify hashes) before freezing. Commit both frozen protocols and their evidence under
   `configs/final_v3/sensitivity_w010/` and `configs/final_v3/sensitivity_w050/` **before** launching their final runs.
2. Run each as a complete 75-run final suite (`rl.suite --phase final --prepare`, then one scheduler job per
   `--run-job NAME`), same seeds 0-4 and participants s02/s08/s09. Run all 75 jobs including sparse/dense even though
   they do not depend on the weight: they serve as a determinism check. **Their metrics and evaluation histories must
   be identical to the existing v2 final sparse/dense runs**; compare and report the result (any mismatch is a finding
   to diagnose and report, not to hide).
3. `rl.summarize` must pass for each suite. Analyze with `rl.analysis` exactly as for v2.

Report for w ∈ {0.1, 0.5, 1.0} side by side: final success, final distance, normalized AUC, paired differences vs
sparse and EEGNet vs CBraMod (same pairing as v2), and the feedback diagnostics table. Also state what the declared v2
rule chose (w = 1.0) and what an AUC-based rule would have chosen from validation (w = 0.1), without promoting either
post hoc.

## 6. Stage 2: centered EEG reward (protocol v3)

**Purpose.** Remove the positive-bonus confound (Section 2, item 3) so that feedback without label information has
approximately zero expected reward. This is a correction motivated by the v2 diagnostics; it is reported as a
post-hoc follow-up, with its own declared validation selection and freeze.

### 6.1 Definition

`r_total = r_task + w * (c_m - p_error)` at feedback events; zero EEG reward elsewhere. Everything else (deadband
labels, feedback interval, sampling, seeds, SAC, evaluation) is identical to v2.

- `c_m` is a fixed per-decoder constant computed **only from the validation participants (s11, s13)** of the frozen
  feedback table, decoder seed 42, calibrated `p_error`:
  `c_m = 0.5 * (mean p_error over label-0 trials + mean p_error over label-1 trials)`, pooled over s11 and s13.
  This is the midpoint between the class-conditional means, so with balanced labels a label-informative decoder gives
  zero expected reward, progress is rewarded and retreat is penalized in expectation.
- `independent_<m>` uses the same `c_m` as decoder `m`. (Its expected reward is `c_m - mean(p)`, about
  0.15 × (μ1 − μ0) ≈ +0.01 per event; report the measured value.)
- `ideal` keeps `p = label` with center 0.5 (rewards ±w/2), as in v2.
- Never compute `c_m` from test participants. Write the computation as a script
  (`scripts/final_v3_compute_centers.py`) that reads the verified feedback table and writes
  `configs/final_v3/centers.json` with the values, the table SHA-256 and the per-label means/counts. Commit it before
  any v3 run.

### 6.2 Implementation

- New schema `csci642-rl-stage2/v3` with `reward_center` as an object, e.g.
  `{"eegnet": c_eegnet, "cbramod": c_cbramod, "ideal": 0.5}`, and a `centers_file` + `centers_sha256` field that
  `rl.train` checks. The v2 code path and its validation (`reward_center == 0.5`) must remain exactly as is, so that
  v2 configs and checkpoints keep validating and reproducing.
- Extend `rl/environment.py`, `rl/train.py`, `rl/smoke.py`, `rl/suite.py`, `rl/summarize.py`, `rl/freeze.py`,
  `rl/analysis.py` and tests as needed. Required tests: v3 EEG reward equals `w (c_m - p)` for every EEG and
  independent condition; ideal equals ±w/2; sparse/dense transitions are bitwise identical between v2 and v3 configs
  for the same seed; wrong/missing centers or a hash mismatch are rejected; v2 configs still load unchanged.
- Run a v3 smoke (`rl.smoke`, all seven conditions, reload and continuation) and record it.

### 6.3 Candidates and selection rule (declared here, before any v3 run)

| candidate | eeg_weight | dense_weight |
|---|---|---|
| `v3_w050` | 0.5 | 10.0 |
| `v3_w100` | 1.0 | 10.0 |
| `v3_w200` | 2.0 | 10.0 |

- Validation: participants s11/s13, seeds 100-102, 150,000 steps, 100 evaluation goals, 33 runs per candidate.
- **Selection: highest mean normalized success AUC** over the 12 EEGNet + CBraMod validation runs (because final
  success saturates, as v2 showed). Ties (|Δ| ≤ 1e-9): higher mean final success, then smaller weight.
- Write the selection script and commit it before the validation results exist (as was done for v2,
  `scripts/rl_select_candidate.py` can be generalized with a `--metric` option). Freeze with `rl.freeze`, commit the
  frozen protocol and evidence under `configs/final_v3/v3_frozen/`, then run the 75-run final suite (seeds 0-4,
  s02/s08/s09).
- Report all three candidates whatever is selected.

### 6.4 Analysis

Same tables and figures as v2 (final success, distance, AUC, paired differences, diagnostics, decoder-quality
relation), plus: (a) mean EEG reward per feedback event and per run by condition, confirming the bonus is removed;
(b) a direct comparison v2 (w = 1.0) vs v3 (selected) for the same seeds and participants; (c) whether centering
changes the independent controls relative to sparse. Interpret carefully: test participants were already used in
v2, so v3 final results are a follow-up evaluation, not an untouched confirmatory test.

## 7. Stage 3: route-preference extension (the unfinished milestone item)

**Question.** Can replayed ErrP feedback teach a preference that the task reward does not contain? This is where
feedback carries information that neither sparse nor dense task reward provides.

### 7.1 Environment (new, separate from the main protocol)

Implement `rl/preference.py` (new wrapper and helpers) with schema `csci642-rl-pref/v1`. Do not alter FetchReach
physics or the policy observation.

- **Task.** `FetchReach-v4`, sparse task reward, 50-step episodes, 5 cm success tolerance (unchanged task reward:
  success only, no preference term).
- **Goal filter.** Let `s0` be the gripper position after reset (`achieved_goal` at reset; FetchReach's arm start is
  fixed). Accept a sampled goal `g` only if the horizontal (x-y) distance `D = ||g_xy - s0_xy|| >= 0.15 m`. Implement it
  reproducibly: after the seeded reset, redraw with the environment's own goal sampler and RNG until the condition
  holds, set the goal on the unwrapped environment, and recompute the observation through the environment's own
  observation function. Unit-test that the filtered goal appears in the returned observation and that identical seeds
  give identical goals. Use this filter for training, evaluation (goal seeds 20000-20099) and trajectory goals
  (30000-30002).
- **Virtual pillar (no collision).** A vertical cylinder of radius `R = 0.025 m`, infinite in z, centered at the x-y
  midpoint `m = (s0_xy + g_xy) / 2`. With `D >= 0.15`, the pillar edge is at least 5 cm (the goal tolerance) from both
  start and goal in x-y. The straight start-goal path goes through it; the robot can physically pass through it.
- **Coordinates.** `u = (g_xy - s0_xy) / D`; along-axis fraction `a(p) = ((p_xy - s0_xy) · u) / D`; lateral offset
  `l(p) = u_x (p_y - s0_y) - u_y (p_x - s0_x)` (positive = left of the start-to-goal direction, viewed from above).
- **Simulated user preference: pass the pillar on the left.** A step is a *violation* if, at its end, the gripper is
  inside the pillar (`||p_xy - m|| < R`) or is in the right-hand corridor (`0.25 <= a <= 0.75` and `l < 0`).
- **Feedback labels** (same 5-step segments, same 1 mm deadband): if any step of the segment is a violation, label 1
  (error); otherwise the v2 distance-change rule (progress 0, retreat 1, deadband → no feedback). Sampling, independent
  controls and probabilities are exactly as in the main study.
- **Reward.** Centered reward from Section 6 (`w (c_m - p_error)`, ideal ±w/2), same `c_m` file. Sparse and dense
  (`10 ×` distance decrease) are task-only baselines with no preference information.
- **Pre-run geometry check (no learning).** Before any training, run a scripted check on the 100 evaluation goals:
  every filtered goal satisfies the distance rule; a straight-line scripted controller violates on every goal; a
  scripted left-detour controller (via a waypoint at `m + 0.06 m × left normal`) reaches the goal without violation on
  every goal; and the labels on both scripted paths match the definitions above. Commit the check output.

### 7.2 Metrics (evaluation uses no feedback and no shaping)

Per evaluation episode: task success (final distance < 5 cm); **preferred pass** = success and no violation at any
step and the gripper crossed `a = 0.5` with `l > 0`; violation-free rate; pillar entry rate; right-corridor rate;
path length. Primary metric: **final-policy preferred-pass rate**; also its normalized AUC over the evaluation
curve. Secondary: task success, final distance, task-success AUC.

### 7.3 Smoke, pilots, validation and final

1. **Smoke**: all seven conditions, 250 steps, reload and continuation, violation/label logging columns
   (`violation`, `pillar`, `right_corridor`, `a`, `l`) checked against recomputation.
2. **Pilots** (validation participants only, pilot seeds 900/901, w = 1.0): sparse, dense, ideal, EEGNet s11, CBraMod
   s13 at 150k steps. Gate: ideal must reach a clearly higher preferred-pass rate than sparse. If it does not, you may
   change only the pillar radius, corridor bounds or budget, using ideal/sparse pilots only (never EEG results), and
   must log each change in `decisions.jsonl` before validation. Keep every pilot.
3. **Declared plan** `reports/final_v3/preference_plan.md` committed before validation: geometry, budget, candidates
   `w ∈ {1.0, 2.0, 4.0}`, selection = highest mean final preferred-pass rate over the 12 EEGNet + CBraMod validation
   runs, ties by higher mean task success, then smaller weight.
4. **Validation**: 3 candidates × 33 runs (s11/s13, seeds 100-102). Freeze the selected candidate.
5. **Final**: 75 runs (s02/s08/s09, seeds 0-4).

Report all candidates; per-condition tables for preferred pass, violations, success and AUC; paired differences
(EEGNet vs sparse, CBraMod vs sparse, EEGNet vs CBraMod, decoder vs independent control); learning curves of
preferred-pass rate; and trajectory plots/GIFs that draw the pillar and corridor (Section 8). If the final suite cannot
be completed within budget, see the priority order in Section 10.

## 8. Trajectories and media for the new suites

Use the v2 rule (declared in advance): RL seed 0, participant s02 for EEG conditions, goal seeds 30000-30002, for the
selected v3 final suite and the preference final suite. Use the existing two-stage rendering (`rl.render_rollout`
record → render, replay checked against the recorded rollout). For the preference suite, overlay the pillar (circle)
and the right corridor on the x-y plots and draw the pillar in the GIF if feasible (otherwise state that only the
plots show it). Also roll out every final policy on the same three goals and produce the all-runs table
(`scripts/rl_trajectory_table.py`, extended with the preference metrics).

## 9. Reports (update in place, keep history)

Create `reports/final_v3/` and write:

- `sensitivity_results.md` (Section 5), `v3_results.md` (Section 6), `preference_results.md` (Section 7),
  `execution.md` (commands, hardware, wall times, run counts, failures/reruns, storage), all tables generated from
  committed CSV/JSON by scripts (no hand-typed numbers; keep a number-to-source audit as on the decoder branch).
- **Update `reports/final_project_report.md`** to be the single final document for the course:
  1. A "Milestone follow-up" table listing every milestone commitment with its status and evidence: sparse/dense
     baselines; EEGNet/CBraMod training and calibration; EEG integration and comparison with baselines; decoder
     quality vs learning; CBraMod frozen / last-layers fine-tuning (decoder study, negative); preference learning
     (this round).
  2. A "Numbers across documents" table that reconciles: decoder five-seed mean vs seed 42 vs controlled refits;
     RL baselines at milestone (3 seeds, 30 eval episodes) vs final (5 seeds, 100 eval episodes). State which numbers
     the RL used (seed 42 calibrated probabilities) and why the figures differ.
  3. Results sections for v2 (unchanged primary), sensitivity, v3 and preference, each labeled with its status
     (declared primary / post-hoc sensitivity / post-hoc follow-up with its own validation selection / extension).
  4. Updated limitations: ceiling effect, repeated use of the test participants across v2/v3/preference, replayed
     rather than live feedback, privileged geometric labels, small samples, abstract feedback timing.
  5. Artifacts section with every Release (old and new) and verification results.
- Update `README.md` status. Do not overwrite or rewrite the historical v2 and decoder reports; add cross-links.

Statistical language: as before, 3 test participants × 5 seeds; report means, SDs, paired differences and sign counts;
no population-level significance claims; decoder-score correlations are not causal evidence.

## 10. Budget, order and what to cut if needed

| stage | runs (150k steps each) | approx. core-hours |
|---|---|---|
| 1. sensitivity finals w = 0.1, 0.5 | 2 × 75 = 150 | ~140 |
| 2. v3 smoke + validation (3 × 33) + final (75) | ~175 | ~165 |
| 3. preference smoke + pilots (~10) + validation (3 × 33) + final (75) | ~185 | ~175 (more if the pilot gate raises the budget) |
| total | ~510 | ~480 |

Execution order: Stage 0 → Stage 1 (can run while Stage 2 code is written) → Stage 2 → Stage 3 → media → reports →
Releases → verification. Stages 2 and 3 code can be developed in parallel, but Stage 3 needs `centers.json` from
Stage 2. If compute or time becomes the binding constraint, record the decision in `decisions.jsonl` **before**
dependent runs and cut in this order: (1) preference validation reduced to 2 candidates (w = 1.0, 2.0); (2) v3
validation reduced to 2 candidates (w = 0.5, 1.0); (3) preference final seeds reduced to 0-2. Never cut checkpoints of
completed runs, the verification, or a final suite after it has started; never report a reduced study as the full
declared one.

## 11. Explicitly out of scope

- Retraining, recalibrating or re-selecting decoders; using `rl_feedback_vnext.json` probabilities for RL.
- New tasks or simulators other than the FetchReach preference wrapper of Section 7 (no FetchPush, no physical
  obstacle model).
- Synthetic decoders, multi-trial averaging, participant-specific decoders, new feedback schedules.
- Changing SAC hyperparameters, budget (except via the declared preference pilot gate), seeds, participants or
  evaluation goals.

## 12. Packaging, Releases and fresh-download verification

Follow the v2 delivery standard (`reports/final_project_report.md`, Section 8; tools `rl.package`,
`scripts/rl_stage_release.sh`, `scripts/rl_upload_missing.sh`, `scripts/rl_verify_release.py`,
`scripts/rl_continuation_check.py`).

- One Release per scope, new unique tags pinned to the actual commits, e.g.
  `final-v3-sensitivity-w010-YYYYMMDD-<sha>`, `final-v3-sensitivity-w050-…`, `final-v3-smoke-…`,
  `final-v3-v3-validation-w050-…` (one per candidate), `final-v3-v3-final-…`, `final-v3-pref-pilots-…`,
  `final-v3-pref-validation-w100-…` (one per candidate), `final-v3-pref-final-…`. Never move or edit existing tags.
- Contents: final snapshot of every run (model with optimizer and entropy state, replay buffer, RNG state, inference
  probe, metadata), every log, evaluation record, transition log, config, plan, frozen protocol and evidence,
  `centers.json`, analysis CSV/JSON, figures and trajectory media (GIFs as Release assets). Periodic snapshots per the
  policy in Section 3, always listed with SHA-256 in the manifest. Split archives above the asset limit losslessly with
  reassembly instructions and full-archive hashes.
- After upload, from a new directory: list assets through the API, download everything, check names, sizes and every
  SHA-256, reassemble split archives and check full-archive hashes, extract, hash-validate every snapshot, reload every
  final snapshot on CPU and reproduce its stored inference probe (tolerance 1e-6), and remove/reupload any incomplete
  ("starter") asset. Run one short continuation from a freshly downloaded v3 checkpoint and one from a preference
  checkpoint, each in a clean checkout of its training commit, into new directories (not part of any result). Commit
  verification and continuation reports under `reports/final_v3/release_verification/`, then upload them too and
  verify those assets after download.
- Push `final-v3-experiments` with all code, configs, reports and small result files. Report the branch URL, the final
  commit SHA and every Release URL in the final handoff message and in `reports/final_v3/status.json`.

## 13. Completion checklist

- [ ] Preflight recorded (environment, hardware, storage, write/upload access); `status.json` and `decisions.jsonl` live.
- [ ] Decoder branch merged; all tests pass; pushed.
- [ ] Sensitivity suites w = 0.1 and w = 0.5 frozen from existing validation evidence, 2 × 75 runs complete, sparse/dense determinism check reported.
- [ ] `centers.json` computed from validation participants only and committed; v3 implemented with tests; v3 smoke passed.
- [ ] v3 selection script committed before validation; 3 × 33 validation runs complete; frozen; 75 final runs complete.
- [ ] Preference wrapper implemented with tests; geometry check passed; smoke and pilots done; plan committed; validation complete; frozen; 75 final runs complete.
- [ ] Analyses, figures, trajectories and GIFs for all new suites generated from committed data.
- [ ] `reports/final_v3/*` written; `reports/final_project_report.md` updated with milestone follow-up and number reconciliation; README updated.
- [ ] All new Releases uploaded; fresh-download inventory, hash, reload/probe and continuation checks passed and published.
- [ ] Final message lists branch, commit, Releases, run counts, wall/core hours, key results and any concrete unresolved limitation.

Do not report the task complete because jobs were submitted, some runs finished, code was pushed or uploads began.
