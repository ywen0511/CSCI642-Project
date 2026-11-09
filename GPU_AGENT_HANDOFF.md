# Complete the remaining robot-learning experiments and delivery

## Assignment and authority

Work in **ywen0511/CSCI642-Project**, starting from branch **rl-proposal-v2**. This is the complete active assignment for the next execution agent. When the user asks you to execute this handoff, carry the work through implementation, runs, analysis, reporting, code push, Release upload and fresh-download verification. Do not stop after preparing scripts or a run plan. All written deliverables must be in English.

**This document is the only handoff the user needs to provide.** Do not ask the user to clone a repository, download data, prepare scripts, run commands, upload checkpoints or publish Releases. Do not run `git clone`. Reuse the existing project workspace when available; otherwise obtain the required source yourself through GitHub's archive/API facilities and prepare the execution workspace and source provenance. Fetch required feedback assets yourself. Reading supporting source files, creating the environment, fixing missing implementation, running experiments and uploading the complete delivery are the executing agent's responsibilities.

Direct entry points, including when this MD is read outside a repository:

- Source branch: https://github.com/ywen0511/CSCI642-Project/tree/rl-proposal-v2
- This assignment: https://github.com/ywen0511/CSCI642-Project/blob/rl-proposal-v2/GPU_AGENT_HANDOFF.md
- Feedback profile: https://github.com/ywen0511/CSCI642-Project/blob/rl-proposal-v2/configs/rl_feedback.json
- Existing EEG assets: https://github.com/ywen0511/CSCI642-Project/releases/tag/eeg-stage1-20261101-bccbb8a
- New code and Release destination: **ywen0511/CSCI642-Project**.

Resolve relative file references against that source branch. The user does not need to supply a second README or repeat the instructions below. At preflight, verify the available compute, durable storage, repository write access and ability to upload Release assets. Use existing authenticated tools without exposing credentials. A document cannot supply missing account permissions or hardware: if genuinely unavailable, report the exact blocker and preserve progress; do not claim a completed upload or merely hand routine work back to the user.

The branch already contains the frozen EEG source and the FetchReach/SAC integration. Required feedback files are in this repository's original EEG Release; no private Savaw checkout is required. Publish new work here. Do not modify Savaw/CS642-EEG-Assisted-RL as part of this assignment. Preserve existing branches, tags, Releases and EEG checkpoints. Use a new working branch such as `rl-v2-experiments` for execution improvements and results. Push that branch and supply its URL; merging into main is not required for delivery.

The preparing agent changed only the reward/feedback protocol, regression tests and documentation, and ran non-training checks. No v2 MuJoCo smoke run, SAC training, validation selection or final experiment has yet been performed. The historical v1 smoke artifacts are not evidence for this version.

## Research question and proposal stages

Can error probabilities decoded from recorded EEG improve robot reaching, and how do EEGNet and CBraMod differ as feedback sources? A stronger decoder is not assumed to produce a stronger policy. Existing EEG results favor EEGNet on this split; report actual RL outcomes even if neither decoder helps.

1. Proposal Stage 1: reaching/SAC setup and sparse/dense baselines. Code exists; run formal baselines with the new common experimental budget. Do not count historical 160-step checks as learned baselines.
2. Proposal Stage 2: EEG audit, training, calibration and evaluation. Completed. Do not retrain, recalibrate, resplit, choose another decoder seed, or change probabilities.
3. Proposal Stage 3: EEG integration, main comparison and optional route preference. v2 mechanics exist; perform fresh verification, pilot, validation and final comparisons.
4. Proposal Stage 4: analyze outcomes, create learning curves and trajectories, write the report and deliver all reproducibility assets.

Historical branch names `eeg-stage1` and `rl-stage2` are delivery names, not these proposal numbers.

## Fixed EEG inputs

Read `configs/rl_feedback.json`; do not edit its pinned identities or hashes.

- Original Release: `eeg-stage1-20261101-bccbb8a` in this repository.
- Decoder seed: 42 for both models; other completed decoder seeds are historical robustness results, not a pool from which to select a better RL result.
- Validation participants: s11, s13; 959 trials per decoder in total.
- Final participants: s02, s08, s09; 1,429 trials per decoder in total.
- Use calibrated `p_error`. Keep raw and calibrated columns unchanged.
- One participant per EEG-assisted policy. Sample with replacement from that participant's appropriate label pool. Replayed trials are not independent new human measurements.
- Predictions are already computed: neither raw EEG nor EEG GPU inference is needed for RL.

The original robot dataset labels are 0=non-error and 1=machine error. Human-error S5 trials were excluded. These recordings are not responses to the current simulated robot. Geometry supplies a proxy label, then recorded EEG probabilities supply a noisy feedback channel. Do not describe this as live human evaluation or offline RL: SAC interacts with the simulator online, using offline recorded feedback.

## Protocol v2: keep these two decisions fixed

The schema is `csci642-rl-stage2/v2`, with `feedback_rule=distance_change_deadband` and `reward_center=0.5`.

Let delta = segment-end distance minus segment-start distance. At each segment boundary:

- delta < -epsilon: label 0, sample a non-error trial.
- delta > epsilon: label 1, sample an error trial.
- abs(delta) <= epsilon: **no label, no sampled trial, no extra reward**.

The default epsilon is 0.001 m. The implementation allows 1e-12 m subtraction tolerance at the boundary. Segment state resets even if feedback is skipped. Eligible-event indices advance only for labeled opportunities. The same rule applies inside the goal region; simulator success and sparse task reward remain independent.

For an actual feedback event:

`r_total = r_task + w_eeg * (0.5 - p_error)`

At all other steps, EEG reward is zero. Ideal feedback uses p=the simulated binary label and therefore yields +/-w_eeg/2. Label-independent controls use the same transform and eligibility gate, but sample from the participant's full pool without conditioning on the binary label. They do not remove all effects of action-dependent gating. Keep these semantics aligned across the main decoders, ideal feedback and independent controls.

Do not infer that true error trials always receive negative rewards; decoder errors and calibration affect the probability. Do not replace probabilities by the known simulator labels for the EEG conditions.

Current defaults: FetchReach-v4, sparse task reward, 50-step episodes, 5 cm goal tolerance, 5-step feedback intervals, 1 mm deadband, EEG weight 0.5, dense weight 10.0. The interval is 0.2 s at 25 Hz, whereas EEG epochs span 2 s. This is an abstract offline replay schedule, not a model of real EEG acquisition latency. Keep that limitation explicit; if exploring feedback frequency, predeclare candidates and select only on validation data.

Logs distinguish `segment_end`, `feedback_eligible`, and `feedback_event`. Neutral rows have blank labels, probabilities, trial IDs and event indices. Sparse/dense policies receive no EEG reward. Event-key pairing does not guarantee identical on-policy sequences when policies choose different actions.

## 1. Setup, preflight and new smoke run

Locate and inspect the existing workspace first. Preserve unrelated changes and use an isolated worktree if needed. Record the full source commit SHA and preserve a clean baseline. If no workspace exists, resolve the branch's current SHA through GitHub, download/extract that exact source archive into a new directory, and prepare the Git metadata required by the runner and eventual publication. Attaching the downloaded tree to its original commit via Git initialization and fetching is permitted; do not create a disconnected history or replace an existing repository. Do this yourself rather than requesting user setup. Check that the workspace contains the v2 fields before doing any training.

Create a unique execution branch such as `rl-v2-experiments` based on the recorded source commit, then run the following from its repository root. These are agent execution steps, not instructions for the user to perform manually.

```bash
python3.12 -m venv venv/rl
. venv/rl/bin/activate
python -m pip install -r rl/requirements.txt
python scripts/download_feedback.py --out-dir artifacts/rl_feedback
python -m unittest rl.test_stage2 -v
python -m rl.smoke --out artifacts/rl_v2_smoke
```

Use an appropriate official PyTorch CUDA build if needed, preserving compatible package versions and recording the actual dependency lock. Never silently upgrade the EEG environment. CPU is supported. `rl.train` and `rl.suite` accept `--device cuda`; `rl.smoke` intentionally runs a small CPU integration check. Validate CUDA availability before launching GPU jobs. Measure throughput on a representative pilot rather than claiming a GPU speedup in advance.

The smoke run must exercise all seven conditions, actual simulator steps, gradients, checkpoint reloads, inference probes, replay loading and a short continuation. Inspect transition logs to confirm neutral segments skip feedback and retreat/progress yield the expected signs for ideal feedback. Verify model observations do not contain EEG probabilities or simulator error labels. Fix failures and rerun affected checks before proceeding.

Record hardware, Python/packages, commands, source hashes, output paths, smoke results and wall-clock runtime in `reports/rl_v2_execution.md`. Old checkpoints cannot initialize a formal v2 run or be silently resumed under v2.

## 2. Finish the experiment configuration before formal runs

The current planner still hardcodes five final seeds and the default JSON has 50,000 steps. These were deliberately not changed by the preparing agent. **Do not launch the old default final suite as if it implemented this assignment.**

Implement configurable, validated seed lists in the protocol and pass them consistently through `rl.suite`, `rl.summarize`, protocol freezing and tests. Suggested fields:

```json
{"validation_seeds": [100, 101, 102], "final_seeds": [0, 1, 2]}
```

Reject empty lists, duplicate/non-integer/negative seeds and overlapping validation/final lists. Ensure frozen config identity includes the lists. Add regression tests for 33 validation jobs and 45 final jobs with three seeds (75 final jobs for five seeds). Sparse/dense/ideal are trained once per seed, not once per participant. Require all jobs in the frozen plan before reporting completion. Do not change only the runner while leaving summary expectations at five seeds.

Use 50,000 steps for pilots. Plan 150,000 steps per formal run initially. Inspect representative validation/pilot learning curves through that budget before finalizing it. If 150,000 is inadequate or infeasible, declare and justify a different common budget using validation-only evidence, before looking at final test outcomes. Never extend only the condition that looks worse on test data.

Default final seeds: 0/1/2. If resources support five seeds, choose 0/1/2/3/4 before starting final runs and record the choice in the frozen config. More seeds do not compensate for a budget too short to learn.

At 150,000 steps: a full validation candidate is 33 runs / 4,950,000 steps; a three-seed final suite is 45 runs / 6,750,000 steps. Five final seeds require 75 runs / 11,250,000 steps. These totals include the ideal and independent controls added beyond the proposal's four main conditions. Evaluation episodes add extra simulator work. Check disk capacity for all periodic replay snapshots before launching; retain them on durable storage.

## 3. Pilots and validation-only selection

Example individual pilot (use a unique new output directory):

```bash
python -m rl.train --condition eegnet --subject s11 --phase pilot --seed 100 \
  --device cuda --steps 50000 --out artifacts/rl_v2_pilots/eegnet_s11_s100
```

Run representative sparse, dense, ideal, EEGNet and CBraMod pilots. Check meaningful learning, neutral-segment fractions, error/non-error proportions, feedback event frequency, reward components, probability distributions and runtime. A flat sparse curve alone is not a reason to tune on test data. Keep pilot artifacts and failure records.

Before validation sweeps, commit `reports/rl_v2_experiment_plan.md` with all candidate weights, fixed seeds, step budget, interval/deadband, evaluation goals, metrics and selection rule. Suggested initial grid: EEG weights [0.1, 0.5, 1.0], dense weight fixed at 10.0. This means three full validation suites (14,850,000 training steps at the initial 150k budget). If pilot evidence calls for another grid or dense weight, declare it and the expanded cost before running it. Do not disguise computational constraints or retroactively select a favorable subset.

Use one common EEG weight for both decoders and independent controls; ideal uses that weight too. Select primarily by mean final-step success across both main decoders, both validation participants and all validation seeds; ties use lower mean final distance, then the smaller EEG weight. Dense-weight selection, if explored, uses dense validation results. Preserve every candidate and all evaluated outcomes. The freeze tool does not perform or certify this scientific selection.

After implementing configurable seeds, create each candidate JSON from the v2 protocol, add the explicit seed lists and selected formal budget, then use existing commands:

```bash
python -m rl.suite --phase validation --config artifacts/rl_v2_candidates/candidate_01.json \
  --device cuda --out artifacts/rl_v2_validation/candidate_01
python -m rl.suite --phase validation --config artifacts/rl_v2_candidates/candidate_01.json \
  --device cuda --out artifacts/rl_v2_validation/candidate_01 --execute
```

The first command only prints a plan. Repeat execution for the predeclared candidates. Confirm each suite is complete, then freeze the selected candidate with a rationale identifying every candidate and its metrics:

```bash
python -m rl.freeze --validation-suite artifacts/rl_v2_validation/candidate_01 \
  --rationale "Replace with the actual complete validation-only selection rationale." \
  --out artifacts/rl_v2_frozen
```

`candidate_01` is an example, not a predetermined winner. Commit the frozen protocol, validation summary and selection rationale before final execution. Preserve the evidence file beside the frozen JSON; its hash and configuration are checked by the final gate.

## 4. Final comparison

Required conditions: sparse, dense, ideal, eegnet, cbramod, independent_eegnet and independent_cbramod. Main paper-facing comparisons are sparse/dense/EEGNet/CBraMod; the other conditions diagnose the feedback mechanism.

```bash
python -m rl.suite --phase final --config artifacts/rl_v2_frozen/protocol.json \
  --device cuda --out artifacts/rl_v2_final
python -m rl.suite --phase final --config artifacts/rl_v2_frozen/protocol.json \
  --device cuda --out artifacts/rl_v2_final --execute
python -m rl.summarize artifacts/rl_v2_final
```

Train each formal policy from scratch with the complete frozen budget. The final EEG participants are s02/s08/s09. These feedback pools train robot policies; they are held out from EEG training, not withheld from the robot's feedback stream. Robot evaluation uses fresh deterministic goal episodes without EEG/dense shaping. State both meanings of held out accurately.

Preserve the shared evaluation goal seeds and cadence across conditions. Compare fixed-budget final policies, not the best test checkpoint. Resumed formal runs are currently rejected by the summary: continuation starts a fresh simulator episode and is not bitwise equivalent. Keep interrupted results for diagnosis and rerun that formal job cleanly. Do not weaken completion checks to hide missing runs.

If adding parallel scheduling for independent jobs, use separate output directories and bounded GPU/CPU resources. Record all commands and do not change seed identities or sampling rules. Avoid counting reused shared controls as additional independent measurements.

## 5. Analysis, plots, trajectories and final report

Implement any missing analysis code in this repository, with meaningful tests for metric calculations.

- Per run: final-step success (primary), ever-success, final distance, sparse task return and normalized area under the success-rate curve (learning speed).
- Compute normalized AUC by trapezoidal integration over evaluation steps, divided by the full training budget. Include step-zero and final evaluations. The runner may record the final step twice; deduplicate identical final-step records before integration and fail on inconsistent duplicates.
- Plot success and distance against environment steps; show participant-level curves and RL-seed variation. Report seed mean/SD within participant, then an unweighted participant macro-average for EEG conditions. Shared controls are not three independent participant observations.
- Show feedback availability and reward/probability diagnostics; check whether the independent controls explain apparent improvements.
- Relate the fixed EEG AUROC/balanced accuracy/Brier results to RL outcomes, including negative findings. With three EEG participants, do not claim population-wide significance or infer causality from decoder-score correlations.
- Use the same prespecified goal seeds for representative trajectories across conditions. Produce trajectory plots and GIFs/videos. `python -m rl.view_policy <checkpoint-dir> --out <new.gif>` is available; configure headless rendering if necessary and record any unresolved rendering limitation.
- Write `reports/rl_v2_results.md` and `reports/final_project_report.md` covering methods, proposal deviations, actual budgets, results, limitations, reproducibility and artifact links. Do not claim strict Kim replication, live human feedback or a CBraMod improvement unsupported by results.

Declare the curve/trajectory selection rule before choosing examples. Preserve underlying numeric CSV/JSON data so figures can be regenerated. A negative result is a valid completed experiment.

## 6. Optional route-preference extension

The proposal makes this conditional on time. Complete and preserve the required main study first. If resources permit, implement a separate environment/protocol with an obstacle and approximately equal left/right routes; task reward includes success/collision but no left preference. The simulated user prefers the left side relative to the start-to-goal direction. Define the geometry and collision detection explicitly; do not pretend current FetchReach already implements this extension.

Use the decoder with better main-study robot results, selected by a declared aggregate success rule, then lower distance and finally a fixed tie-break. Label movement away from the target or toward the undesired route as error; define neutral segments and obstacle-side transitions before running. Run the chosen EEG condition for three final participants and two RL seeds, and dense for three seeds (nine runs total). Evaluate success, collisions and preferred-side fraction using shared goals. This is a secondary, model-selected extension, not an independent confirmation of the main decoder comparison.

Validate the new geometry and feedback before training and use separate configs, outputs and Release assets. If time/compute does not permit the extension, explicitly mark it not performed in the final report; do not leave core delivery unfinished while chasing it.

## 7. Preserve checkpoints and publish verified Releases

**Checkpoints must be Release assets, not merely local files or code links.** Preserve all periodic and final snapshots for every pilot, validation candidate and final run, including failures/interrupted work where available. Required snapshot contents include `model.zip`, `replay_buffer.pkl`, `rng.pt`, `metadata.json` and available fixed inference probes. Preserve optimizer/entropy state, configs, feedback IDs, source hashes, logs, evaluation histories and dependency records. Never delete artifacts just because Git ignores them.

Existing tools package successful suites:

```bash
python -m rl.package --runs artifacts/rl_v2_smoke --out artifacts/rl_v2_release_smoke --smoke
python -m rl.package --runs artifacts/rl_v2_validation/candidate_01 --out artifacts/rl_v2_release_validation_01
python -m rl.package --runs artifacts/rl_v2_final --out artifacts/rl_v2_release_final
```

Package every other validation candidate separately. Individual pilots and interrupted runs are not full suites: extend packaging or archive them separately with an explicit scope, complete file inventory and SHA-256 checksums; do not mislabel them as passed suites. Include pilot stdout/stderr, selection evidence, frozen config, all candidate configs, reports, figure source data and trajectory media. Preserve preexisting EEG Releases unchanged.

Before uploading, inspect total archive sizes. If any asset exceeds hosting limits, use lossless numbered parts plus checksums and exact reassembly instructions. Do not drop replay buffers or periodic checkpoints to shrink delivery.

Upload code and small reports to `ywen0511/CSCI642-Project` yourself using Git or authenticated GitHub tools. Publish all large artifacts as Release assets yourself; a folder prepared for someone else to upload is not a completed delivery. Use new, unique tags pinned to actual source commits; never move the historical tags. Separate smoke, validation and final delivery scopes in tags/Release titles. The user has requested code and Release delivery here; complete it using the available authenticated account. If credentials lack access, preserve all prepared artifacts and report the exact access blocker without pretending upload succeeded.

Example GitHub CLI flow (choose actual unique tags and an existing prepared notes file):

```bash
git push -u origin rl-v2-experiments
gh release create rl-v2-final-YYYYMMDD artifacts/rl_v2_release_final/* \
  --repo ywen0511/CSCI642-Project --target "$(git rev-parse HEAD)" \
  --title "RL protocol v2: completed final comparison" \
  --notes-file reports/rl_v2_final_release_notes.md
gh release download rl-v2-final-YYYYMMDD --repo ywen0511/CSCI642-Project \
  --dir artifacts/rl_v2_fresh_download
```

Use distinct tags/directories for each candidate and scope. Record the exact training source commit in every run, even if the delivery commit adds only reports. A report update is not a new training execution.

In a fresh directory, verify the exact asset inventory, byte sizes and every SHA-256, including any split parts. Extract archives without overwriting originals. Validate hashes for **every** downloaded snapshot. Run `rl.verify_checkpoint` on every final-policy snapshot and at least one periodic snapshot per condition; preserve verification reports. GPU-to-CPU probe roundoff must be measured and documented; do not silently relax tolerances to hide failures.

Run a separate short continuation from a freshly downloaded **v2 pilot/smoke** checkpoint with its exact config, identity and training source, into a new output directory. Verify increasing steps, optimizer updates and replay contents; do not include that continuation in formal performance summaries. The source-hash gate may require a clean checkout of the checkpoint's recorded training commit. Publish the verification/continuation evidence too, and verify those added assets after download. Keep durable local copies until all delivery checks pass.

## Completion checklist

- [ ] v2 smoke, interface checks and checkpoint continuation passed on the execution environment.
- [ ] Seed configuration, budget and suite-completeness checks implemented and tested.
- [ ] Pilots completed; protocol and candidate grid declared before formal selection.
- [ ] All validation candidates completed; common protocol frozen using validation only.
- [ ] All required final runs completed at the frozen budget with correct participants and seeds.
- [ ] Curves, normalized AUC, participant/seed results, diagnostics and trajectories generated.
- [ ] Final report written, including proposal deviations and optional-extension status.
- [ ] Code and small reports pushed; exact branch and commit URLs supplied.
- [ ] ALL checkpoints/replay/logs/configs packaged and uploaded to new Releases in this repository.
- [ ] Fresh-download inventory, hash, inference and continuation checks completed and uploaded.
- [ ] Final handoff lists commands, hardware/runtime, actual run counts, result tables, Release links and any concrete unresolved limitation.

Do not report the task complete merely because a process started, a few runs finished, code was pushed, or uploads began. Finish all required stages, or clearly identify the concrete blocker and retained artifacts.
