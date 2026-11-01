# CSCI 642 — Complete EEG Stage Execution Instructions

**Instructions for the executing agent: When the user asks you to read this README and execute the task, treat this file as the complete specification. Carry out the EEG stage and remote delivery described below. Do not stop after summarizing the document or proposing a plan. No additional prompt or document from `docs/` is required.**

This project builds on Kim's EEG-to-reward / SAC framework to compare EEGNet and CBraMod in a robot reaching task. The current assignment is the EEG stage: train and evaluate both decoders, freeze them, and export error probabilities for subsequent reinforcement learning.

## Entry Point and Definition of Completion

1. Inspect existing work and applicable repository instructions; preserve user changes. Check GPU availability, network downloads, repository push access, and GitHub Release upload access. Record available hardware and permissions.
2. Read this entire README, then complete the data audit, implementation, smoke tests, full training, calibration, testing, export, and delivery. Make and document reasonable implementation decisions. Establish critical event and label semantics from evidence.
3. Both models must actually be trained and evaluated. Code alone, demonstrations alone, EEGNet alone, or artifacts stored only on temporary GPU storage do not satisfy this assignment.
4. Commit and push code, configurations, dependencies, and reports to this project's remote repository. Upload complete checkpoints, calibration parameters, logs, prediction tables, and manifests to a **GitHub Release in the same repository**. Download the assets again, verify them, and validate inference and training resumption.
5. The authorized scope covers EEG implementation, experiments, code commits, and Release delivery. It does not include robot/SAC experiments, purchasing compute, changing access permissions, or publishing to unrelated repositories. Verify the destination using this checkout's remote; do not assume another authenticated account belongs to the user.
6. If a genuine blocker arises, explain what is missing and what has been completed, preserve artifacts, and continue unaffected work. Do not report unexecuted experiments or unsuccessful uploads as complete.
7. The final response must include the repository URL, code commit, Release URL, metrics for both models, checkpoint download verification and resumption results, reproduction commands, and any unfinished work.
8. Write code comments, documentation, reports, and Release notes in English.

## Initial State and References

This handoff initially provides a task specification only. Implementation, EEG data downloads, and training have not started. Inspect the actual checkout before proceeding to avoid repeating work completed since this handoff, and update the status after execution.

The original local project contains `Kim_2.pdf`, `Kim_master.pdf`, and `Proposal/`. Keep these original files read-only. They do not need to be uploaded to GitHub. Their absence from a remote checkout is not a blocker: this README specifies the EEG stage, and Kim's papers are available from the public sources below. Record any differences between paper versions.

- [Kim's short paper](https://arxiv.org/abs/2507.13171)
- [Kim's master's thesis](https://arxiv.org/abs/2512.00050)
- [ErrP-HRI dataset and documentation](https://github.com/stefan-ehrlich/dataset-ErrP-HRI)
- [Official EEGNet implementation](https://github.com/vlawhern/arl-eegmodels)
- [Official CBraMod implementation](https://github.com/wjq-learning/CBraMod)
- [CBraMod pretrained weights](https://huggingface.co/weighting666/CBraMod)

Suggested implementation structure; these directories do not imply that implementation already exists:

```text
configs/          Data splits, preprocessing, training, and calibration settings
src/              Data loading, models, training, calibration, evaluation, and export
scripts/          Executable entry points
reports/          Audits, protocol decisions, results, artifact index, and commands
data/             Raw data and preprocessing caches; excluded from ordinary Git
artifacts/        Checkpoints, logs, and predictions; upload to Release assets
```

Ignoring artifacts in `.gitignore` excludes them from ordinary Git commits. **It does not waive the requirement to upload checkpoints to GitHub Releases.** This README takes precedence over older handoff documents if they differ.

## 1. Objective and Scope

Complete a fair comparison of EEGNet and CBraMod on ErrP-HRI and produce a fixed library of feedback probabilities for subsequent RL experiments.

This stage includes data auditing, participant splits, preprocessing and event epoching, model training, probability calibration, evaluation, checkpoint preservation, per-trial probability export, reproduction documentation, code pushes, and GitHub Release uploads. Saving artifacts only on temporary GPU storage is insufficient.

This stage excludes SAC, robot environments, reward-weight tuning, the route-preference extension, and new human EEG collection. Preserve existing papers, proposal files, and other original materials. CBraMod outperforming EEGNet is not an acceptance criterion.

## 2. Distinguish Paper Settings from Project Decisions

Kim explicitly describes EEGNet, cross-entropy training, leave-one-subject-out validation (LOSO), a 1–20 Hz bandpass, re-referencing, downsampling to 256 Hz, event-aligned 2-second windows, and decoded error probabilities used in RL. The methods section explicitly states that EEG was replayed from offline recordings.

The papers do not fully specify exact epoch start/end times, reference details, all training hyperparameters, or the mapping between EEG trials and simulated robot actions. Do not present assumptions as original paper settings. Paper descriptions also do not replace inspection of the downloaded files.

The current project proposal uses separate training, validation, and test participants, with three test participants for the main comparison. This is not a full 12-fold LOSO reproduction. Follow the project protocol by default; full LOSO is not mandatory for this stage.

In `reports/protocol_decisions.md`, classify each setting by its source: explicitly stated in the paper, confirmed by dataset documentation/files, required by the official model interface, or chosen for this project. Freeze decisions affecting results before formal training.

## 3. Data Download and Audit

Dataset: https://github.com/stefan-ehrlich/dataset-ErrP-HRI

1. Read `documentation_dataset-ErrP-HRI.pdf` in the dataset repository. Download and inspect paired `.set` / `.fdt` files for the `robot` condition. Verify each file's condition rather than relying solely on its directory; exclude cursor-condition files that may be present there.
2. Record the source, repository commit, filename, and SHA-256 of each raw file. Keep raw data read-only and store derived data separately.
3. Inspect participant IDs, actual channels and channel types, sampling rates, numerical units, event codes and times, existing filtering/referencing/epoching, bad channels, and missing files.
4. Establish the mapping from robot-action events to error/non-error labels. Never assume a numerical event code means error without documentation or file evidence.
5. Confirm the event origin and exact epoch start/end times. If critical labels or event definitions cannot be established reliably, stop training that depends on them and report the blocker. Continue independent implementation and model-interface work.
6. Produce `reports/data_audit.md`, including original, retained, and excluded trial counts by participant and label, with all exclusion reasons. Do not invent dataset sizes.

## 4. Data Splits and Leakage Prevention

### Default Protocol

- Split by participant, using the same split file for both models. Do not randomly distribute one participant's trials across training, validation, and test sets.
- Reserve three test participants. If the audit confirms all 12 participants are usable, use two validation participants and seven training participants by default.
- Specific participant IDs have not been selected. After auditing, generate one split from sorted eligible IDs with fixed seed 42. Record the random-number library, version, and resulting IDs. Do not redraw splits based on model performance.
- If too few participants are usable, report the reason and document the alternative protocol. Do not silently reduce the test cohort or reuse test participants for fitting.
- Store actual IDs in `configs/splits.json`, record its checksum, and freeze it before formal training.

### Usage Rules

- Training set: fit networks and any preprocessing statistics estimated across samples. Apply augmentation only to training data.
- Validation set: early stopping, hyperparameter selection, classification thresholds, and probability calibration. These may share the validation set, but document the selection procedure and do not present validation scores as unbiased final results.
- Test set: final evaluation and feedback probability export after the protocol is fixed. Do not use test labels for training, calibration, checkpoint selection, or preprocessing decisions.
- A predefined per-trial normalization rule may be applied to test trials during inference. Do not refit normalization using the overall distribution of test participants.
- Subsequent feedback simulation may sample test trials by their true labels. Explain that this is different from using those labels to fit the EEG decoder.

## 5. Preprocessing and Comparable Inputs

- Use the same retained trials, labels, channel-selection principles, and physical event windows for both models.
- Treat Kim's 256 Hz / 1–20 Hz / 2-second description as a reference. Inspect existing data processing before choosing executable settings to avoid redundant or conflicting processing.
- Verify CBraMod sampling rate, patch length, channels, and amplitude scale against the selected official checkpoint's input requirements. Do not blindly reuse EEGNet input tensors.
- If the models require different sampling rates or tensor layouts, perform and document the necessary adaptations while retaining the same physical time windows. Record bandwidth or normalization differences as potential comparison factors.
- Save the full preprocessing configuration: event mapping, epoch boundaries, filtering, re-referencing, resampling, channel order, unit conversion, normalization, and quality rules.
- If input issues prevent one model from using certain trials, first attempt to fix the adaptation. If exclusion is necessary, apply it consistently to both models in the main comparison and report it. Do not silently create different test sets.
- Create stable trial IDs, preferably combining participant, condition, source file, and source event index. Use the same IDs for both models; temporary DataLoader row numbers are insufficient.

Save a `trial_manifest` containing at least `subject_id`, `trial_id`, source file, original event index/time, original event code, label, split, retention status, and exclusion reason.

## 6. Decoders

### EEGNet

- Train the feature extractor and binary classification head from scratch.
- Reference https://github.com/vlawhern/arl-eegmodels and record the architecture, hyperparameters, implementation source, and version.
- If using a PyTorch port, document its source and verify key layers, input layout, and output class ordering. A shared model name does not establish implementation equivalence.

### CBraMod

- Code: https://github.com/wjq-learning/CBraMod
- Weights: https://huggingface.co/weighting666/CBraMod
- Record the exact code commit, weight filename, revision, and SHA-256.
- Load the pretrained backbone and attach a binary classification head. Default to full-model fine-tuning to implement the proposal's fine-tuning plan.
- If memory limitations appear to require freezing layers or training only the head, first try smaller batches and gradient accumulation. Document any remaining change as a protocol deviation; do not describe a linear probe as full-model fine-tuning.

### Shared Rules

- Use `0 = non-error` and `1 = error`. Exported probabilities must always represent `P(error)`.
- Model-specific learning rates and necessary input adaptations are allowed. Fairness requires shared data splits, transparent validation selection, and an explicit search budget, rather than mechanically identical hyperparameters.
- Initially complete one formal run per model with decoder seed 42. This is the minimum delivery requirement and does not measure variability across decoder-training seeds. Report additional decoder seeds separately from future RL seeds.
- Choose class weighting or sampling strategies using training-set information only, and document their implications for probability calibration.
- Save the best validation checkpoint according to a declared metric and early-stopping rule. Do not assume the final epoch is the best model.

## 7. GPU Execution Sequence

1. Record hardware, GPU memory, driver, framework, and dependency versions. Use an isolated environment.
2. Finish the data audit and freeze the split and protocol decisions before formal training.
3. Use small batches to verify both models' input shapes, units, label direction, finite losses, and checkpoint loading. Check that the models can fit a small training subset.
4. Run a short training benchmark for each model. Measure steady-state epoch time, peak memory, and throughput to estimate full-run duration. Do not report smoke-test results as formal comparison results.
5. Train EEGNet and fine-tune CBraMod from configuration files, preserving logs and best checkpoints. Support interruption and resumption.
6. Fit calibration on validation data, then perform final test evaluation and export.
7. Freeze the networks and calibration parameters. Generate predictions in evaluation mode with dropout disabled and BatchNorm statistics held fixed.

Do not infer a GPU-hour budget from this document. Specify the actual training budget, epoch cap, early-stopping patience, and number of tuning candidates in configurations, and report measured duration. Without a usable GPU, data work and EEGNet CPU debugging can proceed, but unexecuted CBraMod training must not be reported as complete.

## 8. Calibration and Evaluation

- Report at least AUROC, balanced accuracy, and Brier score, along with positive/negative counts, confusion matrices, and classification thresholds.
- Use `P(error) >= 0.5` for balanced accuracy by default. If also selecting a threshold on validation data, report it separately and save the threshold.
- Report pooled test results, results for each test participant, and participant macro-averages. Specify sample sizes and averaging methods. Do not invent seed standard deviations or significance claims without independent repetitions.
- When AUROC is undefined for a single-class subset, report that fact and the reason rather than substituting zero or silently dropping it.
- Temperature scaling fitted only on validation data is an acceptable default; other calibration methods must be documented in advance. Save calibration parameters. Explicitly use an identity calibrator if no calibration is applied.
- Report Brier scores for raw and calibrated probabilities. Do not choose a calibration method because it performs best on the test set.
- This stage evaluates EEG decoding. Do not claim that a decoder necessarily improves subsequent RL performance.

## 9. Interface for Subsequent RL

Export a long-format table at `artifacts/predictions/test_predictions.csv`. Each row represents one trial's prediction from one model/decoder seed. Required fields:

```text
subject_id,trial_id,label,split,model,decoder_seed,p_error_raw,p_error,checkpoint_id,preprocessing_id,calibration_id
```

- Use `eegnet` or `cbramod` for `model`; `split` must be `test` in this file.
- `p_error_raw` is the uncalibrated error probability; `p_error` is the final feedback probability. They are equal when using identity calibration.
- All probabilities must be finite and within [0, 1]. Labels and subject IDs must join correctly to the trial manifest.
- Both models must cover the same test trial IDs for the same decoder seed, with identical corresponding labels.
- Resolve `checkpoint_id` and related identifiers through `artifacts/manifest.json` to configurations, file paths, versions, and checksums.
- Export validation predictions to `artifacts/predictions/validation_predictions.csv` using the same schema as test predictions, with `split=validation`. Generate both tables using the same selected checkpoints and fitted calibration parameters. Include both validation and test prediction tables in the GitHub Release. Validation predictions will support subsequent RL hyperparameter selection; keep them separate from the final test feedback library.

The later RL stage will sample a trial according to a simulated action-segment label and retrieve that trial's probability. Do not apply `1-p`, `0.5-p`, or a feedback weight during this EEG stage. Preserve probabilities so that reward design remains independently configurable.

## 10. Deliverables and Acceptance Checklist

- [ ] Data audit, event/label mapping, and stable trial manifest.
- [ ] Frozen participant splits and preprocessing, model, training, and calibration configurations.
- [ ] Installable dependency specification and complete data preparation, training, evaluation, and export commands.
- [ ] Best checkpoints for both models, calibration parameters, preprocessing settings, and artifact locations/checksums.
- [ ] Best and latest resumable checkpoints for both models uploaded to this project's GitHub Release and verified by downloading them; see Section 11.
- [ ] Metrics, participant-level results, and sample counts on the same test trials for both models.
- [ ] Both `validation_predictions.csv` and `test_predictions.csv`, plus the artifact manifest, satisfying Section 9 and included in the GitHub Release.
- [ ] Verification of disjoint participants, evidence-based label mapping, correct model class ordering, valid probabilities, and trial-aligned predictions.
- [ ] Predictions reproducible within a declared numerical tolerance after reloading saved checkpoints and configurations.
- [ ] `reports/eeg_results.md` describing actual executions, results, duration, limitations, protocol deviations, and unfinished work.
- [ ] README updated with working entry points and a clear distinction between implemented and planned functionality.

CBraMod does not have to outperform EEGNet. Report training failures, missing weights, unexecuted experiments, and unresolved data semantics honestly. Do not replace formal results with example data or simulated metrics.

## 11. Checkpoint Retention, Releases, and Remote Delivery — Required

### What to Preserve

Assign a unique run ID to each formal model / decoder seed / split run. Preserve at least:

- `best.ckpt`: the best checkpoint selected using the predefined validation metric, used for final predictions.
- `last.ckpt`: the latest completed training state for resumption. Saving only the classification head is insufficient.
- Full model parameters, including backbone and head; optimizer; scheduler; AMP scaler if used; epoch/global step; best metric; Python/NumPy/framework CPU/GPU random states; and sampler/generator state where applicable. Mark inapplicable fields explicitly and do not claim unsupported resumption guarantees.
- Model architecture, training configuration, label direction, data split and trial manifest, preprocessing parameters, normalization statistics, calibration parameters, dependency versions, and source commit.
- Training logs, final metrics, and both validation and test per-trial probability tables. Preserve best and latest checkpoints for all formal runs, not only the highest-performing model.

Save resumable states at epoch boundaries using atomic writes to prevent corruption. Save additional periodic checkpoints at a configured interval and declare the retention count. Update `last.ckpt` at least after each completed epoch and `best.ckpt` whenever the validation criterion improves. Include all retained periodic checkpoints in final delivery. Do not clean up local artifacts before verifying their uploads.

### Where to Upload

- Code, configurations, dependency files, audits, and result reports: commit and push to this project's GitHub repository. Do not use an unrelated repository belonging to another account.
- Checkpoints, calibration parameters, manifests, logs, and prediction tables: upload as **GitHub Release assets**. Their exclusion by `.gitignore` is not a reason to omit them.
- Use a unique tag pointing to the actual source commit, such as `eeg-stage1-YYYYMMDD-<short-sha>`. Release notes must include the exact source SHA, run IDs, splits, metrics, hardware, duration, and known limitations.
- Prefer one artifact archive per model/seed/run, plus shared metadata and `SHA256SUMS`. If platform limits require splitting assets, verify the current limit, use multiple parts, and provide reconstruction commands and full-file checksums.
- Do not overwrite or delete existing Releases or assets. Use new tags/run IDs for new experiments. Publish a completed delivery Release only after training and acceptance checks pass; mark interruption backups clearly as incomplete.
- For raw third-party EEG and original pretrained weights, preserve sources, revisions, checksums, and acquisition instructions by default. Preserve the complete newly trained checkpoints. Check relevant model/data licenses before upload and retain required attribution. Do not package unrelated papers or access credentials.

### Verification and Handoff

1. Check GPU, repository push, and Release upload access before long training runs. Report missing permissions before temporary compute is destroyed.
2. During long training runs, periodically back up recent checkpoints to persistent storage. The final destination remains this project's GitHub Release.
3. Download uploaded Release assets into a fresh local directory and verify sizes and SHA-256 checksums. A successful upload command alone is insufficient evidence.
4. Use downloaded `best.ckpt` files to reproduce predictions on a fixed sample set within a declared tolerance. Run a separate short resumption check with downloaded `last.ckpt` files, preserving the originals, to confirm that training state loads and training can continue.
5. Commit `reports/artifact_index.md` containing the Release URL, tag, source SHA, asset inventory, sizes, checksums, and download/extraction/inference/resumption commands. The Release must reference the training source SHA; the artifact index may be committed afterward without moving the original tag.
6. The final response must include repository, code commit, and Release links, plus unfinished items. If checkpoint upload or download verification fails, mark delivery incomplete and retain local artifacts.
