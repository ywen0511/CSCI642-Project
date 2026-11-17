# Execute: CBraMod / EEGNet improvement and complete reproducible delivery

Prepared: 2026-11-10. Destination: **https://github.com/ywen0511/CSCI642-Project**.
Planning branch: **decoder-improvement-v1**.
Historical source: `rl-v2-experiments` at **f6decb7a99628638743a4c126d83668149d5c5c8**.

## 1. Assignment, authority and finish line

**This is the complete instruction for the execution agent. When directed to this document, perform the work; do not merely summarize it or return a proposed plan.** The user should not need to supply another prompt, clone/download anything, prepare an environment, run a command or upload results. Read supporting repository files yourself. Resolve routine implementation choices from this plan and record them. Use the existing authorized compute and authenticated GitHub access. A document cannot supply unavailable hardware, missing credentials or lost files: report a precise blocker only after checking available alternatives, preserve progress, and complete unaffected work.

The user authorizes implementation, training, analysis, code commits/pushes, and publication of all needed project artifacts as GitHub Releases in **ywen0511/CSCI642-Project**, on a new branch. Work in English. Do not change the private Savaw repository, merge into main, buy compute, change access permissions, disclose credentials, overwrite successful historical artifacts, or delete historical branches/tags. Do not send messages to other people.

Your objectives are:

1. Establish the actual completeness of the historical EEG / RL v2 artifacts, recover and publish missing historical files when available, and independently verify that delivery.
2. Implement and execute the bounded decoder study below. Investigate whether a smaller classification head, conservative fine-tuning, filtering and loss choices improve CBraMod; give EEGNet a declared comparison budget.
3. Freeze the selected recipes, train/evaluate both references and candidates, calibrate/export probabilities, and report positive or negative outcomes honestly.
4. Publish source, numerical data, all retained checkpoints/training states, predictions, logs, configs, reports, indices and runnable reproduction tools; verify everything from fresh GitHub downloads.

**A completed experiment is not a completed delivery.** Completion requires the acceptance gate in Section 13. A local upload-ready folder, an existing Release title, a successful API request, or a table of means is insufficient. A candidate beating EEGNet or exceeding any AUROC target is NOT required. Do not keep tuning until a desired story appears.

Scope ends at improved-decoder evaluation and frozen-feedback export/interface checks. Do not rerun the historical full RL suite, add route preference, collect human EEG, or launch new formal robot-learning experiments in this assignment. No new robot-performance claim follows from better decoding alone.

This file supersedes historical execution instructions, including `GPU_AGENT_HANDOFF.md` and the execution directives embedded in `EEG_README.md`. Historical instructions remain useful provenance, not a second active task. `configs/decoder_improvement/study.json` mirrors the numerical plan; verify consistency before any formal run. Document preparation is not experimental evidence.

## 2. Existing evidence and starting points

The historical EEG study is already trained and published. See `reports/eeg_results.md`, `reports/data_audit.md`, `reports/artifact_index.md`, `configs/preprocessing.yaml`, `configs/splits.json`, `configs/models/{eegnet,cbramod}.yaml` and `src/errp/`.

Historical pooled test results on the same three participants:

| model | seed-42 AUROC | five-seed AUROC | five-seed balanced accuracy | five-seed calibrated Brier |
|---|---:|---:|---:|---:|
| EEGNet | 0.637 | 0.613 +/- 0.016 | 0.577 +/- 0.012 | 0.238 +/- 0.006 |
| CBraMod | 0.573 | 0.568 +/- 0.007 | 0.546 +/- 0.010 | 0.245 +/- 0.004 |

These are historical reported results, not measurements from the new study. Seed-42 validation AUROC was 0.645 / 0.638 for EEGNet / CBraMod. CBraMod's temperature was about 9.90. The current CBraMod head flattens 27 x 2 x 200 features and applies Linear(10800,400), ELU, Dropout(0.1), Linear(400,200), ELU, Dropout(0.1), Linear(200,2): **4,401,002 head parameters**. Verify actual checkpoint shapes and counts; some comments in the inherited file describe a different width. Executable shapes and archived tensors take precedence over stale prose.

CBraMod uses official pretrained weights, full fine-tuning, 200 Hz input and microvolts / 100. EEGNet is trained from scratch. Neither a 200 Hz sampling rate nor pretraining alone explains the performance gap. The hypotheses in this document are to be tested, not asserted as established causes. Official sources:

- CBraMod: https://github.com/wjq-learning/CBraMod
- Paper: https://arxiv.org/abs/2412.07236
- Initialization: https://huggingface.co/weighting666/CBraMod
- ErrP-HRI: https://github.com/stefan-ehrlich/dataset-ErrP-HRI

Historical RL v2 reports are under `reports/rl_v2_*`. They report 15 pilots, 99 validation runs and 75 final runs, with formal runs at 150,000 environment steps. The final run CSV was independently checked against the published aggregate means during handoff preparation. This does not independently establish that every checkpoint still exists.

## 3. Workspace, environment and preflight

Start from the current commit of `decoder-improvement-v1`; resolve and record that SHA. Use a clean worktree/new directory and an execution branch **decoder-improvement-v1-experiments** (append a timestamp if that name belongs to different work). Preserve unrelated dirty files and the original branches. Do not blindly pull or reset another working checkout. Prefer reusing an existing repository. If none exists, initialize an empty checkout, add the specified origin, fetch the planning branch and attach to the fetched history; do not create disconnected history. Do not run `git clone` as a user-side prerequisite. All acquisition is your responsibility.

Historical cluster hints, read from prior repository records (check existence rather than assuming):

- EEG project: `/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg`; prior checkout `repo/`, environment `env/`, prepared data `data/derived/prep_v1`, weights `weights/`, runs `artifacts/runs/`.
- RL project: `/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-rl`; prior checkout `repo/`, environment `repo/venv/rl`, runs `repo/artifacts/`.
- Historical scheduler: LSF, GPU queue `gpu_large_shared`, CPU queue `cpu_ubuntu_wrn`. Earlier EEG runs used one NVIDIA A100 per job. These are hints, not a requirement to use an unavailable host or queue.

Inspect scheduler/account resources and durable storage. Use a dedicated new environment; do not upgrade the old one. Root `requirements.txt` / `requirements-lock.txt` describe EEG dependencies; `requirements-closeout.txt` is for CPU diagnostics; `rl/requirements*.txt` belong to RL. Choose an available compatible official PyTorch CUDA build, record actual Python/package/CUDA/driver versions, and prove tensor and checkpoint compatibility with smoke checks. Preserve the exact lock/container description used. A different environment from the historical one must be disclosed, not silently labeled bitwise equivalent.

Preflight report must include source SHA, environment lock, GPU availability, scheduler limits, read/write permissions, input locations, disk estimates, GitHub repository identity/access, and a real small artifact round-trip. Test uploads in a new clearly labeled draft smoke Release owned by this study; do not create an empty final Release. Do not print tokens or dump environment variables. Record commands with secret values omitted. Use one GPU per training job and bounded scheduler concurrency; benchmark before choosing job packing. Do not assume more GPUs improve this small-data workload. Data preparation, metrics and calibration can use CPU.

Maintain `reports/decoder_improvement/status.json` throughout execution, with phase, planned/completed/failed/pending runs, code/config hashes, artifact status, blockers and UTC timestamps. Write an append-only decision log and a job ledger on durable storage. Update and push progress after each major stage. The initial status shipped with this handoff must remain explicitly `prepared_not_executed` until real work occurs.

## 4. Historical artifact recovery: first establish what exists

Inspect the current state anew; assets may have changed since this document was written. Paginate BOTH release lists and each release's asset endpoint (100 per page until an empty/short page), including non-uploaded states. Do not use an embedded/truncated assets array as the complete inventory.

Required historical scopes/tags in this repository:

| tag | scope |
|---|---|
| `eeg-stage1-20261101-bccbb8a` | original EEG: 25 assets, including 19 run archives, predictions and metadata |
| `eeg-stage1-closeout-20261102` | 3 CPU closeout assets |
| `rl-v2-smoke-20261110-cfe0305` | latest and superseded integration checks, labeled separately |
| `rl-v2-pilots-20261110-50888a9` | 15 completed pilot/benchmark runs plus preserved failed attempts |
| `rl-v2-validation-w010-20261110-cfe0305` | 33 validation runs |
| `rl-v2-validation-w050-20261110-cfe0305` | 33 validation runs |
| `rl-v2-validation-w100-20261110-cfe0305` | 33 validation runs |
| `rl-v2-final-20261110-25bb0f4` | 75 final runs, evidence and media |

At handoff inspection, validation Releases contained `starter` assets and the final Release exposed `.part-02` entries without `.part-00` / `.part-01`. This is evidence of incomplete publication then, not proof that training failed or that files remain missing now.

Read the on-disk `status.json`, `run.json`, evaluations, checkpoint metadata and file inventories for EVERY expected historical run. Check actual final steps and files, rather than trusting prose. Compare local files to the historical manifests and remote checksum lists. Categorize each run as complete/recoverable upload, incomplete execution, missing local artifacts, or independently verified remote artifact. Preserve all available periodic/final snapshots, optimizer states, replay buffers, RNG states, logs and failures.

Recover missing archive parts from preserved exact archives or local run directories. A part is only interchangeable with an old part if its bytes/hash match the existing full archive specification. If recompression changes bytes, create a new complete archive with new identity and checksums; do not mix it with old parts. Do not rerun expensive historical training to fill a file hole and silently present it as the original.

Preserve published tags and successfully uploaded assets unchanged. By default create unique **recovery Releases** such as `rl-v2-recovery-final-YYYYMMDD-<sha>` and one per validation candidate needing repair. Prefer complete self-contained recovered suites; if storage/network requires a delta recovery, publish an explicit resolver manifest mapping every expected file/part to its exact old or new Release URL/hash and test the reconstruction automatically. No user should have to guess which parts to combine. Successful original assets are immutable. Existing failed `starter` entries need not be deleted to make a new recovery release.

For historical recovery, verify every downloaded file and archive member; load/probe every final policy and at least one periodic checkpoint per condition per suite; test short isolated continuation for at least one recovered EEGNet and CBraMod decoder and every RL condition in the final suite. Report coverage explicitly. If an upstream input is needed for a probe, acquire it as described below. Never claim a probe was verified from metadata alone.

If originals are truly lost, publish all recoverable files and an exact missing-file report. Continue the new EEG study if independent inputs are available, but historical delivery remains incomplete and the overall status must not say fully complete. A historical recovery blocker must not prevent useful independent decoder work or be concealed in a footnote.

## 5. Data, inputs and leakage controls

### 5.1 Fixed sources and semantics

- Raw dataset repository: https://github.com/stefan-ehrlich/dataset-ErrP-HRI
- Pinned dataset commit: **4bc3e52469934aa0551a2261a6e4e899d156c28f**.
- Read `documentation_dataset-ErrP-HRI.pdf` in that snapshot. Use paired robot `.set` / `.fdt` files; validate the recording metadata. `robot/s12_cursor.set` is a cursor session and is excluded; `s12_robot.set` is retained.
- Marker mapping: **S4 = non-error = 0; S6 = machine error = 1; exclude S5 human-error trials**. S6 cannot be reconstructed from FB direction equality; preserve the documented mapping.
- Expected retained trials: **5,822** from 12 participants; 3,767 label 0 and 2,055 label 1. The 178 S5 trials are the exclusions from 6,000 robot trials.
- Channels, order and linked-mastoid reference: exactly `configs/preprocessing.yaml`; drop EOG channels, preserve the 27 EEG channels. EEG windows are `[0,2)` seconds after FB. No amplitude-based rejection in this study.
- Official CBraMod initialization: HF repository **weighting666/CBraMod**, file **pretrained_weights.pth**, revision **500543c7e30bda1b22bfd51a49301b238dee21fd**. Record SHA-256 and size and check against any existing acquisition records. Use the vendored historical backbone or pin/document an explicitly verified equivalent; do not silently take changing upstream main.
- Baseline feedback and checkpoint assets: original EEG Release above. `configs/rl_feedback.json` pins feedback checksums. Preserve it unchanged.

The agent obtains these inputs directly from pinned archives / official APIs or existing verified copies. Use `data/raw/dataset-ErrP-HRI/`, new versioned prepared directories under `data/derived/decoder_improvement/`, and an initialization directory referenced via `ERRP_WEIGHTS_DIR`. Do not overwrite `prep_v1`.

Existing preparation commands are a starting point, not evidence of new execution:

```bash
python scripts/prepare_data.py --raw-root data/raw/dataset-ErrP-HRI --config configs/preprocessing.yaml --out-dir data/derived/decoder_improvement/band_1_20
```

Implement explicit split views; do not invoke a split generator that redraws participants. Derived epoch order, trial IDs and labels must match across model views and all filter candidates. Record shapes, units, preprocessing hash, raw hashes, reference, filter settings and library versions. Preserve epoch arrays locally until verification succeeds.

### 5.2 Participant split and train-only development folds

Keep the original outer split:

- **Training:** s03, s04, s05, s06, s07, s10, s12 (3,434 trials).
- **Validation/calibration:** s11, s13 (959 trials).
- **Historical test:** s02, s08, s09 (1,429 trials).

All architecture/LR/filter/loss search below uses **only the seven training participants**, with these fixed inner folds:

| fold | inner validation participants | inner training participants |
|---|---|---|
| cv0 | s03, s06 | s04, s05, s07, s10, s12 |
| cv1 | s04, s10 | s03, s05, s06, s07, s12 |
| cv2 | s05, s07, s12 | s03, s04, s06, s10 |

Each training participant is held out once. Estimate normalization and class weights from the current inner-training subset only. EEGNet's global standard deviation must be fitted separately per fold, not on all seven participants. CBraMod's fixed microvolt divisor remains 100. No normalization fit on validation/test data, no random trial split mixing people, no augmentation before splitting, and no self-supervised adaptation on outer validation/test data.

Outer validation s11/s13 is reserved for calibration and fixed-recipe diagnostics after candidate selection. It must not be used to change architecture/LR/band/loss after the train-only search. Test participants have already been inspected in the historical study: call this a **follow-up study on a previously evaluated test split**, not a newly untouched confirmatory test. Do not inspect new test predictions or tune based on new test metrics before the freeze. Do not change splits to get higher scores.

## 6. Implementation and diagnostic gate

Add isolated decoder-improvement code/configs and retain baseline compatibility. Do not change old result files, overwrite old prediction tables, or reinterpret old checkpoints. Provide exact portable CLI entry points for planning, executing one run, summarizing, freezing, final fitting, calibration, export, packaging, download/verification, inference and resume. Every example in the final reproduction guide must run against the delivered source; placeholders in published commands must be resolved automatically or documented as a local path argument, not an unspecified missing script.

Before formal training, verify and record:

1. Dataset counts/labels/trial identities, split disjointness and identical windows across views.
2. Finite voltages and distributions by participant; pre/post-resampling waveform diagnostics and correctly shaped tensors: EEGNet `[N,27,512]`, CBraMod `[N,27,2,200]`.
3. Official weight loading with explicit key/shape checks; correct removal of reconstruction head; actual parameter counts.
4. Trainable parameter names for each adaptation mode; frozen parameters remain byte-identical after an optimizer step; intended unfrozen groups receive finite gradients and change.
5. Frozen modules remain in eval mode during classifier training (no accidental dropout changes when parent `.train()` runs); partial fine-tuning explicitly sets frozen/unfrozen module modes. Do not freeze CBraMod's internal transformer blocks by a guessed name: inspect the actual module tree and save the chosen names.
6. A small, training-only subset can be overfit with the reference full-fine-tuning model and EEGNet. A frozen representation need not perfectly separate the subset; disclose it rather than weaken that check.
7. Save/reload reproduces a fixed inference probe. Resume restores optimizer/scheduler, RNG/sampler/scaler, trainability mask and epoch counters. Interrupted new EEG jobs can resume only if immutable run identity/config/source/input hashes match; identify resumed attempts in the ledger.
8. Metrics/calibration tests cover class direction, single-class undefined metrics, participant macro aggregation, finite probabilities, positive calibration slope and no test access during selection. Cached features must be invalidated by weights/preprocessing/split/input hashes.

A check failure is a diagnostic outcome. Fix implementation defects, log them and rerun affected runs. Do not silently change scientific settings or erase failed attempts. Commit implementation, resolved configs, stage ledger and this fixed plan before running the corresponding formal stage.

## 7. Bounded experiment plan

### 7.1 Common training and selection rules

- Screening decoder seeds: **42,43,44**. Confirmation/final seeds: **42,43,44,45,46**.
- Each candidate is evaluated on all three inner folds and all screening seeds: **9 fits per candidate**. Do not prune based on a single seed or participant.
- Selection score: for each of the seven inner-validation participants, average AUROC across seeds, then average equally across the seven participants. **Do not give the three-person fold the same weight as a two-person fold, and do not choose by pooled trial AUROC.** Report pooled AUROC separately.
- Early checkpoint selection within a fit: highest inner-validation participant-macro AUROC, earliest epoch on exact ties. Undefined per-person AUROC must be reported and treated as a protocol/data failure here (both classes are expected), not replaced with 0.5.
- Candidate ties: score difference <= 1e-6, then fewer trainable parameters, then lexicographically smaller candidate ID. No test score enters selection.
- CBraMod defaults: AdamW; head LR 5e-4; backbone LR as below; weight decay 0.05; batch 64; max 50 epochs; patience 15; cosine schedule over the 50-epoch horizon; minimum LR 1e-6; gradient clip 1.0; no label smoothing; inverse-frequency class-weighted CE unless the loss stage changes it. Use training-only weights. Preserve class direction 0=non-error,1=error.
- EEGNet defaults: historical F1=8,D=2,F2=16,kernel=128,separable kernel=16,pool=4/8,norm_rate=0.25; Adam; weight decay 0; batch64; max300 epochs; patience40; no scheduler; no label smoothing; inverse-frequency class-weighted CE. Scaling is fold-training global SD.
- Start every independent fit from the same pinned pretrained initialization (CBraMod) or declared random initialization (EEGNet). Do not warm-start one candidate from another's selected weights. Frozen-backbone deterministic features can be reused only when the cache provenance matches exactly; classifier initialization still follows the seed.
- Record epochs as both zero-based internal index and one-based completed epoch count to prevent final refit off-by-one errors. Preserve every epoch's metrics and best/last/periodic checkpoints. Log wall time, GPU model, peak memory and trainable counts.

### 7.2 Stage A: CBraMod head x adaptation (7 candidates; 63 fits)

Keep the historical 1-20 Hz band and two-second window. Compare:

| candidate ID | head | adaptation | backbone LR |
|---|---|---|---:|
| cb_flat_frozen | historical flatten MLP | freeze all backbone parameters | none |
| cb_flat_last2 | historical flatten MLP | unfreeze only final two transformer blocks | 3e-5 |
| cb_flat_full | historical flatten MLP | full backbone fine-tuning | 3e-5 |
| cb_timepool_frozen | temporal-mean linear | freeze all backbone parameters | none |
| cb_timepool_last2 | temporal-mean linear | unfreeze only final two transformer blocks | 3e-5 |
| cb_timepool_full | temporal-mean linear | full backbone fine-tuning | 3e-5 |
| cb_reference | historical flatten MLP | historical full fine-tuning | 1e-4 |

The **temporal-mean linear head** averages only the two time-patch representations, retains the 27 channel positions, flattens 27 x 200 features, and applies Linear(5400,2), i.e. **10,802 parameters**. It is not global channel-and-time pooling, and it is not the inherited `head=avgpool` implementation. Use a new explicit name. The last-two-block mode freezes patch embeddings, positional/spatial encodings, and every earlier block; any extra final normalization outside those blocks stays frozen. Record exact names. The head is always trainable.

Select the Stage A champion using the rule above; keep all outcomes. A frozen large head is still a high-capacity model, so do not equate frozen backbone with low overall capacity.

### 7.3 Stage B: conservative backbone LR refinement (at most 18 additional fits)

If the champion is frozen, skip with a recorded reason. Otherwise hold its head/adaptation fixed and compare backbone LRs **1e-5,3e-5,1e-4** with head LR unchanged. Reuse exact matching prior fits; at most two new candidates, 18 fits. Select across these LRs including the incumbent. Do not expand the grid opportunistically. This tests a local refinement; do not claim a global optimum.

### 7.4 Stage C: EEGNet comparison search (4 candidates; 36 fits)

At 1-20 Hz, keep architecture and other defaults unchanged. Cross learning rate **[1e-3,3e-4]** with dropout **[0.5,0.25]**, giving four candidates. The `1e-3,0.5` recipe is the EEGNet reference. Select by the same participant-macro rule. This search budget differs from CBraMod's and must be disclosed; the study is an improvement investigation, not an equal-budget model leaderboard.

### 7.5 Stage D: shared filtering comparison (at most 36 additional fits)

For the selected recipe of each architecture, compare **1-20,0.3-40,0.3-75 Hz**. Reuse the exact 1-20 results; two new bands x two models x nine fits = at most36. Refilter the original continuous recordings; never try to recover discarded frequencies by refiltering already 1-20 Hz epochs. Keep two-second windows, trial inclusion, channel order/reference, sampling rates and amplitude conventions fixed.

Use the historical zero-phase FIR approach with explicit resolved transition bands/padding/version in the config. Keep notch filtering unchanged (none newly introduced) in this stage so it is a band comparison; report line-noise diagnostics for the broad band and do not call 0.3-75 an exact reproduction of upstream preprocessing. Persist a new preprocessing ID for every band. Select each architecture's band using train-only CV. Report all six model/band results, including when model-specific choices differ. The narrower/broader input is a hypothesis, not a presumed improvement.

### 7.6 Stage E: loss comparison (at most 18 additional fits)

For each architecture's selected recipe/band, compare the incumbent inverse-frequency weighted CE with **unweighted CE**. Reuse the incumbent; two new candidates x nine fits =18. Keep the selected recipe only if it wins the same AUROC-based rule. Record Brier/NLL/probability separation for both even if the AUROC-selected candidate has worse probability quality. Do not change the primary selection rule after seeing that tradeoff. Calibration is separate.

### 7.7 Stage F: five-seed confirmation and freeze (at most 24 additional CV fits)

Freeze the selected candidate of each architecture from Stages A-E. Extend each candidate AND its historical-recipe reference to seeds45/46 on all three folds; reuse seeds42-44. At most four recipes x three folds x two extra seeds =24. Do not reselect architecture/LR/band/loss using this confirmation. If a winner is identical to its reference, deduplicate and explicitly report no distinct candidate. Show per-person/per-seed paired differences and the degree of selection optimism.

Then commit a **pre-test freeze** recording selected/reference resolved configs, CV predictions/scores, seeds, code/input hashes, final training epoch counts, calibration procedure, metrics and report conventions. Epoch count for each recipe is the rounded-up median of that recipe's selected one-based best epochs over its 15 CV fits, minimum1, capped at the original maximum. All five final seeds for that recipe use that fixed count. The scheduler retains its original configured maximum-epoch horizon (50 for CBraMod, none for EEGNet), rather than silently compressing the cosine cycle to the selected count. Different recipes may have different selected counts; report them.

### 7.8 Stage G: full-training refits (at most20 fits)

Train the selected candidate and the reference recipe of each architecture on ALL seven training participants for seeds42-46 using the fixed epoch counts. No early stopping or checkpoint selection using outer validation/test; the fixed-epoch final snapshot is the evaluation model. Fit normalization and weights using those seven participants only. These are **controlled new refits**, not reproductions of the historical validation-selected training runs. Present historical archived results in a separate row/table. Reuse a fit only if complete recipe, epoch schedule, seed, input/split and source identities match exactly.

### 7.9 Budget and run identity

Maximum distinct formal fits: Stage A63 + B18 + C36 + D36 + E18 + F24 + G20 = **215**. Exact duplicates reduce this count. Calibration fits, feature extraction, short smoke tests and explicitly recorded failed attempts/retries are accounted for separately. Do not silently spend this allowance on extra hyperparameter searches. Do not count reused runs multiple times or omit losing candidates.

At preflight benchmark a representative full CBraMod fold and EEGNet fold, estimate total GPU hours and disk needs, and record them. Use existing allocation; do not buy resources. If hard quotas make the required plan infeasible, stage/deliver completed work with a precise blocker and an explicitly amended plan BEFORE dependent runs; never report a reduced search as the full declared study. Do not ask the user to make ordinary scheduling decisions.

Every run has a canonical ID derived from stage, recipe hash, preprocessing hash, fold/outer split and seed. Save full resolved config, source commit AND training-file hashes, data lineage, initialization hash, resume provenance, status and actual epochs. Cache hits require validated file membership/hashes and successful completion, not just an existing directory.

## 8. Calibration, evaluation and frozen probability exports

Use margin z = logit_error - logit_nonerror. For the controlled comparison, fit the existing **temperature calibration** on s11/s13 for every final model and report raw and temperature-calibrated metrics for all recipes/seeds. It preserves ranking; a threshold shift is not an AUROC improvement.

For selected candidates only, investigate a deployment calibrator among:

- identity: p=expit(z);
- temperature: p=expit(z/T), T in[0.05,20], minimize mean NLL;
- affine: p=expit(a*z+b), a>0, log(a) in[-log20,log20], b in[-5,5], minimize mean NLL + 1e-3*(log(a)^2+b^2), using stable numerical functions.

Choose a single method per architecture with validation-participant cross-fitting: fit on s11 and score NLL on s13, then reverse; average equally over participants and all five decoder seeds. Ties within1e-6 prefer identity, then temperature, then affine. Identity has no fitted parameters. Fit the selected method separately for each seed on both validation participants using equal participant weighting. Record parameters, optimization success/bounds and fitting IDs. If a fit fails, try the next method in the pre-test validation ranking, for the whole architecture rather than cherry-picking by test performance; log the fallback and refreeze. Do not silently relabel identity as a successful fit.

The two validation participants are a small calibration-development set; cross-fitting them does not yield a new independent test. Their probabilities are calibration-development outputs. They must not later be presented as unbiased held-out feedback quality; a future RL tuning study must account for this reuse.

Commit the calibration-method decision and all fitted parameters before new test evaluation. Evaluate all predeclared final models once as a batch on s02/s08/s09. For each recipe/seed report pooled AND per-participant AUROC, participant-macro AUROC, balanced accuracy at0.5, sensitivity/specificity, confusion matrix, Brier, NLL, 10-bin ECE, mean p by label, label-mean gap, and fraction of each class below0.5. Report identity/temperature for controlled comparisons; selected deployment calibration is a clearly separate probability-quality comparison. Never choose the best test seed, threshold, band or calibrator.

Summaries: seed mean +/- sample SD for pooled and macro metrics; participant-level tables; paired differences on the same people/seeds. State that seed SD is not population uncertainty and trials are not independent participants. With three test participants, do not claim population-level significance, equivalence or causation from a favorable average. No universal success threshold such as AUROC0.70 is imposed. If gains are absent, conclude that the tested changes did not establish improvement.

Export both raw and calibrated predictions for **every final seed**, with subject_id, trial_id, label, split, model, recipe_id, decoder_seed, logits/margin, p_error_raw, p_error_temperature, p_error (selected deployment calibration for candidates; temperature for references), checkpoint_id, preprocessing_id, scaler_id and calibration_id. Write a new schema version and manifest mapping IDs to immutable files/hashes. Keep historical `validation_predictions.csv`, `test_predictions.csv` and `configs/rl_feedback.json` unchanged.

For downstream interface delivery designate **seed42 in advance** for each selected candidate (even if another seed scores higher), and export a separate new `configs/decoder_improvement/rl_feedback_vnext.json` and corresponding validation/test probability tables. Make loaders accept an explicit profile rather than replacing the old default; validate source identity, exact expected participants/trials, label direction, finite [0,1] probabilities and hashes. Run CPU unit/interface checks of matching-label and independent sampling and reward calculation. No new formal RL training is required. Declare in the profile which outputs reuse calibration participants. Keep these exports available even if performance does not improve, labeled experimental follow-up.

## 9. Save every artifact needed to understand and reproduce the study

Required for ALL smoke, screened, losing, selected, reference, final, failed and resumed attempts:

- Resolved configs; input/preprocessing/initialization/split/source hashes; architecture and trainable parameter names/counts; exact CLI; environment and hardware; seed; status and timing; stdout/stderr and scheduler logs.
- Every epoch's training/validation metrics; per-trial inner-validation logits/probabilities at the selected checkpoint; checkpoint selection rationale; learning curves as numerical CSV/JSON and figures.
- **Best, last, final where distinct, and every periodic checkpoint generated in this study.** Save periodic CBraMod snapshots every5 epochs and EEGNet every25; disable inherited periodic-file pruning for new runs. Preserve all distinct best snapshots if the runner versions them. Do not retain only the winning candidate. Content-addressed deduplication is allowed only with an explicit complete logical-to-physical mapping and verified reconstruction.
- Full resumable state: model (backbone+head), optimizer, scheduler, RNG for Python/NumPy/CPU/CUDA, sampler/generator state, scaler, epoch/step, early-stop counters, adaptation mask/modes, class weights and initialization provenance. Inference-only weights alone are insufficient.
- Fixed probe input/outputs or a reproducible trial-ID/input-hash recipe where signal redistribution is restricted; calibration parameters and all validation/test predictions.
- CV fold definitions, trial manifests/exclusion reasons, raw-file inventory, all preprocessing configs/hashes, candidate/selection/freeze evidence, figures and their source tables, final report, exact reproduction commands and verification logs.
- Deterministic feature-cache lineage and generation command. Publish caches/derived arrays if redistribution is permitted and useful; otherwise provide hashes, shapes, derivation recipes and automated regeneration from the exact upstream inputs. No cache may be the only copy of an irreplaceable output.

Keep inputs read-only, version outputs, and store on durable storage. Do not delete local artifacts until a complete remote download has passed verification, including resumed-state checks. An ignored file is not an exempt deliverable.

## 10. Data availability and external dependencies

The request is for a reproducible delivery, including needed data. Package every distributable project-generated numerical output, prediction, metadata file, trained checkpoint and log. For upstream raw EEG and pretrained initialization, inspect the original terms before redistributing; the historical audit records no declared license for the raw dataset. Do not assume public download grants permission to rehost raw EEG or derived signal arrays. Do not fabricate a license or bundle third-party data merely to make a Release look self-contained.

If redistribution is allowed, include the relevant inputs/derived arrays with attribution and terms. If not established, deliver **an executable pinned acquisition/regeneration path**, not a vague web link: exact repository/revision, relative files, expected sizes and SHA-256, source URLs, fetch command/tool, required access, and deterministic preprocessing instructions. Verify a fresh acquisition and regeneration. Publish `external_inputs.json` and `DATA_AVAILABILITY.md` describing exactly which bytes are external and why. Public metadata/probability exports already distributed in the original project remain part of delivery; review any newly proposed raw-signal-bearing output separately.

Use a single portable acquisition command implemented by you that obtains these external inputs and fails loudly on checksum/version mismatches. Network loss or an unavailable upstream file must yield an explicit blocked status. Do not claim an entirely self-contained offline bundle if some inputs are external. Do not ask the user to locate or download them manually. Any private cluster path in public runtime logs must be replaced with a documented placeholder while preserving commands and scientific values; secrets and unrelated files must never be packaged.

## 11. Publication and robust upload requirements

Push source/configs/tests/small reports to the execution branch in **ywen0511/CSCI642-Project**. New study tags must point to actual pushed commits. Record experiment-source SHA per run and delivery-tool SHA separately. Do not move tags. Original branches/figures/results remain historical; put new results under `reports/decoder_improvement/`.

Use scoped Releases such as:

- `decoder-v1-smoke-YYYYMMDD-<sha>`;
- `decoder-v1-cv-YYYYMMDD-<sha>` (all screening, refinement, confirmation fits and failures, not just winners);
- `decoder-v1-final-YYYYMMDD-<sha>` (all final candidate/reference fits, calibration, predictions, report and reproduction bundle);
- separate historical recovery Releases as needed.

Choose actual unique tags, not literal placeholders. Large CV payloads may be sharded into additional scoped Releases, with one top-level resolver manifest. Prepare final Releases as drafts until their uploads and authenticated round-trip checks pass; then publish and independently verify public URLs. A draft or partial Release is not a final deliverable. Preserve old published releases.

Implement idempotent, bounded-retry upload handling with backoff and an on-disk ledger. List actual remote assets with pagination and inspect `state`, size, hashes/digests when available. A `starter` or zero-size entry is not uploaded. For this study's own draft uploads, repair failed entries or use unique replacement names and update the unpublished manifest; never overwrite a successful published asset with different content. Do not run a helper that silently deletes arbitrary historical entries. Treat repeated network failure as a delivery blocker with preserved artifacts, not an excuse to claim completion.

If large transfers are unreliable, use complete archives split into **50 MiB (52,428,800-byte)** parts, consistently numbered from00, with an explicit ordered part list, part count, part size/hash and full concatenated archive size/hash. Smaller parts are permitted only with a recorded upload-only decision; scientific contents must remain identical. Upload every part; shell wildcards or `.part-02` alone are not an inventory. Provide an automated downloader/reassembler, not an instruction for the user to hunt for parts. Archive extraction must reject absolute/traversal paths.

Manifest requirements:

- Every release asset: logical name, scope, actual URL, byte size, SHA-256, uploaded state.
- Every archive: full-file size/hash, ordered parts if any; every member's relative path/size/hash.
- Every run: source/input/config IDs, fit identity/status, epoch counts, all logical checkpoints and exact asset/member locations.
- Every checkpoint: role, epoch/step, trainability mode, architecture, model/optimizer/RNG availability, SHA-256; explicit alias if deduplicated.
- All externally acquired inputs and reproduction routes; failures/missing files explicitly listed.
- `SHA256SUMS` covers payload assets excluding itself. Avoid self-referential hashes: keep the payload manifest independent of its own checksum; checksum it in a separate checksum file. Verification evidence can be an additive follow-up asset/release with its own manifest.

Preserve exact unredacted runtime provenance privately where needed; publish a sanitized but reproducible form. Do not upload raw credentials, original papers unrelated to reproduction, environment directories, or dataset copies with unestablished redistribution permission.

## 12. Fresh-download reproduction and verification

Use a NEW empty verification directory, distinct from training outputs and local staging. Download the actual GitHub assets; do not copy the local originals and call that a download. Once published, use anonymous URLs where the repository is public. Record URL/status/size/hash and any required authentication; never record secrets.

Required checks:

1. Paginate remote inventories, compare exact expected names/counts/states/sizes. No unexplained missing, duplicate, zero-byte or `starter` asset. Check every file's SHA-256, reassemble every split archive, check full-archive SHA-256, and verify every extracted member against the complete manifest.
2. Obtain the exact delivered source commit and install the locked CPU/GPU environments. Execute the documented acquisition, preprocessing and evaluation/reproduction entry points verbatim. Check trial membership/labels/order and prepared-array hashes where byte determinism is expected; document measured numerical tolerance for cross-version regeneration rather than claiming byte identity without evidence.
3. Recompute all numerical tables from downloaded predictions/logs. Assert counts, means/SDs, primary score, candidate selection, epochs and calibration choices. Regenerate all report figures; numerical source data must match. Verify every number in the final report via a machine-readable table-to-source map.
4. Load/probe **every selected/best checkpoint for every completed CV fit and every final refit**, plus the first periodic checkpoint for each unique architecture/head/adaptation combination. Verify SHA-256 for ALL other snapshots even if not loaded. Use downloaded files and pinned original inputs. Reproduce final validation/test probabilities from every final checkpoint, compare raw/calibrated probabilities to the published exports, and measure maximum differences.
5. Predeclared same-software/device inference tolerance is max absolute probability difference <=1e-6. CPU-vs-GPU comparison may use <=1e-5, reported separately with measured differences. Do not relax tolerances after failure to hide a mismatch; investigate and preserve failing evidence. Cross-version replay limitations must be explicit.
6. In isolated copies, resume a downloaded last checkpoint for TWO additional epochs for each unique CBraMod head/adaptation combination that actually ran, both EEGNet dropout settings, and the final selected recipes. Verify restored optimizer/scheduler/RNG/scaler/trainability, epoch/update progression, finite loss and unchanged original hashes. A frozen backbone must remain unchanged; trainable layers must update. Where deterministic continuation is supported, compare against an uninterrupted short reference; otherwise state the exact tested guarantee.
7. Verify the new feedback schema/profile on CPU, including deterministic paired sampling and reward calculations; verify old feedback checks still pass.

Download verification can be disk-heavy. Verify/extract sequentially and remove ONLY regenerable verification copies after their checks are durably recorded. Never delete the sole original or an unverified archive to free space. Publish complete machine-readable and readable verification reports, including failure logs and eventual resolution.

Avoid a circular completion claim: publish payloads, verify the public payloads, commit the verification report, then publish/download-check that small verification evidence separately. The final status links the evidence commit/asset and all payload releases. Historical payload tags are not moved to include later verification.

## 13. Required reporting and acceptance gate

Write/update:

- `reports/decoder_improvement/preflight.md` and `input_audit.json`;
- `reports/decoder_improvement/historical_delivery_audit.json` and `.md`;
- `reports/decoder_improvement/decisions.jsonl`, `run_index.json`, `status.json`;
- resolved candidate configs, CV fold file, selection/freeze/calibration decisions with hashes;
- `reports/decoder_improvement/results.md`, `final_report.md`, figure/data tables and `number_audit.json`;
- `reports/decoder_improvement/REPRODUCE.md`, `DATA_AVAILABILITY.md`, `external_inputs.json`, `artifact_index.md` and verification reports;
- top-level README pointing to actual new results, current completion state and Releases;
- a new `reports/decoder_improvement/rl_feedback_handoff.md` explaining the new probabilities, fixed seed42 interface, label/calibration semantics, original-vs-new distinction and the absence of new RL performance results.

The final report must distinguish: historical archived results; controlled new reference refits; selected candidate results; train-only CV selection; five-seed confirmation; calibration-development scores; previously evaluated outer test results. Explain why improvements may or may not occur; avoid causal claims from one changed recipe with several cumulative differences. Include the head/adaptation matrix and each later stage's full outcomes. Explain sequential-search limitations, unequal architecture search budgets, seven-person CV and three-person test limitations, and any departures from original input/epoch settings. Do not describe old RL results as using new decoders.

Mark completion only if all applicable items pass:

- [ ] Actual historical run/artifact audit performed; every recoverable missing item published and verified; any irrecoverable gap remains explicitly unresolved rather than checked off.
- [ ] Implementation/input/weight/trainability/metric/resume checks passed with saved evidence.
- [ ] Every required candidate/fold/seed fit completed or exact duplicate reused with provenance; all failures/retries retained and counts reconciled with the <=215-fit plan.
- [ ] Candidate/epoch/calibration decisions committed before new test access; no test-driven reselection.
- [ ] All required final models evaluated; all seeds and participant outcomes reported; no cherry-picking.
- [ ] New probabilities/profile exported and tested without replacing the historical feedback.
- [ ] All required data outputs, checkpoints, full resumable states, logs, configs and manifests uploaded to the specified repository; external inputs have a tested acquisition route and explicit availability statement.
- [ ] Fresh remote downloads, all hashes/member inventories, table/figure regeneration, all required inference probes and continuation checks passed.
- [ ] README, reports, status, indices, Release notes and verification evidence agree; no placeholders, fake success flags or dangling artifact links.

If the scientific study completes but delivery is blocked, status must say **experiments_complete_delivery_incomplete**, with exact missing items and durable locations. If new decoding does not improve, a complete verified negative result is an acceptable outcome. If a historical gap cannot be recovered, distinguish `new_study_complete` from `historical_delivery_incomplete`; do not label the combined assignment fully complete.

Final response to the user must include: execution branch URL and SHA, all Release URLs, actual run counts and compute usage, concise reference-vs-candidate metrics with uncertainty definitions, what changed, positive/negative conclusions and limits, reproducible commands, verification coverage/results, and every concrete unresolved item. Do not ask the user to perform the final uploads or accept a statement of intent as delivery.
