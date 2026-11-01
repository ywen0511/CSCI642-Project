# EEG Stage Results: EEGNet vs CBraMod on ErrP-HRI (robot condition)

Date: 2026-11-01. All numbers come from executed runs. The full generated tables are in
[`reports/generated/results_tables.md`](generated/results_tables.md), produced by
`scripts/write_results_tables.py` from `artifacts/metrics_summary.json`.
Data audit: [`reports/data_audit.md`](data_audit.md). Frozen protocol: [`reports/protocol_decisions.md`](protocol_decisions.md).

## 1. What was executed

| step | status |
|---|---|
| Data download, audit, label mapping, trial manifest | done (5,822 retained trials, 12 participants) |
| Frozen split (seed 42) and protocol | done before formal training (commit `76809dd`) |
| Smoke tests: shapes, units, label direction, finite loss, checkpoint loading, subset overfitting | done (EEGNet and CBraMod both fit a 64-trial subset) |
| Short benchmark (3 epochs) and resume check | done; resuming from `last.ckpt` is bitwise identical to an uninterrupted run (max weight difference 0.0) for both models |
| Formal training, 2 candidates per model, selection on validation AUROC | done |
| Temperature calibration on validation, test evaluation, export | done; all export checks passed |
| Extra decoder seeds 43–46 (selected settings) | done (reported separately) |
| Supplementary baselines (7) | done (reported separately) |
| Commit and push to `ywen0511/CSCI642-Project`, GitHub Release, download verification | done (Section 8) |

Source commits: training code `3b95649` (recorded in every checkpoint); evaluation, export and packaging
scripts `bccbb8a` (training code identical, `git diff 3b95649 bccbb8a -- src scripts/train.py` is empty).

## 2. Hardware and software

- GPU: 1× NVIDIA A100-SXM4-40GB per job (node `dcwipphhgc175`, GM Warren LSF, queue `gpu_large_shared`),
  driver 580.105.08. CPU-only steps (downloads, environment, data preparation, xDAWN, packaging) ran on
  the `cpu_ubuntu_wrn` queue.
- Python 3.11, torch 2.15.0+cu132 (CUDA 13.2), braindecode 1.8.1, MNE 1.13.2, NumPy 2.2.6, SciPy 1.17.1,
  scikit-learn 1.9.1, pyriemann 0.12 (`requirements.txt`, `requirements-lock.txt`).
- fp32, cuDNN deterministic algorithms, single process, data held on the GPU.

## 3. Main comparison (decoder seed 42, test participants s02, s08, s09)

Test set: 1,429 trials (501 error, 928 non-error). Threshold P(error) ≥ 0.5.

| | EEGNet | CBraMod |
|---|---|---|
| selected run | `eegnet_lr1e-3_s42` | `cbramod_bblr1e-4_s42` |
| best validation AUROC (selection) | 0.645 (epoch 114) | 0.638 (epoch 24) |
| **test AUROC (pooled)** | **0.637** | **0.573** |
| test balanced accuracy @ 0.5 | 0.590 | 0.559 |
| sensitivity / specificity @ 0.5 | 0.463 / 0.717 | 0.463 / 0.654 |
| confusion @ 0.5 (TN / FP / FN / TP) | 665 / 263 / 269 / 232 | 607 / 321 / 269 / 232 |
| test Brier, raw → calibrated | 0.228 → 0.227 | 0.363 → 0.244 |
| test NLL, raw → calibrated | 0.656 → 0.649 | 2.219 → 0.684 |
| test ECE (10 bins), raw → calibrated | 0.098 → 0.109 | 0.343 → 0.124 |
| temperature T (fitted on validation) | 1.39 | 9.90 |
| validation-selected threshold (Youden) / test bal. acc there | 0.457 / 0.592 | 0.514 / 0.556 |
| participant macro-average AUROC / bal. acc (3 participants, unweighted) | 0.649 / 0.589 | 0.574 / 0.559 |

Per participant (AUROC / balanced accuracy @ 0.5): EEGNet s02 0.639 / 0.590, s08 0.614 / 0.557,
s09 0.694 / 0.619; CBraMod s02 0.553 / 0.557, s08 0.536 / 0.519, s09 0.632 / 0.601. AUROC is defined for
every participant (both classes present).

Interpretation (descriptive only):

- With this protocol, EEGNet decodes held-out participants better than fine-tuned CBraMod on every test
  participant. Both models reached similar validation AUROC (0.645 vs 0.638), so CBraMod's
  validation-to-test drop was larger (−0.065 vs −0.008).
- Fine-tuned CBraMod is strongly overconfident (raw NLL 2.2; T = 9.9). Validation temperature scaling
  removes most of this on the test set (Brier 0.363 → 0.244), but EEGNet stays better calibrated.
- For EEGNet, temperature scaling improved test NLL and Brier slightly but made the 10-bin ECE slightly
  worse (0.098 → 0.109). The calibrator was fixed in advance and fitted on validation only, so this is
  reported as is.
- Both decoders are weak in absolute terms (AUROC 0.57–0.64 on unseen participants). Kim reports
  70–90% accuracy, but with LOSO, on unspecified epochs and labels; those figures are not comparable to
  ours.

## 4. Decoder-seed variability (selected settings, seeds 42–46)

| model | pooled test AUROC | bal. acc @ 0.5 | Brier (cal.) | participant-macro AUROC |
|---|---|---|---|---|
| EEGNet (n = 5) | 0.613 ± 0.016 | 0.577 ± 0.012 | 0.238 ± 0.006 | 0.616 ± 0.021 |
| CBraMod (n = 5) | 0.568 ± 0.007 | 0.546 ± 0.010 | 0.245 ± 0.004 | 0.569 ± 0.008 |

Mean ± sample sd over 5 decoder-training seeds on the same split and the same 3 test participants. Each
EEGNet seed (0.598–0.637) scored above each CBraMod seed (0.560–0.579) in pooled test AUROC. Seed 42 is
EEGNet's best seed, so the formal-run gap (0.064) is larger than the mean gap (0.045). The sd measures
variability across decoder seeds, not across participants or splits. No significance test was run;
with 3 test participants, participant-level inference is not possible.

## 5. Supplementary baselines (same split, trials, windows, loss, selection rule and calibration; one run each, no tuning)

| model | pretrained | val AUROC | test AUROC | test bal. acc | Brier raw → cal. |
|---|---|---|---|---|---|
| EEG-Conformer (2023) | no | 0.641 | 0.616 | 0.584 | 0.251 → 0.247 |
| EEG-Inception-ERP (2020) | no | 0.652 | 0.612 | 0.565 | 0.236 → 0.234 |
| EEGNeX (2024) | no | 0.659 | 0.607 | 0.563 | 0.237 → 0.237 |
| CTNet (2024) | no | 0.631 | 0.607 | 0.559 | 0.243 → 0.233 |
| xDAWN + Riemannian TS + LR | no | 0.603 | 0.574 | 0.558 | 0.258 → 0.250 |
| LaBraM-base (ICLR 2024) | yes | 0.581 | 0.572 | 0.561 | 0.293 → 0.239 |
| EEGPT (NeurIPS 2024) | yes | 0.647 | 0.566 | 0.555 | 0.273 → 0.242 |
| *EEGNet (main)* | no | 0.645 | 0.637 | 0.590 | 0.228 → 0.227 |
| *CBraMod (main)* | yes | 0.638 | 0.573 | 0.559 | 0.363 → 0.244 |

The supplementary results follow the same pattern: compact models trained from scratch (0.607–0.637) do
better on held-out participants than the three fully fine-tuned foundation models (0.566–0.573), which also
generalise worse from validation to test. These runs use default hyperparameters without any search, so
they are a sanity check, not a tuned leaderboard. REVE was not run because its weights are gated.

## 6. Runtime (measured)

| job | GPU wall time |
|---|---|
| smoke tests + benchmark (2 jobs) | 105 s + 146 s |
| EEGNet grid (2 candidates) | 104 s |
| CBraMod grid (2 candidates) | 252 s |
| 4 from-scratch supplementary models | 322 s |
| LaBraM + EEGPT | 442 s |
| selection, calibration, evaluation, 8 extra-seed runs, export | 741 s |
| **total** | **≈ 35 GPU-minutes** (always 1 GPU per job) |

Steady state: EEGNet 0.31 s/epoch (2,562 parameters, 446 MB peak); CBraMod 2.7–2.9 s/epoch
(9.28 M parameters, 1.0 GB peak); EEGPT 17.9 s/epoch (25.3 M, 8.3 GB). Wall time also includes atomic
checkpoint writes to the NFS filer. Formal-run training budgets: EEGNet 154 epochs (cap 300, patience 40),
CBraMod 39 epochs (cap 50, patience 15).

## 7. Protocol deviations and limitations

Deviations from the plan in `protocol_decisions.md`:

1. The environment ended up on torch 2.15.0 + CUDA 13.2 instead of the planned 2.6 + CUDA 12.4:
   installing braindecode upgraded torch. The smoke tests ran on that environment and every run used it.
2. The first split file was generated from data with a buggy marker-consistency rule (all S6 trials were
   wrongly excluded, so the eligible list was empty). It was regenerated once after the fix, before any
   training (`protocol_decisions.md`, Splits).

Limitations:

- Only 3 test participants (1,429 trials). Participant-level conclusions are tentative, and differences
  between models are not tested for significance.
- Epoch origin: we use the FB marker. The documentation places the start of the robot's head movement at
  the key press, 47–130 ms earlier. Both origins capture the ErrP latency range, but they are not identical.
- S6 (machine error) cannot be checked against the direction markers, because FB always records the
  commanded direction. The label rests on the documentation and the per-block error-rate evidence.
- In some trials the 2-s window includes the robot's return movement and the next stimulus. These events
  are independent of the label.
- EOG channels are dropped, but frontal EEG channels still carry ocular activity. Eye movements that differ
  between conditions could contribute to decoding for every model.
- Shared 1–20 Hz input. CBraMod was pretrained on 0.3–75 Hz data, so the narrow band may handicap it.
  A CBraMod-native band was not tried, because both main models must use the same physical windows and band.
- Inverse-frequency class weights move raw probabilities toward 0.5. Temperature scaling has no bias term,
  so it cannot fully undo that shift (for example, validation mean calibrated P(error) is 0.469 for EEGNet
  and 0.456 for CBraMod, against an error rate of 0.361).
- The EEGPT amplitude unit (mV) and the LaBraM temporal-embedding truncation are adaptation choices for
  the supplementary baselines and were not verified against those papers' pipelines.

## 8. Delivery status and unfinished work

- Code pushed to branch `eeg-stage1` of `ywen0511/CSCI642-Project`.
- GitHub Release [`eeg-stage1-20261101-bccbb8a`](https://github.com/ywen0511/CSCI642-Project/releases/tag/eeg-stage1-20261101-bccbb8a) (tag on source commit `bccbb8a`) holds all 25 assets
  (5.5 GB): checkpoints (`best.ckpt`, `last.ckpt`, periodic) and logs for every run, calibration parameters,
  both prediction tables, `manifest.json`, `metrics_summary.json`, shared metadata, `SHA256SUMS`.
- Download verification: all assets were downloaded from GitHub into a fresh directory on the project filer
  (LSF job 7988573). Every SHA-256 and every size reported by the GitHub API matched. A 1-GPU job on gc175
  (LSF job 7988574, 87 s) then extracted the archives, reproduced validation and test predictions from the
  downloaded `best.ckpt` files (max |Δp| 5e-11, tolerance 1e-6) and resumed training from the downloaded
  `last.ckpt` files for 2 more epochs (EEGNet 154 → 156, CBraMod 39 → 41; finite losses, source files unchanged).
  Details in `reports/artifact_index.md`.
- Out of scope for this stage and not done: SAC/robot experiments, LOSO reproduction.
