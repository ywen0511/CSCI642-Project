# Protocol Decisions (frozen before formal training)

Frozen on 2026-11-01, before any formal training run, in the git commit that adds this file.
Smoke tests (subset overfitting, a 3-epoch throughput benchmark, resume checks) were run before
freezing; their scores were not used to choose any setting below.

Frozen artifacts and SHA-256:

| file | SHA-256 |
|---|---|
| `configs/splits.json` | `7d34b0b975cb2a2d36cde1b18b2079526b35095631cac159c93e1bc6c722aa49` |
| `configs/preprocessing.yaml` (preprocessing id `prep_v1-13fe09dfbe64`) | `48808542393a8e383e0143e2f2b9d5ba82dc0e4e7a4b80219cc926f092ad3ed0` |
| `configs/models/eegnet.yaml` | `823329ca1e979ec53d10ebbbfaaef771341c4183976fc51bab2b146aec0044bc` |
| `configs/models/cbramod.yaml` | `faf76cf8226f3aba4b831c5e82fc42abddeef2ab25159542baf072e4008bc262` |

Source codes used in the table:
**P** = explicitly stated in Kim's paper(s); **D** = confirmed by dataset documentation and/or the files;
**M** = required by the official model interface / checkpoint; **C** = chosen for this project.

## Kim's papers

- Short paper: arXiv 2507.13171 v2 ("Aligning Humans and Robots via Reinforcement Learning from Implicit
  Human Feedback"). Master's thesis: arXiv 2512.00050 v1 ("Reinforcement Learning from Implicit Neural
  Feedback for Human-Aligned Robot Control", Korea University, 2025).
- The EEG paragraphs are identical in both versions: "32-channel actiChamp system at 1000 Hz and
  downsampled to 256 Hz. We applied bandpass filtering between 1–20 Hz and re-referencing, and
  segmented the data into 2-second windows aligned to the feedback onset. Each segment was labeled
  as either an error or non-error event ... EEGNet-based classifier in a leave-one-subject-out (LOSO)
  cross-validation setting." Both state that EEG was replayed from an offline dataset, that the
  classifier is trained with cross-entropy, and that r = 1 − p_ErrP.
- Differences found: titles only. The robot task is described as a Kinova Gen2 pick-and-place in the
  short paper (the README of this project calls it reaching; this does not affect the EEG stage).
- Not specified in either version: epoch start/end, event code, handling of S5 human-error trials,
  reference scheme, filter design, channel selection, artifact handling, normalisation, all EEGNet
  hyperparameters, class weighting, and the trial-to-action mapping.

## Data and labels

| setting | value | source |
|---|---|---|
| Dataset | ErrP-HRI, commit `4bc3e524…`, robot condition | D (task spec) |
| Condition check | from the EEGLAB `comments` field (`*_robot.vhdr` vs `*_shooter.vhdr`); `robot/s12_cursor.set` excluded | D (files) |
| Participants | all 12 (s02–s13) usable | D (files) |
| Labels | S4 → 0 non-error, S6 → 1 error | D (documentation p. 3 + per-block rate evidence, see data_audit.md §4) |
| S5 human-error trials | excluded (178 trials) | C |
| Epoch origin | FB1/FB2/FB3 marker | C, informed by D (see data_audit.md §5) |
| Window | [0, 2) s after FB | P (2 s, aligned to feedback onset); start time C |
| Baseline correction | none | C |
| Trial exclusion | only S5 (label policy); no amplitude rejection | C |

## Preprocessing (shared by all models)

| setting | value | source |
|---|---|---|
| Sampling rate | 256 Hz as released (no resampling for EEGNet) | P, D (files are already 256 Hz) |
| Band-pass | 1–20 Hz, MNE zero-phase FIR (firwin, Hamming, default transitions), applied to the continuous recording before epoching | P (band); design C |
| Re-referencing | none added: files are already referenced to linked mastoids (TP9/TP10 removed) | D; re-referencing again would be redundant |
| Channels | 27 scalp EEG in a fixed order; EOG1–3 dropped | C |
| Units | microvolts | D |
| CBraMod view | same filtered windows resampled 256 → 200 Hz (`resample_poly(25, 32)`), reshaped to 2 patches × 200 samples | M (200 Hz, 1-s patches of 200 samples) |
| EEGNet amplitude scaling | divide by one scalar (std of all training-split samples, in µV); fitted on train only, applied unchanged to val/test | C |
| CBraMod amplitude scaling | µV / 100 | M (official dataset loaders return `data / 100`) |

Comparison factors that differ between the two models (documented, not removed): CBraMod was
pretrained on 0.3–75 Hz TUEG data at 200 Hz, while our shared input is band-limited to 1–20 Hz;
EEGNet sees 256 Hz samples and CBraMod 200 Hz samples of the same physical window; the amplitude
scalings differ as listed.

## Splits

| setting | value | source |
|---|---|---|
| Protocol | participant-level train / validation / test (not LOSO) | C (project proposal; task spec §2, §4) |
| Sizes | 7 train / 2 validation / 3 test participants | C (task spec default) |
| Procedure | sorted eligible IDs, `numpy.random.default_rng(42).permutation`, first 3 test, next 2 validation | C (task spec) |
| Result | test s02, s08, s09; validation s11, s13; train s03–s07, s10, s12 | — |

A first split file was generated with empty lists because a bug in the marker-consistency rule excluded
all S6 trials. It was deleted, the rule was fixed (data_audit.md §4) and the split was generated once
from the corrected data. No model had been trained on either version.

## Models

| setting | EEGNet | CBraMod | source |
|---|---|---|---|
| Implementation | PyTorch port of `EEGModels.EEGNet` (arl-eegmodels `4a512e50…`): same layer order, 'same' padding, max-norm constraints, Keras BN constants | official code vendored unmodified (CBraMod `b9e96100…`, only an import path changed) | M |
| Initialisation | from scratch (Glorot uniform, like Keras) | official weights `weighting666/CBraMod` rev `500543c7…`, `pretrained_weights.pth` SHA-256 `0792cb80…`; strict load | M |
| Architecture | F1 = 8, D = 2, F2 = 16, kernLength = 128 (half the sampling rate, as the reference recommends), dropout 0.5 (reference value for cross-subject), norm rate 0.25 | 12 layers, d_model 200, official `all_patch_reps` head (27·2·200 → 400 → 200 → 2), dropout 0.1 | M (reference defaults), C (kernLength rule applied at 256 Hz) |
| Fine-tuning | — | full model (backbone + head) | task spec default |
| Loss | cross-entropy, inverse-frequency class weights from the training split | same | P (cross-entropy); weights C |
| Label smoothing | 0 | 0 (official default 0.1 changed to match EEGNet) | C |
| Optimiser | Adam, lr 1e-3, no weight decay | AdamW, wd 0.05, backbone lr 1e-4, head lr 5e-4 = 1e-3·√(64/256) | M (CBraMod official defaults); C (EEGNet: standard Adam) |
| Schedule | constant | cosine to 1e-6 over max epochs | M (CBraMod) |
| Batch size | 64 | 64 | C / M |
| Gradient clipping | none | 1.0 | M (CBraMod) |
| Epoch cap / patience | 300 / 40 | 50 / 15 | C |
| Search budget | 2 candidates: lr ∈ {1e-3, 3e-4} | 2 candidates: backbone lr ∈ {1e-4, 3e-5} | C (equal budget) |
| Model selection | best validation AUROC epoch → `best.ckpt`; among candidates, highest best-validation AUROC (ties → first listed) | same | C |
| Decoder seed (formal) | 42 | 42 | task spec |
| Precision | fp32, cuDNN deterministic | fp32, cuDNN deterministic | C |

Class order: logit index 0 = non-error, index 1 = error; P(error) = softmax(logits)[1] = sigmoid(l1 − l0).

## Calibration, thresholds and evaluation

| setting | value | source |
|---|---|---|
| Calibration | temperature scaling on the error-logit margin, fitted on validation only by NLL minimisation (log T ∈ [log 0.05, log 20]); identity if the fit fails | task spec default |
| Threshold | P(error) ≥ 0.5 for the main balanced accuracy; Youden-optimal threshold on calibrated validation probabilities reported separately | task spec |
| Metrics | AUROC, balanced accuracy, Brier (raw and calibrated), NLL, ECE (10 bins), confusion matrix, counts | task spec |
| Aggregation | pooled test, per test participant, unweighted participant macro-average | task spec |
| Test set use | final evaluation and export only | task spec |

Note on class weighting and calibration: inverse-frequency weights train the networks as if the
classes were balanced, so raw probabilities are shifted towards P(error) = 0.5 relative to the true
35% error rate. Temperature scaling rescales the margin but does not add a bias term, so it can only
partly correct this; the remaining shift is visible in the Brier score and ECE. Because T > 0
preserves the sign of the margin, calibration does not change balanced accuracy at 0.5 or AUROC.

## Additional runs (reported separately)

- Extra decoder seeds 43–46 for the selected configuration of each main model, after the formal seed-42
  runs and selection are complete. They do not enter the main feedback library.
- Supplementary baselines with the same splits, trials, windows, loss, class weighting, selection rule
  and calibration (one run each, seed 42, no tuning):
  xDAWN + Riemannian tangent space + logistic regression (pyriemann);
  from scratch: EEG-Inception-ERP, EEGNeX, CTNet, EEG-Conformer (braindecode 1.8.1);
  pretrained and fully fine-tuned: LaBraM (`braindecode/labram-pretrained` rev `0563b6c6…`; temporal
  embedding truncated from 16 to 3 positions, channel position embeddings selected by name) and EEGPT
  (`braindecode/eegpt-pretrained` rev `e41cb3ae…`, pretrained at 250 Hz, so it gets the same window resampled
  to 250 Hz; amplitude in millivolts is a project choice that was not checked against the EEGPT paper).
  REVE was not used because its weights are gated.
