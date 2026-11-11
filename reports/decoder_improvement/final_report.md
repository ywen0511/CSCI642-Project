# CBraMod / EEGNet improvement study: final report

Repository `ywen0511/CSCI642-Project`, execution branch `decoder-improvement-v1-experiments` (from the planning commit
`c459bd5` of `decoder-improvement-v1`). Date 2026-11-11. Assignment: `DECODER_IMPROVEMENT_HANDOFF.md`; numerical plan:
`configs/decoder_improvement/study.json`. Detailed tables: [`results.md`](results.md) (all copied from
[`results/generated/results_tables.md`](results/generated/results_tables.md); value-to-source map
[`results/generated/number_audit.json`](results/generated/number_audit.json)). Decisions with timestamps:
[`decisions.jsonl`](decisions.jsonl); committed decision files: [`decisions/`](decisions/); plans: [`plans/`](plans/).

## 1. Question and answer

Can a smaller classification head, conservative fine-tuning, a different input band or a different loss improve the
pretrained CBraMod decoder on ErrP-HRI (robot condition), and how does a comparably searched EEGNet fare?

**No tested change established an improvement on held-out participants.** Train-only cross-validation selected a
CBraMod with the historical flatten head on a *frozen* backbone, 1-20 Hz input and unweighted cross-entropy
(five-seed CV score 0.578 vs 0.567 for the historical CBraMod recipe). On the previously evaluated test participants
the selected CBraMod (pooled AUROC 0.564 ± 0.005 over five seeds) and the controlled refit of the historical recipe
(0.563 ± 0.006) were indistinguishable; the paired difference was -0.003 AUROC and favoured the candidate in only 5
of 15 participant-seed pairs. EEGNet's own search (learning rate x dropout, then band and loss) returned the
historical recipe, which remained the better decoder (test AUROC 0.599 ± 0.014). The selected affine calibration
improved probability quality for both models (test Brier 0.227 and 0.223 vs 0.240 with temperature scaling) but, as a
monotone transform, not ranking. This is a complete negative result for decoding improvement.

## 2. What is being compared (keep these apart)

| quantity | where | what it is |
|---|---|---|
| historical archived results | `reports/eeg_results.md`, Release `eeg-stage1-20261101-bccbb8a` | original validation-selected runs (EEGNet seed-42 test AUROC 0.637, five-seed 0.613 ± 0.016; CBraMod 0.573, 0.568 ± 0.007); not re-run here |
| train-only CV selection | Stages A-E, `decisions/stage_*.json` | 7 training participants, 3 fixed participant folds, seeds 42-44, participant-macro AUROC |
| five-seed confirmation | Stage F, `decisions/freeze.json` | same folds, seeds 42-46, no reselection |
| controlled new reference refits | Stage G, role `reference` | the historical CBraMod recipe retrained exactly like the candidate (all 7 training participants, fixed epochs, 5 seeds); not a reproduction of the historical validation-selected runs |
| selected candidate results | Stage G, role `selected` | frozen recipes, fixed epochs, 5 seeds |
| calibration-development scores | `decisions/calibration.json` | s11/s13 were used to choose and fit calibrators; their probabilities are not unbiased held-out quality |
| previously evaluated outer test | `results/test/` | s02, s08, s09 were already inspected in the historical study: a follow-up evaluation, not a new untouched confirmatory test |

## 3. Data, splits and leakage controls

ErrP-HRI commit `4bc3e524`, robot condition, S4 = 0 / S6 = 1, S5 excluded, `s12_cursor.set` excluded by content;
5,822 trials (3,767 / 2,055) from 12 participants; 27 EEG channels, linked-mastoid reference as released,
`[0, 2)` s after the feedback marker. Outer split unchanged (train s03-s07, s10, s12: 3,434 trials; validation
s11, s13: 959; test s02, s08, s09: 1,429). All architecture / learning-rate / band / loss choices used only the seven
training participants with fixed folds cv0 {s03, s06}, cv1 {s04, s10}, cv2 {s05, s07, s12}; scalers and class
weights were fitted on each inner-training subset (EEGNet global SD per fold; CBraMod fixed /100). Outer validation was
used only for calibration after the freeze; test predictions were computed once, after the calibration decision was
committed (`f8031ec`), and every test-margin file records the calibration file hash it was produced under.

Preprocessing: three bands (1-20, 0.3-40, 0.3-75 Hz) re-filtered from the original continuous recordings with the
historical zero-phase FIR (MNE firwin/hamming; transition bands written explicitly = MNE 'auto'; no notch). The new
1-20 Hz arrays are byte-identical to the historical `prep_v1`; a fresh acquisition into an empty directory reproduced
them byte for byte. The 0.3-75 Hz band is not a reproduction of CBraMod's pretraining preprocessing (no notch, ErrP
windows), only a broader input.

## 4. Implementation checks (diagnostic gate, all passed)

Official weights loaded with no missing, unexpected or shape-mismatched keys (211 tensors), reconstruction head removed;
backbone 4,883,800 parameters; heads 4,401,002 (historical flatten MLP) and 10,802 (temporal-mean linear, mean over
the two time patches only, channels kept). Trainable-parameter lists were saved for every adaptation mode (frozen: head
only; last2: `backbone.encoder.layers.10/11` + head; full: whole backbone except the official fixed `mask_encoding`);
frozen tensors were byte-identical after an optimizer step and frozen modules stayed in eval mode; small training
subsets were overfit; reload reproduced logits exactly and an interrupted-and-resumed fit was bitwise identical to an
uninterrupted one. Details: [`preflight.md`](preflight.md), [`diagnostics/diagnostics.json`](diagnostics/diagnostics.json).

## 5. Study outcomes by stage

- **Stage A (CBraMod head x adaptation, 7 candidates, 63 fits).** CV scores 0.555-0.579; frozen flatten head 0.579,
  last-two-blocks flatten 0.575, historical full fine-tuning 0.568, temporal-mean heads 0.555-0.567. The small
  temporal-mean head did not help in any adaptation mode. Frozen-backbone models peaked within ~3 epochs.
- **Stage B (backbone LR).** Skipped: the Stage A winner has a frozen backbone (handoff rule).
- **Stage C (EEGNet, 4 candidates, 36 fits).** Historical lr 1e-3 / dropout 0.5 won (0.612); 0.594-0.609 otherwise.
- **Stage D (bands, 36 new fits).** 1-20 Hz kept for both: CBraMod 0.579 vs 0.567 (0.3-40) and 0.567 (0.3-75);
  EEGNet 0.612 vs 0.603 and 0.597.
- **Stage E (loss, 18 new fits).** CBraMod: unweighted CE 0.5794 vs weighted 0.5787 (above the 1e-6 tie tolerance,
  so unweighted was selected; it also had lower CV Brier, 0.254 vs 0.278). EEGNet: weighted CE kept (0.6118 vs 0.6106).
- **Stage F (five-seed confirmation, 18 new fits).** CBraMod selected 0.578 (screening seeds 0.579, new seeds 0.577)
  vs reference 0.567; paired +0.012, positive in 19 of 35 participant-seed pairs. EEGNet selection = reference.
- **Freeze.** Final epoch counts (ceil of the median one-based CV best epoch over 15 fits): CBraMod selected 5,
  CBraMod reference 17, EEGNet 47; cosine horizon kept at 50 epochs for CBraMod.
- **Stage G (15 refits on all 7 training participants).** Then calibration on s11/s13: cross-fitting selected the
  positive affine calibrator for both architectures (held-out-participant NLL 0.639 vs temperature 0.660 and
  identity 0.744 for CBraMod; 0.641 vs 0.679 and 0.684 for EEGNet).

Full tables, including per-participant and per-seed values and all probability metrics: [`results.md`](results.md).

## 6. Test results (s02, s08, s09; mean ± sample SD over five final refits)

| recipe | pooled AUROC | participant-macro AUROC | Brier: temperature / selected calibration | bal. acc @0.5: raw / selected calibration |
|---|---|---|---|---|
| CBraMod selected (frozen flatten head, 1-20 Hz, unweighted CE) | 0.564 ± 0.005 | 0.562 ± 0.006 | 0.240 / 0.227 | 0.529 / 0.513 |
| CBraMod reference (historical recipe, controlled refit) | 0.563 ± 0.006 | 0.566 ± 0.006 | 0.244 / (temperature) | 0.544 |
| EEGNet (selected = historical recipe) | 0.599 ± 0.014 | 0.601 ± 0.015 | 0.240 / 0.223 | 0.571 / 0.522 |

Per participant, the selected CBraMod gained on s02 (0.602 vs 0.547) and lost on s09 (0.565 vs 0.619). The seed SD
describes decoder-training variability on the same three people, not population uncertainty; trials are not
independent participants. With three test participants no population-level significance, equivalence or causal
claim is made. Balanced accuracy at a fixed 0.5 threshold fell with the selected calibration because the fitted
offsets move probabilities toward the 35 % error prevalence; this is a threshold effect, not a ranking change.

## 7. Why the changes did not help (interpretation, not causal proof)

The CV gain of the selected CBraMod was small (+0.012), concentrated in some participants, and not reproduced on the
test participants; with seven CV participants and three test participants, differences of this size are within
participant-to-participant variation. Fine-tuning more of the backbone did not help and freezing it mostly made
training stop earlier (best epoch ~3-5), consistent with a pretrained representation that is not well matched to
this short, low-amplitude ErrP window, but the study cannot separate that from overfitting of the large head. The
stages changed several settings sequentially (adaptation, then loss), so the selected recipe's small CV advantage
cannot be attributed to one change. A broader input band did not help either model.

## 8. Limitations and departures

- Sequential search (A -> B/C -> D -> E) with greedy carry-forward; not a global optimum. Unequal search budgets:
  CBraMod 7 + 2 + 1 candidates vs EEGNet 4 + 2 + 1; this is an improvement investigation, not an equal-budget
  leaderboard.
- Seven-participant CV and three-participant test; the test participants were already evaluated historically.
- The calibration participants s11/s13 were used to choose and fit calibrators; their outputs are
  calibration-development probabilities.
- Environment: a new environment from the exact historical lock; one install-order fix (NCCL library) recorded in
  `preflight.md`. GPU host gc176 instead of gc175 (identical A100 model; gc175 GPUs were reserved by another job).
- Controlled refits differ from the historical runs by design (fixed epochs on all seven training participants, no
  validation-based checkpoint selection), so they are reported separately from the historical results.
- No RL experiment was run with the new probabilities; no new robot-performance claim follows from this study.

## 9. Run accounting and compute

Distinct formal fits: **186** of the 215 allowed (Stage A 63, C 36, D 36 new + 18 reused, E 18 new + 18 reused,
F 18 new + 27 reused, G 15; B skipped). Every fit completed on the first attempt (0 failures, 0 resumed attempts;
ledger `<PROJECT_ROOT>/ledger/jobs.jsonl`, published in the Release evidence). Compute: 20 scheduled one-GPU jobs
(19 stage shards + the diagnostic gate), 2.66 GPU-hours of job time (2.22 h of pure training) on NVIDIA
A100-SXM4-40GB (gc176), plus two short interactive GPU jobs for final-model margins, and CPU jobs for preparation,
audit, packaging and verification.

## 10. Historical delivery

See [`historical_delivery_audit.md`](historical_delivery_audit.md). All historical Releases were complete; the RL v2
final (75 runs) and validation w010 (33 runs) suites had their periodic snapshots on the filer only, and are now
published as complete recovery Releases.

## 11. Feedback export

New seed-42 probabilities for the selected candidates: [`rl_feedback_handoff.md`](rl_feedback_handoff.md),
`configs/decoder_improvement/rl_feedback_vnext.json`. The historical `configs/rl_feedback.json` and its tables are
unchanged.

## 12. Artifacts, reproduction and verification

ARTIFACTS_PLACEHOLDER
